"""
datasets.py
-----------
OpenML CC-18 dataset loading, preprocessing, and splitting.

Split protocol — standard 60/20/20 following Guo et al. (2017) and
Gorishniy et al. (2021, 2024):
    - 54% train      : model weight learning
    -  6% val_es     : early stopping only (10% of training fold, internal)
    - 20% val_cal    : calibration fitting only — strictly held out from model
    - 20% test       : final evaluation — strictly held out throughout

Described as 60/20/20 in the paper because val_es is carved internally
from the training fold and does not constitute a separate experimental
partition. This matches the standard tabular DL benchmark protocol while
preserving the val_cal / val_es separation that prevents calibration leakage.

References
----------
Guo et al. (2017) — temperature scaling, separate calibration set
Gorishniy et al. (2021, 2024) — 60/20/20 tabular benchmark standard
Kadra et al. (2021) — 60/20/20, same standardisation pipeline
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

CC18_STUDY_ID       = 99
MIN_SAMPLES_PER_CLASS = 20


# ─────────────────────────────────────────────────────────────────────────────
# Task listing
# ─────────────────────────────────────────────────────────────────────────────

def get_cc18_task_ids() -> List[int]:
    """Fetch all task IDs from the OpenML-CC18 benchmark suite (study ID 99)."""
    logger.info("Fetching OpenML CC-18 task list...")
    benchmark = openml.study.get_suite(CC18_STUDY_ID)
    task_ids  = list(benchmark.tasks)
    logger.info(f"Found {len(task_ids)} tasks in CC-18.")
    return task_ids


def categorise_size(n_samples: int) -> str:
    """Assign a dataset to a size regime."""
    if n_samples < 1_000:
        return "small"
    if n_samples < 10_000:
        return "medium"
    return "large"


def filter_datasets_by_size(
    task_ids: List[int],
    max_per_regime: int = 12,
    cache_dir: str = "results",
) -> List[int]:
    """
    Stratified dataset selection: up to max_per_regime tasks per size regime.

    Metadata is cached in dataset_sizes_cache.json.
    After selection a full dataset manifest is written to dataset_manifest.csv
    and dataset_manifest.json so you always know exactly which datasets were
    used, their sizes, feature counts, class counts, and size regime.

    Returns
    -------
    List of selected task IDs (sorted within each regime for reproducibility).
    """
    cache_path = Path(cache_dir) / "dataset_sizes_cache.json"

    # ── Load or build size cache ──────────────────────────────────────────────
    if cache_path.exists():
        with open(cache_path) as f:
            raw_cache = json.load(f)
        # Support both old format {tid: n_samples} and new {tid: {meta}}
        size_cache: Dict[int, dict] = {}
        for k, v in raw_cache.items():
            tid = int(k)
            if isinstance(v, dict):
                size_cache[tid] = v
            else:
                size_cache[tid] = {"n_samples": int(v), "name": "", "n_features": 0, "n_classes": 0}
        logger.info(f"Loaded size cache ({len(size_cache)} entries).")
    else:
        size_cache = {}

    needs_fetch = [t for t in task_ids if t not in size_cache]
    if needs_fetch:
        logger.info(f"Fetching metadata for {len(needs_fetch)} tasks...")
        for tid in needs_fetch:
            try:
                task    = openml.tasks.get_task(tid)
                dataset = task.get_dataset()
                X, y, _, _ = dataset.get_data(
                    dataset_format="dataframe",
                    target=dataset.default_target_attribute,
                )
                if X is not None and y is not None:
                    from sklearn.preprocessing import LabelEncoder
                    n_classes = len(LabelEncoder().fit(y.astype(str)).classes_)
                    size_cache[tid] = {
                        "n_samples":  int(len(X)),
                        "n_features": int(X.shape[1]),
                        "n_classes":  int(n_classes),
                        "name":       dataset.name,
                    }
                else:
                    size_cache[tid] = {"n_samples": 0, "n_features": 0, "n_classes": 0, "name": ""}
            except Exception as exc:
                logger.debug(f"Task {tid} metadata fetch failed: {exc}")
                size_cache[tid] = {"n_samples": 0, "n_features": 0, "n_classes": 0, "name": ""}

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in size_cache.items()}, f, indent=2)

    # ── Stratified selection ──────────────────────────────────────────────────
    def regime(tid): return categorise_size(size_cache.get(tid, {}).get("n_samples", 0))

    small  = sorted([t for t in task_ids if regime(t) == "small"  and size_cache.get(t, {}).get("n_samples", 0) > 0])
    medium = sorted([t for t in task_ids if regime(t) == "medium" and size_cache.get(t, {}).get("n_samples", 0) > 0])
    large  = sorted([t for t in task_ids if regime(t) == "large"  and size_cache.get(t, {}).get("n_samples", 0) > 0])

    rng = np.random.RandomState(0)

    def sample(lst, n):
        if len(lst) <= n:
            return lst
        return sorted(rng.choice(lst, size=n, replace=False).tolist())

    sel_small  = sample(small,  max_per_regime)
    sel_medium = sample(medium, max_per_regime)
    sel_large  = sample(large,  max_per_regime)
    selected   = sel_small + sel_medium + sel_large

    logger.info(
        f"Selected {len(selected)} tasks: "
        f"{len(sel_small)} small (n<1k) / "
        f"{len(sel_medium)} medium (1k≤n<10k) / "
        f"{len(sel_large)} large (n≥10k)."
    )

    # ── Save dataset manifest ─────────────────────────────────────────────────
    manifest_rows = []
    for tid in selected:
        meta = size_cache.get(tid, {})
        n    = meta.get("n_samples", 0)
        manifest_rows.append({
            "task_id":    tid,
            "name":       meta.get("name", ""),
            "n_samples":  n,
            "n_features": meta.get("n_features", 0),
            "n_classes":  meta.get("n_classes", 0),
            "size_regime": categorise_size(n),
        })

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["size_regime", "n_samples"])
    manifest_csv  = Path(cache_dir) / "dataset_manifest.csv"
    manifest_json = Path(cache_dir) / "dataset_manifest.json"
    manifest_df.to_csv(manifest_csv, index=False)
    manifest_df.to_json(manifest_json, orient="records", indent=2)

    logger.info(f"Dataset manifest saved → {manifest_csv}")
    logger.info("\n" + manifest_df.to_string(index=False))

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_task(task_id: int) -> Optional[Dict]:
    """Load, preprocess, and quality-filter one OpenML classification task."""
    try:
        task        = openml.tasks.get_task(task_id)
        dataset     = task.get_dataset()
        target_attr = dataset.default_target_attribute

        X, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=target_attr,
        )

        if X is None or y is None:
            logger.warning(f"Task {task_id}: empty data.")
            return None

        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        n_classes = len(le_target.classes_)

        if n_classes < 2:
            logger.warning(f"Task {task_id}: single class.")
            return None

        X = X.copy()
        for col in X.columns:
            if X[col].dtype == "object" or hasattr(X[col], "cat"):
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        X = X.dropna(axis=1, how="all")
        X = X.loc[:, X.nunique() > 1]

        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        X_arr      = X.values.astype(np.float32)
        y_arr      = y_encoded.astype(np.int64)
        n_samples, n_features = X_arr.shape

        counts = np.bincount(y_arr)
        if counts.min() < MIN_SAMPLES_PER_CLASS:
            logger.warning(f"Task {task_id}: class too small.")
            return None

        return {
            "task_id":       task_id,
            "name":          dataset.name,
            "X":             X_arr,
            "y":             y_arr,
            "n_classes":     n_classes,
            "n_features":    n_features,
            "n_samples":     n_samples,
            "feature_names": list(X.columns),
            "class_names":   list(le_target.classes_),
        }

    except Exception as exc:
        logger.warning(f"Task {task_id}: failed — {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    data: Dict,
    seed: int = 0,
    es_frac_of_train: float = 0.10,
) -> Dict:
    """
    Standard 60/20/20 stratified split (Guo et al. 2017; Gorishniy et al. 2021).

    The training fold (60%) is further split 90/10 internally to produce:
        X_train   (~54% of total) — model weight updates
        X_val_es  (~ 6% of total) — early stopping signal only

    The held-out 40% is split 50/50 into:
        X_val_cal (20% of total) — calibrator fitting; never seen by model
        X_test    (20% of total) — final evaluation; strictly held out

    The paper describes this as a 60/20/20 split because val_es is an
    internal implementation detail of model training, not a separate
    experimental partition. This matches the standard tabular benchmark
    protocol while preserving the calibration-leakage prevention that
    distinguishes this work from prior comparisons.

    Parameters
    ----------
    data             : dict from load_task()
    seed             : random seed (identical across all models for a given task)
    es_frac_of_train : fraction of the training fold reserved for early stopping
                       (default 0.10 → ~6% of total)
    """
    X, y = data["X"], data["y"]
    stratify = y if np.bincount(y).min() >= 5 else None

    # Step 1: 60% train fold vs 40% held-out
    X_train_fold, X_held, y_train_fold, y_held = train_test_split(
        X, y,
        test_size=0.40,
        random_state=seed,
        stratify=stratify,
    )

    # Step 2: held-out → 50/50 val_cal / test
    stratify_held = y_held if np.bincount(y_held).min() >= 2 else None
    X_val_cal, X_test, y_val_cal, y_test = train_test_split(
        X_held, y_held,
        test_size=0.50,
        random_state=seed,
        stratify=stratify_held,
    )

    # Step 3: training fold → (1-es_frac) train / es_frac val_es
    stratify_train = y_train_fold if np.bincount(y_train_fold).min() >= 2 else None
    X_train, X_val_es, y_train, y_val_es = train_test_split(
        X_train_fold, y_train_fold,
        test_size=es_frac_of_train,
        random_state=seed,
        stratify=stratify_train,
    )

    # Standardise — fit only on X_train
    scaler    = StandardScaler()
    X_train   = scaler.fit_transform(X_train).astype(np.float32)
    X_val_es  = scaler.transform(X_val_es).astype(np.float32)
    X_val_cal = scaler.transform(X_val_cal).astype(np.float32)
    X_test    = scaler.transform(X_test).astype(np.float32)

    return {
        **{k: v for k, v in data.items() if k not in ("X", "y")},
        "X_train":   X_train,
        "X_val_es":  X_val_es,
        "X_val_cal": X_val_cal,
        "X_test":    X_test,
        "y_train":   y_train,
        "y_val_es":  y_val_es,
        "y_val_cal": y_val_cal,
        "y_test":    y_test,
        # backward-compat aliases
        "X_val":     X_val_cal,
        "y_val":     y_val_cal,
        "seed":      seed,
        "n_train":   len(y_train),
        "n_val_es":  len(y_val_es),
        "n_val_cal": len(y_val_cal),
        "n_test":    len(y_test),
    }
