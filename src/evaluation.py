"""
evaluation.py
-------------
Full factorial benchmark runner.

Factors:
    Model       : lgbm | xgboost | single_mlp | deep_ensemble
    Calibrator  : none | temp | logistic | isotonic
    Seed        : 5 seeds (0–4) — minimum for stable median on tabular data

Deep ensemble configuration:
    M=5 members, random initialisation only (no bootstrap).
    Follows Lakshminarayanan et al. (2017) and the current tabular
    benchmark standard (TabM, ICLR 2025).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.calibration import build_calibrator, CALIBRATOR_LABELS
from src.datasets import load_task, split_dataset
from src.metrics import evaluate_all
from src.models import LightGBMModel, XGBoostModel, DeepEnsemble, SingleMLP, get_device

logger = logging.getLogger(__name__)

# ── Shared MLP / Ensemble config ─────────────────────────────────────────────
MLP_HIDDEN_DIMS = (256, 128)
MLP_EPOCHS      = 200
MLP_LR          = 1e-3
MLP_BATCH_SIZE  = 256
MLP_PATIENCE    = 20
ENSEMBLE_SIZE   = 5       # M=5 — standard (Lakshminarayanan et al.; TabM)

# ── Experiment config ─────────────────────────────────────────────────────────
SEEDS            = [0, 1, 2, 3, 4]     # 5 seeds — stable median for tabular data
CALIBRATOR_NAMES = ["none", "temp", "logistic", "isotonic"]


# ─────────────────────────────────────────────────────────────────────────────
# Single experimental cell
# ─────────────────────────────────────────────────────────────────────────────

def run_single_experiment(
    split: Dict[str, Any],
    model_name: str,
    calibrator_name: str,
    seed: int,
    device=None,
    save_probs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    One (dataset × model × calibrator × seed) cell.

    1. Train model on X_train, early-stop on X_val_es
    2. Fit calibrator on X_val_cal predictions (never seen during training)
    3. Evaluate calibrated predictions on X_test
    """
    if device is None:
        device = get_device()

    X_train   = split["X_train"];   y_train   = split["y_train"]
    X_val_es  = split["X_val_es"];  y_val_es  = split["y_val_es"]
    X_val_cal = split["X_val_cal"]; y_val_cal = split["y_val_cal"]
    X_test    = split["X_test"];    y_test    = split["y_test"]
    n_classes  = split["n_classes"]
    n_features = split["n_features"]

    t0 = time.time()

    if model_name == "lgbm":
        model = LightGBMModel(n_classes=n_classes)
        model.fit(X_train, y_train, X_val_es, y_val_es)

    elif model_name == "xgboost":
        model = XGBoostModel(n_classes=n_classes)
        model.fit(X_train, y_train, X_val_es, y_val_es)

    elif model_name == "single_mlp":
        model = SingleMLP(
            input_dim=n_features, n_classes=n_classes,
            hidden_dims=MLP_HIDDEN_DIMS, lr=MLP_LR,
            epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
            patience=MLP_PATIENCE, device=device,
        )
        model.fit(X_train, y_train, X_val_es, y_val_es, seed=seed)

    elif model_name == "deep_ensemble":
        model = DeepEnsemble(
            n_members=ENSEMBLE_SIZE,
            input_dim=n_features, n_classes=n_classes,
            hidden_dims=MLP_HIDDEN_DIMS, lr=MLP_LR,
            epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE,
            patience=MLP_PATIENCE, device=device,
        )
        model.fit(X_train, y_train, X_val_es, y_val_es, base_seed=seed)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_time = time.time() - t0

    probs_val_cal = model.predict_proba(X_val_cal)
    probs_test    = model.predict_proba(X_test)

    calibrator = build_calibrator(calibrator_name)
    calibrator.fit(probs_val_cal, y_val_cal)
    probs_cal = calibrator.calibrate(probs_test)

    # Optionally save probability vectors for reliability diagrams
    if save_probs_dir is not None:
        save_probs_dir.mkdir(parents=True, exist_ok=True)
        fname = (f"{split['task_id']}__{model_name}__"
                 f"{calibrator_name}__seed{seed}.npz")
        np.savez_compressed(
            save_probs_dir / fname,
            probs_cal=probs_cal, y_test=y_test,
        )

    metrics     = evaluate_all(probs_cal,  y_test)
    raw_metrics = evaluate_all(probs_test, y_test)

    return {
        "task_id":          split["task_id"],
        "dataset_name":     split["name"],
        "n_samples":        split["n_samples"],
        "n_train":          split["n_train"],
        "n_val_es":         split["n_val_es"],
        "n_val_cal":        split["n_val_cal"],
        "n_test":           split["n_test"],
        "n_features":       n_features,
        "n_classes":        n_classes,
        "seed":             seed,
        "model":            model_name,
        "calibrator":       calibrator_name,
        "calibrator_label": CALIBRATOR_LABELS.get(calibrator_name, calibrator_name),
        **{f"cal_{k}": v for k, v in metrics.items()},
        **{f"raw_{k}": v for k, v in raw_metrics.items()},
        "train_time_s":     round(train_time, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    task_ids: List[int],
    model_names: Optional[List[str]] = None,
    calibrator_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    device=None,
    skip_on_error: bool = True,
    save_probs_dir: Optional[Path] = None,
    done_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full factorial experiment.

    Parameters
    ----------
    done_keys : set of (task_id, model, calibrator, seed) tuples already
                completed — used to skip rows when resuming from checkpoint.
    """
    if model_names      is None: model_names      = ["lgbm", "xgboost", "single_mlp", "deep_ensemble"]
    if calibrator_names is None: calibrator_names = CALIBRATOR_NAMES
    if seeds            is None: seeds            = SEEDS
    if device           is None: device           = get_device()
    if done_keys        is None: done_keys        = set()

    results: List[Dict[str, Any]] = []
    n_total = len(task_ids) * len(model_names) * len(calibrator_names) * len(seeds)
    logger.info(
        f"Benchmark: {len(task_ids)} tasks × {len(model_names)} models × "
        f"{len(calibrator_names)} calibrators × {len(seeds)} seeds = {n_total} cells."
    )
    if done_keys:
        logger.info(f"Resuming: {len(done_keys)} cells already done.")

    run_idx = 0
    for task_id in task_ids:
        data = load_task(task_id)
        if data is None:
            logger.warning(f"Task {task_id}: skipped.")
            continue

        for seed in seeds:
            split = split_dataset(data, seed=seed)

            for model_name in model_names:
                for cal_name in calibrator_names:
                    run_idx += 1
                    key = (task_id, model_name, cal_name, seed)
                    if key in done_keys:
                        continue

                    logger.info(
                        f"[{run_idx}/{n_total}] "
                        f"{data['name'][:25]} | seed={seed} | "
                        f"{model_name}/{cal_name}"
                    )
                    try:
                        result = run_single_experiment(
                            split, model_name, cal_name, seed, device,
                            save_probs_dir=save_probs_dir,
                        )
                        results.append(result)
                        logger.info(
                            f"  NLL={result['cal_nll']:.4f} "
                            f"ECE={result['cal_ece_mean']:.4f} "
                            f"Brier={result['cal_brier_score']:.4f} "
                            f"Acc={result['cal_accuracy']:.4f} "
                            f"({result['train_time_s']:.1f}s)"
                        )
                    except Exception as exc:
                        if skip_on_error:
                            logger.error(f"  FAILED: {exc}")
                        else:
                            raise

    logger.info(f"Done. {len(results)} new results.")
    return results
