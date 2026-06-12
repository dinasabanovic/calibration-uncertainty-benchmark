"""
run_ood_diagnostics.py
----------------------
TabPFN, OOD, label noise, conformal, and post-hoc analyses for the major
revision.

Reviewer requests addressed
---------------------------
R1.2   OOD evaluation                 -> Gaussian noise on ALL 36 datasets
                                          x 3 sigmas x 8 models x 5 calibrators
R1.3   TabPFN v2 missing              -> Inference on the 24 compatible datasets
R1.3   Conformal prediction missing   -> Split-conformal APS at 1-alpha = 0.90
R1.6   Noisy-label robustness         -> Symmetric label noise eta in {0.1, 0.2}
                                          on a 6-dataset subset
R1.6 (partial) Imbalance behavior     -> Stratification by max/min class ratio
R1.8   Practical-significance         -> % rel. change + practical_sig flag (H7)
R1.9   Computational cost analysis    -> train_time_s + peak_gpu_mem_mb per cell
R2.4   Cost-function calibrator       -> Composite-score weight-simplex sweep

Sections (selectable with --section)
------------------------------------
    tabpfn        TabPFN v2 on its 24 compatible datasets
                  -> tabpfn_results.csv
    ood           Gaussian feature-shift on all 36 datasets
                  -> ood_results.csv  (+ ood_noise/task_*.npz)
    label_noise   Symmetric label noise on 6-dataset subset
                  -> label_noise_results.csv
    conformal     Split-conformal at coverage 0.90 from cached probs
                  -> conformal_results.csv
    postanalysis  Composite weight sweep, imbalance, cost, practical H7
                  -> computational_cost.csv, imbalance_*.csv,
                     composite_score_dominance.csv,
                     H7_with_practical_significance.csv
    all           Run every section in the order above (default)

Usage
-----
    python experiments/run_ood_diagnostics.py                       # all sections
    python experiments/run_ood_diagnostics.py --section ood         # just OOD
    python experiments/run_ood_diagnostics.py --section tabpfn      # just TabPFN

TabPFN requires `pip install tabpfn` and `export TABPFN_TOKEN=<your-token>`
before running. If TABPFN_TOKEN is unset, the tabpfn section is skipped
with a warning rather than raising.

Wall-clock budget on A100: ~5-6 h for `all`. Per-section:
    tabpfn       ~30 min      ood    ~4 h    label_noise   ~45 min
    conformal    <10 min      postanalysis  <5 min
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.calibration import CALIBRATOR_LABELS                # noqa: E402
from src.datasets import load_task, split_dataset            # noqa: E402
from src.metrics import evaluate_all                          # noqa: E402
from src.models import get_device                             # noqa: E402

from experiments.revision_helpers import (                    # noqa: E402
    CALIBRATOR_LABELS_EXTENDED,
    TabPFNModel,
    build_calibrator_extended,
    build_model,
    tabpfn_compatible,
)


# =============================================================================
# Shared config
# =============================================================================

OOD_MODELS      = ["lgbm", "xgboost", "catboost", "single_mlp", "mc_dropout",
                   "deep_ensemble_m3", "deep_ensemble", "deep_ensemble_m10"]
OOD_CALIBRATORS = ["none", "temp", "logistic", "isotonic", "dirichlet"]
OOD_SIGMAS      = [0.5, 1.0, 2.0]
OOD_SEED        = 0
NOISE_LEVELS    = [0.1, 0.2]
NOISE_SEED      = 0
CONFORMAL_ALPHA = 0.1
TABPFN_SEEDS    = [0, 1, 2, 3, 4]


# =============================================================================
# Section: TabPFN
# =============================================================================

def run_tabpfn(work_dir: Path, manifest: pd.DataFrame, device_str: str) -> None:
    """TabPFN v2 on the 24 compatible datasets, 5 seeds, 5 calibrators."""
    print(f"\n========== Section: TabPFN ==========")
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        print("tabpfn not installed. Install: pip install tabpfn")
        return
    if not os.environ.get("TABPFN_TOKEN"):
        print("TABPFN_TOKEN not set in environment. Export it and rerun, or "
              "skip the tabpfn section with --section ood (etc.).")
        return

    compatible = manifest[manifest.apply(tabpfn_compatible, axis=1)]
    print(f"TabPFN-compatible datasets: {len(compatible)} / {len(manifest)}")

    results, failures = [], []
    t0 = time.time()
    for i, task_id in enumerate(compatible["task_id"].tolist()):
        data = load_task(int(task_id))
        if data is None:
            failures.append({"task_id": int(task_id), "reason": "load failed"})
            continue
        print(f"\n[{i+1}/{len(compatible)}] {data['name']} n={data['n_samples']}")
        for seed in TABPFN_SEEDS:
            split = split_dataset(data, seed=seed)
            try:
                tt0 = time.time()
                m   = TabPFNModel(n_classes=split["n_classes"], device=device_str)
                m.fit(split["X_train"], split["y_train"])
                train_time = time.time() - tt0
                probs_vc = m.predict_proba(split["X_val_cal"])
                probs_te = m.predict_proba(split["X_test"])
            except Exception as exc:                           # noqa: BLE001
                print(f"  x seed={seed} TabPFN fit failed: {exc}")
                failures.append({"task_id": int(task_id), "seed": seed,
                                 "reason": str(exc)})
                continue

            for cal_name in OOD_CALIBRATORS:
                try:
                    cal = build_calibrator_extended(cal_name)
                    cal.fit(probs_vc, split["y_val_cal"])
                    probs_cal = cal.calibrate(probs_te)
                    metrics     = evaluate_all(probs_cal, split["y_test"])
                    raw_metrics = evaluate_all(probs_te,  split["y_test"])
                    results.append({
                        "task_id": int(task_id), "dataset_name": split["name"],
                        "n_samples": split["n_samples"], "n_train": split["n_train"],
                        "n_val_es": split["n_val_es"], "n_val_cal": split["n_val_cal"],
                        "n_test":   split["n_test"],  "n_features": split["n_features"],
                        "n_classes": split["n_classes"], "seed": seed,
                        "model": "tabpfn", "calibrator": cal_name,
                        "calibrator_label": CALIBRATOR_LABELS_EXTENDED.get(
                            cal_name, cal_name),
                        **{f"cal_{k}": v for k, v in metrics.items()},
                        **{f"raw_{k}": v for k, v in raw_metrics.items()},
                        "train_time_s": round(train_time, 2),
                    })
                except Exception as exc:                       # noqa: BLE001
                    print(f"  x seed={seed} {cal_name} cal failed: {exc}")
                    failures.append({"task_id": int(task_id), "seed": seed,
                                     "calibrator": cal_name, "reason": str(exc)})
        print(f"  cumulative rows: {len(results)} "
              f"({(time.time()-t0)/60:.1f} min elapsed)")

    print(f"\nTabPFN: {len(results)} rows, {len(failures)} failures")
    if results:
        pd.DataFrame(results).to_csv(work_dir / "tabpfn_results.csv", index=False)
        print(f"saved: {work_dir/'tabpfn_results.csv'}")
    if failures:
        pd.DataFrame(failures).to_csv(work_dir / "tabpfn_failures.csv", index=False)


# =============================================================================
# Section: OOD (Gaussian feature shift)
# =============================================================================

def _make_or_load_noises(task_id: int, n_test: int, n_feat: int,
                         sigmas: List[float], base_seed: int,
                         noise_dir: Path) -> tuple[dict, str]:
    """Per-task noise tensors, persisted to noise_dir/task_<id>.npz.

    Seed formula:
        seed = (task_id * 100003 + round(sigma * 1000) * 1009 + base_seed) mod 2^32
    Ensures noise is byte-identical across reruns, NumPy versions, partial
    resumes, and machine restarts.
    """
    noise_path = noise_dir / f"task_{int(task_id)}.npz"
    keys_needed = [f"sigma_{s}" for s in sigmas]
    if noise_path.exists():
        try:
            cached = np.load(noise_path)
            if all(k in cached.files for k in keys_needed):
                noises = {s: cached[f"sigma_{s}"].astype(np.float32) for s in sigmas}
                if all(noises[s].shape == (n_test, n_feat) for s in sigmas):
                    return noises, "loaded"
        except Exception:                                       # noqa: BLE001
            pass
    noises = {}
    for s in sigmas:
        seed = (int(task_id) * 100003
                + int(round(s * 1000)) * 1009
                + int(base_seed)) % (2**32)
        rng = np.random.RandomState(seed)
        noises[s] = (rng.randn(n_test, n_feat) * s).astype(np.float32)
    noise_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(noise_path,
                        **{f"sigma_{s}": noises[s] for s in sigmas})
    return noises, "generated"


def run_ood(work_dir: Path, manifest: pd.DataFrame, device) -> None:
    """Full OOD sweep: all 36 datasets x 8 models x 5 cals x 3 sigmas."""
    print(f"\n========== Section: OOD (Gaussian feature shift) ==========")
    noise_dir = work_dir / "ood_noise"
    noise_dir.mkdir(exist_ok=True)
    out_path  = work_dir / "ood_results.csv"
    partial   = work_dir / "ood_partial.csv"

    if partial.exists():
        prev = pd.read_csv(partial)
        results = prev.to_dict("records")
        done = {(int(r["task_id"]), r["model"], r["calibrator"], float(r["ood_sigma"]))
                for r in results}
        print(f"resumed: {len(results)} rows already done")
    else:
        results, done = [], set()
    failures = []
    t0 = time.time()

    task_ids = manifest["task_id"].tolist()
    for i, task_id in enumerate(task_ids):
        expected_per_task = len(OOD_MODELS) * len(OOD_CALIBRATORS) * len(OOD_SIGMAS)
        if sum(1 for k in done if k[0] == int(task_id)) >= expected_per_task:
            continue
        data = load_task(int(task_id))
        if data is None:
            failures.append({"task_id": int(task_id), "reason": "load failed"})
            continue
        print(f"\n[{i+1}/{len(task_ids)}] {data['name']} n={data['n_samples']}")
        split = split_dataset(data, seed=OOD_SEED)
        n_test, n_feat = split["X_test"].shape

        noises, origin = _make_or_load_noises(
            int(task_id), n_test, n_feat, OOD_SIGMAS, OOD_SEED, noise_dir,
        )
        print(f"  noise: {origin}")

        for model_name in OOD_MODELS:
            already = sum(1 for k in done if k[0] == int(task_id) and k[1] == model_name)
            if already >= len(OOD_CALIBRATORS) * len(OOD_SIGMAS):
                continue
            try:
                peak_mem_mb = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                model, fit_kwargs = build_model(
                    model_name, split["n_classes"], split["n_features"],
                    OOD_SEED, device,
                )
                mt0 = time.time()
                model.fit(split["X_train"], split["y_train"],
                          split["X_val_es"], split["y_val_es"], **fit_kwargs)
                train_time = time.time() - mt0

                probs_val_cal = model.predict_proba(split["X_val_cal"])

                # Pre-fit every calibrator on clean val_cal
                calibrators = {}
                for cal_name in OOD_CALIBRATORS:
                    cal = build_calibrator_extended(cal_name)
                    cal.fit(probs_val_cal, split["y_val_cal"])
                    calibrators[cal_name] = cal

                rows_for_backfill = []
                for sigma in OOD_SIGMAS:
                    X_shifted = split["X_test"] + noises[sigma]
                    probs_te_shifted = model.predict_proba(X_shifted)
                    for cal_name, cal in calibrators.items():
                        key = (int(task_id), model_name, cal_name, sigma)
                        if key in done:
                            continue
                        probs_cal   = cal.calibrate(probs_te_shifted)
                        metrics     = evaluate_all(probs_cal, split["y_test"])
                        raw_metrics = evaluate_all(probs_te_shifted, split["y_test"])
                        row = {
                            "task_id": int(task_id), "dataset_name": data["name"],
                            "n_samples": split["n_samples"],
                            "n_features": split["n_features"],
                            "n_classes":  split["n_classes"], "seed": OOD_SEED,
                            "model": model_name, "calibrator": cal_name,
                            "ood_sigma": sigma,
                            **{f"cal_{k}": v for k, v in metrics.items()},
                            **{f"raw_{k}": v for k, v in raw_metrics.items()},
                            "train_time_s": round(train_time, 2),
                            "peak_gpu_mem_mb": None,
                        }
                        results.append(row)
                        rows_for_backfill.append(row)
                        done.add(key)

                if torch.cuda.is_available():
                    peak_mem_mb = round(
                        torch.cuda.max_memory_allocated() / (1024 * 1024), 2
                    )
                    for row in rows_for_backfill:
                        row["peak_gpu_mem_mb"] = peak_mem_mb

            except Exception as exc:                            # noqa: BLE001
                print(f"  x {model_name} failed: {exc}")
                failures.append({
                    "task_id": int(task_id), "model": model_name,
                    "reason": str(exc),
                })

        pd.DataFrame(results).to_csv(partial, index=False)
        print(f"  {len(results)} rows so far; "
              f"{(time.time()-t0)/60:.1f} min elapsed")

    print(f"\nOOD: {len(results)} rows, {len(failures)} failures")
    expected = len(task_ids) * len(OOD_MODELS) * len(OOD_CALIBRATORS) * len(OOD_SIGMAS)
    print(f"expected: {expected} rows")
    if results:
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"saved: {out_path}")
        if partial.exists():
            partial.unlink()
    if failures:
        pd.DataFrame(failures).to_csv(work_dir / "ood_failures.csv", index=False)


# =============================================================================
# Section: Label noise
# =============================================================================

def _inject_label_noise(y: np.ndarray, eta: float, n_classes: int,
                        rng: np.random.RandomState) -> np.ndarray:
    """Symmetric label noise: with prob eta replace label with uniform random class."""
    y_noisy = y.copy()
    n = len(y)
    n_corrupt = int(eta * n)
    if n_corrupt == 0:
        return y_noisy
    corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)
    new_labels  = rng.randint(0, n_classes, size=n_corrupt)
    y_noisy[corrupt_idx] = new_labels
    return y_noisy


def run_label_noise(work_dir: Path, manifest: pd.DataFrame, device) -> None:
    """Symmetric label noise on a 6-dataset subset (2 small / 2 medium / 2 large),
    1 seed, 8 models, 5 calibrators, 2 eta values = 480 rows."""
    print(f"\n========== Section: Label noise ==========")
    # 6-dataset subset: smallest 2 per size regime
    noise_tasks = []
    for regime in ("small", "medium", "large"):
        cand = manifest[manifest.size_regime == regime].nsmallest(2, "n_samples")
        noise_tasks.extend(cand["task_id"].tolist())
    print(f"label-noise subset: {len(noise_tasks)} datasets")

    partial = work_dir / "label_noise_partial.csv"
    if partial.exists():
        prev = pd.read_csv(partial)
        results = prev.to_dict("records")
        done = {(int(r["task_id"]), r["model"], r["calibrator"],
                 float(r["label_noise_eta"])) for r in results}
        print(f"resumed: {len(results)} rows already done")
    else:
        results, done = [], set()
    failures = []
    t0 = time.time()

    for i, task_id in enumerate(noise_tasks):
        data = load_task(int(task_id))
        if data is None:
            failures.append({"task_id": int(task_id), "reason": "load failed"})
            continue
        print(f"\n[{i+1}/{len(noise_tasks)}] {data['name']} n={data['n_samples']}")
        split = split_dataset(data, seed=NOISE_SEED)

        for eta in NOISE_LEVELS:
            rng = np.random.RandomState(int(task_id) + int(eta * 100))
            y_train_noisy = _inject_label_noise(split["y_train"], eta,
                                                split["n_classes"], rng)

            for model_name in OOD_MODELS:
                if sum(1 for k in done if k[0] == int(task_id)
                       and k[1] == model_name
                       and abs(k[3] - eta) < 1e-6) >= len(OOD_CALIBRATORS):
                    continue
                try:
                    model, fit_kwargs = build_model(
                        model_name, split["n_classes"], split["n_features"],
                        NOISE_SEED, device,
                    )
                    mt0 = time.time()
                    model.fit(split["X_train"], y_train_noisy,
                              split["X_val_es"], split["y_val_es"], **fit_kwargs)
                    train_time = time.time() - mt0
                    # Calibrator fitted on CLEAN val_cal (realistic deployment)
                    probs_val_cal = model.predict_proba(split["X_val_cal"])
                    probs_test    = model.predict_proba(split["X_test"])

                    for cal_name in OOD_CALIBRATORS:
                        key = (int(task_id), model_name, cal_name, eta)
                        if key in done:
                            continue
                        cal = build_calibrator_extended(cal_name)
                        cal.fit(probs_val_cal, split["y_val_cal"])
                        probs_cal   = cal.calibrate(probs_test)
                        metrics     = evaluate_all(probs_cal, split["y_test"])
                        raw_metrics = evaluate_all(probs_test, split["y_test"])
                        results.append({
                            "task_id": int(task_id), "dataset_name": data["name"],
                            "n_samples": split["n_samples"],
                            "n_features": split["n_features"],
                            "n_classes":  split["n_classes"], "seed": NOISE_SEED,
                            "model": model_name, "calibrator": cal_name,
                            "label_noise_eta": eta,
                            **{f"cal_{k}": v for k, v in metrics.items()},
                            **{f"raw_{k}": v for k, v in raw_metrics.items()},
                            "train_time_s": round(train_time, 2),
                        })
                        done.add(key)
                except Exception as exc:                       # noqa: BLE001
                    print(f"  x {model_name} eta={eta} failed: {exc}")
                    failures.append({"task_id": int(task_id), "model": model_name,
                                     "eta": eta, "reason": str(exc)})

        pd.DataFrame(results).to_csv(partial, index=False)
        print(f"  {len(results)} rows so far; "
              f"{(time.time()-t0)/60:.1f} min elapsed")

    print(f"\nLabel noise: {len(results)} rows, {len(failures)} failures")
    if results:
        pd.DataFrame(results).to_csv(work_dir / "label_noise_results.csv",
                                      index=False)
        if partial.exists():
            partial.unlink()
        print(f"saved: {work_dir/'label_noise_results.csv'}")
    if failures:
        pd.DataFrame(failures).to_csv(
            work_dir / "label_noise_failures.csv", index=False,
        )


# =============================================================================
# Section: Conformal coverage from cached probs
# =============================================================================

def _aps_score(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    n            = probs.shape[0]
    sorted_idx   = np.argsort(-probs, axis=1)
    sorted_probs = -np.sort(-probs, axis=1)
    cum_probs    = np.cumsum(sorted_probs, axis=1)
    scores = np.zeros(n)
    for i in range(n):
        rank      = np.where(sorted_idx[i] == y[i])[0][0]
        scores[i] = cum_probs[i, rank]
    return scores


def _conformal_eval(probs_val, y_val, probs_test, y_test,
                    alpha: float) -> tuple[float, float]:
    n_val      = len(y_val)
    val_scores = _aps_score(probs_val, y_val)
    q_level    = min(np.ceil((n_val + 1) * (1 - alpha)) / n_val, 1.0)
    threshold  = np.quantile(val_scores, q_level, method="higher")
    n_test, K  = probs_test.shape
    sorted_idx   = np.argsort(-probs_test, axis=1)
    sorted_probs = -np.sort(-probs_test, axis=1)
    cum_probs    = np.cumsum(sorted_probs, axis=1)
    covered, sizes = 0, 0
    for i in range(n_test):
        idx = min(np.searchsorted(cum_probs[i], threshold) + 1, K)
        pred_set = set(sorted_idx[i, :idx].tolist())
        sizes += len(pred_set)
        if y_test[i] in pred_set:
            covered += 1
    return covered / n_test, sizes / n_test


def run_conformal(work_dir: Path) -> None:
    """Split-conformal coverage and avg set size for every cached prob file.
    No retraining; sweeps probs_dir.
    """
    print(f"\n========== Section: Conformal coverage ==========")
    probs_dir = work_dir / "probs"
    if not probs_dir.exists():
        print(f"no probs dir at {probs_dir}; nothing to do.")
        return

    paths = sorted(probs_dir.glob("*.npz"))
    print(f"found {len(paths)} cached prob files")
    results = []
    t0 = time.time()
    for i, path in enumerate(paths):
        fname = path.stem
        if "__members" in fname:
            continue
        m = re.match(r"^(\d+)__([\w]+)__seed(\d+)$", fname)
        if not m:
            continue
        task_id, model_name, seed = int(m.group(1)), m.group(2), int(m.group(3))
        try:
            data = np.load(path)
            cov, sz = _conformal_eval(
                data["probs_val_cal"], data["y_val_cal"],
                data["probs_test"],    data["y_test"],
                alpha=CONFORMAL_ALPHA,
            )
            results.append({
                "task_id": task_id, "model": model_name, "seed": seed,
                "target_coverage": 1 - CONFORMAL_ALPHA,
                "empirical_coverage": cov,
                "avg_set_size":      sz,
                "n_classes":         data["probs_test"].shape[1],
                "n_test":            len(data["y_test"]),
            })
        except Exception as exc:                                # noqa: BLE001
            print(f"  x {fname}: {exc}")
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(paths)} processed "
                  f"({(time.time()-t0)/60:.1f} min)")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(work_dir / "conformal_results.csv", index=False)
        print(f"saved: {work_dir/'conformal_results.csv'}")
        print("\nCoverage by model (target = "
              f"{1-CONFORMAL_ALPHA:.2f}):")
        summary = df.groupby("model").agg(
            avg_coverage=("empirical_coverage", "mean"),
            median_coverage=("empirical_coverage", "median"),
            avg_set_size=("avg_set_size", "mean"),
            median_set_size=("avg_set_size", "median"),
        ).round(4)
        print(summary.to_string())


# =============================================================================
# Section: Post-analyses (cost, imbalance, composite, practical H7)
# =============================================================================

def _per_dataset_medians(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["task_id", "model", "calibrator"])
              [["cal_nll", "cal_ece_mean", "cal_brier_score"]]
              .median().reset_index())


def _practical_h7(per_ds: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    rows = []
    for m in models:
        for metric in ("cal_nll", "cal_ece_mean", "cal_brier_score"):
            a = per_ds[(per_ds.model == m) & (per_ds.calibrator == "dirichlet")
                      ].set_index("task_id")[metric]
            b = per_ds[(per_ds.model == m) & (per_ds.calibrator == "logistic")
                      ].set_index("task_id")[metric]
            a, b = a.align(b, join="inner")
            if len(a) < 5:
                continue
            diff = a - b
            if (diff == 0).all():
                p_val = 1.0
            else:
                try:
                    _, p_val = wilcoxon(a, b, zero_method="wilcox")
                except ValueError:
                    continue
            pos = int((diff > 0).sum())
            neg = int((diff < 0).sum())
            rbc = (pos - neg) / max(pos + neg, 1)
            med_a, med_b = float(a.median()), float(b.median())
            med_diff = float((a - b).median())
            rel_change = (med_a - med_b) / max(abs(med_b), 1e-9) * 100
            if metric == "cal_ece_mean":
                pract = abs(med_diff) > 0.005
            else:
                pract = abs(rel_change) > 2.0
            rows.append({
                "model": m, "metric": metric,
                "med_dirichlet": med_a, "med_mlr": med_b,
                "med_diff": med_diff, "rel_change_pct": rel_change,
                "p": float(p_val), "rbc": float(rbc),
                "practically_significant": pract,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    _, p_adj, _, _ = multipletests(df["p"].values, alpha=0.05, method="holm")
    df["p_adj"] = p_adj
    df["statistically_significant"] = df["p_adj"] < 0.05
    return df


def run_postanalysis(work_dir: Path, manifest: pd.DataFrame,
                     results_path: Path) -> None:
    """Cost, imbalance, composite, practical H7 — all from results_raw.csv."""
    print(f"\n========== Section: Post-analyses ==========")
    if not results_path.exists():
        print(f"results_raw.csv not found at {results_path}; skipping.")
        return
    results = pd.read_csv(results_path)
    print(f"results_raw.csv: {len(results)} rows")

    # ── Computational cost ───────────────────────────────────────────────────
    print("\n-- computational cost --")
    unique_train = results.drop_duplicates(subset=["task_id", "model", "seed"])
    tt = unique_train.groupby("model")["train_time_s"].describe().round(2)
    totals = unique_train.groupby("model")["train_time_s"].sum() / 3600
    cost_table = tt[["mean", "std", "50%", "min", "max", "count"]]
    cost_table.columns = ["mean_s", "std_s", "median_s", "min_s", "max_s", "n_runs"]
    cost_table["total_h"] = totals.round(3)
    cost_table.to_csv(work_dir / "computational_cost.csv")
    print(cost_table.to_string())
    print(f"saved: {work_dir/'computational_cost.csv'}")

    # ── Imbalance buckets ────────────────────────────────────────────────────
    print("\n-- imbalance stratification --")
    imb_rows = []
    for _, row in manifest.iterrows():
        data = load_task(int(row["task_id"]))
        if data is None:
            continue
        counts = np.bincount(data["y"])
        counts = counts[counts > 0]
        imb_rows.append({"task_id": int(row["task_id"]),
                         "imbalance_ratio": float(counts.max() / counts.min())})
    imb_df = pd.DataFrame(imb_rows)
    imb_df["imbalance_bucket"] = pd.cut(
        imb_df["imbalance_ratio"],
        bins=[0, 2, 5, 1e9],
        labels=["low (<=2)", "medium (2-5)", "high (>5)"],
    )
    print(imb_df.groupby("imbalance_bucket", observed=True).size().to_string())
    imb_df.to_csv(work_dir / "imbalance_buckets.csv", index=False)

    res_imb = results.merge(imb_df[["task_id", "imbalance_ratio",
                                     "imbalance_bucket"]],
                            on="task_id")
    per_ds_imb = (res_imb.groupby(
        ["imbalance_bucket", "task_id", "model", "calibrator"],
        observed=True,
    )[["cal_nll", "cal_ece_mean", "cal_brier_score"]].median().reset_index())
    bucket_medians = (per_ds_imb.groupby(
        ["imbalance_bucket", "model", "calibrator"], observed=True,
    )["cal_nll"].median().unstack("calibrator").round(4))
    bucket_medians.to_csv(work_dir / "imbalance_stratified_nll.csv")
    print(f"saved: {work_dir/'imbalance_stratified_nll.csv'}")

    # ── Practical H7 (Dirichlet vs MLR) ──────────────────────────────────────
    print("\n-- H7: Dirichlet vs MLR with practical significance --")
    per_ds = _per_dataset_medians(results)
    H7_models = ["lgbm", "xgboost", "catboost", "single_mlp", "mc_dropout",
                 "deep_ensemble_m3", "deep_ensemble", "deep_ensemble_m10"]
    H7_df = _practical_h7(per_ds, H7_models)
    if not H7_df.empty:
        H7_df.to_csv(work_dir / "H7_with_practical_significance.csv", index=False)
        print(H7_df.to_string(index=False,
                              float_format=lambda x: f"{x:.4f}"
                              if isinstance(x, float) else str(x)))
        print(f"saved: {work_dir/'H7_with_practical_significance.csv'}")

    # ── Composite-score weight-simplex sweep ─────────────────────────────────
    print("\n-- composite score weight-simplex sweep --")
    medians = (per_ds.groupby(["model", "calibrator"])
                     [["cal_nll", "cal_ece_mean", "cal_brier_score"]]
                     .median().reset_index())
    out_parts = []
    for _, grp in medians.groupby("model"):
        g = grp.copy()
        for metric in ("cal_nll", "cal_ece_mean", "cal_brier_score"):
            vmin, vmax = g[metric].min(), g[metric].max()
            g[f"{metric}_norm"] = ((g[metric] - vmin) / (vmax - vmin)
                                   if vmax > vmin else 0.0)
        out_parts.append(g)
    med_norm = pd.concat(out_parts, ignore_index=True)

    weights = []
    for w1 in np.round(np.arange(0, 1.01, 0.1), 6):
        for w2 in np.round(np.arange(0, 1.01 - w1, 0.1), 6):
            w3 = round(1 - w1 - w2, 6)
            if w3 < -1e-9:
                continue
            weights.append((round(float(w1), 1), round(float(w2), 1),
                            round(max(float(w3), 0.0), 1)))
    dominance = []
    for w1, w2, w3 in weights:
        med_norm["S"] = (w1 * med_norm["cal_nll_norm"]
                         + w2 * med_norm["cal_ece_mean_norm"]
                         + w3 * med_norm["cal_brier_score_norm"])
        best = (med_norm.loc[med_norm.groupby("model")["S"].idxmin()]
                        [["model", "calibrator", "S"]])
        for _, row in best.iterrows():
            dominance.append({"w_nll": w1, "w_ece": w2, "w_brier": w3,
                              "model": row["model"],
                              "winning_calibrator": row["calibrator"]})
    dom_df = pd.DataFrame(dominance)
    dom_df.to_csv(work_dir / "composite_score_dominance.csv", index=False)
    print(f"saved: {work_dir/'composite_score_dominance.csv'}")
    print("\nCalibrator dominance fractions per model:")
    for model in sorted(dom_df["model"].unique()):
        sub = dom_df[dom_df.model == model]
        counts = sub["winning_calibrator"].value_counts(normalize=True).round(3)
        print(f"\n{model}:")
        print(counts.to_string())


# =============================================================================
# Main dispatcher
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work_dir", type=Path,
                        default=REPO_ROOT / "results",
                        help="Output directory. Default: <repo>/results")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Path to dataset_manifest.csv. "
                             "Default: <work_dir>/dataset_manifest.csv")
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to results_raw.csv (used by --section "
                             "postanalysis). Default: <work_dir>/results_raw.csv")
    parser.add_argument("--section",
                        choices=("tabpfn", "ood", "label_noise",
                                 "conformal", "postanalysis", "all"),
                        default="all",
                        help="Which section(s) to run. Default: all.")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (work_dir / "dataset_manifest.csv")
    if not manifest_path.exists():
        fallback = REPO_ROOT / "results" / "dataset_manifest.csv"
        if fallback.exists():
            manifest_path = fallback
        else:
            print(f"ERROR: dataset_manifest.csv not found", file=sys.stderr)
            return 2
    results_path = args.results or (work_dir / "results_raw.csv")

    logging.basicConfig(level=logging.WARNING)
    print(f"work_dir : {work_dir}")
    print(f"manifest : {manifest_path}")
    print(f"section  : {args.section}")

    manifest    = pd.read_csv(manifest_path)
    device      = get_device()
    device_str  = "cuda" if (hasattr(device, "type") and device.type == "cuda") else "cpu"
    print(f"device   : {device}")

    section = args.section
    if section in ("tabpfn", "all"):
        run_tabpfn(work_dir, manifest, device_str)
    if section in ("ood", "all"):
        run_ood(work_dir, manifest, device)
    if section in ("label_noise", "all"):
        run_label_noise(work_dir, manifest, device)
    if section in ("conformal", "all"):
        run_conformal(work_dir)
    if section in ("postanalysis", "all"):
        run_postanalysis(work_dir, manifest, results_path)

    print("\nDiagnostics complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
