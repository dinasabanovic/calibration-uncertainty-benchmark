#!/usr/bin/env python3
"""
run_experiments.py
------------------
Entry point for the full calibration benchmark.
Designed for server execution (H100 or any CUDA GPU).

Usage
-----
# Full run — 36 datasets, all models, all calibrators, 5 seeds
python run_experiments.py --output_dir results/

# Quick test — 5 datasets, 2 seeds
python run_experiments.py --max_datasets 5 --seeds 0 1 --output_dir results_test/

# Resume interrupted run
python run_experiments.py --output_dir results/   # skips already-done rows

# Only GBDT models
python run_experiments.py --models lgbm xgboost --output_dir results_gbdt/

# Save probability vectors for reliability diagrams
python run_experiments.py --save_probs --output_dir results/

Arguments
---------
--max_datasets   INT    Total datasets (0 = all CC-18). Default: 36
--max_per_regime INT    Per size regime (small/medium/large). Default: 12
--models         LIST   Default: lgbm xgboost single_mlp deep_ensemble
--calibrators    LIST   Default: none temp logistic isotonic
--seeds          LIST   Default: 0 1 2 3 4
--output_dir     PATH   Default: results/
--save_probs            Save per-sample probability vectors for reliability diagrams
--log_level      STR    Default: INFO
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure project root (parent of experiments/) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets import get_cc18_task_ids, filter_datasets_by_size
from src.evaluation import run_benchmark, SEEDS, CALIBRATOR_NAMES
from src.models import get_device
from src.statistical_analysis import run_hypothesis_tests, print_hypothesis_summary
from src.visualization import build_summary_table, generate_all_figures
from experiments.postprocess import (
    build_training_time_table, latex_training_time_table,
    build_inter_seed_variance_table, print_variance_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibration vs Deep Ensembles — Tabular Uncertainty Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max_datasets",   type=int,  default=36)
    p.add_argument("--max_per_regime", type=int,  default=12)
    p.add_argument("--models",   nargs="+",
                   default=["lgbm", "xgboost", "single_mlp", "deep_ensemble"])
    p.add_argument("--calibrators", nargs="+",
                   default=["none", "temp", "logistic", "isotonic"])
    p.add_argument("--seeds",    nargs="+", type=int, default=SEEDS)
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--save_probs", action="store_true", default=False)
    p.add_argument("--log_level",  type=str, default="INFO")
    return p.parse_args()


def setup_logging(level: str, log_file: Path) -> None:
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "probs").mkdir(exist_ok=True)

    setup_logging(args.log_level, out_dir / "run.log")
    logger = logging.getLogger("run_experiments")

    # ── GPU info ──────────────────────────────────────────────────────────────
    device = get_device()
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  VRAM: {mem:.0f} GB")
        # H100 optimisations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.backends.cudnn.benchmark         = True
        logger.info("  TF32 + cudnn.benchmark enabled.")

    logger.info(f"Output: {out_dir.resolve()}")

    # ── Dataset selection ─────────────────────────────────────────────────────
    logger.info("Fetching CC-18 task list...")
    all_task_ids = get_cc18_task_ids()

    if args.max_datasets > 0:
        task_ids = filter_datasets_by_size(
            all_task_ids,
            max_per_regime=args.max_per_regime,
            cache_dir=str(out_dir),
        )
        np.random.seed(0)
        if len(task_ids) > args.max_datasets:
            task_ids = list(np.random.choice(
                task_ids, size=args.max_datasets, replace=False
            ))
    else:
        task_ids = all_task_ids

    logger.info(f"Selected {len(task_ids)} tasks.")
    with open(out_dir / "task_ids.json", "w") as f:
        json.dump({"task_ids": task_ids}, f, indent=2)

    # ── Checkpoint: load existing results and compute done_keys ──────────────
    ckpt_path    = out_dir / "results_raw.csv"
    existing_rows = []
    done_keys     = set()

    if ckpt_path.exists():
        df_existing  = pd.read_csv(ckpt_path)
        existing_rows = df_existing.to_dict("records")
        done_keys     = {
            (int(r["task_id"]), r["model"], r["calibrator"], int(r["seed"]))
            for r in existing_rows
        }
        logger.info(f"Checkpoint: {len(done_keys)} runs already done.")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    probs_dir = (out_dir / "probs") if args.save_probs else None

    new_results = run_benchmark(
        task_ids         = task_ids,
        model_names      = args.models,
        calibrator_names = args.calibrators,
        seeds            = args.seeds,
        device           = device,
        skip_on_error    = True,
        save_probs_dir   = probs_dir,
        done_keys        = done_keys,
    )

    # Merge and save
    all_results = existing_rows + new_results
    df = pd.DataFrame(all_results).drop_duplicates(
        subset=["task_id", "model", "calibrator", "seed"]
    )
    df.to_csv(ckpt_path, index=False)
    logger.info(f"Saved {len(df)} total rows to {ckpt_path}.")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = build_summary_table(df)
    summary.to_csv(out_dir / "table1_summary.csv")
    logger.info("\n── SUMMARY TABLE ──\n" + summary.to_string())
    logger.info("\nLaTeX:\n" + summary.to_latex())

    # ── Statistical analysis ──────────────────────────────────────────────────
    logger.info("Running hypothesis tests...")
    hyp = run_hypothesis_tests(df, include_supplementary=True)
    print_hypothesis_summary(hyp)
    for key, hdf in hyp.items():
        if not hdf.empty:
            hdf.to_csv(out_dir / f"stats_{key}.csv", index=False)
    logger.info("Hypothesis tests saved.")

    # ── Training time table ───────────────────────────────────────────────────
    tt = build_training_time_table(df)
    tt.to_csv(out_dir / "table_training_times.csv", index=False)
    (out_dir / "table_training_times.tex").write_text(
        latex_training_time_table(tt)
    )
    logger.info("\n── TRAINING TIME (medians) ──\n" +
                tt[["model", "mean ± std (s)", "median_s", "n_runs"]].to_string(index=False))

    # ── Inter-seed variance ───────────────────────────────────────────────────
    vdf, fdf = build_inter_seed_variance_table(df)
    if not vdf.empty:
        vdf.to_csv(out_dir / "table_inter_seed_variance.csv", index=False)
    if not fdf.empty:
        fdf.to_csv(out_dir / "table_high_variance_datasets.csv", index=False)
    print_variance_summary(vdf, fdf)

    # ── Figures ───────────────────────────────────────────────────────────────
    logger.info("Generating figures...")
    try:
        generate_all_figures(
            df,
            fig_dir=out_dir / "figures",
            manifest_path=out_dir / "dataset_manifest.csv",
        )
    except Exception as exc:
        logger.error(f"Figure generation failed: {exc}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "n_tasks":            len(task_ids),
        "n_models":           len(args.models),
        "n_calibrators":      len(args.calibrators),
        "n_seeds":            len(args.seeds),
        "total_runs":         len(df),
        "models":             args.models,
        "calibrators":        args.calibrators,
        "seeds":              args.seeds,
        "ensemble_size":      5,
        "ensemble_bootstrap": False,
        "split_protocol":     "60/20/20 (train/val_cal/test); "
                              "val_es is 10% of train fold (internal)",
        "correction":         "Holm-Bonferroni global across all tests",
        "device":             str(device),
        "gpu":                torch.cuda.get_device_name(0)
                              if device.type == "cuda" else "cpu",
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("All done.")


if __name__ == "__main__":
    main()
