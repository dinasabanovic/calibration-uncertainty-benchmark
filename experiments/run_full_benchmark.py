"""
run_full_benchmark.py
---------------------
Full re-baseline for the major revision.

What this script does
---------------------
Trains each of 5 models once per (dataset, seed) and applies 5 calibrators
to the saved probabilities, producing 4,500 rows:

    5 models  ['lgbm', 'xgboost', 'catboost', 'single_mlp', 'deep_ensemble']
    x 5 calibrators ['none', 'temp', 'logistic', 'isotonic', 'dirichlet']
    x 5 seeds [0, 1, 2, 3, 4]
    x 36 datasets (from dataset_manifest.csv)
    = 4,500 rows.

Why re-run from scratch (rather than reuse the original results_raw.csv
shipped in this repo) is documented in the revision response letter:
the new run unifies the hardware so all numbers in the revised manuscript
come from the same machine, eliminating any baseline drift between
hardware generations across the original and new model classes.

How it runs
-----------
1. Each (task, model, seed) trains exactly once. Probabilities (val_cal
   and test) are cached to <work_dir>/probs/<task>__<model>__seed<s>.npz.
   Re-running the script reuses any cached probs, so adding a new
   calibrator does not retrain anything.
2. After every dataset, the partial CSV is written to
   <work_dir>/full_benchmark_partial.csv so a Ctrl-C or disconnect loses
   at most one dataset's progress. When the loop finishes the partial is
   promoted to <work_dir>/results_raw.csv and deleted.
3. If <work_dir>/results_raw.csv already exists, it is renamed to
   results_raw_original_v1.csv so it is preserved (not deleted).

Inputs
------
* dataset_manifest.csv at <work_dir>/dataset_manifest.csv (or the repo's
  results/dataset_manifest.csv, which the default --work_dir points at).
* OpenML credentials in ~/.openml/config (the src.datasets loader uses
  the openml package, which fetches each dataset on first use and caches
  it under ~/.openml/).

Outputs
-------
* <work_dir>/results_raw.csv               the 4,500-row result table
* <work_dir>/probs/                        per-cell .npz caches (~5 GB total)
* <work_dir>/full_benchmark_failures.csv   any rows that errored, with reasons

Usage
-----
    # from the repository root
    python experiments/run_full_benchmark.py
    # or with a custom work dir
    python experiments/run_full_benchmark.py --work_dir /path/to/output

Estimated wall-clock on a single A100 / H100: ~4-6 h.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import load_task, split_dataset           # noqa: E402
from src.models import get_device                            # noqa: E402

from experiments.revision_helpers import (                   # noqa: E402
    evaluate_cell,
    get_predictions,
)


# =============================================================================
# Config (matches paper §3.4)
# =============================================================================

MODELS      = ["lgbm", "xgboost", "catboost", "single_mlp", "deep_ensemble"]
CALIBRATORS = ["none", "temp", "logistic", "isotonic", "dirichlet"]
SEEDS       = [0, 1, 2, 3, 4]


# =============================================================================
# Main
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
    parser.add_argument("--max_datasets", type=int, default=None,
                        help="Smoke-test: process only the first N datasets.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override default seed list (default: 0 1 2 3 4).")
    args = parser.parse_args()

    work_dir: Path = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    probs_dir = work_dir / "probs"
    probs_dir.mkdir(exist_ok=True)

    manifest_path = args.manifest or (work_dir / "dataset_manifest.csv")
    if not manifest_path.exists():
        # Fallback: the manifest shipped in the repo's results/ folder
        fallback = REPO_ROOT / "results" / "dataset_manifest.csv"
        if fallback.exists():
            manifest_path = fallback
        else:
            print(f"ERROR: dataset_manifest.csv not found at {manifest_path}",
                  file=sys.stderr)
            return 2

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(f"work_dir : {work_dir}")
    print(f"manifest : {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    task_ids = manifest["task_id"].tolist()
    if args.max_datasets is not None:
        task_ids = task_ids[:args.max_datasets]
        print(f"smoke-test mode: only first {len(task_ids)} datasets")

    seeds = args.seeds if args.seeds is not None else SEEDS

    # ── Archive any pre-revision results_raw.csv ─────────────────────────────
    out_path     = work_dir / "results_raw.csv"
    archive_path = work_dir / "results_raw_original_v1.csv"
    if out_path.exists() and not archive_path.exists():
        out_path.rename(archive_path)
        print(f"archived: results_raw.csv -> {archive_path.name}")
    elif archive_path.exists():
        print(f"v1 archive already present: {archive_path.name}")

    # ── Resume from partial checkpoint if present ────────────────────────────
    partial_ckpt = work_dir / "full_benchmark_partial.csv"
    if partial_ckpt.exists():
        prev = pd.read_csv(partial_ckpt)
        new_results = prev.to_dict("records")
        done_keys = {
            (int(r["task_id"]), r["model"], r["calibrator"], int(r["seed"]))
            for r in new_results
        }
        print(f"resumed from partial: {len(new_results)} rows already done")
    else:
        new_results = []
        done_keys   = set()

    fail_log    = []
    device      = get_device()
    print(f"device   : {device}")

    total_cells = len(task_ids) * len(seeds) * len(MODELS) * len(CALIBRATORS)
    processed   = len(done_keys)
    print(f"target   : {total_cells} cells "
          f"({len(task_ids)} datasets x {len(seeds)} seeds "
          f"x {len(MODELS)} models x {len(CALIBRATORS)} calibrators)")
    start = time.time()

    # ── Main loop ────────────────────────────────────────────────────────────
    for task_id in task_ids:
        print(f"\n-- task {task_id} -------------------------------------")
        data = load_task(int(task_id))
        if data is None:
            print("   skipped (load_task returned None)")
            fail_log.append({"task_id": int(task_id), "reason": "load_task None"})
            continue
        print(f"   {data['name']}  n={data['n_samples']}  "
              f"d={data['n_features']}  K={data['n_classes']}")

        for seed in seeds:
            split = split_dataset(data, seed=seed)

            for model_name in MODELS:
                try:
                    preds = get_predictions(split, model_name, seed,
                                            probs_dir, device,
                                            need_members=False)
                except Exception as exc:                       # noqa: BLE001
                    print(f"   x {model_name} seed={seed} TRAIN FAILED: {exc}")
                    fail_log.append({
                        "task_id": int(task_id), "model": model_name,
                        "seed": seed, "reason": f"train: {exc}",
                    })
                    continue

                for cal_name in CALIBRATORS:
                    key = (int(task_id), model_name, cal_name, seed)
                    if key in done_keys:
                        continue
                    try:
                        row = evaluate_cell(split, model_name, cal_name,
                                            seed, preds)
                        new_results.append(row)
                        done_keys.add(key)
                        processed += 1
                    except Exception as exc:                   # noqa: BLE001
                        print(f"   x {model_name}/{cal_name} seed={seed} "
                              f"CAL FAILED: {exc}")
                        fail_log.append({
                            "task_id": int(task_id), "model": model_name,
                            "seed": seed, "calibrator": cal_name,
                            "reason": f"calibrate: {exc}",
                        })

        # Checkpoint after every dataset
        if new_results:
            pd.DataFrame(new_results).to_csv(partial_ckpt, index=False)
        elapsed = time.time() - start
        pct     = 100 * processed / max(total_cells, 1)
        eta_s   = elapsed / max(processed, 1) * (total_cells - processed)
        print(f"   done. {processed}/{total_cells} cells ({pct:.1f}%) | "
              f"elapsed {elapsed/60:.1f} min | ETA {eta_s/60:.1f} min")

    # ── Finalise ─────────────────────────────────────────────────────────────
    total_min = (time.time() - start) / 60
    print(f"\n=====================================")
    print(f"DONE. Rows produced: {len(new_results)}")
    print(f"Failures: {len(fail_log)}")
    print(f"Total wall-clock: {total_min:.1f} min ({total_min/60:.2f} h)")

    if not new_results:
        print("No results produced; not writing CSV.")
        return 1

    final_df = pd.DataFrame(new_results)
    final_df.to_csv(out_path, index=False)
    print(f"final CSV: {out_path}")

    if partial_ckpt.exists():
        partial_ckpt.unlink()

    if fail_log:
        pd.DataFrame(fail_log).to_csv(
            work_dir / "full_benchmark_failures.csv", index=False,
        )
        print(f"failures: {work_dir/'full_benchmark_failures.csv'}")

    # ── Sanity matrix ────────────────────────────────────────────────────────
    matrix = final_df.groupby(["model", "calibrator"]).size().unstack(fill_value=0)
    print("\nRows by (model, calibrator) -- each cell should be 5 seeds x N datasets:")
    print(matrix.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
