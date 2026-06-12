"""
run_ensembles_mcdropout.py
--------------------------
Add ensemble M-sweep, MC-Dropout, and member-level temperature scaling
to the benchmark.

Reviewer requests addressed
---------------------------
R2.3   "more experiments for different M than M=5"  -> M=3 and M=10 added
R1.7   "Temperature scaling on ensembles uses an approximation"
                                                     -> Member-level TS added
                                                        for all 3 ensemble sizes
R1.3   "Methods like BNN are not included"           -> MC-Dropout (T=30) as
                                                        Bayesian-NN proxy

What this script does
---------------------
Reads the results_raw.csv produced by run_full_benchmark.py and ADDS the
new (model, calibrator) cells. Only rows that are missing from the
existing CSV are computed.

New cells per (dataset, seed):
    deep_ensemble_m3      : none, temp, logistic, isotonic, dirichlet, temp_member
    deep_ensemble_m10     : none, temp, logistic, isotonic, dirichlet, temp_member
    deep_ensemble (M=5)   : temp_member only (the other 5 are already present)
    mc_dropout            : none, temp, logistic, isotonic, dirichlet

Total new rows:
    (6 + 6 + 1 + 5) cells x 5 seeds x 36 datasets = 18 x 180 = 3,240
    Existing rows from run_full_benchmark.py : 4,500
    Final total                              : 7,740

For the M=5 ensemble, the existing cache contains averaged probs only
(no per-member probs). So the M=5 ensemble is retrained for temp_member
to produce a *__members.npz* cache file; the existing cache of averaged
probs is left intact and remains the source for the 5 regular M=5
calibrator rows already in results_raw.csv.

Outputs
-------
* <work_dir>/results_raw.csv                              merged 7,740-row table
* <work_dir>/results_raw_pre_ensembles_merge_<ts>.csv     pre-merge backup
* <work_dir>/probs/...__members.npz                       per-member prob caches
* <work_dir>/ensembles_mcdropout_failures.csv             any per-cell failures

Usage
-----
    python experiments/run_ensembles_mcdropout.py
    python experiments/run_ensembles_mcdropout.py --work_dir /path/to/output

Wall-clock on A100: ~1.5 h actual.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import load_task, split_dataset           # noqa: E402
from src.models import get_device                            # noqa: E402

from experiments.revision_helpers import (                   # noqa: E402
    ENSEMBLE_SIZES,
    evaluate_cell,
    get_predictions,
)


# =============================================================================
# Config
# =============================================================================

SEEDS = [0, 1, 2, 3, 4]

REGULAR_CALIBRATORS = ["none", "temp", "logistic", "isotonic", "dirichlet"]

# (model, calibrator) pairs to compute. The skip-existing check below
# handles whether each row is actually new vs already in results_raw.csv.
NEW_CELLS: list[tuple[str, str]] = []
for c in REGULAR_CALIBRATORS + ["temp_member"]:
    NEW_CELLS.append(("deep_ensemble_m3", c))
for c in REGULAR_CALIBRATORS + ["temp_member"]:
    NEW_CELLS.append(("deep_ensemble_m10", c))
NEW_CELLS.append(("deep_ensemble", "temp_member"))
for c in REGULAR_CALIBRATORS:
    NEW_CELLS.append(("mc_dropout", c))


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
    parser.add_argument("--results", type=Path, default=None,
                        help="Path to results_raw.csv produced by "
                             "run_full_benchmark.py. "
                             "Default: <work_dir>/results_raw.csv")
    args = parser.parse_args()

    work_dir: Path = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    probs_dir = work_dir / "probs"
    probs_dir.mkdir(exist_ok=True)

    manifest_path = args.manifest or (work_dir / "dataset_manifest.csv")
    if not manifest_path.exists():
        fallback = REPO_ROOT / "results" / "dataset_manifest.csv"
        if fallback.exists():
            manifest_path = fallback
        else:
            print(f"ERROR: dataset_manifest.csv not found at {manifest_path}",
                  file=sys.stderr)
            return 2

    results_path = args.results or (work_dir / "results_raw.csv")
    if not results_path.exists():
        print(f"ERROR: results_raw.csv not found at {results_path}.\n"
              "Run run_full_benchmark.py first.", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.WARNING)
    print(f"work_dir : {work_dir}")
    print(f"manifest : {manifest_path}")
    print(f"existing : {results_path}")

    manifest = pd.read_csv(manifest_path)
    task_ids = manifest["task_id"].tolist()
    existing = pd.read_csv(results_path)
    print(f"existing rows: {len(existing)}")

    existing_keys = {
        (int(r["task_id"]), r["model"], r["calibrator"], int(r["seed"]))
        for _, r in existing.iterrows()
    }

    # Resume from partial checkpoint if present
    partial_ckpt = work_dir / "ensembles_mcdropout_partial.csv"
    if partial_ckpt.exists():
        prev = pd.read_csv(partial_ckpt)
        new_results = prev.to_dict("records")
        done_keys = {
            (int(r["task_id"]), r["model"], r["calibrator"], int(r["seed"]))
            for r in new_results
        }
        print(f"resumed: {len(new_results)} rows already done in this run")
    else:
        new_results = []
        done_keys   = set()

    # Plan of work
    cells_to_compute = []
    for task_id in task_ids:
        for seed in SEEDS:
            for model_name, cal_name in NEW_CELLS:
                key = (int(task_id), model_name, cal_name, int(seed))
                if key in existing_keys or key in done_keys:
                    continue
                cells_to_compute.append(key)

    if not cells_to_compute:
        print("nothing to compute -- every new cell already in results_raw.csv "
              "or in the partial checkpoint.")
    else:
        print(f"cells to compute: {len(cells_to_compute)}")

    work_plan = defaultdict(list)
    for (task_id, model_name, cal_name, seed) in cells_to_compute:
        work_plan[(task_id, seed, model_name)].append(cal_name)

    device     = get_device()
    print(f"device   : {device}")
    fail_log   = []
    processed  = len(done_keys)
    target     = len(cells_to_compute) + processed
    start      = time.time()

    # ── Main loop, iterating by dataset for clean checkpointing ──────────────
    for task_id in task_ids:
        data = None
        for seed in SEEDS:
            for model_name in ("deep_ensemble_m3", "deep_ensemble_m10",
                               "deep_ensemble", "mc_dropout"):
                cals = work_plan.get((int(task_id), seed, model_name), [])
                if not cals:
                    continue
                if data is None:
                    data = load_task(int(task_id))
                    if data is None:
                        print(f"   task {task_id}: load failed")
                        fail_log.append({"task_id": int(task_id),
                                         "reason": "load_task None"})
                        break
                    print(f"\n-- task {task_id}: {data['name']} "
                          f"(n={data['n_samples']}) ----------------")
                split = split_dataset(data, seed=seed)

                need_members = "temp_member" in cals
                try:
                    preds = get_predictions(split, model_name, seed,
                                            probs_dir, device,
                                            need_members=need_members)
                except Exception as exc:                       # noqa: BLE001
                    print(f"   x {model_name} seed={seed} TRAIN FAILED: {exc}")
                    fail_log.append({
                        "task_id": int(task_id), "model": model_name,
                        "seed": seed, "reason": f"train: {exc}",
                    })
                    continue

                for cal_name in cals:
                    try:
                        row = evaluate_cell(split, model_name, cal_name,
                                            seed, preds)
                        new_results.append(row)
                        done_keys.add((int(task_id), model_name, cal_name, seed))
                        processed += 1
                    except Exception as exc:                   # noqa: BLE001
                        print(f"   x {model_name}/{cal_name} seed={seed} "
                              f"CAL FAILED: {exc}")
                        fail_log.append({
                            "task_id": int(task_id), "model": model_name,
                            "seed": seed, "calibrator": cal_name,
                            "reason": f"calibrate: {exc}",
                        })

        if new_results:
            pd.DataFrame(new_results).to_csv(partial_ckpt, index=False)
        elapsed = time.time() - start
        if cells_to_compute:
            print(f"   ...{processed}/{target} "
                  f"elapsed {elapsed/60:.1f} min")

    # ── Merge with existing results_raw.csv ──────────────────────────────────
    if not new_results:
        print("\nNo new rows produced — results_raw.csv unchanged.")
        return 0

    new_df = pd.DataFrame(new_results)
    print(f"\nNew rows: {len(new_df)}; existing rows: {len(existing)}")

    merged = pd.concat([existing, new_df], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(
        subset=["task_id", "model", "calibrator", "seed"], keep="last",
    )
    print(f"merged rows: {len(merged)}")

    # Pre-merge backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = work_dir / f"results_raw_pre_ensembles_merge_{ts}.csv"
    existing.to_csv(backup, index=False)
    print(f"pre-merge backup: {backup.name}")

    merged.to_csv(work_dir / "results_raw.csv", index=False)
    print(f"final merged: {work_dir/'results_raw.csv'} ({len(merged)} rows)")

    if partial_ckpt.exists():
        partial_ckpt.unlink()
    if fail_log:
        pd.DataFrame(fail_log).to_csv(
            work_dir / "ensembles_mcdropout_failures.csv", index=False,
        )
        print(f"failures: {work_dir/'ensembles_mcdropout_failures.csv'}")

    print("\nRow count by (model, calibrator):")
    print(merged.groupby(["model", "calibrator"]).size()
          .unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
