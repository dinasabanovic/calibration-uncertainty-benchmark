# Major-revision addon for `calibration-uncertainty-benchmark`

This folder contains the .py scripts that implement the MAKE revision
pipeline, plus the four publication figure scripts. Everything is
designed to drop into the existing repository at
[`dinasabanovic/calibration-uncertainty-benchmark`](https://github.com/dinasabanovic/calibration-uncertainty-benchmark)
without touching the original `src/` package.

## What's in this zip

```
revision_addon/
├── REVISION_README.md                this file
├── requirements_revision.txt         catboost, dirichletcal, statsmodels, tabpfn
└── experiments/
    ├── revision_helpers.py           new models / calibrators / runner helpers
    ├── run_full_benchmark.py         full re-baseline   (4,500 rows)
    ├── run_ensembles_mcdropout.py    M-sweep + MC-Dropout + member-TS (+3,240 → 7,740)
    ├── run_ood_diagnostics.py        TabPFN, OOD, label noise, conformal, post-hoc
    ├── figure1_heatmap.py            Fig. 1: per-(model, calibrator) heatmap
    ├── figure2_paired_delta.py       Fig. 2: Δ-NLL under temperature scaling
    ├── figure3_four_panel.py         Fig. 3: ΔECE vs ΔNLL diagnostic
    └── figure4_shift_reversal.py     Fig. 4: NLL trajectories under covariate shift
```

## Where each file goes in the repo

The existing repo already has `src/`, `experiments/`, `results/`. Drop each
file into the matching directory. **Don't overwrite anything in `src/`** —
the new code lives entirely in `experiments/`.

| File from this zip                          | Place at                                              |
|---------------------------------------------|-------------------------------------------------------|
| `experiments/revision_helpers.py`           | `experiments/revision_helpers.py`                     |
| `experiments/run_full_benchmark.py`         | `experiments/run_full_benchmark.py`                   |
| `experiments/run_ensembles_mcdropout.py`    | `experiments/run_ensembles_mcdropout.py`              |
| `experiments/run_ood_diagnostics.py`        | `experiments/run_ood_diagnostics.py`                  |
| `experiments/figure1_heatmap.py`            | `experiments/figure1_heatmap.py`                      |
| `experiments/figure2_paired_delta.py`       | `experiments/figure2_paired_delta.py`                 |
| `experiments/figure3_four_panel.py`         | `experiments/figure3_four_panel.py`                   |
| `experiments/figure4_shift_reversal.py`     | `experiments/figure4_shift_reversal.py`               |
| `requirements_revision.txt`                 | repo root (next to `requirements.txt`)                |

### One-time additions to existing files

Append the three required lines from `requirements_revision.txt` to your
existing `requirements.txt`, and add this section to the README so users
know which results belong to which scope:

```diff
 ## Reproducibility

 All scripts are run from the **project root**.
+
+### Major-revision pipeline
+
+The .py scripts under `experiments/` produce the revised benchmark
+(8 models, 5 calibrators, OOD/conformal/TabPFN analyses). Run them
+in this order:
+
+```bash
+# 1. Full re-baseline: 4,500 rows
+python experiments/run_full_benchmark.py
+
+# 2. Add M=3, M=10, MC-Dropout, member-level TS: +3,240 → 7,740 rows
+python experiments/run_ensembles_mcdropout.py
+
+# 3. TabPFN, OOD, label noise, conformal, post-hoc analyses
+python experiments/run_ood_diagnostics.py
+# or just one section:
+python experiments/run_ood_diagnostics.py --section ood
+```
+
+After all three finish, regenerate the four publication figures:
+
+```bash
+python experiments/figure1_heatmap.py
+python experiments/figure2_paired_delta.py
+python experiments/figure3_four_panel.py
+python experiments/figure4_shift_reversal.py
+```
+
+Wall-clock on a single A100/H100: ~10-12 hours total.
```

## What to commit to git (and what NOT to)

### Commit these (small, frozen, source-of-truth)

| Path                                            | Why                                                                    |
|-------------------------------------------------|------------------------------------------------------------------------|
| `experiments/run_*.py`                          | The pipeline                                                           |
| `experiments/revision_helpers.py`               | New models / calibrators / runner                                      |
| `experiments/figure*.py`                        | Figure scripts                                                         |
| `requirements_revision.txt`                     | New deps                                                               |
| `results/results_raw.csv`                       | **The 7,740-row table** — final source for paper Tables 3–14           |
| `results/ood_results.csv`                       | OOD sweep results (~4,320 rows)                                        |
| `results/label_noise_results.csv`               | Label-noise results (~480 rows)                                        |
| `results/conformal_results.csv`                 | Conformal coverage per (task, model, seed)                             |
| `results/tabpfn_results.csv`                    | TabPFN inference rows (24 datasets × 5 seeds × 5 cals)                 |
| `results/computational_cost.csv`                | Per-model training time + total hours                                  |
| `results/imbalance_buckets.csv`                 | Dataset-level imbalance ratios + bucket labels                         |
| `results/imbalance_stratified_nll.csv`          | Per-(bucket, model, calibrator) median NLL                             |
| `results/composite_score_dominance.csv`         | Weight-simplex sweep, winning calibrator per (model, weight triple)    |
| `results/H7_with_practical_significance.csv`    | Dirichlet vs MLR + practical-sig flag                                  |
| `results/ood_noise/task_*.npz`                  | Per-task noise tensors — *byte-identical OOD reproduction*             |
| `results/dataset_manifest.csv` / `.json`        | Already in repo — keep them                                            |
| `results/figures/figure[1-4]_*.{pdf,png}`       | Final rendered figures (commit both PDF and PNG)                       |

### Do NOT commit these (large, regeneratable)

| Path                                          | Reason                                                       |
|-----------------------------------------------|--------------------------------------------------------------|
| `results/probs/*.npz`                         | ~5 GB cache. Re-trained on demand by the run_*.py scripts.   |
| `*_partial.csv`                               | Mid-run checkpoints. Auto-deleted on successful completion.  |
| `*_failures.csv`                              | Optional — only commit if reviewers ask for diagnostics.     |
| `results_raw_original_v1.csv`                 | Pre-revision archive; optional. Useful only for paper trail. |
| `results_raw_pre_ensembles_merge_*.csv`       | Pre-merge backup. Safe to delete after merge.                |

A suggested patch for the existing `.gitignore`:

```gitignore
# revision: large prob cache and transient files
results/probs/
results/*_partial.csv
results/*_failures.csv
results/results_raw_original_v1.csv
results/results_raw_pre_ensembles_merge_*.csv
```

## Running order

Each script writes outputs into `results/` (override with `--work_dir`).
The three runner scripts are independent at the disk level:

- `run_full_benchmark.py` writes the initial `results_raw.csv`.
- `run_ensembles_mcdropout.py` merges new rows into it in place.
- `run_ood_diagnostics.py` reads it for the `postanalysis` section and
  writes the other artefacts as siblings.

**Run them in this order.**

```bash
# Install once
pip install -r requirements.txt
pip install -r requirements_revision.txt
pip install -e .         # so 'src.*' imports work

# Set token (only needed for the TabPFN section)
export TABPFN_TOKEN=<your-token>

# Then run the pipeline
python experiments/run_full_benchmark.py                          # ~5 h
python experiments/run_ensembles_mcdropout.py                     # ~1.5 h
python experiments/run_ood_diagnostics.py                         # ~5-6 h
# or run individual sections:
python experiments/run_ood_diagnostics.py --section ood           # ~4 h
python experiments/run_ood_diagnostics.py --section label_noise   # ~45 min
python experiments/run_ood_diagnostics.py --section conformal     # <10 min, no GPU needed
python experiments/run_ood_diagnostics.py --section postanalysis  # <5 min, no GPU needed
python experiments/run_ood_diagnostics.py --section tabpfn        # ~30 min
```

Every script is resumable — kill it with Ctrl-C and rerun, it will pick up
from the last per-dataset checkpoint.

## Verifying the integration

After dropping the files into the repo, a one-line sanity check:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from experiments.revision_helpers import (
    CatBoostModel, DirichletODIRCalibrator, MCDropoutMLP,
    TabPFNModel, apply_member_level_ts, build_model,
    get_predictions, evaluate_cell, tabpfn_compatible,
)
print('revision_helpers imports OK')
"
```

If that prints `revision_helpers imports OK` you're good to go.

## Notes for the response letter

Items worth mentioning explicitly to reviewers:

1. **Hardware unification.** `run_full_benchmark.py` retrains every model
   class from scratch so all numbers in the revised manuscript come from
   the same machine, eliminating baseline drift between the original
   H100 run and the new Colab A100/H100 run.

2. **OOD reproducibility.** The Gaussian noise tensors in
   `results/ood_noise/task_*.npz` are generated deterministically from
   `seed = (task_id × 100003 + round(σ × 1000) × 1009 + OOD_SEED) mod 2³²`
   and persisted to disk. On re-runs they are loaded rather than
   regenerated, so the exact noise used in the reported OOD results is
   byte-identical across machine restarts and NumPy versions.

3. **Train-once / calibrate-many.** Calibrators sweep cached probabilities
   for free. The cache lives in `results/probs/`. Re-running any script
   with new calibrator/model combinations only retrains the missing
   models, never retraining models whose probs are already cached.

4. **CatBoost seed handling.** The benchmark intentionally does not set a
   per-seed CatBoost `random_seed`. Seed-to-seed variance for the three
   GBDT classes (LightGBM, XGBoost, CatBoost) is split-induced (driven by
   the per-seed train/val/cal/test partition), matching how LightGBM and
   XGBoost are configured in the original codebase.
