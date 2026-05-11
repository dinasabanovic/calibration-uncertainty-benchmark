# Calibration-Uncertainty-Benchmark: Does Post-Hoc Calibration Close the Predictive Uncertainty Gap?

> **A Systematic Comparison of Gradient-Boosted Trees and Deep Ensembles**
> *Submitted to Machine Learning and Knowledge Extraction (MDPI), 2026*

[![Journal](https://img.shields.io/badge/Journal-MAKE%202026-green)](https://www.mdpi.com/journal/make)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

---

Paper:

**Šabanović, D., Krčmar, T., Krpić, Z., & Lukić, I. (2026).**
*A Systematic Comparison of Gradient-Boosted Trees and Deep Ensembles.*
Machine Learning and Knowledge Extraction.

---

## Overview

This repository contains the code, raw results, and analysis pipeline for a fully crossed factorial benchmark comparing four model classes under four post-hoc calibration regimes across 36 OpenML-CC18 datasets, producing **2,880 experimental runs** evaluated with **60 paired Wilcoxon signed-rank tests** under a single global Holm–Bonferroni correction.

The central question: when both gradient-boosted trees and neural ensembles are calibrated using the same post-hoc method, does the widely assumed uncertainty advantage of deep ensembles persist?

### Key findings

| Finding | Result |
| --- | --- |
| Deep ensemble vs LightGBM (raw NLL) | not significant, *r* = 0.009, *p*<sub>adj</sub> = 1.0 |
| Deep ensemble vs SingleMLP (raw NLL) | **significant**, *r* = −0.87, *p*<sub>adj</sub> < 0.001 |
| Temperature scaling improves NLL | **significant for every model family**, *p*<sub>adj</sub> ≤ 0.049 |
| MLR (multinomial logistic recalibration) | **significantly degrades NLL and ECE across all models** |
| Per-dataset–tuned MLP vs DeepEnsemble | not significant, *p* = 0.451 |

**Bottom line:** under in-distribution conditions, choice of calibration method has greater practical impact on uncertainty quality than the choice between GBDTs and five-member neural ensembles.

---

## Method

The benchmark is a fully crossed factorial design with three independent factors:

- **Model class:** LightGBM, XGBoost, SingleMLP, DeepEnsemble (M = 5)
- **Calibration regime:** none, temperature scaling, multinomial logistic recalibration (MLR), isotonic regression
- **Random seed:** {0, 1, 2, 3, 4}

This gives 4 × 4 × 5 = 80 conditions per dataset × 36 datasets = **2,880 runs**.

### Data split (per dataset, per seed)

| Partition | Fraction | Purpose |
| --- | --- | --- |
| `X_train`    | ~54 % | Model weight learning |
| `X_val_es`   | ~6 %  | Early stopping only; never used for calibration |
| `X_val_cal`  | 20 %  | Calibrator fitting only; never seen during training |
| `X_test`     | 20 %  | Final evaluation; strictly held out throughout |

The `X_val_es` / `X_val_cal` separation prevents the calibration leakage that arises when the same validation set is used both for early stopping and for calibrator fitting.

### Metrics

- **NLL** — strictly proper scoring rule, sensitive to both calibration and sharpness
- **ECE** — averaged across bin counts {10, 15, 20} per Nixon et al. recommendation
- **Brier score** — bounded proper scoring rule
- **Accuracy** — reported for reference only; not subject to hypothesis testing

### Statistical procedure

Per-dataset medians are first computed across 5 seeds, yielding *n* = 36 paired observations per comparison. All pairwise comparisons use the two-sided **paired Wilcoxon signed-rank test**, with a **single global Holm–Bonferroni correction** applied across the entire family of 60 tests at α = 0.05. Effect size is reported as the signed rank-biserial correlation.

---

## Datasets

36 binary and multi-class classification tasks sampled from the **OpenML-CC18 benchmark suite** (study ID 99), stratified into three size regimes (12 tasks each):

- **Small** (*n* < 1 000): `kc2`, `cylinder-bands`, `wdbc`, `credit-approval`, …
- **Medium** (1 000 ≤ *n* < 10 000): `MiceProtein`, `splice`, `satimage`, …
- **Large** (*n* ≥ 10 000): `har`, `pendigits`, `bank-marketing`, `mnist_784`, `Fashion-MNIST`, `CIFAR_10`, …

The full task list with sample/feature/class counts is in [`results/dataset_manifest.csv`](results/dataset_manifest.csv) and [`results/dataset_manifest.json`](results/dataset_manifest.json).

---

## Repository structure

```
calibration-uncertainty-benchmark/
├── src/                                  # Library code (importable package)
│   ├── calibration.py                    # Identity, TempScaling, MLR, Isotonic
│   ├── datasets.py                       # OpenML-CC18 loading + 60/20/20 split
│   ├── evaluation.py                     # Full factorial benchmark runner
│   ├── metrics.py                        # NLL, ECE (multi-bin), Brier, accuracy
│   ├── models.py                         # LightGBM, XGBoost, SingleMLP, DeepEnsemble
│   ├── statistical_analysis.py           # Wilcoxon + global Holm–Bonferroni
│   ├── visualization.py                  # Tables and figures
│   └── utils.py                          # Shared constants (EPS, etc.)
├── experiments/                          # Runnable scripts
│   ├── run_experiments.py                # Main benchmark (2 880 runs)
│   ├── hp_sensitivity_ablation.py        # §4.7 ablation (2 700 runs)
│   ├── postprocess.py                    # Recompute stats from results_raw.csv
│   └── make_figures.py                   # Render publication figures
├── results/                              # Frozen artefacts from the paper
│   ├── results_raw.csv                   # 2 880 main-experiment rows
│   ├── dataset_manifest.csv/.json        # 36 datasets — task IDs, sizes, classes
│   └── hp_ablation/                      # §4.7 outputs (2 700 rows)
│       ├── hp_ablation_all_configs.csv
│       ├── hp_ablation_best_configs.csv
│       ├── hp_ablation_table.csv         # Table 6
│       └── hp_ablation_wilcoxon.csv      # Table 7
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Installation

Requires Python 3.9+ and (for the MLP/ensemble experiments) a CUDA-capable GPU. Tested on H100.

```bash
git clone https://github.com/dinasabanovic/calibration-uncertainty-benchmark.git
cd calibration-uncertainty-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                # makes `src.*` importable from anywhere
```

---

## Reproducibility

All scripts are run from the **project root**.

### 1. Reproduce statistics and figures from frozen results (no GPU needed)

The included `results/results_raw.csv` is the exact 2 880-row output reported in the paper. To regenerate every statistical table and figure from it:

```bash
python experiments/postprocess.py --results_dir results/
python experiments/make_figures.py
```

This rebuilds Tables 3, 4, 5, 8 and Figures 1, 2, 3 in seconds. Outputs land in `results/` and `results/figures/`.

### 2. Re-run the full benchmark from scratch (GPU recommended)

Approximately 8–12 hours on a single H100; the runner checkpoints to `results_raw.csv` after every cell, so it can be killed and resumed safely.

```bash
# Full run (36 datasets × 4 models × 4 calibrators × 5 seeds = 2 880 cells)
python experiments/run_experiments.py --output_dir results/

# Quick smoke test (5 datasets × 2 seeds)
python experiments/run_experiments.py --max_datasets 5 --seeds 0 1 \
    --output_dir results_test/
```

### 3. Re-run the §4.7 hyperparameter ablation

15 MLP configurations × 36 datasets × 5 seeds = 2 700 runs.

---

## Configuration

Fixed hyperparameters used across all 36 datasets — see `src/models.py` and `src/evaluation.py`:

| Component | Setting |
| --- | --- |
| **LightGBM** | lr = 0.05, num_leaves = 63, min_child_samples = 20, feature/bagging fraction = 0.8, L1/L2 = 0.1, up to 500 rounds, early stopping (50) |
| **XGBoost** | lr = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8, L1/L2 = 0.1, up to 500 rounds, early stopping (50) |
| **MLP architecture** | d → 256 → 128 → K, BatchNorm + ReLU + Dropout (0.1) |
| **MLP training** | AdamW (lr = 10⁻³, wd = 10⁻⁴), gradient clip ℓ₂ = 1.0, cosine annealing, 200 epochs, patience 20 |
| **Deep ensemble** | M = 5 members, random init only (no bootstrap), arithmetic mean of softmax outputs |
| **Temperature scaling** | scalar T > 0, L-BFGS on `X_val_cal` NLL |
| **MLR** | sklearn `LogisticRegression`, solver `lbfgs`, C = 1.0 |
| **Isotonic regression** | one-vs-rest per class, renormalised |
| **ECE bins** | mean across {10, 15, 20} equal-width bins |

---

## Notes for reviewers

1. **Calibrator naming.** The paper uses **MLR** (multinomial logistic recalibration); the code uses `"logistic"` as the registry key, with `"platt"` retained as a deprecated alias. They denote the same calibrator.
2. **H3 primary vs. supplementary comparator.** Paper Table 5 frames temperature scaling as the *primary* H3 comparator and isotonic regression as a *robustness check*. The code (`src/statistical_analysis.py`) was originally written with the opposite framing, justified on the grounds that TS on ensemble outputs uses a pseudo-logit approximation. **Both sets of numbers are computed and saved** (look for `stats_H3_*.csv` vs `stats_H3_supp_temp_*.csv`), so all values in Table 5 are reproducible — only the labelling of which row is "primary" differs.
3. The version of `results_raw.csv` shipped here is the exact frozen output used to produce every table and figure in the paper.

---

## License

This work is released under the [MIT License](LICENSE). The OpenML-CC18 datasets used by this benchmark are governed by their own respective licenses on the [OpenML platform](https://www.openml.org).
