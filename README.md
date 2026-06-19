# Calibration, Architecture, and Distribution Shift in Predictive Uncertainty Estimation

> **A Systematic Comparison of Gradient-Boosted Trees, Deep Ensembles, MC-Dropout, and TabPFN**
> *Submitted to Machine Learning and Knowledge Extraction (MDPI), 2026*

[![Journal](https://img.shields.io/badge/Journal-MAKE%202026-green)](https://www.mdpi.com/journal/make)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

---

Paper:

**Šabanović, D., Krčmar, T., Krpić, Z., & Lukić, I. (2026).**
*Calibration, Architecture, and Distribution Shift in Predictive Uncertainty Estimation.*
Machine Learning and Knowledge Extraction *(under review).*

---

## Overview

This repository contains the code, raw results, and analysis pipeline for a fully crossed factorial benchmark comparing **8 model classes** under **5 post-hoc calibration regimes** across **36 OpenML-CC18 datasets**, producing **7,740 in-distribution runs**, **4,320 covariate-shift runs**, **480 label-noise runs**, and split-conformal coverage for every cached prediction.

The central question: when neural ensembles, gradient-boosted trees, MC-Dropout networks, and TabPFN are calibrated using the same post-hoc methods, does the widely assumed uncertainty advantage of deep ensembles persist — in-distribution, under covariate shift, and under label noise?

### Key Results

| Finding | Result |
| --- | --- |
| Deep ensemble (M=5) vs LightGBM, raw NLL | not significant after calibration matching |
| Temperature scaling improves NLL | **significant for every model family** (within-family Holm) |
| Dirichlet ODIR vs MLR (multinomial logistic) | **Dirichlet significantly better** on NLL across all 8 model families |
| MC-Dropout vs SingleMLP | no significant gain at matched calibration |
| Ensemble size M ∈ {3, 5, 10} | larger M monotonically improves calibration; gains saturate at M=5 |
| Under covariate shift (σ ≥ 1.0) | **calibrator ordering reverses** — MLR beats temperature scaling |

**Bottom line:** under in-distribution conditions, the choice of calibration method has greater practical impact on uncertainty quality than the choice between GBDTs and neural ensembles. The ordering is not stable under distribution shift — practitioners should pick calibrators *for the deployment regime they expect*, not the in-distribution one.

---

## Method

The benchmark is a fully crossed factorial design with three independent factors:

1. **Model class** — LightGBM, XGBoost, CatBoost, SingleMLP, MC-Dropout (T=30), DeepEnsemble (M ∈ {3, 5, 10}), TabPFN v2
2. **Calibration regime** — none, temperature scaling, multinomial logistic recalibration (MLR), isotonic regression, Dirichlet ODIR (and member-level TS for ensembles)
3. **Random seed** — {0, 1, 2, 3, 4}

This gives **7,740** main-benchmark runs across the 36 datasets, plus targeted distribution-shift and conformal experiments described below.

### Data split (per dataset, per seed)

| Partition   | Fraction | Purpose                                             |
| ----------- | -------- | --------------------------------------------------- |
| `X_train`   | ~54%     | Model weight learning                               |
| `X_val_es`  | ~6%      | Early stopping only; never used for calibration     |
| `X_val_cal` | 20%      | Calibrator fitting only; never seen during training |
| `X_test`    | 20%      | Final evaluation; strictly held out throughout      |

The `X_val_es` / `X_val_cal` separation prevents the calibration leakage that arises when the same validation set is used both for early stopping and calibrator fitting.

### Distribution-shift experiments

- **Covariate shift** — additive Gaussian noise on `X_test` at σ ∈ {0.5, 1.0, 2.0} (standardised-feature units). All 36 datasets × 8 models × 5 calibrators × 3 σ = **4,320 runs**. Noise tensors are deterministic and persisted to disk for byte-identical reproduction.
- **Label noise** — symmetric label noise η ∈ {0.1, 0.2} on a stratified 6-dataset subset × 8 models × 5 calibrators = **480 runs**. Calibrators are fitted on **clean** `X_val_cal` — the realistic deployment scenario.
- **Conformal coverage** — split-conformal Adaptive Prediction Sets (Romano et al., 2020) at 1−α = 0.90, computed post-hoc from cached probabilities. No retraining required.

### Metrics

- **NLL** — strictly proper scoring rule, sensitive to both calibration and sharpness
- **ECE** — averaged across bin counts {10, 15, 20} per Nixon et al. (2019)
- **Brier score** — bounded proper scoring rule
- **Accuracy** — reported for reference only; not subject to hypothesis testing
- **Empirical coverage** and **average set size** — for the conformal analysis

### Statistical procedure

Per-dataset medians are first computed across 5 seeds, yielding *n* = 36 paired observations per comparison. All pairwise comparisons use the two-sided **paired Wilcoxon signed-rank test**, with **Holm–Bonferroni correction applied within each pre-specified family**:

| Family | k tests | Question |
| --- | --- | --- |
| **H1** | 39 | Raw model comparisons (no calibrator) |
| **H2** | 72 | Calibrator-vs-raw within each main model family |
| **H3** | 45 | Deep ensemble (M=5) vs comparison models under matched calibration |
| **H4** | 18 | Ensemble size effects (M=3 vs M=5 vs M=10) |
| **H5** |  9 | Member-level TS vs pseudo-logit TS for ensembles |
| **H6** | 12 | MC-Dropout vs SingleMLP (BNN-proxy comparison) |
| **H7** | 24 | Dirichlet ODIR vs MLR — with practical-significance flag |

Effect size is reported as the signed rank-biserial correlation; practical significance is flagged at |Δ%| > 2% for NLL/Brier or |Δ| > 0.005 for ECE.

---

## Datasets

36 binary and multi-class classification tasks sampled from the **OpenML-CC18 benchmark suite** (study ID 99), stratified into three size regimes (12 tasks each):

- **Small** (*n* < 1 000): `kc2`, `cylinder-bands`, `wdbc`, `credit-approval`, …
- **Medium** (1 000 ≤ *n* < 10 000): `MiceProtein`, `splice`, `satimage`, …
- **Large** (*n* ≥ 10 000): `har`, `pendigits`, `bank-marketing`, `mnist_784`, `Fashion-MNIST`, `CIFAR_10`, …

The full task list with sample/feature/class counts is in [`results/dataset_manifest.csv`](results/dataset_manifest.csv) and [`results/dataset_manifest.json`](results/dataset_manifest.json).

**TabPFN compatibility:** 24 of the 36 datasets satisfy TabPFN v2's pre-training constraints (≤10 k training samples, ≤500 features, ≤10 classes). TabPFN is evaluated on this subset only; the larger 12 are excluded rather than subsampled.

---

## Installation

Requires Python 3.9+ and (for the neural and TabPFN experiments) a CUDA-capable GPU. Tested on A100 / H100.

```bash
git clone https://github.com/dinasabanovic/calibration-uncertainty-benchmark.git
cd calibration-uncertainty-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_revision.txt   # adds catboost, dirichletcal, statsmodels, tabpfn
pip install -e .                            # makes `src.*` importable
```

For the TabPFN section, also export a token:

```bash
export TABPFN_TOKEN=<your-token>
```

---

## Configuration

Fixed hyperparameters used across all 36 datasets (see `src/models.py` and `experiments/revision_helpers.py`):

| Component               | Setting                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **LightGBM**            | lr = 0.05, num\_leaves = 63, min\_child\_samples = 20, feature/bagging fraction = 0.8, L1/L2 = 0.1, up to 500 rounds, early stopping (50) |
| **XGBoost**             | lr = 0.05, max\_depth = 6, subsample = 0.8, colsample\_bytree = 0.8, L1/L2 = 0.1, up to 500 rounds, early stopping (50)                   |
| **CatBoost**            | lr = 0.05, depth = 6, min\_data\_in\_leaf = 20, rsm = 0.8, subsample = 0.8, L2 = 3.0, early stopping (50)                                 |
| **MLP architecture**    | d → 256 → 128 → K, BatchNorm + ReLU + Dropout (0.1)                                                                                       |
| **MLP training**        | AdamW (lr = 10⁻³, wd = 10⁻⁴), gradient clip ℓ₂ = 1.0, cosine annealing, 200 epochs, patience 20                                           |
| **MC-Dropout**          | T = 30 stochastic forward passes; BatchNorm in eval mode, dropout active                                                                  |
| **Deep ensemble**       | M ∈ {3, 5, 10} members, random init only (no bootstrap), arithmetic mean of softmax outputs                                               |
| **TabPFN v2**           | Default pretrained checkpoint, in-context inference only                                                                                  |
| **Temperature scaling** | scalar T > 0, L-BFGS on `X_val_cal` NLL                                                                                                   |
| **MLR**                 | sklearn `LogisticRegression`, solver `lbfgs`, C = 1.0                                                                                     |
| **Isotonic regression** | one-vs-rest per class, renormalised                                                                                                       |
| **Dirichlet ODIR**      | reg\_lambda = reg\_mu = 1e-3 (library defaults)                                                                                           |
| **Member-level TS**     | per-member scalar T\_m, averaged post-softmax                                                                                             |
| **ECE bins**            | mean across {10, 15, 20} equal-width bins                                                                                                 |
| **Conformal α**         | 0.1 (target coverage 0.90), APS scores on `X_val_cal`                                                                                     |


---

## Affiliation

Faculty of Electrical Engineering, Computer Science and Information Technology Osijek
Josip Juraj Strossmayer University of Osijek, Croatia

---

## License

This repository is released under the [MIT License](LICENSE). The OpenML-CC18 datasets used by this benchmark are governed by their respective licenses on the [OpenML platform](https://www.openml.org).
