"""
Calibration vs Deep Ensembles — Tabular Uncertainty Benchmark
=============================================================

Package layout
--------------
    calibration            Post-hoc calibrators (Identity, Temp, MLR, Isotonic)
    datasets               OpenML-CC18 loading and 60/20/20 splitting
    evaluation             Full factorial benchmark runner
    metrics                NLL, ECE (multi-bin), Brier, accuracy
    models                 LightGBM, XGBoost, SingleMLP, DeepEnsemble (M=5)
    statistical_analysis   Paired Wilcoxon + global Holm–Bonferroni
    visualization          Summary tables and publication figures
    utils                  Shared constants and helpers
"""

__version__ = "1.0.0"
