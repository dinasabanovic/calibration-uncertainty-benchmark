"""
Figure 1: heatmap of per-dataset median uncertainty metrics by (model, calibrator).

Inputs:
    results_raw.csv with columns:
    task_id, model, calibrator, cal_nll, cal_ece_mean, cal_brier_score

Outputs:
    figure1_heatmap.pdf
    figure1_heatmap.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = Path("results_raw.csv")
OUT_PDF = Path("figure1_heatmap.pdf")
OUT_PNG = Path("figure1_heatmap.png")

cal_order = ["none", "temp", "logistic", "isotonic", "dirichlet"]
cal_labels = [
    "Raw",
    "Temp. Scaling",
    "MLR",
    "Isotonic Reg.",
    "Dirichlet Cal.",
]

model_order = [
    "lgbm",
    "xgboost",
    "catboost",
    "single_mlp",
    "mc_dropout",
    "deep_ensemble_m3",
    "deep_ensemble",
    "deep_ensemble_m10",
]

model_labels = [
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "SingleMLP",
    "MC-Dropout",
    "Ens.\n($M{=}3$)",
    "Ens.\n($M{=}5$)",
    "Ens.\n($M{=}10$)",
]

metric_specs = [
    ("cal_nll", r"NLL $\downarrow$"),
    ("cal_ece_mean", r"ECE $\downarrow$"),
    ("cal_brier_score", r"Brier Score $\downarrow$"),
]


# =============================================================================
# Journal-style plotting defaults
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": 17,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 17,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


# =============================================================================
# Data
# =============================================================================

df = pd.read_csv(INPUT_CSV)

required_cols = {
    "task_id",
    "model",
    "calibrator",
    "cal_nll",
    "cal_ece_mean",
    "cal_brier_score",
}

missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Two-stage aggregation:
# 1. median across seeds within (task, model, calibrator)
# 2. median across datasets
agg = (
    df.groupby(["task_id", "model", "calibrator"], observed=True)[
        ["cal_nll", "cal_ece_mean", "cal_brier_score"]
    ]
    .median()
    .reset_index()
)

med = agg.groupby(["model", "calibrator"], observed=True)[
    ["cal_nll", "cal_ece_mean", "cal_brier_score"]
].median()


def build_matrix(metric: str) -> np.ndarray:
    """Return calibrator x model matrix for one metric."""
    matrix = np.full((len(cal_order), len(model_order)), np.nan)

    for j, model in enumerate(model_order):
        for i, calibrator in enumerate(cal_order):
            key = (model, calibrator)
            if key in med.index:
                matrix[i, j] = med.loc[key, metric]

    return matrix


metrics = {
    label: build_matrix(metric)
    for metric, label in metric_specs
}


# =============================================================================
# Plot
# =============================================================================

fig = plt.figure(figsize=(17.5, 15.2), constrained_layout=False)
gs = fig.add_gridspec(
    nrows=3,
    ncols=2,
    width_ratios=[1, 0.035],
    height_ratios=[1, 1, 1],
    wspace=0.08,
    hspace=0.34,
)

axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
cax = fig.add_subplot(gs[:, 1])

cmap = plt.get_cmap("RdYlBu_r").copy()
cmap.set_bad(color="#F2F2F2")

mlr_row = cal_order.index("logistic")

for ax, (title, matrix) in zip(axes, metrics.items()):
    masked = np.ma.masked_invalid(matrix)

    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)

    im = ax.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        interpolation="nearest",
    )

    # Cell annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "—" if np.isnan(value) else f"{value:.3f}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=17,
                color="black",
                fontweight="semibold",
            )

    # Axes labels
    ax.set_xticks(np.arange(len(model_order)))
    ax.set_xticklabels(model_labels)

    ax.set_yticks(np.arange(len(cal_order)))
    ax.set_yticklabels(cal_labels)

    ax.set_title(title, loc="left", pad=10, fontweight="bold")

    # Red outline for MLR row
    ax.add_patch(
        Rectangle(
            (-0.5, mlr_row - 0.5),
            len(model_order),
            1.0,
            fill=False,
            edgecolor="#B22222",
            linewidth=2.8,
            clip_on=False,
        )
    )

    # Add subtle cell gridlines for readability
    ax.set_xticks(np.arange(-0.5, len(model_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cal_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        pad=6,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)


# Shared semantic colorbar: best to worst
gradient = np.linspace(0, 1, 512).reshape(512, 1)[::-1]

cax.imshow(
    gradient,
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
)

cax.set_xticks([])
cax.set_yticks([])

cax.text(
    0.5,
    1.015,
    "best",
    transform=cax.transAxes,
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold",
)

cax.text(
    0.5,
    -0.015,
    "worst",
    transform=cax.transAxes,
    ha="center",
    va="top",
    fontsize=14,
    fontweight="bold",
)

for spine in cax.spines.values():
    spine.set_visible(False)


fig.suptitle(
    "Median uncertainty metrics by model and calibration method\n"
    r"($n = 36$ datasets; per-dataset medians of 5-seed medians; red border = MLR)",
    fontsize=19,
    fontweight="bold",
    y=0.985,
)

fig.subplots_adjust(
    left=0.105,
    right=0.93,
    top=0.90,
    bottom=0.075,
)

# Save outputs
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=600)
plt.close(fig)

print(f"Saved {OUT_PDF} and {OUT_PNG}")
