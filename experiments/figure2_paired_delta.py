"""
Figure 2: per-dataset NLL improvement from Temperature Scaling.

One row per model class.
Each dot is one dataset.
Black vertical tick is the median.

Right-margin annotations report:
    median delta,
    W/T/L counts,
    uncorrected Wilcoxon p-value.

Inputs:
    results_raw.csv

Outputs:
    figure2_paired_delta_nll.pdf
    figure2_paired_delta_nll.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from scipy.stats import wilcoxon


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = Path("results_raw.csv")
OUT_PDF = Path("figure2_paired_delta_nll.pdf")
OUT_PNG = Path("figure2_paired_delta_nll.png")

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
    r"Ens. ($M{=}3$)",
    r"Ens. ($M{=}5$)",
    r"Ens. ($M{=}10$)",
]

TOL = 1e-3


# =============================================================================
# Journal-quality plotting defaults
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    # Large, journal-readable text
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 17,

    # Keep text editable/searchable in vector outputs
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # High-resolution raster output
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


# =============================================================================
# Load and aggregate data
# =============================================================================

df = pd.read_csv(INPUT_CSV)

required_cols = {
    "task_id",
    "model",
    "calibrator",
    "cal_nll",
}

missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Median across seeds within each task/model/calibrator
agg = (
    df.groupby(["task_id", "model", "calibrator"], observed=True)[["cal_nll"]]
    .median()
    .reset_index()
)


# =============================================================================
# Compute paired Temperature Scaling deltas
# =============================================================================

results = []

for model in model_order:
    raw = (
        agg[(agg["model"] == model) & (agg["calibrator"] == "none")]
        .set_index("task_id")["cal_nll"]
    )

    temp = (
        agg[(agg["model"] == model) & (agg["calibrator"] == "temp")]
        .set_index("task_id")["cal_nll"]
    )

    common_tasks = raw.index.intersection(temp.index)

    raw = raw.loc[common_tasks]
    temp = temp.loc[common_tasks]

    delta = temp.values - raw.values

    wins = int((delta < -TOL).sum())
    ties = int((np.abs(delta) <= TOL).sum())
    losses = int((delta > TOL).sum())

    try:
        p_value = wilcoxon(
            temp.values,
            raw.values,
            zero_method="wilcox",
            alternative="two-sided",
        ).pvalue
    except ValueError:
        p_value = 1.0

    results.append({
        "model": model,
        "deltas": delta,
        "median": float(np.median(delta)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p": float(p_value),
        "n": len(delta),
    })


def format_p_value(p: float) -> str:
    """Format p-values cleanly for figure annotations."""
    if p < 1e-4:
        return r"$p < 0.0001$"
    return rf"$p$ = {p:.4f}"


def color_for(model: str) -> str:
    """Assign colors by model family."""
    if model in {"lgbm", "xgboost", "catboost"}:
        return "#3498DB"   # GBDT models
    if model == "single_mlp":
        return "#27AE60"   # Single MLP
    if model == "mc_dropout":
        return "#16A085"   # MC-Dropout
    return "#E67E22"       # Deep ensembles


# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(17.5, 9.8))

n_models = len(model_order)
positions = np.arange(n_models)[::-1]

all_deltas = np.concatenate([r["deltas"] for r in results])

# Keep the familiar range, but expand automatically if any values fall outside it.
xmin = min(-0.05, float(np.nanmin(all_deltas)) - 0.01)
xmax = max(0.05, float(np.nanmax(all_deltas)) + 0.01)

ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.7, n_models - 0.15)


# Background row striping
for idx, y in enumerate(positions):
    row_color = "#F4F6F7" if idx % 2 == 0 else "white"

    ax.axhspan(
        y - 0.45,
        y + 0.45,
        facecolor=row_color,
        edgecolor="none",
        zorder=0,
    )


# Zero reference line
ax.axvline(
    0,
    color="#2C3E50",
    linewidth=1.5,
    zorder=1,
)


# Scatter points and median ticks
rng = np.random.default_rng(0)

for y, r in zip(positions, results):
    color = color_for(r["model"])
    jitter = rng.uniform(-0.12, 0.12, size=len(r["deltas"]))

    ax.scatter(
        r["deltas"],
        np.full_like(r["deltas"], y, dtype=float) + jitter,
        s=68,
        color=color,
        alpha=0.70,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )

    # Black median tick
    ax.plot(
        [r["median"], r["median"]],
        [y - 0.34, y + 0.34],
        color="black",
        linewidth=3.4,
        solid_capstyle="round",
        zorder=3,
    )


# Y-axis labels
ax.set_yticks(positions)
ax.set_yticklabels(model_labels)
ax.tick_params(axis="y", length=0, pad=8)


# X-axis label
ax.set_xlabel(
    r"$\Delta\mathrm{NLL} = "
    r"\mathrm{NLL}(\mathrm{Temp.\ Scaling}) - "
    r"\mathrm{NLL}(\mathrm{Raw})$",
    labelpad=12,
)

ax.tick_params(axis="x", pad=6)


# Direction hints placed inside plotting area, away from title
hint_y = positions[0] + 0.48

ax.text(
    xmin * 0.55,
    hint_y,
    "← TS improves NLL",
    ha="center",
    va="center",
    fontsize=15,
    color="#1E8449",
    fontweight="bold",
)

ax.text(
    xmax * 0.55,
    hint_y,
    "TS hurts NLL →",
    ha="center",
    va="center",
    fontsize=15,
    color="#A93226",
    fontweight="bold",
)


# Right-side annotation block
# x coordinate is in axes fraction; y coordinate is in data coordinates.
annotation_transform = blended_transform_factory(ax.transAxes, ax.transData)

for y, r in zip(positions, results):
    ax.text(
        1.025,
        y,
        f"med $\\Delta$ = {r['median']:+.4f}\n"
        f"W/T/L = {r['wins']}/{r['ties']}/{r['losses']}\n"
        f"{format_p_value(r['p'])}",
        transform=annotation_transform,
        fontsize=14,
        va="center",
        ha="left",
        color="#1B2631",
        linespacing=1.25,
    )


# Title
ax.set_title(
    "Per-dataset NLL improvement from Temperature Scaling\n"
    r"(each dot = one dataset; black tick = median; $n = 36$ datasets per model)",
    pad=20,
    fontweight="bold",
)


# Light x-axis grid
ax.grid(
    axis="x",
    color="#D5D8DC",
    linestyle="-",
    linewidth=0.7,
    alpha=0.7,
    zorder=0,
)


# Clean frame
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

ax.spines["bottom"].set_color("#566573")
ax.spines["bottom"].set_linewidth(1.2)


# Leave enough room for right-side statistics
fig.subplots_adjust(
    left=0.13,
    right=0.72,
    top=0.84,
    bottom=0.16,
)


# Save
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=600)
plt.close(fig)

print(f"Saved {OUT_PDF} and {OUT_PNG}")
