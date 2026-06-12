"""
Figure 3: four-panel diagnostic of calibration effects.

(a) Delta-NLL vs Delta-ECE under isotonic regression
    Shows the "isotonic trap": ECE improves while NLL worsens.

(b) Delta-NLL vs Delta-ECE under temperature scaling
    Shows that both metrics tend to move together.

(c) Raw vs calibrated ECE under temperature scaling

(d) Raw vs calibrated ECE under isotonic regression

Inputs:
    results_raw.csv

Outputs:
    figure3_four_panel.pdf
    figure3_four_panel.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = Path("results_raw.csv")
OUT_PDF = Path("figure3_four_panel.pdf")
OUT_PNG = Path("figure3_four_panel.png")

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

color_map = {
    "lgbm": "#3498DB",
    "xgboost": "#5DADE2",
    "catboost": "#2E86C1",
    "single_mlp": "#27AE60",
    "mc_dropout": "#16A085",
    "deep_ensemble_m3": "#F39C12",
    "deep_ensemble": "#E67E22",
    "deep_ensemble_m10": "#D35400",
}


# =============================================================================
# Journal-quality plotting defaults
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    # Large, journal-readable text
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 13,

    # Keep text editable/searchable in vector outputs
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # High-quality raster output
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,

    # Cleaner minus signs
    "axes.unicode_minus": False,
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
    "cal_ece_mean",
}

missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Median across seeds within each task/model/calibrator
agg = (
    df.groupby(["task_id", "model", "calibrator"], observed=True)[
        ["cal_nll", "cal_ece_mean"]
    ]
    .median()
    .reset_index()
)


# =============================================================================
# Helper functions
# =============================================================================

def get_pair(model: str, cal_a: str, cal_b: str, metric: str):
    """
    Return paired metric values for two calibrators, aligned by task_id.
    """
    a = (
        agg[(agg["model"] == model) & (agg["calibrator"] == cal_a)]
        .set_index("task_id")[metric]
    )

    b = (
        agg[(agg["model"] == model) & (agg["calibrator"] == cal_b)]
        .set_index("task_id")[metric]
    )

    common = a.index.intersection(b.index)

    return a.loc[common], b.loc[common]


def build_delta_data(calibrator: str):
    """
    Build per-model arrays for:
        Delta NLL = NLL(calibrated) - NLL(raw)
        Delta ECE = ECE(calibrated) - ECE(raw)
    """
    data = []

    for model in model_order:
        raw_nll, cal_nll = get_pair(model, "none", calibrator, "cal_nll")
        raw_ece, cal_ece = get_pair(model, "none", calibrator, "cal_ece_mean")

        common = (
            raw_nll.index
            .intersection(cal_nll.index)
            .intersection(raw_ece.index)
            .intersection(cal_ece.index)
        )

        if len(common) == 0:
            continue

        delta_nll = cal_nll.loc[common].values - raw_nll.loc[common].values
        delta_ece = cal_ece.loc[common].values - raw_ece.loc[common].values

        data.append({
            "model": model,
            "delta_nll": delta_nll,
            "delta_ece": delta_ece,
        })

    return data


def build_ece_pair_data(calibrator: str):
    """
    Build per-model arrays for raw ECE and calibrated ECE.
    """
    data = []

    for model in model_order:
        raw_ece, cal_ece = get_pair(model, "none", calibrator, "cal_ece_mean")

        if len(raw_ece) == 0:
            continue

        data.append({
            "model": model,
            "raw_ece": raw_ece.values,
            "cal_ece": cal_ece.values,
        })

    return data


def padded_limits(values, padding=0.08):
    """
    Return padded min/max limits for a list or array of values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return -1.0, 1.0

    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if np.isclose(vmin, vmax):
        margin = max(abs(vmin) * 0.1, 0.01)
    else:
        margin = (vmax - vmin) * padding

    return vmin - margin, vmax + margin


def clean_axis(ax):
    """
    Apply journal-style axis cleanup.
    """
    ax.grid(
        True,
        color="#D5D8DC",
        linestyle="-",
        linewidth=0.7,
        alpha=0.75,
        zorder=0,
    )

    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("#566573")
    ax.spines["bottom"].set_color("#566573")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(axis="both", which="major", pad=5)


def scatter_delta_panel(ax, data, title, shade_trap=False):
    """
    Plot Delta-ECE vs Delta-NLL.
    """
    all_dece = np.concatenate([d["delta_ece"] for d in data])
    all_dnll = np.concatenate([d["delta_nll"] for d in data])

    xlim = padded_limits(all_dece, padding=0.10)
    ylim = padded_limits(all_dnll, padding=0.10)

    # Make sure zero is visible
    xlim = (min(xlim[0], -0.005), max(xlim[1], 0.005))
    ylim = (min(ylim[0], -0.005), max(ylim[1], 0.005))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Shade the isotonic-trap quadrant:
    # Delta ECE < 0 and Delta NLL > 0.
    if shade_trap and xlim[0] < 0 and ylim[1] > 0:
        ax.add_patch(
            Rectangle(
                (xlim[0], 0),
                0 - xlim[0],
                ylim[1] - 0,
                facecolor="#F5B7B1",
                alpha=0.32,
                edgecolor="none",
                zorder=0,
            )
        )

        ax.text(
            xlim[0] + 0.04 * (xlim[1] - xlim[0]),
            ylim[1] - 0.08 * (ylim[1] - ylim[0]),
            "ECE$\\downarrow$  NLL$\\uparrow$\n(isotonic trap)",
            fontsize=14,
            color="#922B21",
            ha="left",
            va="top",
            fontweight="bold",
            zorder=4,
        )

    ax.axhline(0, color="#7F8C8D", linewidth=1.2, linestyle="--", zorder=1)
    ax.axvline(0, color="#7F8C8D", linewidth=1.2, linestyle="--", zorder=1)

    for d in data:
        model = d["model"]
        ax.scatter(
            d["delta_ece"],
            d["delta_nll"],
            s=58,
            alpha=0.75,
            color=color_map[model],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    ax.set_xlabel(r"$\Delta$ECE = ECE(calibrated) $-$ ECE(raw)")
    ax.set_ylabel(r"$\Delta$NLL = NLL(calibrated) $-$ NLL(raw)")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)

    clean_axis(ax)


def scatter_ece_pair_panel(ax, data, title):
    """
    Plot raw ECE vs calibrated ECE.
    """
    all_values = []

    for d in data:
        all_values.extend(d["raw_ece"])
        all_values.extend(d["cal_ece"])

    all_values = np.asarray(all_values, dtype=float)
    all_values = all_values[np.isfinite(all_values)]

    lim = float(np.max(all_values)) * 1.08
    lim = max(lim, 0.01)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    for d in data:
        model = d["model"]
        ax.scatter(
            d["raw_ece"],
            d["cal_ece"],
            s=58,
            alpha=0.75,
            color=color_map[model],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )

    ax.plot(
        [0, lim],
        [0, lim],
        color="#7F8C8D",
        linewidth=1.3,
        linestyle="--",
        zorder=1,
    )

    ax.text(
        0.04 * lim,
        0.93 * lim,
        "below line = lower ECE",
        fontsize=13.5,
        color="#566573",
        ha="left",
        va="top",
    )

    ax.set_xlabel("Raw ECE")
    ax.set_ylabel("Calibrated ECE")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)

    clean_axis(ax)


# =============================================================================
# Build plotting data
# =============================================================================

iso_delta_data = build_delta_data("isotonic")
temp_delta_data = build_delta_data("temp")

temp_ece_data = build_ece_pair_data("temp")
iso_ece_data = build_ece_pair_data("isotonic")


# =============================================================================
# Plot
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16.8, 13.6))

scatter_delta_panel(
    axes[0, 0],
    iso_delta_data,
    "(a) Isotonic Regression",
    shade_trap=True,
)

scatter_delta_panel(
    axes[0, 1],
    temp_delta_data,
    "(b) Temperature Scaling",
    shade_trap=False,
)

scatter_ece_pair_panel(
    axes[1, 0],
    temp_ece_data,
    "(c) Temperature Scaling\nRaw vs Calibrated ECE",
)

scatter_ece_pair_panel(
    axes[1, 1],
    iso_ece_data,
    "(d) Isotonic Regression\nRaw vs Calibrated ECE",
)


# Shared legend
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markersize=9.5,
        markerfacecolor=color_map[model],
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=model_labels[i],
    )
    for i, model in enumerate(model_order)
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.5, 0.915),
    borderpad=0.8,
    columnspacing=1.4,
    handletextpad=0.5,
)


fig.suptitle(
    "Calibration effects across 36 OpenML-CC18 datasets\n"
    "(each point = one dataset, median over 5 seeds)",
    fontsize=21,
    fontweight="bold",
    y=0.985,
)


fig.subplots_adjust(
    left=0.075,
    right=0.985,
    bottom=0.075,
    top=0.80,
    wspace=0.28,
    hspace=0.38,
)


# Save
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=600)
plt.close(fig)

print(f"Saved {OUT_PDF} and {OUT_PNG}")