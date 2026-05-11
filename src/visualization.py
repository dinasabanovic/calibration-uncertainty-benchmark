import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

# ── Typography ────────────────────────────────────────────────────────────────
TITLE_FONT  = 14
LABEL_FONT  = 13
TICK_FONT   = 11
LEGEND_FONT = 11
ANNOT_FONT  = 10

matplotlib.rcParams.update({
    "font.family":           "serif",
    "font.size":             12,
    "axes.titlesize":        TITLE_FONT,
    "axes.labelsize":        LABEL_FONT,
    "xtick.labelsize":       TICK_FONT,
    "ytick.labelsize":       TICK_FONT,
    "legend.fontsize":       LEGEND_FONT,
    "legend.title_fontsize": LEGEND_FONT,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.05,
    "text.usetex":           False,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
})

CM   = 1 / 2.54
FULL = 17.5 * CM
HALF = 8.5  * CM

# ── Colours ───────────────────────────────────────────────────────────────────
MODEL_COLOR = {
    "lgbm":          "#4E79A7",
    "xgboost":       "#9467BD",
    "single_mlp":    "#2CA02C",
    "deep_ensemble": "#FF7F0E",
}

MODEL_LABELS = {
    "lgbm":          "LightGBM",
    "xgboost":       "XGBoost",
    "single_mlp":    "SingleMLP",
    "deep_ensemble": "Ensemble (M=5)",
}

CAL_LABELS = {
    "none":     "Raw (No Calibration)",
    "temp":     "Temperature Scaling",
    "logistic": "Logistic Recal. (MLR)",
    "isotonic": "Isotonic Regression",
}

PALETTE = {f"{m}/{c}": MODEL_COLOR[m]
           for m in MODEL_COLOR
           for c in ["none", "temp", "logistic", "isotonic"]}

CONDITION_ORDER = [
    "lgbm/none",          "lgbm/temp",          "lgbm/logistic",          "lgbm/isotonic",
    "xgboost/none",       "xgboost/temp",       "xgboost/logistic",       "xgboost/isotonic",
    "single_mlp/none",    "single_mlp/temp",    "single_mlp/logistic",    "single_mlp/isotonic",
    "deep_ensemble/none", "deep_ensemble/temp",  "deep_ensemble/logistic", "deep_ensemble/isotonic",
]

CONDITION_LABELS = {
    "lgbm/none":               "LightGBM / Raw",
    "lgbm/temp":               "LightGBM / Temp. Scaling",
    "lgbm/logistic":           "LightGBM / MLR",
    "lgbm/isotonic":           "LightGBM / Isotonic",
    "xgboost/none":            "XGBoost / Raw",
    "xgboost/temp":            "XGBoost / Temp. Scaling",
    "xgboost/logistic":        "XGBoost / MLR",
    "xgboost/isotonic":        "XGBoost / Isotonic",
    "single_mlp/none":         "SingleMLP / Raw",
    "single_mlp/temp":         "SingleMLP / Temp. Scaling",
    "single_mlp/logistic":     "SingleMLP / MLR",
    "single_mlp/isotonic":     "SingleMLP / Isotonic",
    "deep_ensemble/none":      "Ensemble / Raw",
    "deep_ensemble/temp":      "Ensemble / Temp. Scaling",
    "deep_ensemble/logistic":  "Ensemble / MLR",
    "deep_ensemble/isotonic":  "Ensemble / Isotonic",
}

METRIC_LABELS = {
    "cal_nll":         "NLL (Negative Log-Likelihood)",
    "cal_ece_mean":    "ECE (Expected Calibration Error, mean over 3 bin sizes)",
    "cal_brier_score": "Brier Score",
    "cal_accuracy":    "Accuracy",
}

METRIC_SHORT = {
    "cal_nll":         "NLL",
    "cal_ece_mean":    "ECE",
    "cal_brier_score": "Brier",
    "cal_accuracy":    "Acc",
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise calibrator names and add condition column."""
    df = df.copy()
    df["calibrator"] = df["calibrator"].replace({"platt": "logistic"})
    df["condition"]  = df["model"] + "/" + df["calibrator"]
    return df


def _save(fig: plt.Figure, fig_dir: Optional[Path], stem: str) -> None:
    if fig_dir is None:
        plt.close(fig)
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
        logger.info(f"  Saved: {stem}.{ext}")
    plt.close(fig)


def _model_legend(ax, **kwargs):
    handles = [mpatches.Patch(facecolor=MODEL_COLOR[m], label=MODEL_LABELS[m])
               for m in MODEL_COLOR]
    ax.legend(handles=handles, title="Model family",
              fontsize=LEGEND_FONT, title_fontsize=LEGEND_FONT,
              framealpha=0.9, **kwargs)


def _per_dataset_medians(df: pd.DataFrame, condition: str, metric: str) -> pd.Series:
    """Return per-dataset median across seeds for one condition."""
    return (
        df[df["condition"] == condition]
        .groupby("dataset_name")[metric]
        .median()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1  —  Metric heatmap grid  (matches paper Figure 1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_heatmap_grid(
    df_results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    fig_dir: Optional[Path] = None,
) -> None:
    """
    Coloured table: rows = calibration method, columns = model × metric.
    Cell values are per-dataset medians of 5-seed medians (correct aggregation).
    Colour normalised within each metric column (blue = lower = better).
    MLR row is hatched / bordered to signal cautionary status.

    Matches the layout of Figure 1 in the paper.
    """
    if metrics is None:
        metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score"]

    df = _prep(df_results)
    models   = ["lgbm", "xgboost", "single_mlp", "deep_ensemble"]
    cal_rows = ["none", "temp", "logistic", "isotonic"]
    cal_row_labels = {
        "none":     "Raw",
        "temp":     "Temp. Scaling",
        "logistic": "MLR",
        "isotonic": "Isotonic Reg.",
    }

    # Build value matrix: shape (n_cal, n_models × n_metrics)
    col_labels = []
    val_matrix = []

    for cal in cal_rows:
        row_vals = []
        for model in models:
            cond = f"{model}/{cal}"
            for metric in metrics:
                per_ds = _per_dataset_medians(df, cond, metric)
                row_vals.append(per_ds.median() if len(per_ds) > 0 else np.nan)
        val_matrix.append(row_vals)

    for model in models:
        for metric in metrics:
            col_labels.append(f"{MODEL_LABELS[model]}\n{METRIC_SHORT[metric]}")

    val_arr = np.array(val_matrix)  # (4, 4×n_metrics)

    # Normalise per metric-block for colouring
    n_models  = len(models)
    n_metrics = len(metrics)
    norm_arr  = val_arr.copy()
    for m_idx in range(n_metrics):
        cols = list(range(m_idx, n_models * n_metrics, n_metrics))
        block = val_arr[:, cols]
        lo, hi = np.nanmin(block), np.nanmax(block)
        if hi > lo:
            norm_arr[:, cols] = (block - lo) / (hi - lo)
        else:
            norm_arr[:, cols] = 0.5

    n_rows = len(cal_rows)
    n_cols = len(col_labels)
    cell_w = 1.5
    cell_h = 0.75

    fig, ax = plt.subplots(figsize=(n_cols * cell_w + 2.5, n_rows * cell_h + 2.5))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    cmap = matplotlib.cm.RdYlBu_r

    for ri, cal in enumerate(cal_rows):
        for ci in range(n_cols):
            val  = val_arr[ri, ci]
            norm = norm_arr[ri, ci]
            color = cmap(norm) if not np.isnan(norm) else (0.9, 0.9, 0.9, 1.0)
            rect = mpatches.FancyBboxPatch(
                (ci - 0.48, ri - 0.46), 0.96, 0.92,
                boxstyle="round,pad=0.02",
                linewidth=2.5 if cal == "logistic" else 0.5,
                edgecolor="#cc0000" if cal == "logistic" else "white",
                facecolor=color,
            )
            ax.add_patch(rect)
            txt_color = "black" if 0.2 < norm < 0.8 else ("white" if norm >= 0.8 else "black")
            ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                    fontsize=ANNOT_FONT + 1, fontweight="bold", color=txt_color)

    # Row labels
    for ri, cal in enumerate(cal_rows):
        ax.text(-0.55, ri, cal_row_labels[cal], ha="right", va="center",
                fontsize=TICK_FONT + 1, fontweight="bold" if cal == "logistic" else "normal",
                color="#cc0000" if cal == "logistic" else "black")

    # Column labels — group separators
    for ci, label in enumerate(col_labels):
        ax.text(ci, -0.62, label, ha="center", va="bottom",
                fontsize=TICK_FONT - 1, rotation=0)

    # Model family separators
    for sep in range(n_metrics, n_cols, n_metrics):
        ax.axvline(sep - 0.5, color="gray", lw=1.5, alpha=0.5,
                   ymin=0, ymax=1)

    # Metric column group headers
    for m_idx, metric in enumerate(metrics):
        center = m_idx + (n_models - 1) * n_metrics / 2 + m_idx * (n_models - 1)
        # Simpler: use evenly spaced cols
        start_col = m_idx * n_models  # if metrics is the fast index: wrong
        # Recompute: cols for metric m_idx across all models are m_idx, m_idx+n_metrics, ...
        # Actually col index = model_idx * n_metrics + metric_idx
        cols_for_metric = [model_idx * n_metrics + m_idx for model_idx in range(n_models)]
        center_x = np.mean(cols_for_metric)
        ax.text(center_x, -1.1, f"{METRIC_SHORT[metric]} ↓",
                ha="center", va="bottom", fontsize=LABEL_FONT,
                fontweight="bold", color="#333333")

    # Colourbar
    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["best\n(lowest)", "mid", "worst\n(highest)"],
                        fontsize=ANNOT_FONT)
    cbar.set_label("Relative value\n(blue = lower = better)",
                   fontsize=ANNOT_FONT, labelpad=8)

    ax.set_title(
        "Figure 1. Median uncertainty metrics by model and calibration method\n"
        "(per-dataset medians of 5-seed medians, n = 36 datasets; "
        "red border = MLR, consistently worsens both NLL and ECE)",
        fontsize=TITLE_FONT, fontweight="bold", pad=30,
    )
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    _save(fig, fig_dir, "figure1_metric_heatmap_grid")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2  —  Combined 4-panel  (matches paper Figure 2)
#
#   (a) top-left  : ΔNLL vs ΔECE scatter — Isotonic Regression
#   (b) top-right : ΔNLL vs ΔECE scatter — Temperature Scaling
#   (c) bottom-left : Raw vs Cal ECE     — Temperature Scaling
#   (d) bottom-right: Raw vs Cal ECE     — Isotonic Regression
# ─────────────────────────────────────────────────────────────────────────────

def plot_figure2_combined(
    df_results: pd.DataFrame,
    fig_dir: Optional[Path] = None,
) -> None:
    """
    Four-panel diagnostic of calibration effects (paper Figure 2).

    Each point = one dataset, median over 5 seeds.
    Panels (a)(b): ΔNLL vs ΔECE — shows the isotonic trap vs. TS coherence.
    Panels (c)(d): raw vs calibrated ECE — shows ECE improvement rates.

    ΔNLL = calibrated_NLL − raw_NLL   (positive = worsened)
    ΔECE = calibrated_ECE − raw_ECE   (negative = improved)
    """
    df     = _prep(df_results)
    models = ["lgbm", "xgboost", "single_mlp", "deep_ensemble"]

    # Per-dataset medians for raw and each calibrator
    raw_nll = {}
    raw_ece = {}
    for model in models:
        raw_nll[model] = _per_dataset_medians(df, f"{model}/none", "cal_nll")
        raw_ece[model] = _per_dataset_medians(df, f"{model}/none", "cal_ece_mean")

    fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 1.1))

    scatter_pairs = [
        (axes[0, 0], "isotonic", "(a) Isotonic Regression\nΔECE improves, ΔNLL worsens"),
        (axes[0, 1], "temp",     "(b) Temperature Scaling\nBoth metrics move together"),
    ]
    ece_pairs = [
        (axes[1, 0], "temp",     "(c) Temperature Scaling\nRaw vs Calibrated ECE"),
        (axes[1, 1], "isotonic", "(d) Isotonic Regression\nRaw vs Calibrated ECE"),
    ]

    # ── Panels (a) and (b): ΔNLL vs ΔECE ──────────────────────────────────
    for ax, cal, title in scatter_pairs:
        all_dnll, all_dece = [], []
        for model in models:
            cal_nll = _per_dataset_medians(df, f"{model}/{cal}", "cal_nll")
            cal_ece = _per_dataset_medians(df, f"{model}/{cal}", "cal_ece_mean")
            ds      = raw_nll[model].index.intersection(cal_nll.index)
            dnll    = (cal_nll[ds] - raw_nll[model][ds]).values
            dece    = (cal_ece[ds] - raw_ece[model][ds]).values
            ax.scatter(dece, dnll, color=MODEL_COLOR[model], alpha=0.70,
                       s=50, edgecolors="k", linewidths=0.3, zorder=3)
            all_dnll.extend(dnll.tolist())
            all_dece.extend(dece.tolist())

        ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.6)
        ax.axvline(0, color="gray", lw=1.0, ls="--", alpha=0.6)
        ax.set_xlabel("ΔECE (calibrated − raw)", fontsize=LABEL_FONT)
        ax.set_ylabel("ΔNLL (calibrated − raw)", fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT)
        ax.set_title(title, fontsize=TITLE_FONT, fontweight="bold", pad=8)

        # Shade the isotonic-trap quadrant (ΔECE < 0, ΔNLL > 0)
        if cal == "isotonic":
            xl, xr = ax.get_xlim()
            yb, yt = ax.get_ylim()
            ax.axhspan(0, max(yt, max(all_dnll) * 1.1),
                       xmin=0, xmax=0.5, alpha=0.08, color="red", zorder=0)
            n_trap = sum(d > 0 for d in all_dnll) + sum(e < 0 for e in all_dece)
            ax.text(0.03, 0.97,
                    "ECE↓  NLL↑\n(isotonic trap)",
                    transform=ax.transAxes, fontsize=ANNOT_FONT,
                    va="top", ha="left", color="#cc0000", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cc0000", alpha=0.8))

        sns.despine(ax=ax)

    _model_legend(axes[0, 1], loc="lower right")

    # ── Panels (c) and (d): Raw vs Calibrated ECE ─────────────────────────
    all_ece_vals = []
    for model in models:
        for cal in ["temp", "isotonic"]:
            per_ds = _per_dataset_medians(df, f"{model}/{cal}", "cal_ece_mean")
            all_ece_vals.extend(per_ds.values.tolist())
            all_ece_vals.extend(raw_ece[model].values.tolist())

    lo = max(0.0, np.nanpercentile(all_ece_vals, 1) * 0.85)
    hi = np.nanpercentile(all_ece_vals, 99) * 1.10

    for ax, cal, title in ece_pairs:
        win_counts = {}
        for model in models:
            cal_ece_per_ds = _per_dataset_medians(df, f"{model}/{cal}", "cal_ece_mean")
            ds = raw_ece[model].index.intersection(cal_ece_per_ds.index)
            ax.scatter(raw_ece[model][ds], cal_ece_per_ds[ds],
                       color=MODEL_COLOR[model], alpha=0.72, s=50,
                       edgecolors="k", linewidths=0.3, zorder=3,
                       label=MODEL_LABELS[model])
            n_better = int((cal_ece_per_ds[ds] < raw_ece[model][ds]).sum())
            win_counts[model] = (n_better, len(ds))

        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel("Raw ECE", fontsize=LABEL_FONT)
        ax.set_ylabel("Calibrated ECE", fontsize=LABEL_FONT)
        ax.tick_params(labelsize=TICK_FONT)
        ax.set_title(title, fontsize=TITLE_FONT, fontweight="bold", pad=8)
        ax.text(0.04, 0.96, "Below diagonal\n= ECE improved",
                transform=ax.transAxes, fontsize=ANNOT_FONT,
                va="top", color="green")

        # Win counts annotation per model
        legend_text = "\n".join(
            f"{MODEL_LABELS[m]} {wc[0]}/{wc[1]} ↓"
            for m, wc in win_counts.items()
        )
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
                fontsize=ANNOT_FONT - 1, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="gray"))
        sns.despine(ax=ax)

    fig.suptitle(
        "Figure 2. Four-panel diagnostic of calibration effects across 36 OpenML-CC18 datasets\n"
        "(each point = one dataset, median over 5 seeds)",
        fontsize=TITLE_FONT, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _save(fig, fig_dir, "figure2_combined_four_panel")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3  —  NLL boxplots, per-dataset distribution
#              (recommended addition to the paper)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_boxplots(
    df_results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    fig_dir: Optional[Path] = None,
) -> None:
    """
    Per-dataset distribution of calibrated NLL / ECE / Brier Score.

    Uses CORRECT aggregation: each observation = per-dataset median across seeds
    (36 values per condition), not raw pooled (180 values).

    Why this matters: the median line in each box is the estimand used in
    the Wilcoxon tests.  Showing the distribution makes the statistical
    power argument (n=36 paired observations) visually transparent.

    Conditions on Y axis (horizontal) so all 16 labels are readable.
    MLR boxes hatched in red to signal cautionary status.
    """
    if metrics is None:
        metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score"]

    df      = _prep(df_results)
    present = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    labels  = [CONDITION_LABELS.get(c, c) for c in present]
    colors  = [PALETTE.get(c, "#888") for c in present]

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue

        # CORRECT: per-dataset median across seeds, then boxplot over datasets
        data = [
            df[df["condition"] == c].groupby("dataset_name")[metric].median().values
            for c in present
        ]

        fig, ax = plt.subplots(figsize=(FULL, len(present) * 0.62 + 2.0))

        bp = ax.boxplot(
            data, vert=False, patch_artist=True, labels=labels,
            medianprops={"color": "black", "linewidth": 2.5},
            flierprops={"marker": "o", "markersize": 3.5, "alpha": 0.5, "linewidth": 0},
            boxprops={"linewidth": 1.3},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, color, cond in zip(bp["boxes"], colors, present):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)
            if "/logistic" in cond:
                patch.set_hatch("///")
                patch.set_edgecolor("#cc0000")
                patch.set_linewidth(2.0)

        suffix = ["a", "b", "c"][i]
        ax.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=LABEL_FONT)
        ax.set_title(
            f"Figure 3{suffix}. {METRIC_SHORT.get(metric, metric)} "
            f"Distribution Across 36 Datasets per Condition\n"
            f"(each box = 36 per-dataset medians; "
            f"box median = estimand used in Wilcoxon tests; "
            f"hatching = MLR)",
            fontsize=TITLE_FONT, fontweight="bold", pad=10,
        )
        ax.tick_params(axis="y", labelsize=TICK_FONT)
        ax.tick_params(axis="x", labelsize=TICK_FONT)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # Separator lines between model families
        for sep in range(4, len(present), 4):
            ax.axhline(sep + 0.5, color="gray", lw=0.8, ls="--", alpha=0.5)

        _model_legend(ax, loc="lower right")
        sns.despine(ax=ax)
        fig.tight_layout()
        _save(fig, fig_dir, f"figure3{suffix}_boxplot_{metric.replace('cal_', '')}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures 4 & 5  —  Per-dataset ranking heatmaps (appendix)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ranking_heatmap(
    df_results: pd.DataFrame,
    metric: str = "cal_nll",
    fig_dir: Optional[Path] = None,
    fig_num: int = 4,
) -> None:
    """
    Full-width heatmap: datasets × conditions, cell = rank (1 = best).
    Uses per-dataset median across seeds (correct aggregation).
    """
    df  = _prep(df_results)
    # Correct: per-dataset median across seeds
    agg = (
        df.groupby(["dataset_name", "condition"])[metric]
        .median()
        .unstack("condition")
    )
    col_order = [c for c in CONDITION_ORDER if c in agg.columns]
    ranks     = agg[col_order].rank(axis=1, method="min", ascending=True)
    ranks.columns = [CONDITION_LABELS.get(c, c) for c in col_order]
    ranks     = ranks.sort_index()

    n_ds   = len(ranks)
    n_cond = len(col_order)
    fig_h  = max(n_ds * 0.42 + 2.5, 9)
    ann_fs = max(7, min(10, 220 // n_ds))

    fig, ax = plt.subplots(figsize=(FULL, fig_h))

    sns.heatmap(
        ranks, ax=ax, cmap="RdYlGn_r",
        annot=True, fmt=".0f", annot_kws={"size": ann_fs},
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Rank (1 = best)", "shrink": 0.45,
                  "aspect": 25, "pad": 0.02},
        vmin=1, vmax=n_cond,
    )
    ax.set_title(
        f"Figure {fig_num}. Per-Dataset Condition Rankings — "
        f"{METRIC_SHORT.get(metric, metric)} (lower is better)\n"
        f"(rank 1 = best; {n_cond} conditions × {n_ds} datasets; "
        f"per-dataset median over 5 seeds)",
        fontsize=TITLE_FONT, fontweight="bold", pad=12,
    )
    ax.set_ylabel("Dataset", fontsize=LABEL_FONT)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=max(8, TICK_FONT-1))
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=max(8, TICK_FONT-1))

    for sep in range(4, n_cond, 4):
        ax.axvline(sep, color="black", lw=1.8, alpha=0.7)

    fig.tight_layout()
    suffix = metric.replace("cal_", "").replace("_mean", "")
    _save(fig, fig_dir, f"figure{fig_num}_ranking_heatmap_{suffix}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6  —  Training time
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_time(
    df_results: pd.DataFrame,
    fig_dir: Optional[Path] = None,
) -> None:
    """Horizontal boxplot of training time per model, log scale, medians annotated."""
    df    = _prep(df_results)
    dedup = df.drop_duplicates(subset=["task_id", "model", "seed"])[
        ["model", "train_time_s"]
    ]

    order   = ["lgbm", "xgboost", "single_mlp", "deep_ensemble"]
    present = [m for m in order if m in dedup["model"].unique()]
    data    = [dedup[dedup["model"] == m]["train_time_s"].values for m in present]

    fig, ax = plt.subplots(figsize=(FULL * 0.75, 5.0))

    bp = ax.boxplot(
        data, vert=False, patch_artist=True,
        labels=[MODEL_LABELS[m] for m in present],
        medianprops={"color": "black", "linewidth": 3.0},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.4, "linewidth": 0},
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.3},
        capprops={"linewidth": 1.3},
    )
    for patch, model in zip(bp["boxes"], present):
        patch.set_facecolor(MODEL_COLOR[model])
        patch.set_alpha(0.82)

    for i, (vals, model) in enumerate(zip(data, present), start=1):
        med = np.median(vals)
        ax.text(med * 1.4, i, f"median = {med:.1f} s",
                va="center", fontsize=ANNOT_FONT + 2, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Training time (seconds, log scale)", fontsize=LABEL_FONT)
    ax.set_title(
        "Figure 6. Training Time per Model Class\n"
        "(one observation per dataset × seed; log scale; medians annotated)",
        fontsize=TITLE_FONT, fontweight="bold", pad=10,
    )
    ax.tick_params(labelsize=TICK_FONT)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.3g}s")
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, fig_dir, "figure6_training_time")


# ─────────────────────────────────────────────────────────────────────────────
# Figure S1  —  Dataset overview (supplementary)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset_overview(
    manifest_path: Path,
    fig_dir: Optional[Path] = None,
) -> None:
    """Supplementary figure: dataset size scatter + class distribution."""
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return

    df = pd.read_csv(manifest_path)
    regime_colors = {"small": "#4E79A7", "medium": "#FF7F0E", "large": "#2CA02C"}
    regime_ranges = {"small": "n < 1,000", "medium": "1,000 ≤ n < 10,000",
                     "large": "n ≥ 10,000"}

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 6.5))

    ax = axes[0]
    for regime, grp in df.groupby("size_regime"):
        ax.scatter(grp["n_features"], grp["n_samples"],
                   color=regime_colors.get(regime, "gray"),
                   s=90, alpha=0.85, edgecolors="k", linewidths=0.5,
                   label=f"{regime.capitalize()} ({regime_ranges[regime]})",
                   zorder=3)
    for _, row in df.iterrows():
        ax.annotate(row["name"][:14], (row["n_features"], row["n_samples"]),
                    fontsize=7.5, alpha=0.7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of features (log scale)", fontsize=LABEL_FONT)
    ax.set_ylabel("Number of samples (log scale)",  fontsize=LABEL_FONT)
    ax.set_title("(a) Dataset Size Distribution", fontsize=TITLE_FONT, fontweight="bold")
    ax.legend(fontsize=LEGEND_FONT, title="Size regime", title_fontsize=LEGEND_FONT)
    ax.tick_params(labelsize=TICK_FONT)
    sns.despine(ax=ax)

    ax = axes[1]
    cc = df["n_classes"].value_counts().sort_index()
    bars = ax.bar(cc.index.astype(str), cc.values,
                  color="#4E79A7", edgecolor="white", linewidth=1.0)
    for bar, val in zip(bars, cc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom",
                fontsize=ANNOT_FONT + 2, fontweight="bold")
    ax.set_xlabel("Number of classes", fontsize=LABEL_FONT)
    ax.set_ylabel("Number of datasets", fontsize=LABEL_FONT)
    ax.set_title("(b) Class Count Distribution", fontsize=TITLE_FONT, fontweight="bold")
    ax.tick_params(labelsize=TICK_FONT)
    ax.set_ylim(0, cc.max() * 1.3)
    sns.despine(ax=ax)

    n_small  = int((df["size_regime"] == "small").sum())
    n_medium = int((df["size_regime"] == "medium").sum())
    n_large  = int((df["size_regime"] == "large").sum())
    fig.suptitle(
        f"Figure S1. Overview of the {len(df)} Selected OpenML-CC18 Datasets\n"
        f"({n_small} small / {n_medium} medium / {n_large} large; "
        f"selected with seed=0 for reproducibility)",
        fontsize=TITLE_FONT, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, fig_dir, "figureS1_dataset_overview")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table  (CORRECTED: per-dataset-then-median)
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_table(
    df_results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Per-condition summary: median [Q1–Q3] across 36 per-dataset medians.

    CORRECT two-stage aggregation:
        Stage 1: for each (dataset, condition), take the median across 5 seeds
                 → 36 values per condition
        Stage 2: compute median, Q1, Q3 of those 36 values

    The old single-stage approach took the median / IQR of 180 pooled
    observations (36 datasets × 5 seeds), biasing central tendency toward
    large datasets that contribute many observations.  The maximum absolute
    NLL difference between old and new is 0.022 (lgbm/isotonic).

    Returns a LaTeX-ready DataFrame indexed by condition label.
    """
    if metrics is None:
        metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score", "cal_accuracy"]

    df = _prep(df_results)
    rows = []
    for cond in CONDITION_ORDER:
        if cond not in df["condition"].values:
            continue
        # Stage 1: per-dataset median across seeds
        per_ds = (
            df[df["condition"] == cond]
            .groupby("dataset_name")[metrics]
            .median()
        )
        row = {"Condition": CONDITION_LABELS.get(cond, cond)}
        for m in metrics:
            if m not in per_ds.columns:
                row[METRIC_SHORT.get(m, m)] = "N/A"
                continue
            v = per_ds[m].dropna()
            # Stage 2: median and IQR of the 36 per-dataset values
            row[METRIC_SHORT.get(m, m)] = (
                f"{v.median():.4f} [{v.quantile(0.25):.4f}–{v.quantile(0.75):.4f}]"
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Condition")


def build_summary_table_numeric(
    df_results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Same as build_summary_table but returns raw numeric median values
    (no IQR formatting) for programmatic use (e.g., building LaTeX tables).
    """
    if metrics is None:
        metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score", "cal_accuracy"]

    df = _prep(df_results)
    records = []
    for cond in CONDITION_ORDER:
        if cond not in df["condition"].values:
            continue
        per_ds = (
            df[df["condition"] == cond]
            .groupby("dataset_name")[metrics]
            .median()
        )
        rec = {"condition": cond, "label": CONDITION_LABELS.get(cond, cond)}
        for m in metrics:
            if m in per_ds.columns:
                rec[m + "_median"] = per_ds[m].median()
                rec[m + "_q25"]    = per_ds[m].quantile(0.25)
                rec[m + "_q75"]    = per_ds[m].quantile(0.75)
        records.append(rec)
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Win / Tie / Loss table
# ─────────────────────────────────────────────────────────────────────────────

def build_win_rate_table(
    df_results: pd.DataFrame,
    pairs: Optional[List] = None,
    metrics: Optional[List[str]] = None,
    tol: float = 0.001,
) -> pd.DataFrame:
    """
    For each (condition_A, condition_B, metric) pair: count Win / Tie / Loss
    across 36 datasets.

    A "Win" for condition_A means its per-dataset median metric is lower
    than condition_B's by more than `tol` (lower = better for NLL, ECE, Brier).

    Parameters
    ----------
    pairs   : list of (cond_a, cond_b) tuples
    metrics : metric names
    tol     : minimum absolute difference to count as a win vs tie

    Returns
    -------
    DataFrame with columns: condition_A, condition_B, metric, W, T, L, n,
                             win_pct, note
    """
    if metrics is None:
        metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score"]
    if pairs is None:
        pairs = [
            # H1: raw model comparisons
            ("deep_ensemble/none", "lgbm/none"),
            ("single_mlp/none",    "lgbm/none"),
            ("deep_ensemble/none", "single_mlp/none"),
            ("xgboost/none",       "lgbm/none"),
            # H2: calibration effects for LightGBM
            ("lgbm/none",     "lgbm/temp"),      # A = uncalibrated; W = temp helps
            ("lgbm/none",     "lgbm/isotonic"),
            ("lgbm/temp",     "lgbm/isotonic"),
            # H3: post-calibration gap
            ("deep_ensemble/isotonic", "lgbm/isotonic"),
            ("deep_ensemble/temp",     "lgbm/temp"),
        ]

    df = _prep(df_results)
    records = []
    for cond_a, cond_b in pairs:
        for metric in metrics:
            a = df[df["condition"] == cond_a].groupby("dataset_name")[metric].median()
            b = df[df["condition"] == cond_b].groupby("dataset_name")[metric].median()
            ds = a.index.intersection(b.index)
            if len(ds) == 0:
                continue
            diff = a[ds] - b[ds]
            w = int((diff < -tol).sum())
            l = int((diff >  tol).sum())
            t = len(ds) - w - l
            records.append({
                "condition_A": cond_a,
                "condition_B": cond_b,
                "metric":      metric,
                "W":           w,
                "T":           t,
                "L":           l,
                "n":           len(ds),
                "win_pct":     round(100 * w / len(ds), 1),
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Master call
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    df_results: pd.DataFrame,
    fig_dir: Path,
    manifest_path: Optional[Path] = None,
) -> None:
    """
    Generate all paper figures and save to fig_dir.

    Paper figures (recommended):
        figure1_metric_heatmap_grid  — overview heatmap (Table 2 equivalent)
        figure2_combined_four_panel  — isotonic trap + ECE scatter (key finding)
        figure3_boxplot_nll          — NLL distribution across datasets

    Supplementary / appendix:
        figure4_ranking_heatmap_nll
        figure5_ranking_heatmap_ece
        figure6_training_time
        figureS1_dataset_overview
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating figures...")
    plot_metric_heatmap_grid(df_results, fig_dir=fig_dir)
    plot_figure2_combined(df_results, fig_dir=fig_dir)
    plot_metric_boxplots(df_results, metrics=["cal_nll"], fig_dir=fig_dir)
    plot_ranking_heatmap(df_results, metric="cal_nll",      fig_dir=fig_dir, fig_num=4)
    plot_ranking_heatmap(df_results, metric="cal_ece_mean", fig_dir=fig_dir, fig_num=5)
    plot_training_time(df_results, fig_dir=fig_dir)
    if manifest_path and manifest_path.exists():
        plot_dataset_overview(manifest_path, fig_dir=fig_dir)
    logger.info(f"All figures saved to {fig_dir.resolve()}")
