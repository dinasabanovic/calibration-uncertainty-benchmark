"""
make_figures.py
---------------
Generates three publication-ready figures.

Figure 1 — Metric heatmap grid (overview of all 16 conditions)
Figure 2 — Four-panel calibration diagnostic (isotonic trap)
Figure 3 — Paired ΔNLL (TS − Raw), one row per model, one dot per dataset

Run from the project root:
    python experiments/make_figures.py

Outputs land in <project_root>/results/figures/.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns

matplotlib.use("Agg")

# Resolve paths relative to project root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"
FIG_DIR      = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "results_raw.csv")
df["calibrator"] = df["calibrator"].replace({"platt": "logistic"})
df["condition"]  = df["model"] + "/" + df["calibrator"]

CM       = 1 / 2.54
FULL     = 17.5 * CM

FONT_TITLE = 14
FONT_LABEL = 12
FONT_TICK  = 11
FONT_ANNOT = 10
FONT_NOTE  = 10
FONT_VALUE = 11

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  FONT_TITLE,
    "axes.labelsize":  FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_TICK,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.08,
    "text.usetex":     False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

MODEL_COLOR = {
    "lgbm":          "#4E79A7",
    "xgboost":       "#9467BD",
    "single_mlp":    "#2CA02C",
    "deep_ensemble": "#FF7F0E",
}
MODEL_LABEL = {
    "lgbm":          "LightGBM",
    "xgboost":       "XGBoost",
    "single_mlp":    "SingleMLP",
    "deep_ensemble": "Ensemble (M=5)",
}


def per_ds(condition, metric):
    return (
        df[df["condition"] == condition]
        .groupby("dataset_name")[metric]
        .median()
    )


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


# =============================================================================
# FIGURE 1 — Metric heatmap grid (vertically stacked panels)
# =============================================================================
def make_figure1():
    """Three vertically stacked heatmap panels — one per metric."""
    models  = ["lgbm", "xgboost", "single_mlp", "deep_ensemble"]
    cals    = ["none", "temp", "logistic", "isotonic"]
    metrics = [
        ("cal_nll",         "NLL ↓"),
        ("cal_ece_mean",    "ECE ↓"),
        ("cal_brier_score", "Brier Score ↓"),
    ]
    cal_label = {
        "none":     "Raw",
        "temp":     "Temp. Scaling",
        "logistic": "MLR",
        "isotonic": "Isotonic Reg.",
    }

    cmap = matplotlib.cm.RdYlBu_r

    fig, axes = plt.subplots(3, 1, figsize=(FULL, FULL * 1.30))
    plt.subplots_adjust(left=0.22, right=0.83, top=0.88, bottom=0.04,
                        hspace=0.55)

    for panel_idx, (ax, (metric, mlabel)) in enumerate(zip(axes, metrics)):
        n_rows, n_cols = len(cals), len(models)
        vals = np.full((n_rows, n_cols), np.nan)
        for ri, cal in enumerate(cals):
            for ci, model in enumerate(models):
                v = per_ds(f"{model}/{cal}", metric)
                if len(v):
                    vals[ri, ci] = v.median()
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if hi > lo:
            norm = (vals - lo) / (hi - lo)
        else:
            norm = np.full_like(vals, 0.5)

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(-0.5, n_rows - 0.5)
        ax.invert_yaxis()
        ax.axis("off")

        cell_w, cell_h = 0.94, 0.86
        for ri, cal in enumerate(cals):
            for ci in range(n_cols):
                v  = vals[ri, ci]
                nv = norm[ri, ci]
                color = cmap(nv) if not np.isnan(nv) else (0.93, 0.93, 0.93, 1.0)
                lw    = 2.5 if cal == "logistic" else 0.5
                ec    = "#cc0000" if cal == "logistic" else "white"
                rect  = mpatches.FancyBboxPatch(
                    (ci - cell_w/2, ri - cell_h/2), cell_w, cell_h,
                    boxstyle="round,pad=0.02",
                    linewidth=lw, edgecolor=ec, facecolor=color,
                )
                ax.add_patch(rect)
                txt_c = "white" if nv > 0.72 else "black"
                ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                        fontsize=FONT_TITLE, fontweight="bold", color=txt_c)

        # Calibrator row labels
        for ri, cal in enumerate(cals):
            fc = "#cc0000" if cal == "logistic" else "black"
            fw = "bold"    if cal == "logistic" else "normal"
            ax.text(-0.62, ri, cal_label[cal], ha="right", va="center",
                    fontsize=FONT_LABEL, color=fc, fontweight=fw,
                    clip_on=False)

        # Model column labels — only above top panel
        if panel_idx == 0:
            for ci, model in enumerate(models):
                ax.text(ci, -0.5 - 0.45, MODEL_LABEL[model],
                        ha="center", va="bottom",
                        fontsize=FONT_LABEL, fontweight="bold",
                        color=MODEL_COLOR[model], clip_on=False)

        # Metric label as panel title (left-aligned, above panel)
        ax.text(-2.15, -0.5 - 0.05, mlabel,
                ha="left", va="bottom",
                fontsize=FONT_TITLE, fontweight="bold",
                color="#222222", clip_on=False)

    # Shared colourbar on the right
    cbar_ax = fig.add_axes([0.86, 0.18, 0.020, 0.55])
    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["best", "mid", "worst"], fontsize=FONT_ANNOT)
    cb.ax.tick_params(labelsize=FONT_ANNOT)
    cb.set_label("Lower is better", fontsize=FONT_ANNOT, labelpad=8)

    fig.suptitle(
        "Median uncertainty metrics by model and calibration method\n"
        r"($n = 36$ datasets; per-dataset medians of 5-seed medians; red border = MLR)",
        fontsize=FONT_TITLE, fontweight="bold", y=0.985,
    )
    save(fig, "figure1_heatmap")


# =============================================================================
# FIGURE 2 — Four-panel calibration diagnostic
# =============================================================================
def make_figure2():
    models = ["lgbm", "xgboost", "single_mlp", "deep_ensemble"]

    raw_nll = {m: per_ds(f"{m}/none", "cal_nll")     for m in models}
    raw_ece = {m: per_ds(f"{m}/none", "cal_ece_mean") for m in models}

    fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 1.20))
    plt.subplots_adjust(hspace=0.50, wspace=0.34,
                        left=0.10, right=0.97, top=0.86, bottom=0.07)

    # ── Panels (a) and (b): ΔNLL vs ΔECE ─────────────────────────────────────
    for ax, cal, title in [
        (axes[0, 0], "isotonic", "(a) Isotonic Regression"),
        (axes[0, 1], "temp",     "(b) Temperature Scaling"),
    ]:
        for model in models:
            cnll = per_ds(f"{model}/{cal}", "cal_nll")
            cece = per_ds(f"{model}/{cal}", "cal_ece_mean")
            ds   = raw_nll[model].index.intersection(cnll.index)
            dnll = (cnll[ds] - raw_nll[model][ds]).values
            dece = (cece[ds] - raw_ece[model][ds]).values
            ax.scatter(dece, dnll,
                       color=MODEL_COLOR[model], alpha=0.78, s=42,
                       edgecolors="k", linewidths=0.4, zorder=3,
                       label=MODEL_LABEL[model])

        ax.axhline(0, color="gray", lw=0.9, ls="--", alpha=0.55)
        ax.axvline(0, color="gray", lw=0.9, ls="--", alpha=0.55)
        ax.set_xlabel("ΔECE (calibrated − raw)", fontsize=FONT_LABEL)
        ax.set_ylabel("ΔNLL (calibrated − raw)", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_LABEL + 1, fontweight="bold", pad=8)
        ax.tick_params(labelsize=FONT_TICK)

        if cal == "isotonic":
            yhi = ax.get_ylim()[1]
            ax.axhspan(0, max(yhi, 0.05), xmin=0, xmax=0.5,
                       alpha=0.10, color="red", zorder=0)
            ax.text(0.04, 0.96, "ECE↓  NLL↑\n(isotonic trap)",
                    transform=ax.transAxes, fontsize=FONT_NOTE,
                    va="top", color="#cc0000", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#cc0000", alpha=0.92, lw=1.4))

        sns.despine(ax=ax)

    # Single shared legend at the top of the figure
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels_,
        loc="center", bbox_to_anchor=(0.5, 0.945),
        ncol=4, fontsize=FONT_NOTE,
        framealpha=0.95, edgecolor="gray",
        handlelength=1.4, handletextpad=0.5,
        columnspacing=1.5,
    )

    # ── Panels (c) and (d): Raw vs Calibrated ECE ────────────────────────────
    all_ece = []
    for model in models:
        for cal in ["temp", "isotonic"]:
            all_ece.extend(per_ds(f"{model}/{cal}", "cal_ece_mean").values)
            all_ece.extend(raw_ece[model].values)
    lo = max(0.0, np.nanpercentile(all_ece, 1) * 0.9)
    hi = np.nanpercentile(all_ece, 99) * 1.10

    for ax, cal, title in [
        (axes[1, 0], "temp",     "(c) Temperature Scaling"),
        (axes[1, 1], "isotonic", "(d) Isotonic Regression"),
    ]:
        for model in models:
            cece = per_ds(f"{model}/{cal}", "cal_ece_mean")
            ds   = raw_ece[model].index.intersection(cece.index)
            ax.scatter(raw_ece[model][ds], cece[ds],
                       color=MODEL_COLOR[model], alpha=0.78, s=42,
                       edgecolors="k", linewidths=0.4, zorder=3)

        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel("Raw ECE", fontsize=FONT_LABEL)
        ax.set_ylabel("Calibrated ECE", fontsize=FONT_LABEL)
        ax.tick_params(labelsize=FONT_TICK)
        ax.set_title(title + "\nRaw vs Calibrated ECE",
                     fontsize=FONT_LABEL + 1, fontweight="bold", pad=8)
        sns.despine(ax=ax)

    fig.suptitle(
        "Calibration effects across 36 OpenML-CC18 datasets\n"
        "(each point = one dataset, median over 5 seeds)",
        fontsize=FONT_TITLE, fontweight="bold", y=1.02,
    )
    save(fig, "figure2_four_panel")


# =============================================================================
# FIGURE 3 — Paired ΔNLL (Temp. Scaling − Raw), one row per model
# =============================================================================
def make_figure3():
    """
    Paired-difference dot plot.

    For each model, plot the 36 per-dataset values of:
        ΔNLL = NLL(Temp. Scaling) − NLL(Raw)

    Negative ΔNLL = calibration improved NLL (most points should be left of zero).
    The vertical zero line is the null hypothesis (no effect).
    Median ΔNLL is shown as a thick black tick with its value annotated.
    Win/tie/loss counts are shown on the right.

    This makes the small-but-consistent improvement of TS visually obvious,
    which the "absolute NLL" boxplot view obscured.
    """
    from scipy.stats import wilcoxon

    models = ["lgbm", "xgboost", "single_mlp", "deep_ensemble"]

    # Compute per-dataset paired differences for each model
    diffs = {}
    for m in models:
        raw = per_ds(f"{m}/none", "cal_nll")
        ts  = per_ds(f"{m}/temp", "cal_nll")
        ds  = raw.index.intersection(ts.index)
        diffs[m] = (ts[ds] - raw[ds]).values

    # Determine a sensible x range (clip extreme outliers visually but keep them)
    all_vals = np.concatenate(list(diffs.values()))
    x_min = np.percentile(all_vals,  2) - 0.005
    x_max = np.percentile(all_vals, 98) + 0.005
    # Make symmetric around 0 for clarity
    x_abs = max(abs(x_min), abs(x_max))
    x_min, x_max = -x_abs, x_abs

    fig, ax = plt.subplots(figsize=(FULL, FULL * 0.58))
    plt.subplots_adjust(left=0.20, right=0.74, top=0.84, bottom=0.18)

    # Reverse so LightGBM appears at the top
    rows = list(reversed(list(enumerate(models))))
    n = len(models)

    # Zero line first (behind dots)
    ax.axvline(0, color="black", lw=1.2, ls="-", alpha=0.6, zorder=2)
    # Faint tinted background: left = TS helps, right = TS hurts
    ax.axvspan(x_min, 0, alpha=0.04, color="green", zorder=0)
    ax.axvspan(0, x_max, alpha=0.04, color="red",   zorder=0)

    yticks, ylabels = [], []
    for plot_y, (orig_idx, model) in enumerate(rows, start=1):
        d   = diffs[model]
        n_ds = len(d)
        col = MODEL_COLOR[model]

        # Jitter dots vertically a tiny bit so overlapping points are visible
        rng = np.random.default_rng(orig_idx)
        jitter = rng.uniform(-0.16, 0.16, size=n_ds)

        # Clip dots that are off-scale, draw them at the edge with a little arrow tick
        x_clipped = np.clip(d, x_min * 0.985, x_max * 0.985)
        is_clipped = (d < x_min) | (d > x_max)

        ax.scatter(x_clipped[~is_clipped], plot_y + jitter[~is_clipped],
                   color=col, alpha=0.75, s=46,
                   edgecolors="black", linewidths=0.5, zorder=4)
        if is_clipped.any():
            ax.scatter(x_clipped[is_clipped], plot_y + jitter[is_clipped],
                       color=col, alpha=0.75, s=46,
                       marker=">", edgecolors="black", linewidths=0.5,
                       zorder=4)

        # Median line (thick black vertical tick)
        med = np.median(d)
        ax.plot([med, med], [plot_y - 0.32, plot_y + 0.32],
                color="black", lw=3.0, solid_capstyle="round", zorder=5)

        # Win / Tie / Loss + median + p-value on the right margin —
        # all summary stats for this row in one unambiguous place.
        tol = 0.001
        wins   = int((d < -tol).sum())
        losses = int((d >  tol).sum())
        ties   = n_ds - wins - losses
        try:
            _, p = wilcoxon(d, alternative="two-sided")
            p_str = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
        except Exception:
            p_str = "p = —"

        ax.text(1.02, plot_y,
                f"med Δ = {med:+.4f}\nW/T/L = {wins}/{ties}/{losses}\n{p_str}",
                transform=ax.get_yaxis_transform(),
                ha="left", va="center",
                fontsize=FONT_NOTE, family="monospace")

        yticks.append(plot_y)
        ylabels.append(MODEL_LABEL[model])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=FONT_LABEL)
    ax.set_ylim(0.4, n + 0.85)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(
        r"$\Delta$NLL = NLL(Temp. Scaling) $-$ NLL(Raw)" + "\n"
        r"left of zero $\rightarrow$ Temperature Scaling improves NLL",
        fontsize=FONT_LABEL,
    )
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_LABEL)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Subtle horizontal separators between models
    for plot_y in range(1, n):
        ax.axhline(plot_y + 0.5, color="#bbbbbb", lw=0.7, ls="--",
                   alpha=0.6, zorder=1)

    # Annotations describing the two halves of the plot
    ax.text(x_min * 0.97, n + 0.55, "← TS improves NLL",
            ha="left", va="center", fontsize=FONT_NOTE,
            color="#1b7a3e", fontweight="bold")
    ax.text(x_max * 0.97, n + 0.55, "TS hurts NLL →",
            ha="right", va="center", fontsize=FONT_NOTE,
            color="#a8312a", fontweight="bold")

    ax.set_title(
        "Per-dataset NLL improvement from Temperature Scaling\n"
        "(each dot = one dataset; black tick = median; n = 36 datasets per model)",
        fontsize=FONT_TITLE, fontweight="bold", pad=12,
    )
    sns.despine(ax=ax)
    save(fig, "figure3_paired_delta_nll")


# =============================================================================
if __name__ == "__main__":
    print("Generating figures...")
    make_figure1()
    make_figure2()
    make_figure3()
    print(f"\nDone. Figures saved to {FIG_DIR.resolve()}/")
