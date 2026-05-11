"""
postprocess.py
--------------
Post-hoc analysis script that operates entirely on results_raw.csv.
No model retraining required.

Produces
--------
R2  — Corrected hypothesis tests (H3 restricted to isotonic; H3_supp_temp
       for transparency). Overwrites stats_*.csv files.
R3  — Inter-seed variance table. Flags high-variance datasets (CV > 0.1).
R4  — Win / Tie / Loss counts for all primary comparison pairs and metrics.
R5  — Bootstrap 95% CIs on per-condition median NLL / ECE / Brier.
R8  — Training time table (mean ± std per model class).

Usage
-----
    python postprocess.py --results_dir results/
    python postprocess.py --results_dir results/ --alpha 0.05

All output files written to results_dir:
    stats_H1_*.csv, stats_H2_*.csv, stats_H3_*.csv, stats_H3_supp_temp_*.csv
    table1_summary_updated.csv / .tex   — corrected two-stage median summary
    table_win_tie_loss.csv / .tex       — W/T/L counts per pair × metric (R4)
    table_bootstrap_ci.csv              — 95% CI on per-condition median (R5)
    table_training_times.csv / .tex     — R8
    table_inter_seed_variance.csv       — R3
    table_high_variance_datasets.csv    — R3 flagged datasets
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root (parent of experiments/) is on sys.path so `src.*` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.statistical_analysis import run_hypothesis_tests, print_hypothesis_summary
from src.visualization import (
    build_summary_table,
    build_summary_table_numeric,
    build_win_rate_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("postprocess")


# ─────────────────────────────────────────────────────────────────────────────
# R8 — Training time table
# ─────────────────────────────────────────────────────────────────────────────

def build_training_time_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ± std training time in seconds per model class.

    Training time is recorded once per (task, model, seed) regardless of
    calibrator, so we de-duplicate before aggregating.
    """
    dedup = (
        df.drop_duplicates(subset=["task_id", "model", "seed"])
        [["model", "train_time_s"]]
    )
    agg = (
        dedup.groupby("model")["train_time_s"]
        .agg(
            mean_s="mean",
            std_s="std",
            median_s="median",
            min_s="min",
            max_s="max",
            n_runs="count",
        )
        .reset_index()
        .sort_values("mean_s", ascending=False)
    )
    agg = agg.round(2)
    agg["mean ± std (s)"] = (
        agg["mean_s"].map("{:.1f}".format)
        + " ± "
        + agg["std_s"].map("{:.1f}".format)
    )
    return agg


def latex_training_time_table(agg: pd.DataFrame) -> str:
    display = agg[["model", "mean ± std (s)", "median_s", "min_s", "max_s", "n_runs"]].copy()
    display.columns = ["Model", "Mean ± Std (s)", "Median (s)", "Min (s)", "Max (s)", "N"]
    display["Model"] = display["Model"].str.replace("_", "\\_", regex=False)
    return display.to_latex(index=False, escape=False)


# ─────────────────────────────────────────────────────────────────────────────
# R3 — Inter-seed variance table
# ─────────────────────────────────────────────────────────────────────────────

def build_inter_seed_variance_table(
    df: pd.DataFrame,
    metrics: tuple = ("cal_nll", "cal_ece_mean", "cal_brier_score"),
    flag_threshold: float = 0.10,
) -> tuple:
    """
    Coefficient of variation (std / |median|) across seeds per
    (dataset, model, calibrator) cell.

    Returns (variance_df, flagged_df).
    """
    df = df.copy()
    df["condition"] = df["model"] + "/" + df["calibrator"]

    records = []
    for (dataset, condition), grp in df.groupby(["dataset_name", "condition"]):
        for metric in metrics:
            if metric not in grp.columns:
                continue
            vals = grp[metric].dropna().values
            n    = len(vals)
            if n < 2:
                continue
            med = np.median(vals)
            std = np.std(vals, ddof=1)
            cv  = std / (abs(med) + 1e-12)
            records.append({
                "dataset":   dataset,
                "condition": condition,
                "metric":    metric,
                "n_seeds":   n,
                "median":    round(float(med), 5),
                "std":       round(float(std), 5),
                "cv":        round(float(cv),  4),
            })

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    variance_df = pd.DataFrame(records).sort_values("cv", ascending=False)
    high_cv = variance_df[variance_df["cv"] > flag_threshold]
    flagged_datasets = high_cv["dataset"].unique()

    flagged_df = (
        variance_df[variance_df["dataset"].isin(flagged_datasets)]
        .groupby(["dataset", "condition", "metric"])["cv"]
        .max()
        .reset_index()
        .rename(columns={"cv": "max_cv"})
        .sort_values("max_cv", ascending=False)
    )

    n_flagged = len(flagged_datasets)
    n_total   = df["dataset_name"].nunique()
    logger.info(
        f"Inter-seed variance: {n_flagged}/{n_total} datasets have CV > "
        f"{flag_threshold} in at least one condition × metric combination."
    )
    return variance_df, flagged_df


def print_variance_summary(variance_df: pd.DataFrame, flagged_df: pd.DataFrame) -> None:
    print("\n── INTER-SEED VARIANCE SUMMARY ──")
    print(f"Total (dataset × condition × metric) cells: {len(variance_df)}")
    high  = variance_df[variance_df["cv"] > 0.10]
    high2 = variance_df[variance_df["cv"] > 0.20]
    print(f"Cells with CV > 0.10: {len(high)} ({100*len(high)/max(len(variance_df),1):.1f}%)")
    print(f"Cells with CV > 0.20: {len(high2)} ({100*len(high2)/max(len(variance_df),1):.1f}%)")
    if not flagged_df.empty:
        print("\nTop flagged datasets (highest max CV):")
        print(
            flagged_df.drop_duplicates("dataset")
            .head(10)[["dataset", "max_cv"]]
            .to_string(index=False)
        )


# ─────────────────────────────────────────────────────────────────────────────
# R4 — Win / Tie / Loss table
# ─────────────────────────────────────────────────────────────────────────────

def latex_win_rate_table(wtl: pd.DataFrame) -> str:
    """
    Compact LaTeX table for win/tie/loss counts.

    Groups by (condition_A, condition_B) with one sub-row per metric.
    Format: condition_A vs condition_B | metric | W/T/L (win%)
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Win / Tie / Loss counts across 36 datasets. "
        r"A ``Win'' for condition A means its per-dataset median is lower by $>$0.001. "
        r"H2 pairs list the \emph{uncalibrated} condition as A so that W = calibration helps.}",
        r"\label{tab:win_tie_loss}",
        r"\begin{tabular}{llcrrrr}",
        r"\toprule",
        r"Condition A & Condition B & Metric & W & T & L & Win\% \\",
        r"\midrule",
    ]
    prev_pair = None
    for _, row in wtl.iterrows():
        pair = (row["condition_A"], row["condition_B"])
        ca   = row["condition_A"].replace("_", r"\_")
        cb   = row["condition_B"].replace("_", r"\_")
        m    = row["metric"].replace("cal_", "").replace("_mean", "").upper()
        if m == "NLL":
            m = "NLL"
        elif m == "ECE":
            m = "ECE"
        elif m == "BRIER":
            m = "Brier"
        w, t, l = int(row["W"]), int(row["T"]), int(row["L"])
        wp = f"{row['win_pct']:.0f}\\%"
        if pair != prev_pair:
            if prev_pair is not None:
                lines.append(r"\midrule")
            lines.append(f"{ca} & {cb} & {m} & {w} & {t} & {l} & {wp} \\\\")
        else:
            lines.append(f" & & {m} & {w} & {t} & {l} & {wp} \\\\")
        prev_pair = pair
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# R5 — Bootstrap confidence intervals on per-condition medians
# ─────────────────────────────────────────────────────────────────────────────

def build_bootstrap_ci_table(
    df: pd.DataFrame,
    metrics: tuple = ("cal_nll", "cal_ece_mean", "cal_brier_score"),
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap 95% CI on the per-condition median NLL / ECE / Brier.

    Procedure:
        1. For each (dataset, condition): compute per-seed median → 36 values
        2. Bootstrap resample the 36 per-dataset values n_boot times
        3. Compute the median of each bootstrap sample
        4. CI = [alpha/2 percentile, 1-alpha/2 percentile]

    The bootstrap operates at the dataset level (resampling datasets with
    replacement), which is the correct unit of analysis for Wilcoxon tests
    and matches how the statistical tests are structured.
    """
    rng = np.random.RandomState(seed)
    df  = df.copy()
    df["condition"] = df["model"] + "/" + df["calibrator"]

    from visualization import CONDITION_ORDER, CONDITION_LABELS

    records = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        rec = {"condition": cond, "label": CONDITION_LABELS.get(cond, cond)}
        for metric in metrics:
            # Stage 1: per-dataset medians
            per_ds = sub.groupby("dataset_name")[metric].median().dropna().values
            n = len(per_ds)
            if n < 2:
                rec[f"{metric}_median"] = np.nan
                rec[f"{metric}_ci_lo"]  = np.nan
                rec[f"{metric}_ci_hi"]  = np.nan
                continue
            # Stage 2: bootstrap
            boot_medians = np.array([
                np.median(rng.choice(per_ds, size=n, replace=True))
                for _ in range(n_boot)
            ])
            lo = float(np.percentile(boot_medians, 100 * alpha / 2))
            hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
            rec[f"{metric}_median"] = float(np.median(per_ds))
            rec[f"{metric}_ci_lo"]  = lo
            rec[f"{metric}_ci_hi"]  = hi
        records.append(rec)

    return pd.DataFrame(records)


def latex_bootstrap_ci_table(ci_df: pd.DataFrame, metric: str = "cal_nll") -> str:
    """Render a LaTeX table with median [95% CI] for one metric."""
    m_lo  = f"{metric}_ci_lo"
    m_hi  = f"{metric}_ci_hi"
    m_med = f"{metric}_median"
    if m_med not in ci_df.columns:
        return "% metric not found"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Per-condition median NLL with 95\% bootstrap confidence interval "
        r"(bootstrap over 36 per-dataset medians, $B=2000$ resamples).}",
        r"\label{tab:bootstrap_ci}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Condition & Median NLL & 95\% CI \\",
        r"\midrule",
    ]
    prev_model = None
    for _, row in ci_df.iterrows():
        model = row["condition"].split("/")[0]
        if prev_model is not None and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model
        med = row[m_med]
        lo  = row[m_lo]
        hi  = row[m_hi]
        if np.isnan(med):
            continue
        label = row["label"].replace("_", r"\_")
        lines.append(
            f"{label} & {med:.4f} & [{lo:.4f}, {hi:.4f}] \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Corrected summary table: LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def latex_summary_table(numeric_df: pd.DataFrame) -> str:
    """
    Render Table 2 of the paper as LaTeX.

    Uses the CORRECT two-stage aggregation from build_summary_table_numeric().
    Values marked ** are the best (lowest) per metric column.
    """
    metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score", "cal_accuracy"]
    metric_labels = ["NLL $\\downarrow$", "ECE $\\downarrow$",
                     "Brier $\\downarrow$", "Acc $\\uparrow$"]

    # Find best (min for NLL/ECE/Brier, max for Acc) per metric
    best = {}
    for m in metrics:
        col = m + "_median"
        if col not in numeric_df.columns:
            continue
        best[m] = (
            numeric_df[col].max() if m == "cal_accuracy"
            else numeric_df[col].min()
        )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Median primary uncertainty metrics by condition. "
        r"Values are per-dataset medians of 5-seed medians (n=36 datasets). "
        r"Bold = best in column. Lower is better for NLL, ECE, Brier; "
        r"higher is better for Accuracy.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{l" + "r" * len(metrics) + "}",
        r"\toprule",
        "Condition & " + " & ".join(metric_labels) + r" \\",
        r"\midrule",
    ]

    prev_model = None
    for _, row in numeric_df.iterrows():
        model = row["condition"].split("/")[0]
        if prev_model is not None and model != prev_model:
            lines.append(r"\midrule")
        prev_model = model

        label = row["label"]
        cells = [label.replace("_", r"\_")]
        for m in metrics:
            col = m + "_median"
            if col not in row or np.isnan(row[col]):
                cells.append("—")
                continue
            val_str = f"{row[col]:.4f}"
            if abs(row[col] - best.get(m, np.nan)) < 1e-6:
                val_str = r"\textbf{" + val_str + "}"
            cells.append(val_str)
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Console summaries
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_comparison(df: pd.DataFrame, metrics=None) -> None:
    """Print side-by-side comparison of old (pooled) vs new (per-dataset) medians."""
    if metrics is None:
        metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score"]

    df = df.copy()
    df["condition"] = df["model"] + "/" + df["calibrator"]

    print("\n── MEDIAN COMPARISON: pooled (old) vs per-dataset (correct) ──")
    print(f"{'Condition':<35} {'Metric':<20} {'Pooled':>10} {'Correct':>10} {'Δ':>8}")
    print("-" * 88)

    key_conds = [
        "lgbm/none", "lgbm/temp", "lgbm/isotonic",
        "deep_ensemble/none", "deep_ensemble/temp", "deep_ensemble/isotonic",
        "xgboost/temp", "single_mlp/temp",
    ]
    for cond in key_conds:
        sub = df[df["condition"] == cond]
        for metric in metrics:
            pooled  = sub[metric].dropna().median()
            correct = sub.groupby("dataset_name")[metric].median().median()
            delta   = correct - pooled
            flag    = " ←" if abs(delta) > 0.005 else ""
            print(f"{cond:<35} {metric:<20} {pooled:>10.4f} {correct:>10.4f} {delta:>+8.4f}{flag}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc analysis from results_raw.csv")
    p.add_argument("--results_dir",   type=str,   default="results")
    p.add_argument("--alpha",         type=float, default=0.05)
    p.add_argument("--cv_threshold",  type=float, default=0.10)
    p.add_argument("--n_boot",        type=int,   default=2000,
                   help="Bootstrap resamples for CI computation (default: 2000)")
    p.add_argument("--metrics", nargs="+",
                   default=["cal_nll", "cal_ece_mean", "cal_brier_score"])
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.results_dir)
    csv_path = out_dir / "results_raw.csv"

    if not csv_path.exists():
        logger.error(f"results_raw.csv not found at {csv_path}.")
        sys.exit(1)

    logger.info(f"Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(
        f"Loaded {len(df)} rows. "
        f"Datasets: {df['dataset_name'].nunique()}, "
        f"Seeds: {sorted(df['seed'].unique())}."
    )

    # ── Median correction report ─────────────────────────────────────────────
    print_summary_comparison(df, metrics=args.metrics)

    # ── R2: Corrected hypothesis tests ──────────────────────────────────────
    logger.info("Running corrected hypothesis tests (H3 restricted to isotonic)...")
    hyp_results = run_hypothesis_tests(
        df,
        metrics=args.metrics,
        include_supplementary=True,
    )
    print_hypothesis_summary(hyp_results, alpha=args.alpha)

    for key, hdf in hyp_results.items():
        if not hdf.empty:
            out_path = out_dir / f"stats_{key}.csv"
            hdf.to_csv(out_path, index=False)
            logger.info(f"  Saved {out_path}")

    # ── R4: Win / Tie / Loss table ───────────────────────────────────────────
    logger.info("Building win/tie/loss table (R4)...")
    wtl = build_win_rate_table(df, metrics=args.metrics)
    wtl_csv = out_dir / "table_win_tie_loss.csv"
    wtl.to_csv(wtl_csv, index=False)
    logger.info(f"  Saved {wtl_csv}")

    wtl_tex = out_dir / "table_win_tie_loss.tex"
    wtl_tex.write_text(latex_win_rate_table(wtl))
    logger.info(f"  Saved {wtl_tex}")

    print("\n── WIN / TIE / LOSS TABLE (R4) ──")
    print(wtl.to_string(index=False))

    # ── R5: Bootstrap confidence intervals ──────────────────────────────────
    logger.info(f"Computing bootstrap CIs (n_boot={args.n_boot})...")
    ci_df = build_bootstrap_ci_table(
        df,
        metrics=tuple(args.metrics),
        n_boot=args.n_boot,
    )
    ci_csv = out_dir / "table_bootstrap_ci.csv"
    ci_df.to_csv(ci_csv, index=False)
    logger.info(f"  Saved {ci_csv}")

    ci_tex = out_dir / "table_bootstrap_ci.tex"
    ci_tex.write_text(latex_bootstrap_ci_table(ci_df, metric="cal_nll"))
    logger.info(f"  Saved {ci_tex}")

    print("\n── BOOTSTRAP 95% CI — NLL ──")
    nll_cols = ["label", "cal_nll_median", "cal_nll_ci_lo", "cal_nll_ci_hi"]
    present  = [c for c in nll_cols if c in ci_df.columns]
    print(ci_df[present].round(4).to_string(index=False))

    # ── R8: Training time table ──────────────────────────────────────────────
    logger.info("Building training time table (R8)...")
    time_table = build_training_time_table(df)
    (out_dir / "table_training_times.csv").write_text(time_table.to_csv(index=False))
    (out_dir / "table_training_times.tex").write_text(latex_training_time_table(time_table))
    logger.info(f"  Saved table_training_times.{{csv,tex}}")

    print("\n── TRAINING TIME (R8) ──")
    print(time_table[["model", "mean ± std (s)", "median_s", "n_runs"]].to_string(index=False))

    # ── R3: Inter-seed variance ──────────────────────────────────────────────
    logger.info(f"Computing inter-seed variance (R3, CV threshold={args.cv_threshold})...")
    variance_df, flagged_df = build_inter_seed_variance_table(
        df, metrics=tuple(args.metrics), flag_threshold=args.cv_threshold,
    )
    if not variance_df.empty:
        variance_df.to_csv(out_dir / "table_inter_seed_variance.csv", index=False)
    if not flagged_df.empty:
        flagged_df.to_csv(out_dir / "table_high_variance_datasets.csv", index=False)
    print_variance_summary(variance_df, flagged_df)

    # ── Updated summary table (correct medians) ──────────────────────────────
    logger.info("Rebuilding summary table with correct two-stage aggregation...")
    summary_str  = build_summary_table(df, metrics=args.metrics)
    numeric_df   = build_summary_table_numeric(df, metrics=args.metrics)
    summary_str.to_csv(out_dir / "table1_summary_updated.csv")
    numeric_df.to_csv(out_dir / "table1_summary_numeric.csv", index=False)
    (out_dir / "table1_summary_updated.tex").write_text(latex_summary_table(numeric_df))
    logger.info("  Saved table1_summary_updated.{csv,tex}")

    print("\n── CORRECTED SUMMARY TABLE (Table 2) ──")
    print(summary_str.to_string())
    print("\nLaTeX:")
    print(latex_summary_table(numeric_df))

    logger.info("postprocess.py complete.")


if __name__ == "__main__":
    main()
