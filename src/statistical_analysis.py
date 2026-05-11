"""
statistical_analysis.py
------------------------
Paired statistical comparison of model-calibrator conditions across datasets.

Procedure:
    1. Aggregate per-dataset medians across seeds (robust central tendency)
    2. For each targeted pair of conditions, compute the Wilcoxon signed-rank test
    3. Apply ONE global Holm–Bonferroni correction across ALL tests simultaneously
       (previously correction was applied separately per hypothesis, which was
        overly lenient and incorrect)
    4. Report rank-biserial correlation as effect size
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.utils import deprecated

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Effect size
# ─────────────────────────────────────────────────────────────────────────────

def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Signed rank-biserial correlation for paired Wilcoxon data.

    r = (T+ − T−) / (n(n+1)/2)

    where T+ (T−) is the sum of ranks of positive (negative) differences
    between x and y. Range: [−1, 1].

    Interpretation:
        r > 0  → x tends to be larger than y
        r < 0  → x tends to be smaller than y
        |r| < 0.3 → small,  0.3–0.5 → medium,  > 0.5 → large

    Ties (d == 0) are excluded before ranking, consistent with how scipy's
    wilcoxon() handles them.
    """
    from scipy.stats import rankdata

    diffs  = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diffs  = diffs[diffs != 0]
    n      = len(diffs)
    if n == 0:
        return 0.0

    ranks   = rankdata(np.abs(diffs))
    T_plus  = float(ranks[diffs > 0].sum())
    T_minus = float(ranks[diffs < 0].sum())
    max_w   = n * (n + 1) / 2.0
    return float((T_plus - T_minus) / max_w)


# ─────────────────────────────────────────────────────────────────────────────
# Holm–Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """
    Apply the Holm–Bonferroni step-down correction to an array of p-values.

    This function is intended to be called ONCE on the full family of tests,
    not separately per hypothesis group.

    Returns
    -------
    adjusted p-values (same length as input), capped at 1.0
    """
    n       = len(p_values)
    order   = np.argsort(p_values)
    p_adj   = np.zeros(n)
    running_max = 0.0

    for rank, idx in enumerate(order):
        corrected   = (n - rank) * p_values[idx]
        running_max = max(running_max, corrected)
        p_adj[idx]  = min(running_max, 1.0)

    return p_adj


# ─────────────────────────────────────────────────────────────────────────────
# Single-pair Wilcoxon (no correction — correction is applied globally later)
# ─────────────────────────────────────────────────────────────────────────────

def _wilcoxon_single_pair(
    agg: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    metric: str,
    dataset_col: str = "dataset_name",
    min_datasets: int = 5,
) -> Optional[Dict]:
    """
    Run a two-sided Wilcoxon signed-rank test for one (cond_a, cond_b) pair.

    Returns a result dict (WITHOUT p-value correction) or None if there are
    fewer than min_datasets aligned pairs.
    """
    sub_a   = agg[agg["condition"] == cond_a][[dataset_col, metric]].set_index(dataset_col)
    sub_b   = agg[agg["condition"] == cond_b][[dataset_col, metric]].set_index(dataset_col)
    aligned = sub_a.join(sub_b, lsuffix="_a", rsuffix="_b", how="inner")

    n = len(aligned)
    if n < min_datasets:
        logger.debug(f"Skipping ({cond_a}, {cond_b}): only {n} aligned datasets (<{min_datasets}).")
        return None

    vals_a = aligned[f"{metric}_a"].values
    vals_b = aligned[f"{metric}_b"].values
    diffs  = vals_a - vals_b

    try:
        stat, p_val = wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        # All differences are zero — test undefined; set p = 1
        stat, p_val = np.nan, 1.0

    es = rank_biserial_correlation(vals_a, vals_b)

    return {
        "condition_A": cond_a,
        "condition_B": cond_b,
        "n_datasets":  n,
        "median_A":    float(np.median(vals_a)),
        "median_B":    float(np.median(vals_b)),
        "median_diff": float(np.median(vals_a) - np.median(vals_b)),
        "statistic":   stat,
        "p_value_raw": p_val,
        "effect_size": es,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helper
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_by_dataset(
    df: pd.DataFrame,
    metric: str,
    condition_col: str = "condition",
    dataset_col: str = "dataset_name",
) -> pd.DataFrame:
    """
    Aggregate raw results: median across seeds per (dataset, condition).

    Using the median (not mean) makes the central tendency robust to
    outlier seeds caused by degenerate random splits.

    Returns
    -------
    DataFrame with one row per (dataset, condition)
    """
    return (
        df.groupby([dataset_col, condition_col])[metric]
        .median()
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis-targeted tests with GLOBAL correction
# ─────────────────────────────────────────────────────────────────────────────

# H1: does the raw uncertainty advantage of ensembles exist?
#     Also tests single_mlp vs lgbm raw to isolate model family from ensemble size.
#     XGBoost vs LGBM tests whether findings are GBDT-implementation-specific.
_H1_PAIRS: List[Tuple[str, str]] = [
    ("deep_ensemble/none", "lgbm/none"),
    ("single_mlp/none",    "lgbm/none"),        # model family effect (no ensemble)
    ("deep_ensemble/none", "single_mlp/none"),  # ensemble size effect
    ("xgboost/none",       "lgbm/none"),        # GBDT robustness check
]

# H2: does post-hoc calibration improve uncertainty quality?
#     Tested for LGBM, XGBoost, and SingleMLP separately.
_H2_PAIRS: List[Tuple[str, str]] = [
    # LGBM calibration
    ("lgbm/none",     "lgbm/temp"),
    ("lgbm/none",     "lgbm/logistic"),
    ("lgbm/none",     "lgbm/isotonic"),
    ("lgbm/temp",     "lgbm/isotonic"),
    # XGBoost calibration (robustness check — same structure)
    ("xgboost/none",  "xgboost/temp"),
    ("xgboost/none",  "xgboost/logistic"),
    ("xgboost/none",  "xgboost/isotonic"),
    # SingleMLP calibration
    ("single_mlp/none",     "single_mlp/temp"),
    ("single_mlp/none",     "single_mlp/logistic"),
    ("single_mlp/none",     "single_mlp/isotonic"),
]

# H3: after FAIR calibration (isotonic only), does the gap persist?
#
#     Restricted to isotonic regression because:
#       (a) TemperatureScaling on ensembles uses a pseudo-logit approximation
#           (log of averaged member probs), introducing a methodological
#           asymmetry not present when comparing GBDT vs. ensemble with isotonic.
#       (b) IsotonicCalibrator makes no logit-space assumptions and is
#           equally applicable to all model types.
#
#     deep_ensemble/temp comparisons are retained in supplementary_h3_with_temp
#     (see run_hypothesis_tests return value) for transparency, but are not
#     used to draw primary conclusions about the calibration gap.
_H3_PAIRS_PRIMARY: List[Tuple[str, str]] = [
    ("deep_ensemble/isotonic", "lgbm/isotonic"),
    ("deep_ensemble/isotonic", "xgboost/isotonic"),
    ("deep_ensemble/isotonic", "single_mlp/isotonic"),
]

# Supplementary H3 pairs including temp — kept for completeness / transparency
_H3_PAIRS_SUPPLEMENTARY: List[Tuple[str, str]] = [
    ("deep_ensemble/temp",     "lgbm/temp"),
    ("deep_ensemble/temp",     "xgboost/temp"),
    ("deep_ensemble/temp",     "single_mlp/temp"),
]

_HYPOTHESIS_PAIRS = {
    "H1":   _H1_PAIRS,
    "H2":   _H2_PAIRS,
    "H3":   _H3_PAIRS_PRIMARY,
}

# Supplementary: identical procedure, separate output key
_SUPPLEMENTARY_PAIRS = {
    "H3_supp_temp": _H3_PAIRS_SUPPLEMENTARY,
}


def run_hypothesis_tests(
    df_results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    min_datasets: int = 5,
    include_supplementary: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run all targeted hypothesis tests with a single global Holm–Bonferroni
    correction applied across every test simultaneously.

    Correction scope: all hypotheses × all metrics × all pairs.
    This is strictly more conservative (and statistically correct) than
    correcting within each hypothesis separately.

    H3 is restricted to isotonic calibration only. The supplementary H3
    comparisons using TemperatureScaling are included in the global
    correction family (so they do not inflate any p-values) but are
    reported under separate keys prefixed with 'H3_supp_'.

    Parameters
    ----------
    df_results            : DataFrame output from run_benchmark() → pd.DataFrame
    metrics               : list of metric names to test (default: NLL, ECE, Brier)
    min_datasets          : minimum number of datasets required to run a test
    include_supplementary : if True (default), H3_supp_temp pairs are included
                            in the correction family and reported separately

    Returns
    -------
    dict mapping '{H1|H2|H3|H3_supp_temp}_{metric}' → results DataFrame
    Each DataFrame has columns:
        condition_A, condition_B, n_datasets,
        median_A, median_B, median_diff,
        statistic, p_value_raw, p_value_adj, effect_size,
        effect_magnitude, significant
    """
    if metrics is None:
        metrics = ["cal_nll", "cal_ece_mean", "cal_brier_score"]

    df = df_results.copy()

    # Normalise calibrator names: 'platt' → 'logistic' for condition strings
    df["calibrator"] = df["calibrator"].replace({"platt": "logistic"})
    df["condition"]  = df["model"] + "/" + df["calibrator"]

    all_hypothesis_pairs = dict(_HYPOTHESIS_PAIRS)
    if include_supplementary:
        all_hypothesis_pairs.update(_SUPPLEMENTARY_PAIRS)

    # ── Pass 1: collect ALL uncorrected test results ─────────────────────────
    all_records: List[Dict] = []

    for metric in metrics:
        agg = aggregate_by_dataset(df, metric, condition_col="condition")

        for hyp_name, pairs in all_hypothesis_pairs.items():
            for cond_a, cond_b in pairs:
                row = _wilcoxon_single_pair(
                    agg, cond_a, cond_b, metric,
                    min_datasets=min_datasets,
                )
                if row is not None:
                    row["hypothesis"] = hyp_name
                    row["metric"]     = metric
                    all_records.append(row)

    if not all_records:
        logger.warning("No valid test pairs found (insufficient data?).")
        return {}

    # ── Pass 2: single global Holm–Bonferroni correction ────────────────────
    df_all   = pd.DataFrame(all_records)
    adj_p    = holm_bonferroni(df_all["p_value_raw"].values)
    df_all["p_value_adj"] = adj_p
    df_all["significant"] = adj_p < 0.05
    df_all["effect_magnitude"] = df_all["effect_size"].abs().map(
        lambda r: "large" if r > 0.5 else ("medium" if r > 0.3 else "small")
    )

    n_sig = int(df_all["significant"].sum())
    n_total = len(df_all)
    logger.info(
        f"Global correction applied to {n_total} tests "
        f"({n_sig} significant at α=0.05 after Holm–Bonferroni correction)."
    )
    if include_supplementary:
        logger.info(
            "  Note: H3_supp_temp pairs are included in the correction "
            "family but reported separately — do not use for primary conclusions."
        )

    # ── Organise output by hypothesis × metric ───────────────────────────────
    result: Dict[str, pd.DataFrame] = {}
    for hyp_name in all_hypothesis_pairs:
        for metric in metrics:
            key    = f"{hyp_name}_{metric}"
            subset = df_all[
                (df_all["hypothesis"] == hyp_name) &
                (df_all["metric"]     == metric)
            ].copy().reset_index(drop=True)
            result[key] = subset

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Legacy: unrestricted pairwise comparison (DEPRECATED — for exploratory use)
# ─────────────────────────────────────────────────────────────────────────────

@deprecated(
    "Use run_hypothesis_tests() which applies a single global Holm–Bonferroni "
    "correction across all tests. pairwise_wilcoxon applies correction only "
    "within the pairs you pass in, making results incomparable to main results."
)
def pairwise_wilcoxon(
    df: pd.DataFrame,
    condition_col: str,
    metric: str,
    dataset_col: str = "dataset_name",
    min_datasets: int = 5,
) -> pd.DataFrame:
    """
    Perform all pairwise Wilcoxon tests between conditions.

    DEPRECATED: This function applies its own internal Holm–Bonferroni
    correction limited to the pairs passed in. Results are NOT comparable
    to those from run_hypothesis_tests(), which applies correction globally.
    This function is retained only for exploratory / supplementary analysis
    and will be removed in a future version.

    Returns a DataFrame with one row per pair, sorted by adjusted p-value.
    """
    import itertools

    conditions = sorted(df[condition_col].unique())
    pairs      = list(itertools.combinations(conditions, 2))

    rows = []
    for cond_a, cond_b in pairs:
        agg = df  # already aggregated by caller
        row = _wilcoxon_single_pair(agg, cond_a, cond_b, metric, dataset_col, min_datasets)
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    adj_p     = holm_bonferroni(result_df["p_value_raw"].values)
    result_df["p_value_adj"] = adj_p
    result_df["significant"] = adj_p < 0.05
    result_df["effect_magnitude"] = result_df["effect_size"].abs().map(
        lambda r: "large" if r > 0.5 else ("medium" if r > 0.3 else "small")
    )
    return result_df.sort_values("p_value_adj").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printer
# ─────────────────────────────────────────────────────────────────────────────

def print_hypothesis_summary(
    hypothesis_results: Dict[str, pd.DataFrame],
    alpha: float = 0.05,
    include_supplementary: bool = True,
) -> None:
    """
    Print a human-readable summary of all hypothesis test results.

    Columns shown: conditions, n_datasets, medians, adjusted p-value,
    effect size, effect magnitude, and significance flag.

    Supplementary H3 (with temp) is printed in a clearly demarcated section.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print(f"(Holm–Bonferroni correction applied globally across all tests; α = {alpha})")
    print("=" * 80)

    primary_keys = [k for k in hypothesis_results if not k.startswith("H3_supp")]
    supp_keys    = [k for k in hypothesis_results if k.startswith("H3_supp")]

    for key in sorted(primary_keys):
        df = hypothesis_results[key]
        if df.empty:
            print(f"\n{key}: no results (insufficient data).")
            continue
        print(f"\n── {key} ──")
        _print_results(df)

    if include_supplementary and supp_keys:
        print("\n" + "─" * 80)
        print("SUPPLEMENTARY — H3 with TemperatureScaling (approximate for ensembles)")
        print("These pairs are in the global correction family but should not be")
        print("used for primary conclusions. See paper §methods for rationale.")
        print("─" * 80)
        for key in sorted(supp_keys):
            df = hypothesis_results[key]
            if df.empty:
                print(f"\n{key}: no results.")
                continue
            print(f"\n── {key} ──")
            _print_results(df)

    print("\n" + "=" * 80)


def _print_results(df: pd.DataFrame) -> None:
    display_cols = [
        "condition_A", "condition_B",
        "n_datasets", "median_A", "median_B", "median_diff",
        "p_value_adj", "effect_size", "effect_magnitude", "significant",
    ]
    present = [c for c in display_cols if c in df.columns]
    print(df[present].to_string(index=False, float_format="{:.4f}".format))
