"""
metrics.py
----------
Uncertainty evaluation metrics for probabilistic classifiers.

Primary metrics:
    - nll(probs, y)            : Negative Log-Likelihood
    - ece(probs, y, n_bins)    : Expected Calibration Error
    - brier_score(probs, y)    : Multi-class Brier Score

Secondary:
    - accuracy(probs, y)       : Classification accuracy

ECE is computed with three bin sizes (10, 15, 20) and the mean is reported
as the 'ece_mean' field to ensure stability across bin granularities.

Reliability diagram data is also produced for visualisation.

Fix (v2): ECE bin membership now uses `>= lo` instead of `> lo` so that
samples with confidence exactly 0.0 are included in the first bin rather
than silently dropped.  This is a correctness fix; in practice probabilities
are clipped above zero but the boundary case should be handled correctly.
"""

import numpy as np
from typing import Dict, List, Tuple

from src.utils import EPS as _EPS


# ─────────────────────────────────────────────────────────────────────────────
# NLL
# ─────────────────────────────────────────────────────────────────────────────

def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Mean negative log-likelihood (cross-entropy) over test samples.

    Parameters
    ----------
    probs : (n_samples, n_classes) predicted probabilities
    y     : (n_samples,) integer class labels

    Returns
    -------
    float — lower is better
    """
    n       = len(y)
    clipped = np.clip(probs[np.arange(n), y], _EPS, 1.0)
    return float(-np.mean(np.log(clipped)))


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────────────────────

def ece(
    probs: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error using equal-width confidence bins.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) − conf(B_b)|

    Bin membership: sample i belongs to bin b if
        lo_b <= confidence_i <= hi_b   (closed on both ends)

    Using `>= lo` (closed lower bound) ensures the very first bin
    [0, 1/n_bins] includes samples with confidence exactly 0.0 and prevents
    any sample from falling outside all bins.

    Parameters
    ----------
    probs  : (n_samples, n_classes) predicted probabilities
    y      : (n_samples,) integer class labels
    n_bins : number of confidence bins (default: 10)

    Returns
    -------
    float — lower is better
    """
    n            = len(y)
    confidences  = probs.max(axis=1)
    predictions  = probs.argmax(axis=1)
    correct      = (predictions == y).astype(float)

    bin_edges    = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val      = 0.0

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        # FIX: use >= lo (was > lo) so the first bin includes confidence == 0
        mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_bin  = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece_val += (mask.sum() / n) * abs(acc_bin - conf_bin)

    return float(ece_val)


def ece_multi_bin(
    probs: np.ndarray,
    y: np.ndarray,
    bin_sizes: Tuple[int, ...] = (10, 15, 20),
) -> Dict[str, float]:
    """
    Compute ECE for multiple bin sizes and return mean ECE.

    Parameters
    ----------
    probs     : (n_samples, n_classes) predicted probabilities
    y         : (n_samples,) integer class labels
    bin_sizes : bin counts to evaluate

    Returns
    -------
    dict with keys: 'ece_10', 'ece_15', 'ece_20', 'ece_mean'
    """
    results: Dict[str, float] = {}
    ece_vals: List[float] = []
    for b in bin_sizes:
        val = ece(probs, y, n_bins=b)
        results[f"ece_{b}"] = val
        ece_vals.append(val)
    results["ece_mean"] = float(np.mean(ece_vals))
    return results


def reliability_diagram_data(
    probs: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute per-bin accuracy and mean confidence for a reliability diagram.

    Uses the same closed-lower-bound convention as ece() for consistency.

    Returns
    -------
    dict with keys: 'bin_centers', 'mean_confidence', 'mean_accuracy', 'bin_counts'
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == y).astype(float)

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences <= hi)  # closed on both ends
        centers.append((lo + hi) / 2)
        counts.append(mask.sum())
        accs.append(correct[mask].mean() if mask.sum() > 0 else 0.0)
        confs.append(confidences[mask].mean() if mask.sum() > 0 else 0.0)

    return {
        "bin_centers":     np.array(centers),
        "mean_confidence": np.array(confs),
        "mean_accuracy":   np.array(accs),
        "bin_counts":      np.array(counts, dtype=int),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Brier Score
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """
    Multi-class Brier Score (mean squared error between probs and one-hot targets).

    BS = (1/n) Σ_i Σ_k (p_{ik} − 1{y_i = k})^2

    Parameters
    ----------
    probs : (n_samples, n_classes)
    y     : (n_samples,) integer class labels

    Returns
    -------
    float — lower is better
    """
    n, k    = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    """Top-1 classification accuracy."""
    return float((probs.argmax(axis=1) == y).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Composite evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    probs: np.ndarray,
    y: np.ndarray,
    bin_sizes: Tuple[int, ...] = (10, 15, 20),
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a set of predicted probabilities.

    Parameters
    ----------
    probs     : (n_samples, n_classes) calibrated probabilities
    y         : (n_samples,) true integer labels
    bin_sizes : ECE bin granularities

    Returns
    -------
    dict with keys: nll, ece_10, ece_15, ece_20, ece_mean,
                    brier_score, accuracy
    """
    metrics: Dict[str, float] = {}
    metrics["nll"]         = nll(probs, y)
    metrics["brier_score"] = brier_score(probs, y)
    metrics["accuracy"]    = accuracy(probs, y)
    metrics.update(ece_multi_bin(probs, y, bin_sizes=bin_sizes))
    return metrics
