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
"""

import numpy as np
from typing import Dict, List, Tuple

from src.utils import EPS as _EPS


# ─────────────────────────────────────────────────────────────────────────────
# NLL
# ─────────────────────────────────────────────────────────────────────────────

def nll(probs: np.ndarray, y: np.ndarray) -> float:
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
    metrics: Dict[str, float] = {}
    metrics["nll"]         = nll(probs, y)
    metrics["brier_score"] = brier_score(probs, y)
    metrics["accuracy"]    = accuracy(probs, y)
    metrics.update(ece_multi_bin(probs, y, bin_sizes=bin_sizes))
    return metrics
