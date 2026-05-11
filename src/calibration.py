"""
calibration.py
--------------
Post-hoc calibration methods.

Compatible with scikit-learn >= 1.0.
The multi_class parameter was removed in sklearn 1.5 — this file does not
use it. LogisticRegression with solver='lbfgs' handles multinomial
classification automatically.

Calibrators:
    none      → IdentityCalibrator   (raw outputs, no change)
    temp      → TemperatureScaling   (Guo et al. 2017)
    logistic  → LogisticRecalibrator (multinomial logistic on probabilities)
    isotonic  → IsotonicCalibrator   (per-class isotonic regression)
    platt     → alias for logistic (legacy name)
"""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils import EPS, deprecated

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseCalibrator(ABC):

    @abstractmethod
    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "BaseCalibrator":
        ...

    @abstractmethod
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__


# ─────────────────────────────────────────────────────────────────────────────
# Identity
# ─────────────────────────────────────────────────────────────────────────────

class IdentityCalibrator(BaseCalibrator):
    """Pass-through — returns raw model probabilities clipped to [EPS, 1-EPS]."""

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "IdentityCalibrator":
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        return np.clip(probs, EPS, 1.0 - EPS)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaling(BaseCalibrator):
    """
    Guo et al. (2017) — single scalar T minimises NLL on val_cal via L-BFGS.
    T is parameterised as exp(log_T) to enforce T > 0.

    For ensembles, log(averaged_probs) is used as pseudo-logit space.
    This is an acknowledged approximation; isotonic regression is preferred
    for ensembles as it makes no logit-space assumptions.
    """

    def __init__(self, lr: float = 0.01, max_iter: int = 200) -> None:
        self.lr       = lr
        self.max_iter = max_iter
        self._log_T   = None   # scalar torch Parameter after fit()
        self._T_val   = 1.0    # python float for calibrate()

    @property
    def temperature(self) -> float:
        return self._T_val

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaling":
        import torch
        import torch.nn as nn
        import torch.optim as optim

        log_probs = np.log(np.clip(probs_val, EPS, 1.0)).astype(np.float32)
        logits_t  = torch.from_numpy(log_probs)
        labels_t  = torch.from_numpy(y_val.astype(np.int64))

        log_T     = nn.Parameter(torch.zeros(1))
        optimizer = optim.LBFGS([log_T], lr=self.lr, max_iter=self.max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / log_T.exp()
            loss   = criterion(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._T_val = float(log_T.exp().item())
        logger.debug(f"TemperatureScaling: T = {self._T_val:.4f}")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        log_probs = np.log(np.clip(probs, EPS, 1.0)).astype(np.float32)
        scaled    = log_probs / self._T_val
        # softmax in numpy
        shifted   = scaled - scaled.max(axis=1, keepdims=True)
        exp_s     = np.exp(shifted)
        cal       = exp_s / exp_s.sum(axis=1, keepdims=True)
        return np.clip(cal, EPS, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic Recalibrator
# ─────────────────────────────────────────────────────────────────────────────

class LogisticRecalibrator(BaseCalibrator):
    """
    Multinomial logistic regression fitted on predicted probability vectors.

    NOT classical Platt scaling (Platt 1999) — that was binary-only with a
    single affine transformation. This fits a full weight matrix on the
    (n_samples, n_classes) probability output, making it strictly more
    expressive and prone to overfitting on small val_cal sets.

    Compatible with scikit-learn >= 1.0.
    The deprecated multi_class parameter is NOT used — lbfgs handles
    multinomial automatically.

    Parameters
    ----------
    C        : inverse regularisation (default 1.0; use 0.1 for small val_cal)
    max_iter : solver iterations
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C        = C
        self.max_iter = max_iter
        self._model: Optional[LogisticRegression] = None

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "LogisticRecalibrator":
        # solver='lbfgs' handles binary and multiclass automatically.
        # Do NOT pass multi_class — it was deprecated in 1.2 and removed in 1.5.
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
        )
        self._model.fit(probs_val, y_val)
        logger.debug("LogisticRecalibrator fitted.")
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before calibrate()."
        cal = self._model.predict_proba(probs)
        return np.clip(cal, EPS, 1.0 - EPS)


# Backward-compatibility alias
class PlattScaling(LogisticRecalibrator):
    """Deprecated alias for LogisticRecalibrator. Use LogisticRecalibrator."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PlattScaling is deprecated. Use LogisticRecalibrator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Isotonic Regression
# ─────────────────────────────────────────────────────────────────────────────

class IsotonicCalibrator(BaseCalibrator):
    """
    One isotonic regression per class (one-vs-rest), then re-normalise.
    Makes no logit-space assumptions — valid for all model types.
    """

    def __init__(self, n_classes: Optional[int] = None) -> None:
        self.n_classes   = n_classes
        self._regressors = []

    def fit(self, probs_val: np.ndarray, y_val: np.ndarray) -> "IsotonicCalibrator":
        n_classes        = probs_val.shape[1]
        self.n_classes   = n_classes
        self._regressors = []
        for k in range(n_classes):
            target = (y_val == k).astype(np.float64)
            ir     = IsotonicRegression(out_of_bounds="clip", increasing=True)
            ir.fit(probs_val[:, k], target)
            self._regressors.append(ir)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        assert self._regressors, "Call fit() before calibrate()."
        cal = np.column_stack([
            self._regressors[k].predict(probs[:, k])
            for k in range(self.n_classes)
        ])
        cal = np.clip(cal, EPS, None)
        cal /= cal.sum(axis=1, keepdims=True)
        return cal


# ─────────────────────────────────────────────────────────────────────────────
# Registry / factory
# ─────────────────────────────────────────────────────────────────────────────

CALIBRATOR_REGISTRY = {
    "none":     IdentityCalibrator,
    "temp":     TemperatureScaling,
    "logistic": LogisticRecalibrator,
    "platt":    LogisticRecalibrator,   # legacy alias
    "isotonic": IsotonicCalibrator,
}

CALIBRATOR_LABELS = {
    "none":     "No calibration",
    "temp":     "Temperature Scaling",
    "logistic": "Logistic Recalibration",
    "platt":    "Logistic Recalibration",
    "isotonic": "Isotonic Regression",
}


def build_calibrator(name: str, **kwargs) -> BaseCalibrator:
    """Instantiate a calibrator by name key."""
    if name not in CALIBRATOR_REGISTRY:
        raise ValueError(
            f"Unknown calibrator '{name}'. "
            f"Choose from: {list(CALIBRATOR_REGISTRY)}"
        )
    return CALIBRATOR_REGISTRY[name](**kwargs)
