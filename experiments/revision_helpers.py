"""
revision_helpers.py
-------------------
Models and calibrators introduced in the major revision, plus the shared
train-once / calibrate-many runner.

Why these classes live here rather than in src/:
    The submitted repo's src/ tree (calibration.py, models.py, etc.) is
    the v1 codebase that produced the original results_raw.csv (2,880
    rows, 4 models, 4 calibrators). Keeping src/ untouched lets reviewers
    reproduce the v1 numbers exactly. The revision adds new model
    classes (CatBoost, MC-Dropout, TabPFN), a new calibrator (Dirichlet
    ODIR), a new ensemble variant (member-level TS), and a new ensemble
    cardinality sweep (M in {3, 5, 10}). 
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# v1 codebase
from src.calibration import BaseCalibrator, CALIBRATOR_LABELS, build_calibrator
from src.metrics import evaluate_all
from src.models import (
    DeepEnsemble,
    LightGBMModel,
    SingleMLP,
    XGBoostModel,
    get_device,
)
from src.utils import EPS

logger = logging.getLogger(__name__)


# =============================================================================
# CatBoost classifier  
# =============================================================================

class CatBoostModel:
    """CatBoost classifier mirroring the LightGBM/XGBoost AutoML-default profile.

    random_seed is intentionally NOT set: GBDT seed-to-seed variance across
    the 5 benchmark seeds is split-induced (driven by the per-seed
    train/val/cal/test partition), matching how LightGBM/XGBoost are
    configured in this codebase.
    """

    _DEFAULT_PARAMS = {
        "learning_rate":       0.05,
        "depth":               6,
        "min_data_in_leaf":    20,
        "rsm":                 0.8,
        "subsample":           0.8,
        "bootstrap_type":      "Bernoulli",
        "l2_leaf_reg":         3.0,
        "od_type":             "Iter",
        "od_wait":             50,
        "verbose":             False,
        "thread_count":        -1,
        "allow_writing_files": False,
    }

    def __init__(self, n_classes: int, num_boost_round: int = 500,
                 extra_params: Optional[dict] = None) -> None:
        self.n_classes = n_classes
        params = dict(self._DEFAULT_PARAMS)
        params["iterations"] = num_boost_round
        if n_classes > 2:
            params["loss_function"] = "MultiClass"
            params["eval_metric"]   = "MultiClass"
            params["classes_count"] = n_classes
        else:
            params["loss_function"] = "Logloss"
            params["eval_metric"]   = "Logloss"
        if extra_params:
            params.update(extra_params)
        self._params = params
        self._model  = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "CatBoostModel":
        from catboost import CatBoostClassifier
        self._model = CatBoostClassifier(**self._params)
        eval_set = (X_val, y_val) if X_val is not None else None
        self._model.fit(X_train, y_train, eval_set=eval_set,
                        verbose=False, plot=False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before predict_proba()."
        return np.clip(self._model.predict_proba(X), EPS, 1.0)


# =============================================================================
# Dirichlet ODIR calibrator  
# =============================================================================

class DirichletODIRCalibrator(BaseCalibrator):
    """Dirichlet calibration with off-diagonal and intercept regularisation
    (Kull et al. 2019). Wraps dirichletcal.FullDirichletCalibrator.

    Library defaults (reg_lambda=reg_mu=1e-3) are used per the paper's
    AutoML-defaults philosophy (not CV-tuned per dataset).
    """

    def __init__(self, reg_lambda: float = 1e-3, reg_mu: float = 1e-3) -> None:
        self.reg_lambda = reg_lambda
        self.reg_mu     = reg_mu
        self._model     = None

    def fit(self, probs_val, y_val) -> "DirichletODIRCalibrator":
        from dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
        self._model = FullDirichletCalibrator(
            reg_lambda=self.reg_lambda,
            reg_mu=self.reg_mu,
        )
        self._model.fit(probs_val, y_val)
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before calibrate()."
        return np.clip(self._model.predict_proba(probs), EPS, 1.0 - EPS)


def build_calibrator_extended(name: str) -> BaseCalibrator:
    """Calibrator factory that knows the new 'dirichlet' key in addition
    to the four v1 calibrators (none / temp / logistic / isotonic).
    """
    if name == "dirichlet":
        return DirichletODIRCalibrator()
    return build_calibrator(name)


CALIBRATOR_LABELS_EXTENDED = {**CALIBRATOR_LABELS, "dirichlet": "Dirichlet ODIR"}


# =============================================================================
# MC-Dropout MLP  
# =============================================================================

class MCDropoutMLP(SingleMLP):
    """SingleMLP with MC-Dropout at inference (Gal & Ghahramani, 2016).

    Training is identical to SingleMLP. At inference, BatchNorm uses
    running statistics (eval mode) but dropout layers are kept in train
    mode so they stay stochastic; the predictive distribution is the
    arithmetic mean of T stochastic forward passes.
    """

    def __init__(self, T: int = 30, **kwargs) -> None:
        super().__init__(**kwargs)
        self.T = T

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before predict_proba()."
        self._model.eval()
        for module in self._model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        with torch.no_grad():
            Xt = torch.FloatTensor(X).to(self.device)
            samples = []
            for _ in range(self.T):
                logits = self._model(Xt)
                samples.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.mean(np.stack(samples, axis=0), axis=0)


# =============================================================================
# Member-level temperature scaling  
# =============================================================================

def fit_temperature_scalar(probs: np.ndarray, y: np.ndarray,
                           lr: float = 0.01, max_iter: int = 200) -> float:
    """Fit a single temperature T > 0 on (clipped log-probs, labels)
    via L-BFGS. Mirrors src.calibration.TemperatureScaling.fit but
    returns the scalar T so it can be applied per ensemble member.
    """
    log_probs = np.log(np.clip(probs, EPS, 1.0)).astype(np.float32)
    logits_t  = torch.from_numpy(log_probs)
    labels_t  = torch.from_numpy(y.astype(np.int64))

    log_T     = nn.Parameter(torch.zeros(1))
    optimizer = optim.LBFGS([log_T], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / log_T.exp()
        loss   = criterion(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def apply_member_level_ts(
    probs_val_cal_members: np.ndarray,
    probs_test_members:    np.ndarray,
    y_val_cal:             np.ndarray,
) -> Tuple[np.ndarray, List[float]]:
    """Per-member temperature scaling for an M-member ensemble.

    Fit T_m on val_cal probs of member m, apply each T_m to that member's
    test probs, then average across members. This is the non-approximate
    TS variant for ensembles (vs. the pseudo-logit approximation that
    scales the post-averaged probability vector).

    probs_val_cal_members : shape (M, n_val_cal, K)
    probs_test_members    : shape (M, n_test, K)
    y_val_cal             : shape (n_val_cal,)

    Returns (calibrated_test_probs, [T_1, ..., T_M]).
    """
    M = probs_val_cal_members.shape[0]
    scaled_members, Ts = [], []

    for m in range(M):
        T_m = fit_temperature_scalar(probs_val_cal_members[m], y_val_cal)
        Ts.append(T_m)

        log_probs = np.log(np.clip(probs_test_members[m], EPS, 1.0))
        scaled    = log_probs / T_m
        shifted   = scaled - scaled.max(axis=1, keepdims=True)
        exp_s     = np.exp(shifted)
        scaled_members.append(exp_s / exp_s.sum(axis=1, keepdims=True))

    avg = np.mean(np.stack(scaled_members, axis=0), axis=0)
    return np.clip(avg, EPS, 1.0), Ts


# =============================================================================
# TabPFN v2 wrapper  
# =============================================================================

def tabpfn_compatible(task_meta: Dict, train_frac: float = 0.54) -> bool:
    """Check whether a manifest row fits TabPFN v2's inference constraints
    under the standard 54% training fraction:
        train_size <= 10_000, n_features <= 500, n_classes <= 10.
    """
    n_train_est = task_meta["n_samples"] * train_frac
    return (
        n_train_est <= 10_000
        and task_meta["n_features"] <= 500
        and task_meta["n_classes"]  <= 10
    )


class TabPFNModel:
    """TabPFN v2 wrapper with the same fit / predict_proba API as the
    other models in this benchmark.

    Note: TabPFN's `fit` is context-loading, not parameter training.
    `train_time_s` therefore measures context loading + first-inference
    setup, not gradient training (TabPFN v2 is pretrained).

    Set TABPFN_TOKEN in the environment before calling fit().
    """

    def __init__(self, n_classes: int, device: str = "cuda",
                 ignore_pretraining_limits: bool = False) -> None:
        self.n_classes                  = n_classes
        self._device                    = device
        self._ignore_pretraining_limits = ignore_pretraining_limits
        self._model                     = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "TabPFNModel":
        from tabpfn import TabPFNClassifier
        if not os.environ.get("TABPFN_TOKEN"):
            raise RuntimeError(
                "TABPFN_TOKEN environment variable not set. "
                "Export it before running: export TABPFN_TOKEN=<your-token>"
            )
        self._model = TabPFNClassifier(
            device=self._device,
            ignore_pretraining_limits=self._ignore_pretraining_limits,
        )
        self._model.fit(X_train, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before predict_proba()."
        return np.clip(self._model.predict_proba(X), EPS, 1.0)


# =============================================================================
# Model factory and runner  
# =============================================================================

# Ensemble cardinalities swept in the revision.
ENSEMBLE_SIZES = {
    "deep_ensemble_m3":  3,
    "deep_ensemble":     5,
    "deep_ensemble_m10": 10,
}

MLP_HIDDEN_DIMS = (256, 128)
MLP_LR          = 1e-3
MLP_EPOCHS      = 200
MLP_BATCH_SIZE  = 256
MLP_PATIENCE    = 20


def build_model(model_name: str, n_classes: int, n_features: int,
                seed: int, device=None) -> Tuple[Any, Dict[str, Any]]:
    """Return (model, fit_kwargs) for one model by string key."""
    if device is None:
        device = get_device()

    if model_name == "lgbm":
        return LightGBMModel(n_classes=n_classes), {}
    if model_name == "xgboost":
        return XGBoostModel(n_classes=n_classes), {}
    if model_name == "catboost":
        return CatBoostModel(n_classes=n_classes), {}

    if model_name == "single_mlp":
        return (
            SingleMLP(
                input_dim=n_features, n_classes=n_classes,
                hidden_dims=MLP_HIDDEN_DIMS, lr=MLP_LR, epochs=MLP_EPOCHS,
                batch_size=MLP_BATCH_SIZE, patience=MLP_PATIENCE,
                device=device,
            ),
            {"seed": seed},
        )
    if model_name == "mc_dropout":
        return (
            MCDropoutMLP(
                T=30,
                input_dim=n_features, n_classes=n_classes,
                hidden_dims=MLP_HIDDEN_DIMS, lr=MLP_LR, epochs=MLP_EPOCHS,
                batch_size=MLP_BATCH_SIZE, patience=MLP_PATIENCE,
                device=device,
            ),
            {"seed": seed},
        )
    if model_name in ENSEMBLE_SIZES:
        return (
            DeepEnsemble(
                n_members=ENSEMBLE_SIZES[model_name],
                input_dim=n_features, n_classes=n_classes,
                hidden_dims=MLP_HIDDEN_DIMS, lr=MLP_LR, epochs=MLP_EPOCHS,
                batch_size=MLP_BATCH_SIZE, patience=MLP_PATIENCE,
                device=device,
            ),
            {"base_seed": seed},
        )
    raise ValueError(f"Unknown model key: {model_name!r}")


def probs_path(probs_dir: Path, task_id: int, model_name: str, seed: int) -> Path:
    return probs_dir / f"{task_id}__{model_name}__seed{seed}.npz"


def members_path(probs_dir: Path, task_id: int, model_name: str, seed: int) -> Path:
    return probs_dir / f"{task_id}__{model_name}__seed{seed}__members.npz"


def get_predictions(split: Dict, model_name: str, seed: int,
                    probs_dir: Path, device=None,
                    need_members: bool = False) -> Dict[str, Any]:
    """Train or load cached predictions for one (task, model, seed).

    Cache layout under probs_dir:
        <task>__<model>__seed<s>.npz            mean probs (all models)
        <task>__<model>__seed<s>__members.npz   per-member probs (ensembles)

    The second file is written only when need_members=True and the model
    is an ensemble. Returns a dict with probs_val_cal, probs_test,
    y_val_cal, y_test, train_time_s, and (for ensembles when present)
    probs_val_cal_members / probs_test_members.
    """
    task_id     = split["task_id"]
    is_ensemble = model_name in ENSEMBLE_SIZES

    if is_ensemble:
        mpath = members_path(probs_dir, task_id, model_name, seed)
        if mpath.exists():
            data = np.load(mpath)
            return {
                "probs_val_cal":         data["probs_val_cal"],
                "probs_test":            data["probs_test"],
                "probs_val_cal_members": data["probs_val_cal_members"],
                "probs_test_members":    data["probs_test_members"],
                "y_val_cal":             data["y_val_cal"],
                "y_test":                data["y_test"],
                "train_time_s":          float(data["train_time_s"]),
            }

    spath = probs_path(probs_dir, task_id, model_name, seed)
    if spath.exists() and not (is_ensemble and need_members):
        data = np.load(spath)
        return {
            "probs_val_cal": data["probs_val_cal"],
            "probs_test":    data["probs_test"],
            "y_val_cal":     data["y_val_cal"],
            "y_test":        data["y_test"],
            "train_time_s":  float(data["train_time_s"]),
        }

    # Train fresh
    model, fit_kwargs = build_model(
        model_name, split["n_classes"], split["n_features"], seed, device,
    )

    t0 = time.time()
    model.fit(split["X_train"], split["y_train"],
              split["X_val_es"], split["y_val_es"], **fit_kwargs)
    train_time = time.time() - t0

    probs_val_cal = model.predict_proba(split["X_val_cal"])
    probs_test    = model.predict_proba(split["X_test"])

    out: Dict[str, Any] = {
        "probs_val_cal": probs_val_cal,
        "probs_test":    probs_test,
        "y_val_cal":     split["y_val_cal"],
        "y_test":        split["y_test"],
        "train_time_s":  train_time,
    }

    probs_dir.mkdir(parents=True, exist_ok=True)
    if is_ensemble and need_members:
        probs_vc_m = model.predict_proba_members(split["X_val_cal"])
        probs_te_m = model.predict_proba_members(split["X_test"])
        out["probs_val_cal_members"] = probs_vc_m
        out["probs_test_members"]    = probs_te_m
        np.savez_compressed(
            members_path(probs_dir, task_id, model_name, seed),
            probs_val_cal=probs_val_cal, probs_test=probs_test,
            probs_val_cal_members=probs_vc_m, probs_test_members=probs_te_m,
            y_val_cal=split["y_val_cal"], y_test=split["y_test"],
            train_time_s=np.array(train_time),
        )
    else:
        np.savez_compressed(
            spath,
            probs_val_cal=probs_val_cal, probs_test=probs_test,
            y_val_cal=split["y_val_cal"], y_test=split["y_test"],
            train_time_s=np.array(train_time),
        )

    return out


def evaluate_cell(split: Dict, model_name: str, calibrator_name: str,
                  seed: int, preds: Dict) -> Dict[str, Any]:
    """Apply the named calibrator to cached predictions, compute metrics,
    return a flat result row.

    calibrator_name == 'temp_member' uses per-member TS for ensembles and
    requires probs_*_members in `preds`.
    """
    y_test         = preds["y_test"]
    y_val_cal      = preds["y_val_cal"]
    probs_test     = preds["probs_test"]
    probs_val_cal  = preds["probs_val_cal"]

    if calibrator_name == "temp_member":
        if "probs_val_cal_members" not in preds:
            raise ValueError(
                f"calibrator 'temp_member' needs per-member probabilities, "
                f"missing in cache for model {model_name!r}."
            )
        probs_cal, _ = apply_member_level_ts(
            preds["probs_val_cal_members"],
            preds["probs_test_members"],
            y_val_cal,
        )
        cal_label = "Member-level Temperature Scaling"
    else:
        calibrator = build_calibrator_extended(calibrator_name)
        calibrator.fit(probs_val_cal, y_val_cal)
        probs_cal  = calibrator.calibrate(probs_test)
        cal_label  = CALIBRATOR_LABELS_EXTENDED.get(calibrator_name, calibrator_name)

    metrics     = evaluate_all(probs_cal,  y_test)
    raw_metrics = evaluate_all(probs_test, y_test)

    return {
        "task_id":          split["task_id"],
        "dataset_name":     split["name"],
        "n_samples":        split["n_samples"],
        "n_train":          split["n_train"],
        "n_val_es":         split["n_val_es"],
        "n_val_cal":        split["n_val_cal"],
        "n_test":           split["n_test"],
        "n_features":       split["n_features"],
        "n_classes":        split["n_classes"],
        "seed":             seed,
        "model":            model_name,
        "calibrator":       calibrator_name,
        "calibrator_label": cal_label,
        **{f"cal_{k}": v for k, v in metrics.items()},
        **{f"raw_{k}": v for k, v in raw_metrics.items()},
        "train_time_s":     round(preds["train_time_s"], 2),
    }
