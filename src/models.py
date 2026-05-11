"""
models.py
---------
Model implementations for the calibration benchmark.

Deep ensemble definition follows the current standard:
    Lakshminarayanan et al. (2017) — original formulation
    Gorishniy et al. / TabM (ICLR 2025) — "multiple DL models of the same
        architecture trained independently under different random seeds.
"""

import logging
from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils import EPS

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# MLP architecture
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        n_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Single MLP
# ─────────────────────────────────────────────────────────────────────────────

class SingleMLP:
    """
    One MLP with early stopping on X_val_es.

    Architecture: input → 256 → 128 → n_classes
    Optimiser   : AdamW, lr=1e-3, weight_decay=1e-4
    Scheduler   : CosineAnnealingLR over max epochs
    Gradient clip: ℓ2 norm 1.0
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 256,
        patience: int = 20,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.input_dim   = input_dim
        self.n_classes   = n_classes
        self.hidden_dims = hidden_dims
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.patience    = patience
        self.dropout     = dropout
        self.device      = device or get_device()
        self._model: Optional[_MLP] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> "SingleMLP":
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._model = _MLP(
            self.input_dim, self.hidden_dims, self.n_classes, self.dropout
        ).to(self.device)

        optimizer = optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.CrossEntropyLoss()

        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=min(self.batch_size, len(y_train)),
            shuffle=True,
        )

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        for epoch in range(self.epochs):
            self._model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if X_val is not None and y_val is not None:
                val_loss = self._val_loss(X_val, y_val, criterion)
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    best_state    = {k: v.clone() for k, v in
                                     self._model.state_dict().items()}
                    no_improve    = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        logger.debug(f"Early stop at epoch {epoch}.")
                        break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def _val_loss(self, X_val, y_val, criterion) -> float:
        self._model.eval()
        with torch.no_grad():
            Xv = torch.FloatTensor(X_val).to(self.device)
            yv = torch.LongTensor(y_val).to(self.device)
            return criterion(self._model(Xv), yv).item()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.FloatTensor(X).to(self.device))
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def get_logits(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            return self._model(
                torch.FloatTensor(X).to(self.device)
            ).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Deep Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class DeepEnsemble:
    """
    Deep ensemble of M independently initialised MLPs.

    Follows the definition of Lakshminarayanan et al. (2017) and
    the current tabular DL standard (TabM, ICLR 2025):
        "multiple DL models of the same architecture trained independently
         under different random seeds"

    Diversity source: random weight initialisation only.
    No bootstrap sampling — bootstrap was not part of the original
    formulation and conflates diversity from initialisation with
    diversity from training data subsampling.

    Parameters
    ----------
    n_members : M (default 5 — standard in calibration and tabular literature)
    **mlp_kwargs : forwarded to SingleMLP constructor
    """

    def __init__(self, n_members: int = 5, **mlp_kwargs) -> None:
        self.n_members  = n_members
        self.mlp_kwargs = mlp_kwargs
        self.members: List[SingleMLP] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        base_seed: int = 0,
    ) -> "DeepEnsemble":
        self.members = []
        for i in range(self.n_members):
            seed = base_seed + i * 137
            mlp  = SingleMLP(**self.mlp_kwargs)
            mlp.fit(X_train, y_train, X_val, y_val, seed=seed)
            self.members.append(mlp)
            logger.debug(f"  Member {i+1}/{self.n_members} trained.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Arithmetic mean of member softmax outputs."""
        return np.stack(
            [m.predict_proba(X) for m in self.members], axis=0
        ).mean(axis=0)

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """
        Mean of member log-probabilities (pseudo-logit space for TS).

        For M=1 this recovers standard temperature scaling exactly.
        For M>1 this is an approximation; isotonic regression is
        preferred for ensembles as it requires no logit-space assumption.
        """
        log_probs = np.stack(
            [np.log(np.clip(m.predict_proba(X), EPS, 1.0))
             for m in self.members],
            axis=0,
        )
        return log_probs.mean(axis=0)

    def predict_proba_members(self, X: np.ndarray) -> np.ndarray:
        """Per-member probabilities — shape (n_members, n_samples, n_classes)."""
        return np.stack([m.predict_proba(X) for m in self.members], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────────────────────────────────────

class LightGBMModel:
    """
    LightGBM classifier with early stopping.

    Hyperparameters follow standard AutoML defaults.
    """

    _DEFAULT_PARAMS = {
        "learning_rate":     0.05,
        "num_leaves":        63,
        "min_child_samples": 20,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      1,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    def __init__(
        self,
        n_classes: int,
        num_boost_round: int = 500,
        extra_params: Optional[dict] = None,
    ) -> None:
        self.n_classes       = n_classes
        self.num_boost_round = num_boost_round
        self._model          = None

        params = dict(self._DEFAULT_PARAMS)
        if n_classes > 2:
            params.update({"objective": "multiclass",
                           "num_class": n_classes,
                           "metric":    "multi_logloss"})
        else:
            params.update({"objective": "binary",
                           "metric":    "binary_logloss"})
        if extra_params:
            params.update(extra_params)
        self._params = params

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "LightGBMModel":
        dtrain    = lgb.Dataset(X_train, label=y_train)
        valid_sets = []
        callbacks  = [lgb.log_evaluation(period=-1)]
        if X_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            callbacks.append(lgb.early_stopping(50, verbose=False))
        self._model = lgb.train(
            self._params, dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets or None,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None
        raw = self._model.predict(X)
        if self.n_classes == 2:
            raw = np.clip(raw, EPS, 1.0 - EPS)
            return np.column_stack([1.0 - raw, raw])
        return raw


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostModel:
    """XGBoost robustness check — hyperparameters mirror LightGBM."""

    _DEFAULT_PARAMS = {
        "learning_rate":    0.05,
        "max_depth":        6,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "verbosity":        0,
        "n_jobs":          -1,
    }

    def __init__(
        self,
        n_classes: int,
        num_boost_round: int = 500,
        extra_params: Optional[dict] = None,
    ) -> None:
        self.n_classes       = n_classes
        self.num_boost_round = num_boost_round
        self._model          = None

        params = dict(self._DEFAULT_PARAMS)
        if n_classes > 2:
            params.update({"objective":   "multi:softprob",
                           "num_class":   n_classes,
                           "eval_metric": "mlogloss"})
        else:
            params.update({"objective":   "binary:logistic",
                           "eval_metric": "logloss"})
        if extra_params:
            params.update(extra_params)
        self._params = params

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> "XGBoostModel":
        import xgboost as xgb
        dtrain    = xgb.DMatrix(X_train, label=y_train)
        evals, callbacks = [], []
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dval, "val")]
            callbacks = [xgb.callback.EarlyStopping(
                rounds=50, save_best=True, maximize=False)]
        self._model = xgb.train(
            self._params, dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals, callbacks=callbacks, verbose_eval=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        assert self._model is not None
        raw = self._model.predict(xgb.DMatrix(X))
        if self.n_classes == 2:
            raw = np.clip(raw, EPS, 1.0 - EPS)
            return np.column_stack([1.0 - raw, raw])
        return raw
