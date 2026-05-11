"""
hp_sensitivity_ablation.py
--------------------------
MLP Hyperparameter Sensitivity Ablation — Final Version

Addresses the fixed-hyperparameter threat from Section 5.7 of:
"Does Calibration Close the Uncertainty Gap? A Systematic Comparison
of Gradient-Boosted Trees and Deep Ensembles"

Design
------
- Exact same 36 datasets as the main experiment (hardcoded task IDs)
- Random search over 15 MLP hyperparameter configurations per dataset
- Best config selected by val_cal NLL (same metric as temperature scaling)
- Evaluation under temperature scaling only (recommended calibrator)
- Results directly comparable to main experiment Tables 2–7

Usage
-----
GPU 0 (seeds 0, 1, 2):
    CUDA_VISIBLE_DEVICES=0 python hp_sensitivity_ablation.py \\
        --results_csv results/results_raw.csv \\
        --output_dir  results/hp_ablation_seeds012 \\
        --n_hp_configs 15 \\
        --seeds 0 1 2

GPU 1 (seeds 3, 4):
    CUDA_VISIBLE_DEVICES=1 python hp_sensitivity_ablation.py \\
        --results_csv results/results_raw.csv \\
        --output_dir  results/hp_ablation_seeds34 \\
        --n_hp_configs 15 \\
        --seeds 3 4
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hp_ablation")

# ─────────────────────────────────────────────────────────────────────────────
# Exact 36 task IDs from the main experiment — DO NOT CHANGE
# Extracted from results_raw.csv to guarantee identical dataset coverage
# ─────────────────────────────────────────────────────────────────────────────
MAIN_EXPERIMENT_TASK_IDS = [
    6, 14, 15, 18, 23, 29, 32, 45, 49, 219,
    2074, 2079, 3021, 3022, 3549, 3560, 3573, 3913,
    9946, 9960, 9985, 10101, 14952, 14954, 14965, 14970,
    146195, 146800, 146819, 146820, 146821, 146825,
    167119, 167120, 167124, 167125,
]

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space
# Fixed config (index 0) is always included as the first candidate
# ─────────────────────────────────────────────────────────────────────────────
FIXED_CONFIG = {
    "hidden_dims":  (256, 128),
    "lr":           1e-3,
    "dropout":      0.1,
    "weight_decay": 1e-4,
    "batch_size":   256,
    "epochs":       200,
    "patience":     20,
}

HP_SEARCH_SPACE = {
    "hidden_dims": [
        (64, 32),
        (128, 64),
        (256, 128),       # ← fixed config
        (512, 256),
        (256, 128, 64),
        (512, 256, 128),
    ],
    "lr":           [1e-4, 5e-4, 1e-3, 3e-3],
    "dropout":      [0.0, 0.05, 0.1, 0.2, 0.3],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}

EPS = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    return float(-np.mean(np.log(np.clip(probs[np.arange(n), y], EPS, 1.0))))


def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    n, k = probs.shape
    oh = np.zeros_like(probs)
    oh[np.arange(n), y] = 1.0
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    n    = len(y)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok   = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        val += (m.sum() / n) * abs(ok[m].mean() - conf[m].mean())
    return float(val)


def _ece_mean(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([_ece(probs, y, b) for b in (10, 15, 20)]))


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────

def _temperature_scale(
    probs_val: np.ndarray, y_val: np.ndarray,
    probs_test: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit T on val set, apply to test set. Returns (calibrated_probs, T)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    log_p = np.log(np.clip(probs_val, EPS, 1.0)).astype(np.float32)
    lt    = torch.from_numpy(log_p)
    yt    = torch.from_numpy(y_val.astype(np.int64))
    log_T = nn.Parameter(torch.zeros(1))
    opt   = optim.LBFGS([log_T], lr=0.01, max_iter=200)
    crit  = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = crit(lt / log_T.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_T.exp().item())

    log_pt = np.log(np.clip(probs_test, EPS, 1.0)).astype(np.float32)
    return _softmax(log_pt / T), T


# ─────────────────────────────────────────────────────────────────────────────
# MLP training
# ─────────────────────────────────────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val_es: np.ndarray, y_val_es: np.ndarray,
    X_val_cal: np.ndarray, y_val_cal: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_classes: int,
    config: dict,
    seed: int,
) -> dict:
    """
    Train one MLP with given config; evaluate under temperature scaling.
    Returns metrics dict including val_cal_nll for HP selection.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims  = config["hidden_dims"]
    lr           = config["lr"]
    dropout      = config["dropout"]
    weight_decay = config["weight_decay"]
    batch_size   = min(config.get("batch_size", 256), len(y_train))
    epochs       = config.get("epochs", 200)
    patience     = config.get("patience", 20)

    # Build network — identical architecture to main experiment
    layers = []
    prev = X_train.shape[1]
    for h in hidden_dims:
        layers += [
            nn.Linear(prev, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        ]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    net = nn.Sequential(*layers).to(device)

    opt   = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    Xt = torch.FloatTensor(X_train).to(device)
    yt = torch.LongTensor(y_train).to(device)
    loader = DataLoader(TensorDataset(Xt, yt),
                        batch_size=batch_size, shuffle=True)

    Xve = torch.FloatTensor(X_val_es).to(device)
    yve = torch.LongTensor(y_val_es).to(device)

    best_loss, best_state, no_imp = float("inf"), None, 0

    for epoch in range(epochs):
        net.train()
        for Xb, yb in loader:
            opt.zero_grad()
            loss = crit(net(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            vl = crit(net(Xve), yve).item()
        if vl < best_loss - 1e-5:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    if best_state:
        net.load_state_dict(best_state)

    def predict(X: np.ndarray) -> np.ndarray:
        net.eval()
        with torch.no_grad():
            logits = net(torch.FloatTensor(X).to(device))
            return torch.softmax(logits, -1).cpu().numpy()

    probs_val_cal  = predict(X_val_cal)
    probs_test_raw = predict(X_test)

    probs_test_ts, T = _temperature_scale(probs_val_cal, y_val_cal, probs_test_raw)

    return {
        "val_cal_nll": _nll(probs_val_cal, y_val_cal),   # used for HP selection
        "nll_ts":      _nll(probs_test_ts,  y_test),
        "ece_ts":      _ece_mean(probs_test_ts, y_test),
        "brier_ts":    _brier(probs_test_ts, y_test),
        "nll_raw":     _nll(probs_test_raw, y_test),
        "temperature": T,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HP config sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_hp_configs(n: int, rng: np.random.RandomState) -> List[dict]:
    """
    Sample n distinct HP configurations.
    Index 0 is always the fixed config from the main experiment.
    """
    configs = [dict(FIXED_CONFIG)]
    seen = {
        str(FIXED_CONFIG["hidden_dims"]) +
        str(FIXED_CONFIG["lr"]) +
        str(FIXED_CONFIG["dropout"]) +
        str(FIXED_CONFIG["weight_decay"])
    }

    attempts = 0
    while len(configs) < n and attempts < n * 30:
        attempts += 1
        cfg = {
            "hidden_dims":  HP_SEARCH_SPACE["hidden_dims"][
                                rng.randint(len(HP_SEARCH_SPACE["hidden_dims"]))],
            "lr":           float(rng.choice(HP_SEARCH_SPACE["lr"])),
            "dropout":      float(rng.choice(HP_SEARCH_SPACE["dropout"])),
            "weight_decay": float(rng.choice(HP_SEARCH_SPACE["weight_decay"])),
            "batch_size":   256,
            "epochs":       200,
            "patience":     20,
        }
        key = (str(cfg["hidden_dims"]) + str(cfg["lr"]) +
               str(cfg["dropout"]) + str(cfg["weight_decay"]))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    logger.info(f"Sampled {len(configs)} HP configurations "
                f"(index 0 = fixed config from main experiment).")
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(out_dir: Path) -> Tuple[pd.DataFrame, set]:
    """Load existing results and return done keys for resuming."""
    ckpt = out_dir / "hp_ablation_all_configs.csv"
    if ckpt.exists():
        df = pd.read_csv(ckpt)
        done = set(zip(df["task_id"], df["config_idx"], df["seed"]))
        logger.info(f"Checkpoint: {len(done)} runs already done.")
        return df, done
    return pd.DataFrame(), set()


def save_checkpoint(rows: List[dict], out_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(
        out_dir / "hp_ablation_all_configs.csv", index=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    results_csv: str,
    output_dir: str,
    n_hp_configs: int = 15,
    seeds: List[int] = None,
) -> None:
    if seeds is None:
        seeds = [0, 1, 2]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # H100 optimisations
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True
            torch.backends.cudnn.benchmark         = True
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} — TF32 enabled.")
        else:
            logger.info("No GPU found — running on CPU.")
    except Exception:
        pass

    # Load existing results for verification and baselines
    df_existing = pd.read_csv(results_csv)
    existing_task_ids = set(df_existing["task_id"].unique())

    # Verify task IDs match
    missing = set(MAIN_EXPERIMENT_TASK_IDS) - existing_task_ids
    if missing:
        logger.warning(f"Task IDs in script but not in results_csv: {missing}")
    extra = existing_task_ids - set(MAIN_EXPERIMENT_TASK_IDS)
    if extra:
        logger.warning(f"Task IDs in results_csv but not in script: {extra}")
    logger.info(
        f"Task ID verification: {len(MAIN_EXPERIMENT_TASK_IDS)} in script, "
        f"{len(existing_task_ids)} in results_csv, "
        f"{len(set(MAIN_EXPERIMENT_TASK_IDS) & existing_task_ids)} match."
    )

    # Sample HP configs — same seed=42 on both GPUs guarantees identical configs
    rng     = np.random.RandomState(42)
    configs = sample_hp_configs(n_hp_configs, rng)

    # Log the full config table for reproducibility
    logger.info("\nHP configurations to be evaluated:")
    for i, cfg in enumerate(configs):
        tag = "FIXED" if i == 0 else f"cfg{i:02d}"
        logger.info(f"  [{tag}] hidden={cfg['hidden_dims']} "
                    f"lr={cfg['lr']:.0e} dropout={cfg['dropout']:.2f} "
                    f"wd={cfg['weight_decay']:.0e}")

    # Load checkpoint
    df_ckpt, done_keys = load_checkpoint(out)
    all_rows = df_ckpt.to_dict("records") if not df_ckpt.empty else []

    # Import dataset loading
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.datasets import load_task, split_dataset
    except ImportError:
        logger.error(
            "Cannot import src.datasets. "
            "Run from project root so src/ is on the path."
        )
        raise

    total = len(MAIN_EXPERIMENT_TASK_IDS) * len(configs) * len(seeds)
    logger.info(
        f"\nStarting ablation: {len(MAIN_EXPERIMENT_TASK_IDS)} datasets × "
        f"{len(configs)} configs × {len(seeds)} seeds = {total} runs. "
        f"Seeds on this GPU: {seeds}."
    )

    run_idx   = 0
    t_start   = time.time()

    for task_id in MAIN_EXPERIMENT_TASK_IDS:
        data = load_task(task_id)
        if data is None:
            logger.warning(f"Task {task_id}: failed to load, skipping.")
            continue

        dataset_name = data["name"]
        n_classes    = data["n_classes"]
        n_samples    = data["n_samples"]
        regime = ("small"  if n_samples <  1_000 else
                  "medium" if n_samples < 10_000 else "large")

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name} (id={task_id}, "
                    f"n={n_samples}, K={n_classes}, regime={regime})")

        for seed in seeds:
            split = split_dataset(data, seed=seed)

            for ci, cfg in enumerate(configs):
                run_idx += 1
                key = (task_id, ci, seed)

                if key in done_keys:
                    logger.info(f"  [{run_idx}/{total}] SKIP "
                                f"(already done: cfg{ci:02d}, seed={seed})")
                    continue

                tag = "FIXED" if ci == 0 else f"cfg{ci:02d}"
                logger.info(
                    f"  [{run_idx}/{total}] {tag} "
                    f"hidden={cfg['hidden_dims']} lr={cfg['lr']:.0e} "
                    f"drop={cfg['dropout']:.2f} wd={cfg['weight_decay']:.0e} "
                    f"seed={seed}"
                )

                t0 = time.time()
                try:
                    res = train_mlp(
                        split["X_train"],   split["y_train"],
                        split["X_val_es"],  split["y_val_es"],
                        split["X_val_cal"], split["y_val_cal"],
                        split["X_test"],    split["y_test"],
                        n_classes=n_classes,
                        config=cfg,
                        seed=seed,
                    )
                    elapsed = time.time() - t0

                    # Elapsed time and ETA
                    done_so_far  = run_idx - len(done_keys)
                    elapsed_total = time.time() - t_start
                    rate         = elapsed_total / max(done_so_far, 1)
                    remaining    = (total - run_idx) * rate
                    eta_min      = remaining / 60

                    logger.info(
                        f"    NLL_ts={res['nll_ts']:.4f}  "
                        f"ECE_ts={res['ece_ts']:.4f}  "
                        f"valNLL={res['val_cal_nll']:.4f}  "
                        f"T={res['temperature']:.3f}  "
                        f"({elapsed:.1f}s)  ETA: {eta_min:.0f}min"
                    )

                    row = {
                        "task_id":      task_id,
                        "dataset_name": dataset_name,
                        "n_samples":    n_samples,
                        "regime":       regime,
                        "n_classes":    n_classes,
                        "seed":         seed,
                        "config_idx":   ci,
                        "is_fixed":     (ci == 0),
                        "hidden_dims":  str(cfg["hidden_dims"]),
                        "lr":           cfg["lr"],
                        "dropout":      cfg["dropout"],
                        "weight_decay": cfg["weight_decay"],
                        **res,
                    }
                    all_rows.append(row)
                    done_keys.add(key)

                    # Save checkpoint after every run
                    save_checkpoint(all_rows, out)

                except Exception as e:
                    logger.error(f"    FAILED: {e}")

    logger.info(f"\nDone. {len(all_rows)} total rows saved to "
                f"{out / 'hp_ablation_all_configs.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MLP Hyperparameter Sensitivity Ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_csv",   required=True,
                   help="Path to results_raw.csv from main experiment")
    p.add_argument("--output_dir",    required=True,
                   help="Directory to save results")
    p.add_argument("--n_hp_configs",  type=int, default=15,
                   help="Number of HP configs to evaluate (including fixed)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="Seeds to run on this GPU")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(
        results_csv  = args.results_csv,
        output_dir   = args.output_dir,
        n_hp_configs = args.n_hp_configs,
        seeds        = args.seeds,
    )
