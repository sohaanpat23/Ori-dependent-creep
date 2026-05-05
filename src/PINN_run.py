
"""Orientation-aware LMP physics-regularized PINN for creep rupture life prediction.

This revision keeps the overall workflow close to the previous script, but makes
two targeted updates for a mixed-alloy, mixed-orientation dataset:

1. The low-capacity baseline branch is driven by the original measured stress and
   temperature inputs (stress_MPa and T_C), rather than derived stress features.
   This makes the baseline simpler and easier to defend physically.
2. Hyperparameter optimization is broadened into a genuine fresh search with
   pruning, while keeping runtime moderate.

The script preserves:
- the same CSV input path,
- the same engineered residual features,
- paper-aware balanced cross-validation folds,
- Optuna-based hyperparameter tuning,
- final out-of-fold evaluation across 10 folds,
- export of predictions and summary metrics.
"""

from __future__ import annotations

import os
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import optuna
except ImportError as exc:
    raise ImportError("Install optuna first: pip install optuna") from exc

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Runtime configuration for training and evaluation."""

    data_path: str = str(PROJECT_ROOT / "data" / "dataset_processed.csv")
    n_splits: int = 10
    tune_folds: Tuple[int, ...] = (0, 1, 2)
    n_trials: int = 28
    max_epochs_tune: int = 130
    max_epochs_final: int = 650
    patience_tune: int = 32
    patience_final: int = 90
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    oof_filename: str = "pinn_oof_preds.csv"
    fold_metrics_filename: str = "pinn_fold_metrics.csv"
    zone_metrics_filename: str = "pinn_zone_metrics.csv"
    best_params_filename: str = "pinn_best_params.txt"


CONFIG = Config()

MAT_COLS = ["Inv_T_K", "Gamma_prime_size", "L", "G", "SFE", "Misfit"]
ORI_COLS = ["cos_001", "cos_011", "MultislipPropensity_110"]
STRESS_COL = "Log_RSS_110"
BASE_COLS = ["stress_MPa", "T_C"]
TARGET_COL = "y_log10"


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CONFIG.seed)


def assign_orientation_zone(
    theta001: float,
    theta011: float,
    corner_thresh: float = 8.0,
) -> str:
    """Assign an orientation sample to a corner zone or the interior of the IPF."""

    corners = {
        "near_001": np.array([0.0, 45.0]),
        "near_011": np.array([45.0, 0.0]),
        "near_111": np.array([54.7356, 35.2644]),
    }
    point = np.array([theta001, theta011], dtype=float)
    distances = {key: np.linalg.norm(point - value) for key, value in corners.items()}
    best_zone = min(distances, key=distances.get)
    return best_zone if distances[best_zone] <= corner_thresh else "interior"


def load_and_engineer(path: str) -> pd.DataFrame:
    """Load the input CSV and create all engineered model features."""

    df = pd.read_csv(path).copy()

    df = df.rename(
        columns={
            "Gamma prime size": "Gamma_prime_size",
            "Schmid Max": "Schmid_Max",
        }
    )

    required = [
        "paper_id",
        "spec_no",
        "T_C",
        "stress_MPa",
        "t_rupture_h",
        "theta_from_001_deg",
        "theta_from_011_deg",
        "Gamma_prime_size",
        "L",
        "G",
        "SFE",
        "Misfit",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (df["t_rupture_h"] <= 0).any():
        raise ValueError("Found non-positive values in t_rupture_h.")

    df["y_log10"] = np.log10(df["t_rupture_h"].astype(float).values)
    df["T_K"] = df["T_C"].astype(float) + 273.15
    df["Inv_T_K"] = 1000.0 / df["T_K"]
    df["Log_Stress"] = np.log10(df["stress_MPa"].astype(float))
    df["cos_001"] = np.cos(np.radians(df["theta_from_001_deg"].astype(float)))
    df["cos_011"] = np.cos(np.radians(df["theta_from_011_deg"].astype(float)))

    if "Schmid1_110" in df.columns:
        df["Schmid1_110"] = df["Schmid1_110"].astype(float)
    elif "Schmid110_1" in df.columns:
        df["Schmid1_110"] = df["Schmid110_1"].astype(float)
    else:
        raise ValueError("Need either 'Schmid1_110' or 'Schmid110_1' in the dataset.")

    if "MultislipPropensity_110" in df.columns:
        df["MultislipPropensity_110"] = df["MultislipPropensity_110"].astype(float)
    elif "Schmid110_2" in df.columns:
        df["MultislipPropensity_110"] = (
            df["Schmid110_2"].astype(float) / df["Schmid1_110"].astype(float)
        )
    else:
        raise ValueError(
            "Need either 'MultislipPropensity_110' or 'Schmid110_2' in the dataset."
        )

    df["RSS_110"] = df["stress_MPa"].astype(float) * df["Schmid1_110"].astype(float)
    if (df["RSS_110"] <= 0).any():
        raise ValueError("Found non-positive values in RSS_110.")

    df["Log_RSS_110"] = np.log10(df["RSS_110"].astype(float))
    df["orientation_zone"] = df.apply(
        lambda row: assign_orientation_zone(
            row["theta_from_001_deg"], row["theta_from_011_deg"]
        ),
        axis=1,
    )

    return df


def make_paper_balanced_folds(
    df: pd.DataFrame,
    n_splits: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create paper-aware snake-balanced folds across sorted rupture life."""

    out = df.copy()
    out["cv_fold"] = -1
    rng = np.random.RandomState(seed)

    for _, subset in out.groupby("paper_id"):
        subset = subset.copy()
        subset = subset.sample(frac=1.0, random_state=rng.randint(1, 1_000_000))
        subset = subset.sort_values(TARGET_COL).reset_index()

        forward = list(range(n_splits))
        backward = list(range(n_splits - 1, -1, -1))
        snake = forward + backward
        offset = rng.randint(0, n_splits)
        snake = [((fold + offset) % n_splits) for fold in snake]

        subset["cv_fold"] = [snake[i % len(snake)] for i in range(len(subset))]
        out.loc[subset["index"].values, "cv_fold"] = subset["cv_fold"].values

    out["cv_fold"] = out["cv_fold"].astype(int)
    return out


def prepare_fold_data(
    df: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    c_lmp: float,
) -> Dict[str, Dict[str, np.ndarray | StandardScaler]]:
    """Prepare scaled train/validation arrays for a single fold."""

    train_df = df.loc[train_idx].copy()
    val_df = df.loc[val_idx].copy()

    train_df["LMP_target"] = train_df["T_K"] * (c_lmp + train_df[TARGET_COL])
    val_df["LMP_target"] = val_df["T_K"] * (c_lmp + val_df[TARGET_COL])

    mat_scaler = StandardScaler()
    ori_scaler = StandardScaler()
    stress_scaler = StandardScaler()
    base_scaler = StandardScaler()
    lmp_scaler = StandardScaler()

    xmat_train = mat_scaler.fit_transform(train_df[MAT_COLS].values)
    xori_train = ori_scaler.fit_transform(train_df[ORI_COLS].values)
    xs_train = stress_scaler.fit_transform(train_df[[STRESS_COL]].values)
    xb_train = base_scaler.fit_transform(train_df[BASE_COLS].values)
    ylmp_train = lmp_scaler.fit_transform(train_df[["LMP_target"]].values).reshape(-1)

    xmat_val = mat_scaler.transform(val_df[MAT_COLS].values)
    xori_val = ori_scaler.transform(val_df[ORI_COLS].values)
    xs_val = stress_scaler.transform(val_df[[STRESS_COL]].values)
    xb_val = base_scaler.transform(val_df[BASE_COLS].values)
    ylmp_val = lmp_scaler.transform(val_df[["LMP_target"]].values).reshape(-1)

    return {
        "train": {
            "x_mat": xmat_train.astype(np.float32),
            "x_ori": xori_train.astype(np.float32),
            "x_stress": xs_train.astype(np.float32),
            "x_base": xb_train.astype(np.float32),
            "y_lmp": ylmp_train.astype(np.float32),
            "y_log": train_df[TARGET_COL].values.astype(np.float32),
            "T_K": train_df["T_K"].values.astype(np.float32),
            "zone": train_df["orientation_zone"].values,
            "index": train_df.index.values,
        },
        "val": {
            "x_mat": xmat_val.astype(np.float32),
            "x_ori": xori_val.astype(np.float32),
            "x_stress": xs_val.astype(np.float32),
            "x_base": xb_val.astype(np.float32),
            "y_lmp": ylmp_val.astype(np.float32),
            "y_log": val_df[TARGET_COL].values.astype(np.float32),
            "T_K": val_df["T_K"].values.astype(np.float32),
            "zone": val_df["orientation_zone"].values,
            "index": val_df.index.values,
        },
        "scalers": {
            "mat": mat_scaler,
            "ori": ori_scaler,
            "stress": stress_scaler,
            "base": base_scaler,
            "lmp": lmp_scaler,
        },
    }


def make_loader(
    part: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a PyTorch dataloader from a prepared fold partition."""

    dataset = TensorDataset(
        torch.from_numpy(part["x_mat"]),
        torch.from_numpy(part["x_ori"]),
        torch.from_numpy(part["x_stress"]),
        torch.from_numpy(part["x_base"]),
        torch.from_numpy(part["y_lmp"]),
        torch.from_numpy(part["y_log"]),
        torch.from_numpy(part["T_K"]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class MLP(nn.Module):
    """Simple multilayer perceptron used for branch and head networks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        out_dim: int,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        act = nn.SiLU if activation.lower() == "silu" else nn.ReLU

        layers: List[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(prev, hidden_dim), act(), nn.Dropout(dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RawStressTempBaseBranch(nn.Module):
    """Low-capacity baseline in normalized LMP space using raw stress and temperature.

    Inputs are the standardized original measured variables [stress_MPa, T_C].
    The branch remains interpretable and low-capacity:
    - quadratic polynomial in stress and temperature,
    - one interaction term,
    - two ReLU hinge terms in stress.

    This lets the baseline express broad thermo-stress regimes while keeping the
    monotonic prior localized to the baseline branch rather than the full model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.poly = nn.Parameter(torch.zeros(6))
        self.relu_weights = nn.Parameter(torch.zeros(2))
        self.register_buffer("stress_knots", torch.tensor([[-0.75, 0.75]], dtype=torch.float32))
        nn.init.normal_(self.poly, mean=0.0, std=0.05)
        nn.init.normal_(self.relu_weights, mean=0.0, std=0.05)

    def forward(self, x_base: torch.Tensor) -> torch.Tensor:
        stress = x_base[:, [0]]
        temp = x_base[:, [1]]

        poly_term = (
            self.poly[0]
            + self.poly[1] * stress
            + self.poly[2] * stress.pow(2)
            + self.poly[3] * temp
            + self.poly[4] * temp.pow(2)
            + self.poly[5] * stress * temp
        )

        hinge = F.relu(stress - self.stress_knots)
        relu_term = (hinge * self.relu_weights.view(1, -1)).sum(dim=1, keepdim=True)
        return (poly_term + relu_term).squeeze(1)


class OrientationAwareLMPPINN(nn.Module):
    """Orientation-aware LMP predictor with raw stress-temperature baseline branch.

    Final normalized LMP prediction:
        full = raw_stress_temp_base + material/orientation residual corrections
    """

    def __init__(self, n_mat: int, n_ori: int, params: Dict[str, float | int]) -> None:
        super().__init__()

        self.ori_scale = params["ori_scale"]
        self.mat_scale = params["mat_scale"]

        self.base_branch = RawStressTempBaseBranch()

        self.mat_net = MLP(
            in_dim=n_mat,
            hidden_dim=int(params["mat_width"]),
            depth=int(params["mat_depth"]),
            dropout=float(params["dropout"]),
            out_dim=int(params["mat_width"]),
            activation="silu",
        )
        self.ori_net = MLP(
            in_dim=n_ori,
            hidden_dim=int(params["ori_width"]),
            depth=int(params["ori_depth"]),
            dropout=float(params["dropout"]),
            out_dim=int(params["ori_width"]),
            activation="silu",
        )

        self.mat_head = MLP(
            in_dim=int(params["mat_width"]) + 1,
            hidden_dim=int(params["fusion_width"]),
            depth=int(params["fusion_depth"]),
            dropout=float(params["dropout"]),
            out_dim=1,
            activation="silu",
        )
        self.ori_head = MLP(
            in_dim=int(params["ori_width"]) + 1,
            hidden_dim=int(params["fusion_width"]),
            depth=int(params["fusion_depth"]),
            dropout=float(params["dropout"]),
            out_dim=1,
            activation="silu",
        )
        self.mix_head = MLP(
            in_dim=int(params["mat_width"]) + int(params["ori_width"]) + 1,
            hidden_dim=int(params["fusion_width"]),
            depth=max(1, int(params["fusion_depth"]) - 1),
            dropout=float(params["dropout"]),
            out_dim=1,
            activation="silu",
        )

    def forward(
        self,
        x_mat: torch.Tensor,
        x_ori: torch.Tensor,
        x_stress: torch.Tensor,
        x_base: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_pred = self.base_branch(x_base)

        mat_features = self.mat_net(x_mat)
        ori_features = self.ori_net(x_ori)

        mat_corr = self.mat_scale * self.mat_head(torch.cat([mat_features, x_stress], dim=1))
        ori_corr = self.ori_scale * self.ori_head(torch.cat([ori_features, x_stress], dim=1))
        mix_corr = self.mix_head(torch.cat([mat_features, ori_features, x_stress], dim=1))

        residual_pred = (mat_corr + ori_corr + mix_corr).squeeze(1)
        full_pred = base_pred + residual_pred

        if return_components:
            return full_pred, base_pred, residual_pred
        return full_pred


def inverse_lmp_to_loglife(
    lmp_pred_norm: np.ndarray,
    lmp_scaler: StandardScaler,
    t_k: np.ndarray,
    c_lmp: float,
) -> np.ndarray:
    """Convert normalized predicted LMP back to log10 rupture life."""

    lmp_pred = lmp_scaler.inverse_transform(np.asarray(lmp_pred_norm).reshape(-1, 1)).reshape(-1)
    return lmp_pred / t_k - c_lmp


def zone_mae(y_true: np.ndarray, y_pred: np.ndarray, zones: np.ndarray) -> Dict[str, float]:
    """Compute MAE for each orientation zone."""

    zone_df = pd.DataFrame({"zone": zones, "yt": y_true, "yp": y_pred})
    return zone_df.groupby("zone").apply(
        lambda x: mean_absolute_error(x["yt"], x["yp"])
    ).to_dict()


def eval_model(
    model: nn.Module,
    part: Dict[str, np.ndarray],
    lmp_scaler: StandardScaler,
    c_lmp: float,
) -> Dict[str, float | np.ndarray | Dict[str, float]]:
    """Evaluate a trained model on one fold partition."""

    model.eval()
    with torch.no_grad():
        x_mat = torch.from_numpy(part["x_mat"]).to(CONFIG.device)
        x_ori = torch.from_numpy(part["x_ori"]).to(CONFIG.device)
        x_stress = torch.from_numpy(part["x_stress"]).to(CONFIG.device)
        x_base = torch.from_numpy(part["x_base"]).to(CONFIG.device)
        pred_lmp_norm = model(x_mat, x_ori, x_stress, x_base).cpu().numpy()

    y_pred = inverse_lmp_to_loglife(pred_lmp_norm, lmp_scaler, part["T_K"], c_lmp)
    y_true = part["y_log"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    z_mae = zone_mae(y_true, y_pred, part["zone"])
    zone_penalty = np.std(list(z_mae.values())) if len(z_mae) > 1 else 0.0

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "zone_mae": z_mae,
        "zone_penalty": zone_penalty,
    }


def train_one_fold(
    fold_data: Dict[str, Dict[str, np.ndarray | StandardScaler]],
    params: Dict[str, float | int],
    max_epochs: int = 300,
    patience: int = 50,
) -> Tuple[nn.Module, Dict[str, float | np.ndarray | Dict[str, float]]]:
    """Train the model for one fold with a baseline-only stress monotonic prior."""

    model = OrientationAwareLMPPINN(
        n_mat=len(MAT_COLS),
        n_ori=len(ORI_COLS),
        params=params,
    ).to(CONFIG.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    train_loader = make_loader(
        fold_data["train"],
        batch_size=int(params["batch_size"]),
        shuffle=True,
    )

    best_state = None
    best_score = np.inf
    bad_epochs = 0

    for _ in range(max_epochs):
        model.train()

        for xb_mat, xb_ori, xb_stress, xb_base, yb_lmp, _, _ in train_loader:
            xb_mat = xb_mat.to(CONFIG.device).requires_grad_(True)
            xb_ori = xb_ori.to(CONFIG.device).requires_grad_(True)
            xb_stress = xb_stress.to(CONFIG.device)
            xb_base = xb_base.to(CONFIG.device).requires_grad_(True)
            yb_lmp = yb_lmp.to(CONFIG.device)

            pred_lmp, base_lmp, residual_lmp = model(
                xb_mat, xb_ori, xb_stress, xb_base, return_components=True
            )
            data_loss = F.huber_loss(
                pred_lmp,
                yb_lmp,
                delta=float(params["huber_delta"]),
                reduction="mean",
            )

            dBase_dStress = torch.autograd.grad(
                base_lmp.sum(), xb_base, create_graph=True, retain_graph=True
            )[0][:, [0]]
            stress_mono_loss = torch.mean(F.relu(dBase_dStress) ** 2)

            dL_dOri = torch.autograd.grad(
                pred_lmp.sum(), xb_ori, create_graph=True, retain_graph=True
            )[0]
            ori_smooth_loss = torch.mean(dL_dOri ** 2)

            dResidual_dMat = torch.autograd.grad(
                residual_lmp.sum(), xb_mat, create_graph=True, retain_graph=True
            )[0]
            mat_reg_loss = torch.mean(dResidual_dMat ** 2)

            loss = (
                data_loss
                + float(params["lambda_stress_mono"]) * stress_mono_loss
                + float(params["lambda_ori_smooth"]) * ori_smooth_loss
                + float(params["lambda_mat_reg"]) * mat_reg_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_out = eval_model(
            model,
            fold_data["val"],
            fold_data["scalers"]["lmp"],
            float(params["C_lmp"]),
        )
        val_score = val_out["mae"] + float(params["zone_balance_weight"]) * val_out["zone_penalty"]

        if val_score < best_score:
            best_score = val_score
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    final_val = eval_model(
        model,
        fold_data["val"],
        fold_data["scalers"]["lmp"],
        float(params["C_lmp"]),
    )
    return model, final_val


def sample_params(trial: optuna.Trial) -> Dict[str, float | int]:
    """Broader search space for a fresh Optuna sweep with moderate runtime."""

    return {
        "mat_width": trial.suggest_categorical("mat_width", [64, 96, 128, 160]),
        "mat_depth": trial.suggest_categorical("mat_depth", [1, 2, 3]),
        "ori_width": trial.suggest_categorical("ori_width", [32, 48, 64, 96]),
        "ori_depth": trial.suggest_categorical("ori_depth", [1, 2, 3]),
        "fusion_width": trial.suggest_categorical("fusion_width", [32, 64, 96, 128]),
        "fusion_depth": trial.suggest_categorical("fusion_depth", [1, 2, 3]),
        "dropout": trial.suggest_float("dropout", 0.00, 0.15),
        "lr": trial.suggest_float("lr", 5e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "C_lmp": trial.suggest_float("C_lmp", 18.8, 21.2),
        "ori_scale": trial.suggest_float("ori_scale", 1.0, 4.5),
        "mat_scale": trial.suggest_float("mat_scale", 0.8, 3.2),
        "lambda_stress_mono": trial.suggest_float(
            "lambda_stress_mono", 1e-3, 8e-2, log=True
        ),
        "lambda_ori_smooth": trial.suggest_float(
            "lambda_ori_smooth", 5e-5, 3e-3, log=True
        ),
        "lambda_mat_reg": trial.suggest_float(
            "lambda_mat_reg", 1e-4, 5e-3, log=True
        ),
        "zone_balance_weight": trial.suggest_float("zone_balance_weight", 0.02, 0.16),
        "huber_delta": trial.suggest_float("huber_delta", 0.6, 1.8),
    }


def tune_hyperparameters(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float | int], optuna.Study]:
    """Run a fresh Optuna search with fold-wise reporting and pruning."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        fold_scores = []

        for fold_step, fold_id in enumerate(CONFIG.tune_folds):
            train_idx = df.index[df["cv_fold"] != fold_id]
            val_idx = df.index[df["cv_fold"] == fold_id]
            fold_data = prepare_fold_data(df, train_idx, val_idx, float(params["C_lmp"]))

            _, val_out = train_one_fold(
                fold_data,
                params=params,
                max_epochs=CONFIG.max_epochs_tune,
                patience=CONFIG.patience_tune,
            )
            score = val_out["mae"] + float(params["zone_balance_weight"]) * val_out["zone_penalty"]
            fold_scores.append(score)

            running_mean = float(np.mean(fold_scores))
            trial.report(running_mean, step=fold_step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(
        seed=CONFIG.seed,
        n_startup_trials=10,
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=8,
            n_warmup_steps=1,
            interval_steps=1,
        ),
    )

    study.optimize(objective, n_trials=CONFIG.n_trials, show_progress_bar=True)

    print("\nOptuna best tuning score:", study.best_value)
    print("Optuna best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params, study


def final_10fold_run(
    df: pd.DataFrame,
    best_params: Dict[str, float | int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Run final out-of-fold evaluation across all configured folds."""

    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows = []

    for fold_id in range(CONFIG.n_splits):
        print(f"\n========== FINAL FOLD {fold_id} ==========")
        train_idx = df.index[df["cv_fold"] != fold_id]
        val_idx = df.index[df["cv_fold"] == fold_id]

        fold_data = prepare_fold_data(df, train_idx, val_idx, float(best_params["C_lmp"]))
        _, val_out = train_one_fold(
            fold_data,
            params=best_params,
            max_epochs=CONFIG.max_epochs_final,
            patience=CONFIG.patience_final,
        )

        oof[val_idx] = val_out["y_pred"]
        actual_h = 10 ** val_out["y_true"]
        pred_h = 10 ** val_out["y_pred"]
        ape = np.abs(pred_h - actual_h) / actual_h * 100.0

        fold_rows.append(
            {
                "fold": fold_id,
                "n_val": len(val_idx),
                "MAE_log10h": val_out["mae"],
                "RMSE_log10h": val_out["rmse"],
                "R2_log10h": val_out["r2"],
                "MedianAPE_%": np.median(ape),
                "MeanAPE_%": np.mean(ape),
                "ZonePenalty": val_out["zone_penalty"],
            }
        )

    oof_df = df.copy()
    oof_df["Pred_Log10_t_rupture_h"] = oof
    oof_df["Actual_Log10_t_rupture_h"] = oof_df[TARGET_COL]
    oof_df["Pred_t_rupture_h"] = 10 ** oof_df["Pred_Log10_t_rupture_h"]
    oof_df["APE_%"] = (
        np.abs(oof_df["Pred_t_rupture_h"] - oof_df["t_rupture_h"]) / oof_df["t_rupture_h"] * 100.0
    )

    folds_df = pd.DataFrame(fold_rows)
    overall = {
        "MAE_log10h": mean_absolute_error(
            oof_df["Actual_Log10_t_rupture_h"], oof_df["Pred_Log10_t_rupture_h"]
        ),
        "RMSE_log10h": np.sqrt(
            mean_squared_error(
                oof_df["Actual_Log10_t_rupture_h"], oof_df["Pred_Log10_t_rupture_h"]
            )
        ),
        "R2_log10h": r2_score(
            oof_df["Actual_Log10_t_rupture_h"], oof_df["Pred_Log10_t_rupture_h"]
        ),
        "MedianAPE_%": np.median(oof_df["APE_%"]),
        "MeanAPE_%": np.mean(oof_df["APE_%"]),
    }

    zone_summary = (
        oof_df.groupby("orientation_zone")
        .apply(
            lambda x: pd.Series(
                {
                    "n": len(x),
                    "MAE_log10h": mean_absolute_error(
                        x["Actual_Log10_t_rupture_h"], x["Pred_Log10_t_rupture_h"]
                    ),
                    "RMSE_log10h": np.sqrt(
                        mean_squared_error(
                            x["Actual_Log10_t_rupture_h"], x["Pred_Log10_t_rupture_h"]
                        )
                    ),
                    "MedianAPE_%": np.median(x["APE_%"]),
                    "MeanAPE_%": np.mean(x["APE_%"]),
                }
            )
        )
        .reset_index()
    )

    return oof_df, folds_df, overall, zone_summary


def save_outputs(
    oof_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    zone_summary: pd.DataFrame,
    best_params: Dict[str, float | int],
    output_dir: str | Path = ".",
) -> None:
    """Write model outputs and configuration summaries to disk."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df.to_csv(output_dir / CONFIG.oof_filename, index=False)
    folds_df.to_csv(output_dir / CONFIG.fold_metrics_filename, index=False)
    zone_summary.to_csv(output_dir / CONFIG.zone_metrics_filename, index=False)

    with open(output_dir / CONFIG.best_params_filename, "w", encoding="utf-8") as file:
        for key, value in best_params.items():
            file.write(f"{key}: {value}\n")

    print("\nSaved:")
    print(f"  {output_dir / CONFIG.oof_filename}")
    print(f"  {output_dir / CONFIG.fold_metrics_filename}")
    print(f"  {output_dir / CONFIG.zone_metrics_filename}")
    print(f"  {output_dir / CONFIG.best_params_filename}")



# ============================================================================
# Optuna-tuned 20-point most-interpolative holdout extension
# Model architecture, losses, training loop, Optuna search, and fold logic above are unchanged.
# ============================================================================

BEST_PARAMS: Dict[str, float | int] = {
    "mat_width": 64,
    "mat_depth": 1,
    "ori_width": 48,
    "ori_depth": 2,
    "fusion_width": 128,
    "fusion_depth": 1,
    "dropout": 0.05404192321079343,
    "lr": 0.0035139241470565053,
    "weight_decay": 6.342076009997429e-08,
    "batch_size": 8,
    "C_lmp": 20.692797298886845,
    "ori_scale": 1.6670632421301352,
    "mat_scale": 1.2920566820905666,
    "lambda_stress_mono": 0.011392397326613552,
    "lambda_ori_smooth": 0.002853357889064749,
    "lambda_mat_reg": 0.00025431550597345635,
    "zone_balance_weight": 0.13748108178968857,
    "huber_delta": 1.3574112507233114,
}


def sanitize_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with all model-required numeric fields finite."""
    needed = sorted(set(MAT_COLS + ORI_COLS + [STRESS_COL] + BASE_COLS + [TARGET_COL, "T_K", "t_rupture_h"]))
    out = df.copy()
    before = len(out)
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    mask = np.isfinite(out[needed].to_numpy(dtype=float)).all(axis=1)
    out = out.loc[mask].copy()
    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped} rows with non-finite required model features.")
    return out.reset_index(drop=True)


def condition_key_columns() -> List[str]:
    return ["paper_id", "T_C", "stress_MPa", "Gamma_prime_size", "L", "G", "SFE", "Misfit"]


def orientation_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    return float(np.sqrt((float(row_a["theta_from_001_deg"]) - float(row_b["theta_from_001_deg"])) ** 2 + (float(row_a["theta_from_011_deg"]) - float(row_b["theta_from_011_deg"])) ** 2))


def build_interpolation_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Build input-space-only candidates for the 20-point validation holdout.

    Eligibility is based only on INPUT features. A row is eligible only if it is
    an interior orientation point inside an EXACT same-condition group. The
    exact same-condition group fixes these non-orientation descriptors:
    paper_id, T_C, stress_MPa, Gamma_prime_size, L, G, SFE, and Misfit.

    This means most validation rows are not merely similar to training rows;
    they have the same alloy/source, same temperature, same stress, same gamma'
    size, and same material descriptors, while orientation differs.
    """

    rows = []
    cond_cols = condition_key_columns()

    for group_id, (_, group) in enumerate(df.groupby(cond_cols, dropna=False, sort=False)):
        if len(group) < 3:
            continue

        g = group.copy()
        spread_001 = float(g["theta_from_001_deg"].max() - g["theta_from_001_deg"].min())
        spread_011 = float(g["theta_from_011_deg"].max() - g["theta_from_011_deg"].min())
        sort_col = "theta_from_001_deg" if spread_001 >= spread_011 else "theta_from_011_deg"

        g = g.sort_values([sort_col, "theta_from_011_deg", "theta_from_001_deg"]).copy()
        idxs = list(g.index)

        # Exclude end-points. Holding out an interior point keeps neighbouring
        # orientation states in the training/CV pool under the same exact conditions.
        for pos in range(1, len(idxs) - 1):
            idx = idxs[pos]
            row = g.loc[idx]
            left = g.loc[idxs[pos - 1]]
            right = g.loc[idxs[pos + 1]]

            d_left = orientation_distance(row, left)
            d_right = orientation_distance(row, right)
            d_balance = abs(d_left - d_right)
            local_span = d_left + d_right
            centrality = min(pos, len(idxs) - 1 - pos)
            near_001_flag = bool(float(row["theta_from_001_deg"]) <= 35.0)

            # This score uses input-space support only. It does NOT use target
            # rupture life, predictions, errors, or any model-performance signal.
            interpolation_score = (
                10.0 * np.log1p(len(g))
                + 3.0 / (1.0 + local_span)
                + 2.0 / (1.0 + d_balance)
                + 0.15 * centrality
            )

            rows.append(
                {
                    "index": int(idx),
                    "exact_condition_group_id": int(group_id),
                    "paper_id": row["paper_id"],
                    "spec_no": row.get("spec_no", ""),
                    "T_C": row["T_C"],
                    "stress_MPa": row["stress_MPa"],
                    "Gamma_prime_size": row["Gamma_prime_size"],
                    "L": row["L"],
                    "G": row["G"],
                    "SFE": row["SFE"],
                    "Misfit": row["Misfit"],
                    "theta_from_001_deg": row["theta_from_001_deg"],
                    "theta_from_011_deg": row["theta_from_011_deg"],
                    "near_001_validation_flag": near_001_flag,
                    "same_condition_group_size": int(len(g)),
                    "left_orientation_distance": d_left,
                    "right_orientation_distance": d_right,
                    "local_orientation_span": local_span,
                    "orientation_balance_gap": d_balance,
                    "interpolation_score": float(interpolation_score),
                }
            )

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand

    return cand.sort_values(
        ["interpolation_score", "same_condition_group_size"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _select_with_diversity(
    candidates: pd.DataFrame,
    already_selected: List[int],
    needed: int,
    work: pd.DataFrame,
    group_counts: Dict[Tuple[object, ...], int],
    paper_counts: Dict[object, int],
    gamma_counts: Dict[object, int],
    max_per_exact_group: int,
    max_per_paper: int,
    max_per_gamma: int,
) -> List[int]:
    """Select candidates using only fixed input-space diversity limits."""

    selected_now: List[int] = []
    cond_cols = condition_key_columns()

    for _, cand in candidates.iterrows():
        idx = int(cand["index"])
        if idx in already_selected or idx in selected_now:
            continue

        row = work.loc[idx]
        exact_key = tuple(row[c] for c in cond_cols)
        paper_key = row["paper_id"]
        gamma_key = row["Gamma_prime_size"]

        if group_counts.get(exact_key, 0) >= max_per_exact_group:
            continue
        if paper_counts.get(paper_key, 0) >= max_per_paper:
            continue
        if gamma_counts.get(gamma_key, 0) >= max_per_gamma:
            continue

        selected_now.append(idx)
        group_counts[exact_key] = group_counts.get(exact_key, 0) + 1
        paper_counts[paper_key] = paper_counts.get(paper_key, 0) + 1
        gamma_counts[gamma_key] = gamma_counts.get(gamma_key, 0) + 1

        if len(selected_now) >= needed:
            break

    return selected_now


def create_physically_diverse_holdout(
    df: pd.DataFrame,
    holdout_size: int = 20,
    min_near_001: int = 8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the final 20-point internal validation holdout.

    The split is fixed BEFORE Optuna tuning and model training.

    Design:
    - 20 total validation rows.
    - At least 7-8 rows near [001]; implemented as theta_from_001_deg <= 35°.
    - Remaining rows are diverse, not forced into artificial angle bins.
    - Most/eligible rows come from exact same-condition groups:
      paper_id, T_C, stress_MPa, Gamma_prime_size, L, G, SFE, Misfit.
    - Interior orientation rows are preferred so neighbouring orientations under
      the same exact conditions remain in training.
    - Diversity is enforced across paper_id, exact-condition groups, and gamma'
      size to avoid selecting all 20 from one regime.
    - No target value, prediction result, error, or model performance is used.
    """

    rng = np.random.RandomState(seed)
    work = df.copy().reset_index(drop=True)
    candidates = build_interpolation_candidates(work)

    if candidates.empty:
        raise ValueError(
            "No exact-condition interpolation candidates found. Need groups with "
            "at least 3 rows sharing paper_id, T_C, stress_MPa, Gamma_prime_size, "
            "L, G, SFE, and Misfit but differing in orientation."
        )

    near_candidates = candidates[candidates["near_001_validation_flag"] == True].copy()
    all_candidates = candidates.copy()

    selected: List[int] = []
    group_counts: Dict[Tuple[object, ...], int] = {}
    paper_counts: Dict[object, int] = {}
    gamma_counts: Dict[object, int] = {}

    # Passes become progressively less restrictive only if the dataset cannot
    # satisfy strict diversity limits. This is deterministic and input-based.
    diversity_passes = [
        (1, 4, 6),
        (2, 5, 8),
        (3, 7, 10),
        (999, 999, 999),
    ]

    # Stage 1: reserve 7-8 near-[001] samples, but do not make the full holdout
    # near-[001]. This matches the project focus without over-removing [001] data.
    near_target = min(min_near_001, len(near_candidates), holdout_size)
    for max_group, max_paper, max_gamma in diversity_passes:
        if len(selected) >= near_target:
            break
        chosen = _select_with_diversity(
            near_candidates,
            selected,
            near_target - len(selected),
            work,
            group_counts,
            paper_counts,
            gamma_counts,
            max_per_exact_group=max_group,
            max_per_paper=max_paper,
            max_per_gamma=max_gamma,
        )
        selected.extend(chosen)

    # Stage 2: fill the remaining points from all exact-condition candidates,
    # prioritising supported/interior points and diversity, not performance.
    for max_group, max_paper, max_gamma in diversity_passes:
        if len(selected) >= holdout_size:
            break
        chosen = _select_with_diversity(
            all_candidates,
            selected,
            holdout_size - len(selected),
            work,
            group_counts,
            paper_counts,
            gamma_counts,
            max_per_exact_group=max_group,
            max_per_paper=max_paper,
            max_per_gamma=max_gamma,
        )
        selected.extend(chosen)

    # Final fallback: still exact-condition candidates only; randomization is
    # seeded and independent of target/prediction performance.
    if len(selected) < holdout_size:
        remaining = [int(i) for i in all_candidates["index"].tolist() if int(i) not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: holdout_size - len(selected)])

    selected = selected[:holdout_size]

    if len(selected) < holdout_size:
        raise ValueError(f"Only found {len(selected)} eligible exact-condition holdout candidates, requested {holdout_size}.")

    holdout_df = work.loc[selected].copy().reset_index(drop=True)
    train_cv_df = work.drop(index=selected).copy().reset_index(drop=True)

    split_audit = work.copy()
    split_audit["split"] = "train_cv"
    split_audit.loc[selected, "split"] = "internal_validation_20"

    cand_lookup = candidates.set_index("index")

    # Pandas 3.x is stricter about assigning boolean/string values into
    # columns initialized with np.nan (float dtype). Initialize each audit
    # column with the correct dtype to avoid LossySetitemError. This changes
    # only bookkeeping/output, not the split, model, tuning, or training logic.
    object_audit_cols = ["exact_condition_group_id"]
    bool_audit_cols = ["near_001_validation_flag"]
    numeric_audit_cols = [
        "same_condition_group_size",
        "left_orientation_distance",
        "right_orientation_distance",
        "local_orientation_span",
        "orientation_balance_gap",
        "interpolation_score",
    ]

    for col in object_audit_cols:
        split_audit[col] = pd.Series([None] * len(split_audit), dtype="object")
    for col in bool_audit_cols:
        split_audit[col] = pd.Series([False] * len(split_audit), dtype="bool")
    for col in numeric_audit_cols:
        split_audit[col] = np.nan

    for idx in selected:
        if idx in cand_lookup.index:
            for col in object_audit_cols + bool_audit_cols + numeric_audit_cols:
                split_audit.loc[idx, col] = cand_lookup.loc[idx, col]
    near_count = int((holdout_df["theta_from_001_deg"].astype(float) <= 35.0).sum())

    print("\n===== 20-POINT PHYSICALLY DIVERSE HOLDOUT SUMMARY =====")
    print("Train/CV rows:", len(train_cv_df))
    print("Internal holdout rows:", len(holdout_df))
    print(f"Near-[001] rows in holdout, theta_from_001_deg <= 35: {near_count}/{len(holdout_df)}")
    print("Selection rule: exact non-orientation conditions + near-[001] emphasis + diversity; no target/error/performance used.")

    print("\nHoldout counts by paper_id:")
    print(holdout_df["paper_id"].value_counts())
    print("\nHoldout Gamma_prime_size counts:")
    print(holdout_df["Gamma_prime_size"].value_counts().sort_index())
    print("\nHoldout T_C counts:")
    print(holdout_df["T_C"].value_counts().sort_index())

    display_cols = [
        "paper_id",
        "spec_no",
        "T_C",
        "stress_MPa",
        "Gamma_prime_size",
        "theta_from_001_deg",
        "theta_from_011_deg",
        "near_001_validation_flag",
        "same_condition_group_size",
        "left_orientation_distance",
        "right_orientation_distance",
        "interpolation_score",
    ]
    print("\nSelected holdout rows:")
    print(split_audit.loc[selected, display_cols].sort_values(["near_001_validation_flag", "interpolation_score"], ascending=[False, False]).to_string(index=False))

    if near_count < min_near_001:
        print(
            f"\nWARNING: Requested at least {min_near_001} near-[001] holdout rows, "
            f"but only {near_count} were available under the exact-condition eligibility rule."
        )

    return train_cv_df, holdout_df, split_audit.reset_index(drop=True)


# Backward-compatible alias used by main().
def create_most_interpolative_holdout(df: pd.DataFrame, holdout_size: int = 20, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return create_physically_diverse_holdout(
        df=df,
        holdout_size=holdout_size,
        min_near_001=8,
        seed=seed,
    )


def prepare_external_data(df: pd.DataFrame, scalers: Dict[str, StandardScaler], c_lmp: float) -> Dict[str, np.ndarray]:
    part_df = df.copy()
    part_df["LMP_target"] = part_df["T_K"] * (c_lmp + part_df[TARGET_COL])
    x_mat = scalers["mat"].transform(part_df[MAT_COLS].values)
    x_ori = scalers["ori"].transform(part_df[ORI_COLS].values)
    x_stress = scalers["stress"].transform(part_df[[STRESS_COL]].values)
    x_base = scalers["base"].transform(part_df[BASE_COLS].values)
    y_lmp = scalers["lmp"].transform(part_df[["LMP_target"]].values).reshape(-1)
    x_mat = np.nan_to_num(x_mat, nan=0.0, posinf=1e6, neginf=-1e6)
    x_ori = np.nan_to_num(x_ori, nan=0.0, posinf=1e6, neginf=-1e6)
    x_stress = np.nan_to_num(x_stress, nan=0.0, posinf=1e6, neginf=-1e6)
    x_base = np.nan_to_num(x_base, nan=0.0, posinf=1e6, neginf=-1e6)
    y_lmp = np.nan_to_num(y_lmp, nan=0.0, posinf=1e6, neginf=-1e6)
    return {"x_mat": x_mat.astype(np.float32), "x_ori": x_ori.astype(np.float32), "x_stress": x_stress.astype(np.float32), "x_base": x_base.astype(np.float32), "y_lmp": y_lmp.astype(np.float32), "y_log": part_df[TARGET_COL].values.astype(np.float32), "T_K": part_df["T_K"].values.astype(np.float32), "zone": part_df["orientation_zone"].values, "index": part_df.index.values}


def evaluate_internal_holdout(holdout_df: pd.DataFrame, fold_artifacts: List[Dict[str, object]], best_params: Dict[str, float | int]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if holdout_df.empty:
        raise ValueError("Internal validation holdout is empty.")
    all_preds = []
    c_lmp = float(best_params["C_lmp"])
    for artifact in fold_artifacts:
        model = artifact["model"]
        scalers = artifact["scalers"]
        part = prepare_external_data(holdout_df, scalers, c_lmp)
        model.eval()
        with torch.no_grad():
            x_mat = torch.from_numpy(part["x_mat"]).to(CONFIG.device)
            x_ori = torch.from_numpy(part["x_ori"]).to(CONFIG.device)
            x_stress = torch.from_numpy(part["x_stress"]).to(CONFIG.device)
            x_base = torch.from_numpy(part["x_base"]).to(CONFIG.device)
            pred_lmp_norm = model(x_mat, x_ori, x_stress, x_base).cpu().numpy()
        y_pred = inverse_lmp_to_loglife(pred_lmp_norm, scalers["lmp"], part["T_K"], c_lmp)
        y_pred = np.nan_to_num(y_pred, nan=float(np.nanmedian(part["y_log"])), posinf=10.0, neginf=-10.0)
        all_preds.append(y_pred)
    pred_matrix = np.vstack(all_preds)
    result_df = holdout_df.copy()
    result_df["Actual_Log10_t_rupture_h"] = result_df[TARGET_COL]
    result_df["Pred_Log10_t_rupture_h"] = pred_matrix.mean(axis=0)
    result_df["Pred_Log10_Ensemble_STD"] = pred_matrix.std(axis=0)
    result_df["Pred_t_rupture_h"] = 10 ** result_df["Pred_Log10_t_rupture_h"]
    result_df["APE_%"] = np.abs(result_df["Pred_t_rupture_h"] - result_df["t_rupture_h"]) / result_df["t_rupture_h"] * 100.0
    metrics = {"n_holdout": int(len(result_df)), "MAE_log10h": float(mean_absolute_error(result_df["Actual_Log10_t_rupture_h"], result_df["Pred_Log10_t_rupture_h"])), "RMSE_log10h": float(np.sqrt(mean_squared_error(result_df["Actual_Log10_t_rupture_h"], result_df["Pred_Log10_t_rupture_h"]))), "R2_log10h": float(r2_score(result_df["Actual_Log10_t_rupture_h"], result_df["Pred_Log10_t_rupture_h"])), "MedianAPE_%": float(np.median(result_df["APE_%"])), "MeanAPE_%": float(np.mean(result_df["APE_%"])), "MeanEnsembleSTD_log10h": float(np.mean(result_df["Pred_Log10_Ensemble_STD"]))}
    return result_df, pd.DataFrame([metrics]), metrics


def final_10fold_run(df: pd.DataFrame, best_params: Dict[str, float | int]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], pd.DataFrame, List[Dict[str, object]]]:
    oof = np.full(len(df), np.nan, dtype=float)
    fold_rows = []
    fold_artifacts: List[Dict[str, object]] = []
    for fold_id in range(CONFIG.n_splits):
        print(f"\n========== FINAL FOLD {fold_id} ==========")
        train_idx = df.index[df["cv_fold"] != fold_id]
        val_idx = df.index[df["cv_fold"] == fold_id]
        fold_data = prepare_fold_data(df, train_idx, val_idx, float(best_params["C_lmp"]))
        model, val_out = train_one_fold(fold_data, params=best_params, max_epochs=CONFIG.max_epochs_final, patience=CONFIG.patience_final)
        fold_artifacts.append({"fold": fold_id, "model": model, "scalers": fold_data["scalers"]})
        oof[val_idx] = val_out["y_pred"]
        actual_h = 10 ** val_out["y_true"]
        pred_h = 10 ** val_out["y_pred"]
        ape = np.abs(pred_h - actual_h) / actual_h * 100.0
        fold_rows.append({"fold": fold_id, "n_val": len(val_idx), "MAE_log10h": val_out["mae"], "RMSE_log10h": val_out["rmse"], "R2_log10h": val_out["r2"], "MedianAPE_%": np.median(ape), "MeanAPE_%": np.mean(ape), "ZonePenalty": val_out["zone_penalty"]})
    oof_df = df.copy()
    oof_df["Pred_Log10_t_rupture_h"] = oof
    oof_df["Actual_Log10_t_rupture_h"] = oof_df[TARGET_COL]
    oof_df["Pred_t_rupture_h"] = 10 ** oof_df["Pred_Log10_t_rupture_h"]
    oof_df["APE_%"] = np.abs(oof_df["Pred_t_rupture_h"] - oof_df["t_rupture_h"]) / oof_df["t_rupture_h"] * 100.0
    folds_df = pd.DataFrame(fold_rows)
    overall = {"MAE_log10h": mean_absolute_error(oof_df["Actual_Log10_t_rupture_h"], oof_df["Pred_Log10_t_rupture_h"]), "RMSE_log10h": np.sqrt(mean_squared_error(oof_df["Actual_Log10_t_rupture_h"], oof_df["Pred_Log10_t_rupture_h"])), "R2_log10h": r2_score(oof_df["Actual_Log10_t_rupture_h"], oof_df["Pred_Log10_t_rupture_h"]), "MedianAPE_%": np.median(oof_df["APE_%"]), "MeanAPE_%": np.mean(oof_df["APE_%"])}
    zone_summary = oof_df.groupby("orientation_zone").apply(lambda x: pd.Series({"n": len(x), "MAE_log10h": mean_absolute_error(x["Actual_Log10_t_rupture_h"], x["Pred_Log10_t_rupture_h"]), "RMSE_log10h": np.sqrt(mean_squared_error(x["Actual_Log10_t_rupture_h"], x["Pred_Log10_t_rupture_h"])), "MedianAPE_%": np.median(x["APE_%"]), "MeanAPE_%": np.mean(x["APE_%"])})).reset_index()
    return oof_df, folds_df, overall, zone_summary, fold_artifacts


def save_outputs(oof_df: pd.DataFrame, folds_df: pd.DataFrame, zone_summary: pd.DataFrame, best_params: Dict[str, float | int], internal_val_df: pd.DataFrame | None = None, internal_metrics_df: pd.DataFrame | None = None, split_audit_df: pd.DataFrame | None = None, output_dir: str | Path = ".") -> None:
    """Save all outputs into a fresh run folder. If a file is locked, use an alternate run folder."""
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        output_dir = output_dir.with_name(output_dir.name + "_new")
        output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv_safely(df_to_write: pd.DataFrame, filename: str) -> Path:
        path = output_dir / filename
        try:
            df_to_write.to_csv(path, index=False)
            return path
        except PermissionError:
            fallback_dir = output_dir.with_name(output_dir.name + "_permission_safe")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / filename
            df_to_write.to_csv(fallback_path, index=False)
            return fallback_path

    saved_paths = []
    saved_paths.append(write_csv_safely(oof_df, CONFIG.oof_filename))
    saved_paths.append(write_csv_safely(folds_df, CONFIG.fold_metrics_filename))
    saved_paths.append(write_csv_safely(zone_summary, CONFIG.zone_metrics_filename))
    if internal_val_df is not None:
        saved_paths.append(write_csv_safely(internal_val_df, "pinn_internal_validation_20_unseen.csv"))
    if internal_metrics_df is not None:
        saved_paths.append(write_csv_safely(internal_metrics_df, "pinn_internal_validation_metrics.csv"))
    if split_audit_df is not None:
        saved_paths.append(write_csv_safely(split_audit_df, "pinn_split_assignments.csv"))

    params_path = output_dir / CONFIG.best_params_filename
    try:
        with open(params_path, "w", encoding="utf-8") as file:
            for key, value in best_params.items():
                file.write(f"{key}: {value}\n")
    except PermissionError:
        fallback_dir = output_dir.with_name(output_dir.name + "_permission_safe")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        params_path = fallback_dir / CONFIG.best_params_filename
        with open(params_path, "w", encoding="utf-8") as file:
            for key, value in best_params.items():
                file.write(f"{key}: {value}\n")
    saved_paths.append(params_path)

    print("\nSaved outputs in fresh run folder:")
    for path in saved_paths:
        print(f"  {path}")


def main() -> None:
    df_full = load_and_engineer(CONFIG.data_path)
    df_full = sanitize_model_dataframe(df_full)
    train_cv_df, holdout_df, split_audit_df = create_most_interpolative_holdout(df_full, holdout_size=20, seed=CONFIG.seed)
    df = make_paper_balanced_folds(train_cv_df, n_splits=CONFIG.n_splits, seed=CONFIG.seed)
    print("Input file:", CONFIG.data_path)
    print("Total usable rows:", len(df_full))
    print("Rows used for 10-fold CV:", len(df))
    print("Rows held out for internal validation:", len(holdout_df))
    print("\nFold counts:")
    print(df["cv_fold"].value_counts().sort_index())
    print("\nPaper x fold table:")
    print(pd.crosstab(df["paper_id"], df["cv_fold"]))
    print("\nStarting Optuna hyperparameter optimization on the TRAIN/CV pool only.")
    print("The 20-point internal validation holdout remains fully unseen during tuning and 10-fold CV.")
    best_params, _ = tune_hyperparameters(df)
    oof_df, folds_df, overall, zone_summary, fold_artifacts = final_10fold_run(df, best_params)
    internal_val_df, internal_metrics_df, internal_metrics = evaluate_internal_holdout(holdout_df, fold_artifacts, best_params)
    print("\n===== OVERALL OOF METRICS ON TRAIN/CV POOL =====")
    for key, value in overall.items():
        print(f"{key}: {value:.6f}")
    print("\n===== 20-POINT INTERNAL VALIDATION METRICS =====")
    for key, value in internal_metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")
    print("\n===== FOLD-WISE METRICS =====")
    print(folds_df)
    print("\n===== ORIENTATION-ZONE METRICS =====")
    print(zone_summary)
    print("\n===== INTERNAL VALIDATION POINTS =====")
    cols = ["paper_id", "spec_no", "T_C", "stress_MPa", "Gamma_prime_size", "theta_from_001_deg", "theta_from_011_deg", "t_rupture_h", "Pred_t_rupture_h", "APE_%", "Pred_Log10_Ensemble_STD"]
    print(internal_val_df[cols].to_string(index=False))
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / f"pinn_run_{run_stamp}"
    save_outputs(oof_df, folds_df, zone_summary, best_params, internal_val_df=internal_val_df, internal_metrics_df=internal_metrics_df, split_audit_df=split_audit_df, output_dir=output_dir)


if __name__ == "__main__":
    main()
