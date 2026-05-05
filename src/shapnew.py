
"""SHAP analysis for the raw-stress/temperature baseline PINN.

This script reuses the same:
- dataset preparation
- fold construction
- architecture
- fixed hyperparameters
- final-training logic

and computes SHAP explanations on predicted log10 rupture life over the
out-of-fold validation partitions.
"""

from __future__ import annotations

import os
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import shap
except ImportError as exc:
    raise ImportError("Install shap first: pip install shap") from exc

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data_path: str = str(PROJECT_ROOT / "data" / "dataset_processed.csv")
    n_splits: int = 10
    max_epochs_final: int = 650
    patience_final: int = 90
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    shap_background_size: int = 64
    shap_eval_size_per_fold: int = 64

    shap_values_filename: str = "pinn_shap_values.csv"
    shap_importance_filename: str = "pinn_shap_mean_abs.csv"
    shap_beeswarm_filename: str = "pinn_shap_beeswarm.png"
    shap_bar_filename: str = "pinn_shap_bar.png"
    fold_metrics_filename: str = "pinn_shap_fold_metrics.csv"


CONFIG = Config()

MAT_COLS = ["Inv_T_K", "Gamma_prime_size", "L", "G", "SFE", "Misfit"]
ORI_COLS = ["cos_001", "cos_011", "MultislipPropensity_110"]
STRESS_COL = "Log_RSS_110"
BASE_COLS = ["stress_MPa", "T_C"]
TARGET_COL = "y_log10"
FEATURE_NAMES = MAT_COLS + ORI_COLS + [STRESS_COL] + BASE_COLS

BEST_PARAMS: Dict[str, float | int] = {
    "mat_width": 128,
    "mat_depth": 1,
    "ori_width": 96,
    "ori_depth": 1,
    "fusion_width": 96,
    "fusion_depth": 1,
    "dropout": 0.012911091331098929,
    "lr": 0.0034151336870370066,
    "weight_decay": 8.64926877683776e-08,
    "batch_size": 16,
    "C_lmp": 20.486634034474243,
    "ori_scale": 2.232453172897684,
    "mat_scale": 1.2069661281645983,
    "lambda_stress_mono": 0.04379065030206242,
    "lambda_ori_smooth": 0.0003118441618051569,
    "lambda_mat_reg": 0.000247409611688879,
    "zone_balance_weight": 0.14795986794117055,
    "huber_delta": 1.7121605246227074,
}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CONFIG.seed)


def assign_orientation_zone(theta001: float, theta011: float, corner_thresh: float = 8.0) -> str:
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
    df = pd.read_csv(path).copy()

    df = df.rename(columns={"Gamma prime size": "Gamma_prime_size", "Schmid Max": "Schmid_Max"})

    required = [
        "paper_id", "spec_no", "T_C", "stress_MPa", "t_rupture_h",
        "theta_from_001_deg", "theta_from_011_deg", "Gamma_prime_size",
        "L", "G", "SFE", "Misfit",
    ]
    missing = [c for c in required if c not in df.columns]
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
        df["MultislipPropensity_110"] = df["Schmid110_2"].astype(float) / df["Schmid1_110"].astype(float)
    else:
        raise ValueError("Need either 'MultislipPropensity_110' or 'Schmid110_2' in the dataset.")

    df["RSS_110"] = df["stress_MPa"].astype(float) * df["Schmid1_110"].astype(float)
    if (df["RSS_110"] <= 0).any():
        raise ValueError("Found non-positive values in RSS_110.")

    df["Log_RSS_110"] = np.log10(df["RSS_110"].astype(float))
    df["orientation_zone"] = df.apply(
        lambda row: assign_orientation_zone(row["theta_from_001_deg"], row["theta_from_011_deg"]),
        axis=1,
    )
    return df


def make_paper_balanced_folds(df: pd.DataFrame, n_splits: int = 10, seed: int = 42) -> pd.DataFrame:
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


def prepare_fold_data(df: pd.DataFrame, train_idx: pd.Index, val_idx: pd.Index, c_lmp: float):
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
            "x_all": np.concatenate([xmat_train, xori_train, xs_train, xb_train], axis=1).astype(np.float32),
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
            "x_all": np.concatenate([xmat_val, xori_val, xs_val, xb_val], axis=1).astype(np.float32),
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


def make_loader(part: Dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
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
    def __init__(self, in_dim: int, hidden_dim: int, depth: int, dropout: float, out_dim: int, activation: str = "silu") -> None:
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
    def __init__(self, n_mat: int, n_ori: int, params: Dict[str, float | int]) -> None:
        super().__init__()
        self.ori_scale = params["ori_scale"]
        self.mat_scale = params["mat_scale"]

        self.base_branch = RawStressTempBaseBranch()

        self.mat_net = MLP(n_mat, int(params["mat_width"]), int(params["mat_depth"]), float(params["dropout"]), int(params["mat_width"]), "silu")
        self.ori_net = MLP(n_ori, int(params["ori_width"]), int(params["ori_depth"]), float(params["dropout"]), int(params["ori_width"]), "silu")

        self.mat_head = MLP(int(params["mat_width"]) + 1, int(params["fusion_width"]), int(params["fusion_depth"]), float(params["dropout"]), 1, "silu")
        self.ori_head = MLP(int(params["ori_width"]) + 1, int(params["fusion_width"]), int(params["fusion_depth"]), float(params["dropout"]), 1, "silu")
        self.mix_head = MLP(int(params["mat_width"]) + int(params["ori_width"]) + 1, int(params["fusion_width"]), max(1, int(params["fusion_depth"]) - 1), float(params["dropout"]), 1, "silu")

    def forward(self, x_mat: torch.Tensor, x_ori: torch.Tensor, x_stress: torch.Tensor, x_base: torch.Tensor, return_components: bool = False):
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


class FoldLogLifeWrapper(nn.Module):
    def __init__(self, model: OrientationAwareLMPPINN, mat_scaler: StandardScaler, lmp_scaler: StandardScaler, c_lmp: float) -> None:
        super().__init__()
        self.model = model
        self.c_lmp = float(c_lmp)
        self.register_buffer("mat_inv_t_mean", torch.tensor(float(mat_scaler.mean_[0]), dtype=torch.float32))
        self.register_buffer("mat_inv_t_scale", torch.tensor(float(mat_scaler.scale_[0]), dtype=torch.float32))
        self.register_buffer("lmp_mean", torch.tensor(float(lmp_scaler.mean_[0]), dtype=torch.float32))
        self.register_buffer("lmp_scale", torch.tensor(float(lmp_scaler.scale_[0]), dtype=torch.float32))

    def forward(self, x_all: torch.Tensor) -> torch.Tensor:
        x_mat = x_all[:, : len(MAT_COLS)]
        x_ori = x_all[:, len(MAT_COLS): len(MAT_COLS) + len(ORI_COLS)]
        x_stress = x_all[:, len(MAT_COLS) + len(ORI_COLS): len(MAT_COLS) + len(ORI_COLS) + 1]
        x_base = x_all[:, -len(BASE_COLS):]

        pred_norm = self.model(x_mat, x_ori, x_stress, x_base).unsqueeze(1)
        lmp_raw = pred_norm * self.lmp_scale + self.lmp_mean

        inv_t_std = x_mat[:, [0]]
        inv_t_raw = inv_t_std * self.mat_inv_t_scale + self.mat_inv_t_mean
        temp_k = 1000.0 / inv_t_raw

        return lmp_raw / temp_k - self.c_lmp


def zone_mae(y_true: np.ndarray, y_pred: np.ndarray, zones: np.ndarray) -> Dict[str, float]:
    zone_df = pd.DataFrame({"zone": zones, "yt": y_true, "yp": y_pred})
    return zone_df.groupby("zone").apply(lambda x: mean_absolute_error(x["yt"], x["yp"])).to_dict()


def eval_model(model: nn.Module, part: Dict[str, np.ndarray], lmp_scaler: StandardScaler, c_lmp: float):
    model.eval()
    with torch.no_grad():
        x_mat = torch.from_numpy(part["x_mat"]).to(CONFIG.device)
        x_ori = torch.from_numpy(part["x_ori"]).to(CONFIG.device)
        x_stress = torch.from_numpy(part["x_stress"]).to(CONFIG.device)
        x_base = torch.from_numpy(part["x_base"]).to(CONFIG.device)
        pred_lmp_norm = model(x_mat, x_ori, x_stress, x_base).cpu().numpy()

    lmp_pred = lmp_scaler.inverse_transform(np.asarray(pred_lmp_norm).reshape(-1, 1)).reshape(-1)
    y_pred = lmp_pred / part["T_K"] - c_lmp
    y_true = part["y_log"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    z_mae = zone_mae(y_true, y_pred, part["zone"])
    zone_penalty = np.std(list(z_mae.values())) if len(z_mae) > 1 else 0.0

    return {"y_true": y_true, "y_pred": y_pred, "mae": mae, "rmse": rmse, "r2": r2, "zone_mae": z_mae, "zone_penalty": zone_penalty}


def train_one_fold(fold_data, params, max_epochs=300, patience=50):
    model = OrientationAwareLMPPINN(len(MAT_COLS), len(ORI_COLS), params).to(CONFIG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    train_loader = make_loader(fold_data["train"], batch_size=int(params["batch_size"]), shuffle=True)

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

            pred_lmp, base_lmp, residual_lmp = model(xb_mat, xb_ori, xb_stress, xb_base, return_components=True)

            data_loss = F.huber_loss(pred_lmp, yb_lmp, delta=float(params["huber_delta"]), reduction="mean")

            dBase_dStress = torch.autograd.grad(base_lmp.sum(), xb_base, create_graph=True, retain_graph=True)[0][:, [0]]
            stress_mono_loss = torch.mean(F.relu(dBase_dStress) ** 2)

            dL_dOri = torch.autograd.grad(pred_lmp.sum(), xb_ori, create_graph=True, retain_graph=True)[0]
            ori_smooth_loss = torch.mean(dL_dOri ** 2)

            dResidual_dMat = torch.autograd.grad(residual_lmp.sum(), xb_mat, create_graph=True, retain_graph=True)[0]
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

        val_out = eval_model(model, fold_data["val"], fold_data["scalers"]["lmp"], float(params["C_lmp"]))
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
    final_val = eval_model(model, fold_data["val"], fold_data["scalers"]["lmp"], float(params["C_lmp"]))
    return model, final_val


def sample_rows(x: np.ndarray, size: int, seed: int):
    rng = np.random.RandomState(seed)
    if len(x) <= size:
        idx = np.arange(len(x))
        return x, idx
    idx = rng.choice(len(x), size=size, replace=False)
    return x[idx], idx


def run_shap_analysis(df: pd.DataFrame, params: Dict[str, float | int], output_dir: Path) -> None:
    all_shap_rows = []
    fold_metric_rows = []
    feature_blocks = []
    shap_blocks = []

    for fold_id in range(CONFIG.n_splits):
        print(f"\n========== SHAP FOLD {fold_id} ==========")
        train_idx = df.index[df["cv_fold"] != fold_id]
        val_idx = df.index[df["cv_fold"] == fold_id]

        fold_data = prepare_fold_data(df, train_idx, val_idx, float(params["C_lmp"]))
        model, val_out = train_one_fold(
            fold_data,
            params=params,
            max_epochs=CONFIG.max_epochs_final,
            patience=CONFIG.patience_final,
        )

        fold_metric_rows.append(
            {
                "fold": fold_id,
                "n_val": len(val_idx),
                "MAE_log10h": val_out["mae"],
                "RMSE_log10h": val_out["rmse"],
                "R2_log10h": val_out["r2"],
                "ZonePenalty": val_out["zone_penalty"],
            }
        )

        wrapper = FoldLogLifeWrapper(
            model=model,
            mat_scaler=fold_data["scalers"]["mat"],
            lmp_scaler=fold_data["scalers"]["lmp"],
            c_lmp=float(params["C_lmp"]),
        ).to(CONFIG.device)
        wrapper.eval()

        background_x, _ = sample_rows(fold_data["train"]["x_all"], CONFIG.shap_background_size, CONFIG.seed + fold_id)
        eval_x, eval_local_idx = sample_rows(fold_data["val"]["x_all"], CONFIG.shap_eval_size_per_fold, 1000 + CONFIG.seed + fold_id)

        background_t = torch.tensor(background_x, dtype=torch.float32, device=CONFIG.device)
        eval_t = torch.tensor(eval_x, dtype=torch.float32, device=CONFIG.device)

        explainer = shap.GradientExplainer(wrapper, background_t)
        shap_vals = explainer.shap_values(eval_t)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
            shap_vals = shap_vals[:, :, 0]

        feature_blocks.append(eval_x)
        shap_blocks.append(shap_vals)

        selected_global_idx = fold_data["val"]["index"][eval_local_idx]
        pred_log = val_out["y_pred"][eval_local_idx]
        true_log = val_out["y_true"][eval_local_idx]

        for row_i, global_idx in enumerate(selected_global_idx):
            row = {
                "global_index": int(global_idx),
                "fold": fold_id,
                "paper_id": df.loc[global_idx, "paper_id"],
                "spec_no": df.loc[global_idx, "spec_no"],
                "orientation_zone": df.loc[global_idx, "orientation_zone"],
                "y_true_log10h": float(true_log[row_i]),
                "y_pred_log10h": float(pred_log[row_i]),
            }
            for feat_j, feat_name in enumerate(FEATURE_NAMES):
                row[f"{feat_name}_value_scaled"] = float(eval_x[row_i, feat_j])
                row[f"{feat_name}_shap"] = float(shap_vals[row_i, feat_j])
            all_shap_rows.append(row)

    shap_features_all = np.vstack(feature_blocks)
    shap_values_all = np.vstack(shap_blocks)

    pd.DataFrame(all_shap_rows).to_csv(output_dir / CONFIG.shap_values_filename, index=False)

    importance_df = pd.DataFrame(
        {"feature": FEATURE_NAMES, "mean_abs_shap": np.mean(np.abs(shap_values_all), axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(output_dir / CONFIG.shap_importance_filename, index=False)

    pd.DataFrame(fold_metric_rows).to_csv(output_dir / CONFIG.fold_metrics_filename, index=False)

    shap.summary_plot(shap_values_all, features=shap_features_all, feature_names=FEATURE_NAMES, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / CONFIG.shap_bar_filename, dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values_all, features=shap_features_all, feature_names=FEATURE_NAMES, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / CONFIG.shap_beeswarm_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved SHAP outputs:")
    print(output_dir / CONFIG.shap_values_filename)
    print(output_dir / CONFIG.shap_importance_filename)
    print(output_dir / CONFIG.fold_metrics_filename)
    print(output_dir / CONFIG.shap_bar_filename)
    print(output_dir / CONFIG.shap_beeswarm_filename)


def main() -> None:
    df = load_and_engineer(CONFIG.data_path)
    df = make_paper_balanced_folds(df, n_splits=CONFIG.n_splits, seed=CONFIG.seed)

    print("Input file:", CONFIG.data_path)
    print("Total rows:", len(df))
    print("\nUsing fixed hyperparameters:")
    for key, value in BEST_PARAMS.items():
        print(f"  {key}: {value}")

    output_dir = PROJECT_ROOT / "output_shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_shap_analysis(df, BEST_PARAMS, output_dir)


if __name__ == "__main__":
    main()
