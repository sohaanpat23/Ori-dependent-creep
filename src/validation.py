from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# =========================================================
# EDIT THESE PATHS
# =========================================================
TRAIN_PATH = "dataset_processed.csv"   # training dataset with Leng rows removed
VAL_PATH = "validationset.xlsx"                     # external validation set containing Leng rows
OUTPUT_DIR = "output_external_validation"

# =========================================================
# FIXED BEST HYPERPARAMETERS FROM YOUR TRIAL 13
# =========================================================
BEST_PARAMS: Dict[str, float | int] = {
    "mat_width": 128,
    "mat_depth": 3,
    "ori_width": 96,
    "ori_depth": 1,
    "fusion_width": 32,
    "fusion_depth": 1,
    "dropout": 0.013990327652214543,
    "lr": 0.002393777695561849,
    "weight_decay": 4.5585004222393665e-07,
    "batch_size": 8,
    "C_lmp": 19.65435690613274,
    "ori_scale": 2.0984001405657695,
    "mat_scale": 2.2068873736990002,
    "lambda_stress_mono": 0.0015850667647819914,
    "lambda_ori_smooth": 7.576356667024342e-05,
    "lambda_mat_reg": 0.00016685867205702565,
    "zone_balance_weight": 0.09981010915364186,
    "huber_delta": 1.6042534216685036,
}

# =========================================================
# CONFIG
# =========================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 650  # same final-training budget as your current code

MAT_COLS = ["Inv_T_K", "Gamma_prime_size", "L", "G", "SFE", "Misfit"]
ORI_COLS = ["cos_001", "cos_011", "MultislipPropensity_110"]
STRESS_COL = "Log_RSS_110"
BASE_COLS = ["stress_MPa", "T_C"]
TARGET_COL = "y_log10"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


def assign_orientation_zone(
    theta001: float,
    theta011: float,
    corner_thresh: float = 8.0,
) -> str:
    corners = {
        "near_001": np.array([0.0, 45.0]),
        "near_011": np.array([45.0, 0.0]),
        "near_111": np.array([54.7356, 35.2644]),
    }
    point = np.array([theta001, theta011], dtype=float)
    distances = {key: np.linalg.norm(point - value) for key, value in corners.items()}
    best_zone = min(distances, key=distances.get)
    return best_zone if distances[best_zone] <= corner_thresh else "interior"


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_and_engineer(path: str | Path) -> pd.DataFrame:
    df = load_table(path).copy()

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


def prepare_train_external_data(
    train_df: pd.DataFrame,
    external_df: pd.DataFrame,
    c_lmp: float,
):
    train_df = train_df.copy()
    external_df = external_df.copy()

    train_df["LMP_target"] = train_df["T_K"] * (c_lmp + train_df[TARGET_COL])
    external_df["LMP_target"] = external_df["T_K"] * (c_lmp + external_df[TARGET_COL])

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

    xmat_ext = mat_scaler.transform(external_df[MAT_COLS].values)
    xori_ext = ori_scaler.transform(external_df[ORI_COLS].values)
    xs_ext = stress_scaler.transform(external_df[[STRESS_COL]].values)
    xb_ext = base_scaler.transform(external_df[BASE_COLS].values)
    ylmp_ext = lmp_scaler.transform(external_df[["LMP_target"]].values).reshape(-1)

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
            "x_mat": xmat_ext.astype(np.float32),
            "x_ori": xori_ext.astype(np.float32),
            "x_stress": xs_ext.astype(np.float32),
            "x_base": xb_ext.astype(np.float32),
            "y_lmp": ylmp_ext.astype(np.float32),
            "y_log": external_df[TARGET_COL].values.astype(np.float32),
            "T_K": external_df["T_K"].values.astype(np.float32),
            "zone": external_df["orientation_zone"].values,
            "index": external_df.index.values,
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
    def __init__(self) -> None:
        super().__init__()
        self.poly = nn.Parameter(torch.zeros(6))
        self.relu_weights = nn.Parameter(torch.zeros(2))
        self.register_buffer(
            "stress_knots", torch.tensor([[-0.75, 0.75]], dtype=torch.float32)
        )
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
    lmp_pred = lmp_scaler.inverse_transform(
        np.asarray(lmp_pred_norm).reshape(-1, 1)
    ).reshape(-1)
    return lmp_pred / t_k - c_lmp


def zone_mae(y_true: np.ndarray, y_pred: np.ndarray, zones: np.ndarray) -> Dict[str, float]:
    zone_df = pd.DataFrame({"zone": zones, "yt": y_true, "yp": y_pred})
    return zone_df.groupby("zone").apply(
        lambda x: mean_absolute_error(x["yt"], x["yp"])
    ).to_dict()


def eval_model(
    model: nn.Module,
    part: Dict[str, np.ndarray],
    lmp_scaler: StandardScaler,
    c_lmp: float,
):
    model.eval()
    with torch.no_grad():
        x_mat = torch.from_numpy(part["x_mat"]).to(DEVICE)
        x_ori = torch.from_numpy(part["x_ori"]).to(DEVICE)
        x_stress = torch.from_numpy(part["x_stress"]).to(DEVICE)
        x_base = torch.from_numpy(part["x_base"]).to(DEVICE)
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


def train_final_model(train_data, params):
    model = OrientationAwareLMPPINN(
        n_mat=len(MAT_COLS),
        n_ori=len(ORI_COLS),
        params=params,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )

    train_loader = make_loader(
        train_data,
        batch_size=int(params["batch_size"]),
        shuffle=True,
    )

    model.train()
    for epoch in range(MAX_EPOCHS):
        epoch_loss = 0.0

        for xb_mat, xb_ori, xb_stress, xb_base, yb_lmp, _, _ in train_loader:
            xb_mat = xb_mat.to(DEVICE).requires_grad_(True)
            xb_ori = xb_ori.to(DEVICE).requires_grad_(True)
            xb_stress = xb_stress.to(DEVICE)
            xb_base = xb_base.to(DEVICE).requires_grad_(True)
            yb_lmp = yb_lmp.to(DEVICE)

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

            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {epoch_loss / len(train_loader):.6f}")

    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = load_and_engineer(TRAIN_PATH)
    val_df = load_and_engineer(VAL_PATH)

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("\nValidation paper counts:")
    print(val_df["paper_id"].value_counts())

    data = prepare_train_external_data(
        train_df=train_df,
        external_df=val_df,
        c_lmp=float(BEST_PARAMS["C_lmp"]),
    )

    model = train_final_model(data["train"], BEST_PARAMS)

    val_out = eval_model(
        model=model,
        part=data["val"],
        lmp_scaler=data["scalers"]["lmp"],
        c_lmp=float(BEST_PARAMS["C_lmp"]),
    )

    pred_df = val_df.copy()
    pred_df["Pred_Log10_t_rupture_h"] = val_out["y_pred"]
    pred_df["Actual_Log10_t_rupture_h"] = pred_df[TARGET_COL]
    pred_df["Pred_t_rupture_h"] = 10 ** pred_df["Pred_Log10_t_rupture_h"]
    pred_df["APE_%"] = (
        np.abs(pred_df["Pred_t_rupture_h"] - pred_df["t_rupture_h"])
        / pred_df["t_rupture_h"]
        * 100.0
    )

    overall = {
        "MAE_log10h": val_out["mae"],
        "RMSE_log10h": val_out["rmse"],
        "R2_log10h": val_out["r2"],
        "MedianAPE_%": np.median(pred_df["APE_%"]),
        "MeanAPE_%": np.mean(pred_df["APE_%"]),
        "ZonePenalty": val_out["zone_penalty"],
        "n_validation_samples": len(pred_df),
    }

    zone_summary = (
        pred_df.groupby("orientation_zone")
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

    paper_summary = (
        pred_df.groupby("paper_id")
        .apply(
            lambda x: pd.Series(
                {
                    "n_samples": len(x),
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

    pred_df.to_csv(Path(OUTPUT_DIR) / "external_validation_predictions.csv", index=False)
    pd.DataFrame([overall]).to_csv(Path(OUTPUT_DIR) / "external_validation_metrics.csv", index=False)
    zone_summary.to_csv(Path(OUTPUT_DIR) / "external_validation_zone_metrics.csv", index=False)
    paper_summary.to_csv(Path(OUTPUT_DIR) / "external_validation_paper_metrics.csv", index=False)

    print("\n===== EXTERNAL VALIDATION METRICS =====")
    for k, v in overall.items():
        if isinstance(v, (int, np.integer)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")

    print("\n===== EXTERNAL VALIDATION ZONE METRICS =====")
    print(zone_summary)

    print("\n===== EXTERNAL VALIDATION PAPER METRICS =====")
    print(paper_summary)

    torch.save(model.state_dict(), Path(OUTPUT_DIR) / "final_model_state_dict.pt")
    print("\nSaved files:")
    print(Path(OUTPUT_DIR) / "external_validation_predictions.csv")
    print(Path(OUTPUT_DIR) / "external_validation_metrics.csv")
    print(Path(OUTPUT_DIR) / "external_validation_zone_metrics.csv")
    print(Path(OUTPUT_DIR) / "external_validation_paper_metrics.csv")
    print(Path(OUTPUT_DIR) / "final_model_state_dict.pt")


if __name__ == "__main__":
    main()