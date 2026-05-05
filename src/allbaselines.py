"""
Three-baseline runner using the same split logic as the PINN.

Baselines:
1. Random Forest
2. SVR
3. XGBoost

Logic:
- Load raw processed dataset.
- Engineer same features.
- Drop rows with non-finite required features.
- Select 20-point internal validation holdout first.
- Assign paper-balanced 10-fold CV only on remaining Train/CV pool.
- Tune each baseline using GridSearchCV + PredefinedSplit.
- Generate OOF predictions on Train/CV pool.
- Evaluate 20 unseen internal validation points using 10 fold-model ensemble.

Feature logic:
- T_C is NOT used as a direct feature because Inv_T_K is already present.
- stress_MPa is included, so total = 11 features.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data_path: str = str(PROJECT_ROOT / "data" / "dataset_processed.csv")
    output_root: str = str(PROJECT_ROOT / "output")
    n_splits: int = 10
    holdout_size: int = 20
    min_near_001: int = 8
    seed: int = 42
    grid_n_jobs: int = -1


CONFIG = Config()

TARGET_COL = "y_log10"

FEATURE_COLS = [
    "Inv_T_K",
    "Gamma_prime_size",
    "L",
    "G",
    "SFE",
    "Misfit",
    "cos_001",
    "cos_011",
    "MultislipPropensity_110",
    "Log_RSS_110",
    "stress_MPa",
]

BASE_COLS = ["stress_MPa", "T_C"]


# =========================================================
# REPRODUCIBILITY
# =========================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(CONFIG.seed)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

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
    distances = {
        key: np.linalg.norm(point - value)
        for key, value in corners.items()
    }

    best_zone = min(distances, key=distances.get)

    if distances[best_zone] <= corner_thresh:
        return best_zone

    return "interior"


def load_and_engineer(path: str) -> pd.DataFrame:
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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (df["t_rupture_h"] <= 0).any():
        raise ValueError("Found non-positive values in t_rupture_h.")

    df["y_log10"] = np.log10(df["t_rupture_h"].astype(float))
    df["T_K"] = df["T_C"].astype(float) + 273.15
    df["Inv_T_K"] = 1000.0 / df["T_K"]
    df["Log_Stress"] = np.log10(df["stress_MPa"].astype(float))

    df["cos_001"] = np.cos(
        np.radians(df["theta_from_001_deg"].astype(float))
    )
    df["cos_011"] = np.cos(
        np.radians(df["theta_from_011_deg"].astype(float))
    )

    if "Schmid1_110" in df.columns:
        df["Schmid1_110"] = df["Schmid1_110"].astype(float)
    elif "Schmid110_1" in df.columns:
        df["Schmid1_110"] = df["Schmid110_1"].astype(float)
    else:
        raise ValueError(
            "Need either 'Schmid1_110' or 'Schmid110_1' in the dataset."
        )

    if "MultislipPropensity_110" in df.columns:
        df["MultislipPropensity_110"] = df[
            "MultislipPropensity_110"
        ].astype(float)
    elif "Schmid110_2" in df.columns:
        df["MultislipPropensity_110"] = (
            df["Schmid110_2"].astype(float)
            / df["Schmid1_110"].astype(float)
        )
    else:
        raise ValueError(
            "Need either 'MultislipPropensity_110' or 'Schmid110_2' "
            "in the dataset."
        )

    df["RSS_110"] = (
        df["stress_MPa"].astype(float)
        * df["Schmid1_110"].astype(float)
    )

    if (df["RSS_110"] <= 0).any():
        raise ValueError("Found non-positive values in RSS_110.")

    df["Log_RSS_110"] = np.log10(df["RSS_110"].astype(float))

    df["orientation_zone"] = df.apply(
        lambda row: assign_orientation_zone(
            row["theta_from_001_deg"],
            row["theta_from_011_deg"],
        ),
        axis=1,
    )

    return df


def sanitize_model_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    needed = sorted(
        set(
            FEATURE_COLS
            + BASE_COLS
            + [
                TARGET_COL,
                "T_K",
                "t_rupture_h",
                "theta_from_001_deg",
                "theta_from_011_deg",
            ]
        )
    )

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


# =========================================================
# 20-POINT HOLDOUT SELECTION
# =========================================================

def condition_key_columns() -> List[str]:
    return [
        "paper_id",
        "T_C",
        "stress_MPa",
        "Gamma_prime_size",
        "L",
        "G",
        "SFE",
        "Misfit",
    ]


def orientation_distance(row_a: pd.Series, row_b: pd.Series) -> float:
    return float(
        np.sqrt(
            (
                float(row_a["theta_from_001_deg"])
                - float(row_b["theta_from_001_deg"])
            ) ** 2
            + (
                float(row_a["theta_from_011_deg"])
                - float(row_b["theta_from_011_deg"])
            ) ** 2
        )
    )


def build_interpolation_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cond_cols = condition_key_columns()

    for group_id, (_, group) in enumerate(
        df.groupby(cond_cols, dropna=False, sort=False)
    ):
        if len(group) < 3:
            continue

        g = group.copy()

        spread_001 = float(
            g["theta_from_001_deg"].max()
            - g["theta_from_001_deg"].min()
        )
        spread_011 = float(
            g["theta_from_011_deg"].max()
            - g["theta_from_011_deg"].min()
        )

        if spread_001 >= spread_011:
            sort_col = "theta_from_001_deg"
        else:
            sort_col = "theta_from_011_deg"

        g = g.sort_values(
            [sort_col, "theta_from_011_deg", "theta_from_001_deg"]
        ).copy()

        idxs = list(g.index)

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

            near_001_flag = bool(
                float(row["theta_from_001_deg"]) <= 35.0
            )

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

    cand = cand.sort_values(
        ["interpolation_score", "same_condition_group_size"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return cand


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

    rng = np.random.RandomState(seed)
    work = df.copy().reset_index(drop=True)

    candidates = build_interpolation_candidates(work)

    if candidates.empty:
        raise ValueError(
            "No exact-condition interpolation candidates found. "
            "Need groups with at least 3 rows sharing paper_id, T_C, stress_MPa, "
            "Gamma_prime_size, L, G, SFE, and Misfit but differing in orientation."
        )

    near_candidates = candidates[
        candidates["near_001_validation_flag"] == True
    ].copy()

    all_candidates = candidates.copy()

    selected: List[int] = []
    group_counts: Dict[Tuple[object, ...], int] = {}
    paper_counts: Dict[object, int] = {}
    gamma_counts: Dict[object, int] = {}

    diversity_passes = [
        (1, 4, 6),
        (2, 5, 8),
        (3, 7, 10),
        (999, 999, 999),
    ]

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

    if len(selected) < holdout_size:
        remaining = [
            int(i)
            for i in all_candidates["index"].tolist()
            if int(i) not in selected
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: holdout_size - len(selected)])

    selected = selected[:holdout_size]

    if len(selected) < holdout_size:
        raise ValueError(
            f"Only found {len(selected)} eligible holdout candidates, "
            f"requested {holdout_size}."
        )

    holdout_df = work.loc[selected].copy().reset_index(drop=True)
    train_cv_df = work.drop(index=selected).copy().reset_index(drop=True)

    split_audit = work.copy()
    split_audit["split"] = "train_cv"
    split_audit.loc[selected, "split"] = "internal_validation_20"

    cand_lookup = candidates.set_index("index")

    # =====================================================
    # FIXED AUDIT COLUMN INITIALIZATION
    # This prevents Pandas dtype errors such as:
    # TypeError: Invalid value 'True' for dtype 'float64'
    # =====================================================

    object_audit_cols = [
        "exact_condition_group_id",
    ]

    bool_audit_cols = [
        "near_001_validation_flag",
    ]

    numeric_audit_cols = [
        "same_condition_group_size",
        "left_orientation_distance",
        "right_orientation_distance",
        "local_orientation_span",
        "orientation_balance_gap",
        "interpolation_score",
    ]

    for col in object_audit_cols:
        split_audit[col] = pd.Series(
            [None] * len(split_audit),
            dtype="object",
        )

    for col in bool_audit_cols:
        split_audit[col] = pd.Series(
            [False] * len(split_audit),
            dtype="bool",
        )

    for col in numeric_audit_cols:
        split_audit[col] = np.nan

    for idx in selected:
        if idx in cand_lookup.index:
            for col in object_audit_cols + bool_audit_cols + numeric_audit_cols:
                split_audit.loc[idx, col] = cand_lookup.loc[idx, col]

    near_count = int(
        (holdout_df["theta_from_001_deg"].astype(float) <= 35.0).sum()
    )

    print("\n===== 20-POINT INTERNAL VALIDATION HOLDOUT SUMMARY =====")
    print("Train/CV rows:", len(train_cv_df))
    print("Internal holdout rows:", len(holdout_df))
    print(
        f"Near-[001] rows, theta_from_001_deg <= 35: "
        f"{near_count}/{len(holdout_df)}"
    )
    print(
        "Selection uses input-space diversity only; "
        "no target, prediction, or error used."
    )

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
    print(
        split_audit.loc[selected, display_cols]
        .sort_values(
            ["near_001_validation_flag", "interpolation_score"],
            ascending=[False, False],
        )
        .to_string(index=False)
    )

    if near_count < min_near_001:
        print(
            f"\nWARNING: Requested at least {min_near_001} near-[001] holdout rows, "
            f"but only {near_count} were available under the exact-condition "
            f"eligibility rule."
        )

    return train_cv_df, holdout_df, split_audit.reset_index(drop=True)


# =========================================================
# PAPER-BALANCED 10-FOLD CV
# =========================================================

def make_paper_balanced_folds(
    df: pd.DataFrame,
    n_splits: int = 10,
    seed: int = 42,
) -> pd.DataFrame:

    out = df.copy()
    out["cv_fold"] = -1

    rng = np.random.RandomState(seed)

    for _, subset in out.groupby("paper_id"):
        subset = subset.copy()

        subset = subset.sample(
            frac=1.0,
            random_state=rng.randint(1, 1_000_000),
        )

        subset = subset.sort_values(TARGET_COL).reset_index()

        forward = list(range(n_splits))
        backward = list(range(n_splits - 1, -1, -1))
        snake = forward + backward

        offset = rng.randint(0, n_splits)
        snake = [((fold + offset) % n_splits) for fold in snake]

        subset["cv_fold"] = [
            snake[i % len(snake)]
            for i in range(len(subset))
        ]

        out.loc[subset["index"].values, "cv_fold"] = subset[
            "cv_fold"
        ].values

    out["cv_fold"] = out["cv_fold"].astype(int)

    if (out["cv_fold"] < 0).any():
        raise RuntimeError("Some rows did not receive a CV fold.")

    return out.reset_index(drop=True)


# =========================================================
# METRICS
# =========================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    actual_h = 10 ** y_true
    pred_h = 10 ** y_pred

    ape = np.abs(pred_h - actual_h) / actual_h * 100.0

    return {
        "MAE_log10h": float(mae),
        "RMSE_log10h": float(rmse),
        "R2_log10h": float(r2),
        "MedianAPE_%": float(np.median(ape)),
        "MeanAPE_%": float(np.mean(ape)),
    }


def compute_zone_metrics(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    ape_col: str,
) -> pd.DataFrame:

    rows = []

    for zone, g in df.groupby("orientation_zone"):
        rows.append(
            {
                "orientation_zone": zone,
                "n": len(g),
                "MAE_log10h": mean_absolute_error(
                    g[actual_col],
                    g[pred_col],
                ),
                "RMSE_log10h": np.sqrt(
                    mean_squared_error(
                        g[actual_col],
                        g[pred_col],
                    )
                ),
                "MedianAPE_%": np.median(g[ape_col]),
                "MeanAPE_%": np.mean(g[ape_col]),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def get_model_and_grid(model_name: str) -> Tuple[Pipeline, Dict]:

    model_name = model_name.upper()

    if model_name == "RF":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        random_state=CONFIG.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        param_grid = {
            "rf__n_estimators": [200, 400],
            "rf__max_depth": [None, 8, 12],
            "rf__min_samples_split": [2, 4],
            "rf__min_samples_leaf": [1, 2],
            "rf__max_features": ["sqrt", 0.7],
            "rf__bootstrap": [True],
        }

        return pipe, param_grid

    if model_name == "SVR":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("svr", SVR()),
            ]
        )

        param_grid = [
            {
                "svr__kernel": ["rbf"],
                "svr__C": [1, 3, 10, 30, 100, 300],
                "svr__epsilon": [0.01, 0.03, 0.05, 0.1, 0.2],
                "svr__gamma": ["scale", 0.01, 0.03, 0.1, 0.3, 1.0],
            },
            {
                "svr__kernel": ["poly"],
                "svr__degree": [2, 3],
                "svr__C": [1, 3, 10, 30, 100],
                "svr__epsilon": [0.01, 0.03, 0.05, 0.1],
                "svr__gamma": ["scale", 0.01, 0.1],
                "svr__coef0": [0.0, 0.5, 1.0],
            },
            {
                "svr__kernel": ["linear"],
                "svr__C": [0.3, 1, 3, 10, 30, 100],
                "svr__epsilon": [0.01, 0.03, 0.05, 0.1, 0.2],
            },
        ]

        return pipe, param_grid

    if model_name == "XGB":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "xgb",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=CONFIG.seed,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )

        param_grid = {
            "xgb__n_estimators": [100, 200],
            "xgb__max_depth": [2, 3, 4],
            "xgb__learning_rate": [0.03, 0.05, 0.1],
            "xgb__subsample": [0.8, 1.0],
            "xgb__colsample_bytree": [0.8, 1.0],
            "xgb__min_child_weight": [1, 3],
            "xgb__reg_lambda": [1.0, 3.0],
        }

        return pipe, param_grid

    raise ValueError(f"Unknown model_name: {model_name}")


# =========================================================
# BASELINE TRAINING
# =========================================================

def run_one_baseline(
    model_name: str,
    train_cv_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, float]:

    model_name = model_name.upper()

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    X = train_cv_df[FEATURE_COLS].copy()
    y = train_cv_df[TARGET_COL].astype(float).values
    fold_ids = train_cv_df["cv_fold"].astype(int).values

    pipe, param_grid = get_model_and_grid(model_name)

    ps = PredefinedSplit(test_fold=fold_ids)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=ps,
        n_jobs=CONFIG.grid_n_jobs,
        verbose=2,
        refit=True,
        return_train_score=False,
    )

    print(f"\n\n================ TUNING {model_name} ================")
    search.fit(X, y)

    print(f"\nBest {model_name} CV score, neg MAE: {search.best_score_}")
    print(f"Best {model_name} params:")

    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    unique_folds = sorted(np.unique(fold_ids))

    oof_pred = np.full(len(train_cv_df), np.nan, dtype=float)
    fold_rows = []
    fold_models = []

    for fold in unique_folds:
        print(f"\n========== {model_name} FINAL FOLD {fold} ==========")

        train_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        model_fold = clone(search.best_estimator_)
        model_fold.fit(X_train, y_train)

        y_pred = model_fold.predict(X_test)

        oof_pred[test_idx] = y_pred
        fold_models.append(model_fold)

        fold_metric = compute_metrics(y_test, y_pred)
        fold_metric["fold"] = int(fold)
        fold_metric["n_val"] = int(len(test_idx))

        fold_rows.append(fold_metric)

        for k, v in fold_metric.items():
            if k not in ["fold", "n_val"]:
                print(f"{k}: {v:.6f}")

    if np.isnan(oof_pred).any():
        raise RuntimeError(f"{model_name}: Some OOF predictions are NaN.")

    oof_df = train_cv_df.copy()

    actual_col = f"{model_name}_Actual_Log10_t_rupture_h"
    pred_col = f"{model_name}_Pred_Log10_t_rupture_h"
    pred_h_col = f"{model_name}_Pred_t_rupture_h"
    ape_col = f"{model_name}_APE_%"

    oof_df[actual_col] = y
    oof_df[pred_col] = oof_pred
    oof_df[pred_h_col] = 10 ** oof_df[pred_col]
    oof_df[ape_col] = (
        np.abs(oof_df[pred_h_col] - oof_df["t_rupture_h"])
        / oof_df["t_rupture_h"]
        * 100.0
    )

    overall_oof = compute_metrics(y, oof_pred)

    fold_metrics_df = (
        pd.DataFrame(fold_rows)
        .sort_values("fold")
        .reset_index(drop=True)
    )

    zone_metrics_df = compute_zone_metrics(
        oof_df,
        actual_col=actual_col,
        pred_col=pred_col,
        ape_col=ape_col,
    )

    print(f"\n================ OVERALL {model_name} OOF METRICS ================")

    for k, v in overall_oof.items():
        print(f"{k}: {v:.6f}")

    # =====================================================
    # INTERNAL VALIDATION USING 10-FOLD ENSEMBLE
    # =====================================================

    X_holdout = holdout_df[FEATURE_COLS].copy()
    y_holdout = holdout_df[TARGET_COL].astype(float).values

    holdout_pred_matrix = []

    for model_fold in fold_models:
        holdout_pred_matrix.append(model_fold.predict(X_holdout))

    holdout_pred_matrix = np.vstack(holdout_pred_matrix)

    holdout_pred_mean = holdout_pred_matrix.mean(axis=0)
    holdout_pred_std = holdout_pred_matrix.std(axis=0)

    holdout_result_df = holdout_df.copy()

    h_actual_col = f"{model_name}_Actual_Log10_t_rupture_h"
    h_pred_col = f"{model_name}_Pred_Log10_t_rupture_h"
    h_std_col = f"{model_name}_Pred_Log10_Ensemble_STD"
    h_pred_h_col = f"{model_name}_Pred_t_rupture_h"
    h_ape_col = f"{model_name}_APE_%"

    holdout_result_df[h_actual_col] = y_holdout
    holdout_result_df[h_pred_col] = holdout_pred_mean
    holdout_result_df[h_std_col] = holdout_pred_std
    holdout_result_df[h_pred_h_col] = 10 ** holdout_result_df[h_pred_col]

    holdout_result_df[h_ape_col] = (
        np.abs(
            holdout_result_df[h_pred_h_col]
            - holdout_result_df["t_rupture_h"]
        )
        / holdout_result_df["t_rupture_h"]
        * 100.0
    )

    holdout_metrics = compute_metrics(y_holdout, holdout_pred_mean)
    holdout_metrics["n_holdout"] = int(len(holdout_df))
    holdout_metrics["MeanEnsembleSTD_log10h"] = float(
        np.mean(holdout_pred_std)
    )

    holdout_metrics_df = pd.DataFrame([holdout_metrics])

    print(
        f"\n================ {model_name} "
        f"20-POINT INTERNAL VALIDATION ================"
    )

    for k, v in holdout_metrics.items():
        if isinstance(v, int):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.6f}")

    # Also train one final model on all Train/CV data.
    # This is saved for reference only.
    # The main holdout metric above uses fold ensemble.

    final_full_model = clone(search.best_estimator_)
    final_full_model.fit(X, y)

    full_pred = final_full_model.predict(X_holdout)

    holdout_result_df[
        f"{model_name}_FullTrainCV_Pred_Log10_t_rupture_h"
    ] = full_pred

    holdout_result_df[
        f"{model_name}_FullTrainCV_Pred_t_rupture_h"
    ] = 10 ** full_pred

    holdout_result_df[
        f"{model_name}_FullTrainCV_APE_%"
    ] = (
        np.abs(
            holdout_result_df[
                f"{model_name}_FullTrainCV_Pred_t_rupture_h"
            ]
            - holdout_result_df["t_rupture_h"]
        )
        / holdout_result_df["t_rupture_h"]
        * 100.0
    )

    # =====================================================
    # SAVE OUTPUTS
    # =====================================================

    oof_path = model_dir / f"{model_name.lower()}_oof_preds_train_cv.csv"
    fold_metrics_path = model_dir / f"{model_name.lower()}_fold_metrics.csv"
    zone_metrics_path = model_dir / f"{model_name.lower()}_zone_metrics.csv"
    holdout_path = model_dir / f"{model_name.lower()}_internal_validation_20_unseen.csv"
    holdout_metrics_path = model_dir / f"{model_name.lower()}_internal_validation_metrics.csv"
    best_params_path = model_dir / f"{model_name.lower()}_best_params.txt"

    oof_df.to_csv(oof_path, index=False)
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    zone_metrics_df.to_csv(zone_metrics_path, index=False)
    holdout_result_df.to_csv(holdout_path, index=False)
    holdout_metrics_df.to_csv(holdout_metrics_path, index=False)

    with open(best_params_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best CV score, neg MAE: {search.best_score_}\n\n")

        f.write("Feature columns:\n")
        for col in FEATURE_COLS:
            f.write(f"  {col}\n")

        f.write("\nBest params:\n")
        for k, v in search.best_params_.items():
            f.write(f"{k}: {v}\n")

        f.write("\nOverall OOF metrics on Train/CV pool:\n")
        for k, v in overall_oof.items():
            f.write(f"{k}: {v}\n")

        f.write("\n20-point internal validation metrics, fold-ensemble prediction:\n")
        for k, v in holdout_metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"\nSaved {model_name} outputs:")
    print(" ", oof_path)
    print(" ", fold_metrics_path)
    print(" ", zone_metrics_path)
    print(" ", holdout_path)
    print(" ", holdout_metrics_path)
    print(" ", best_params_path)

    summary = {
        "model": model_name,
        "n_train_cv": int(len(train_cv_df)),
        "n_holdout": int(len(holdout_df)),
        "best_cv_neg_mae": float(search.best_score_),
    }

    for k, v in overall_oof.items():
        summary[f"OOF_{k}"] = v

    for k, v in holdout_metrics.items():
        summary[f"Holdout_{k}"] = v

    return summary


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("Input file:", CONFIG.data_path)

    df_full = load_and_engineer(CONFIG.data_path)
    df_full = sanitize_model_dataframe(df_full)

    train_cv_df, holdout_df, split_audit_df = create_physically_diverse_holdout(
        df_full,
        holdout_size=CONFIG.holdout_size,
        min_near_001=CONFIG.min_near_001,
        seed=CONFIG.seed,
    )

    train_cv_df = make_paper_balanced_folds(
        train_cv_df,
        n_splits=CONFIG.n_splits,
        seed=CONFIG.seed,
    )

    print("\n===== FINAL SPLIT SUMMARY =====")
    print("Total usable rows:", len(df_full))
    print("Rows used for Train/CV:", len(train_cv_df))
    print("Rows held out for internal validation:", len(holdout_df))

    print("\nFold counts:")
    print(train_cv_df["cv_fold"].value_counts().sort_index())

    print("\nPaper x fold table:")
    print(pd.crosstab(train_cv_df["paper_id"], train_cv_df["cv_fold"]))

    print("\nFeatures used:")
    for i, col in enumerate(FEATURE_COLS, start=1):
        print(f"{i:02d}. {col}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(CONFIG.output_root) / f"baseline_same_split_run_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_audit_path = output_dir / "baseline_split_assignments.csv"
    train_cv_path = output_dir / "baseline_train_cv_with_folds.csv"
    holdout_path = output_dir / "baseline_internal_validation_20_raw.csv"

    split_audit_df.to_csv(split_audit_path, index=False)
    train_cv_df.to_csv(train_cv_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)

    print("\nSaved split files:")
    print(" ", split_audit_path)
    print(" ", train_cv_path)
    print(" ", holdout_path)

    summaries = []

    for model_name in ["RF", "SVR", "XGB"]:
        summary = run_one_baseline(
            model_name=model_name,
            train_cv_df=train_cv_df,
            holdout_df=holdout_df,
            output_dir=output_dir,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_path = output_dir / "all_baseline_summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n================ ALL BASELINE SUMMARY ================")
    print(summary_df.to_string(index=False))

    print("\nSaved combined summary:")
    print(" ", summary_path)


if __name__ == "__main__":
    main()