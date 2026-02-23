from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("artifacts") / "synthetic_dataset.csv"
FEATURE_NAMES_PATH = Path("artifacts") / "feature_names.json"
FEATURE_COLUMNS = [
    "growth_rate",
    "referral_ratio",
    "payout_dependency",
    "centralization_index",
    "avg_holding_time",
    "reinvestment_rate",
    "gini_coefficient",
    "transaction_entropy",
    "structural_depth",
]


def load_feature_names(path: str | Path = FEATURE_NAMES_PATH) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature names file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_population_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Population dataset not found: {p}")
    df = pd.read_csv(p)

    required = FEATURE_COLUMNS + ["label", "is_borderline"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Population dataset missing columns: {missing}")
    return df


def check_no_nan(df: pd.DataFrame, feature_names: list[str]) -> None:
    arr = df[feature_names].to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise ValueError("Population dataset contains NaN in feature columns.")
    if not np.isfinite(arr).all():
        raise ValueError("Population dataset contains Inf values in feature columns.")


def check_feature_order(feature_names: list[str]) -> bool:
    return feature_names == FEATURE_COLUMNS


def fit_population_pca(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, StandardScaler, PCA]:
    check_no_nan(df, feature_names)

    x = df[feature_names].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    projected = pd.DataFrame(
        {
            "PCA1": x_pca[:, 0],
            "PCA2": x_pca[:, 1],
            "label": df["label"].astype(int).to_numpy(),
            "is_borderline": df["is_borderline"].astype(bool).to_numpy(),
        }
    )
    return projected, scaler, pca


def project_current_case(
    features: dict[str, Any],
    feature_names: list[str],
    scaler: StandardScaler,
    pca: PCA,
) -> tuple[float, float]:
    row = np.array([[float(features[name]) for name in feature_names]], dtype=float)
    row_scaled = scaler.transform(row)
    row_pca = pca.transform(row_scaled)
    return float(row_pca[0, 0]), float(row_pca[0, 1])


def check_pca_dimensions(projected: pd.DataFrame, pca: PCA) -> None:
    if projected.shape[1] < 4:
        raise ValueError("Projected dataframe does not contain expected columns.")
    if pca.n_components_ != 2:
        raise ValueError(f"PCA has invalid n_components_: {pca.n_components_}")
    if "PCA1" not in projected.columns or "PCA2" not in projected.columns:
        raise ValueError("Projected dataframe must contain PCA1 and PCA2.")


def build_population_map_figure(
    projected: pd.DataFrame,
    current_point: tuple[float, float] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.8, 6.2))

    legit = projected[projected["label"] == 0]
    pyramid = projected[projected["label"] == 1]
    borderline = projected[projected["is_borderline"]]

    ax.scatter(
        legit["PCA1"],
        legit["PCA2"],
        s=22,
        alpha=0.45,
        color="#2563EB",
        label="legit",
        edgecolors="none",
    )
    ax.scatter(
        pyramid["PCA1"],
        pyramid["PCA2"],
        s=24,
        alpha=0.45,
        color="#DC2626",
        label="pyramid",
        edgecolors="none",
    )
    ax.scatter(
        borderline["PCA1"],
        borderline["PCA2"],
        s=34,
        alpha=0.72,
        color="#F59E0B",
        label="borderline",
        marker="^",
        edgecolors="white",
        linewidths=0.3,
    )

    if current_point is not None:
        ax.scatter(
            [current_point[0]],
            [current_point[1]],
            s=220,
            color="black",
            marker="X",
            linewidths=1.4,
            edgecolors="white",
            label="current_case",
            zorder=10,
        )

    ax.set_title("Population Risk Map")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    return fig

