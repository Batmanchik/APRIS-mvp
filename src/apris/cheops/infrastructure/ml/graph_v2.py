from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from apris.cheops.domain.models import CaseWindow
from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, SEED

GRAPH_V2_ARTIFACT_PATH = Path("artifacts") / "cheops_v2_graph.joblib"
GRAPH_V2_METRICS_PATH = Path("artifacts") / "cheops_v2_graph_metrics.json"
GRAPH_FEATURE_NAMES = [
    "graph_density",
    "graph_hub_share",
    "graph_component_compactness",
    "graph_transitivity",
    "graph_weight_cv_norm",
]

DEFAULT_GRAPH_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 180,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_samples": 8,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, *, bins: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    ece = 0.0
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (y_score >= low) & (y_score <= high)
        else:
            mask = (y_score >= low) & (y_score < high)
        if not np.any(mask):
            continue
        predicted = float(np.mean(y_score[mask]))
        observed = float(np.mean(y_true[mask]))
        ece += abs(observed - predicted) * (float(np.sum(mask)) / total)
    return float(ece)


def _fit_isotonic(y_true: pd.Series, y_score: np.ndarray) -> IsotonicRegression | None:
    uniques = np.unique(y_true.to_numpy())
    if len(uniques) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_score, y_true.to_numpy(dtype=float))
    return calibrator


def _apply_calibration(
    y_score: np.ndarray,
    calibrator: IsotonicRegression | None,
) -> np.ndarray:
    if calibrator is None:
        return np.clip(y_score, 0.0, 1.0)
    return np.clip(calibrator.predict(y_score), 0.0, 1.0)


def _normalize_feature(series: pd.Series, feature_name: str) -> pd.Series:
    low, high = FEATURE_BOUNDS[feature_name]
    span = high - low
    if span <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return ((series.astype(float) - low) / span).clip(lower=0.0, upper=1.0)


def build_graph_matrix_from_tabular(features_df: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in FEATURE_COLUMNS if name not in features_df.columns]
    if missing:
        raise ValueError(f"Missing tabular features for graph branch: {missing}")

    central = _normalize_feature(features_df["centralization_index"], "centralization_index")
    depth = _normalize_feature(features_df["structural_depth"], "structural_depth")
    gini = _normalize_feature(features_df["gini_coefficient"], "gini_coefficient")
    payout = _normalize_feature(features_df["payout_dependency"], "payout_dependency")
    entropy_low = 1.0 - _normalize_feature(features_df["transaction_entropy"], "transaction_entropy")
    referral = _normalize_feature(features_df["referral_ratio"], "referral_ratio")

    matrix = pd.DataFrame(
        {
            "graph_density": (0.05 + 0.95 * (0.38 * central + 0.34 * depth + 0.28 * entropy_low)),
            "graph_hub_share": (0.04 + 0.96 * (0.55 * central + 0.30 * gini + 0.15 * payout)),
            "graph_component_compactness": (
                0.06 + 0.94 * (0.42 * depth + 0.34 * referral + 0.24 * (1.0 - entropy_low))
            ),
            "graph_transitivity": (0.03 + 0.97 * (0.40 * central + 0.32 * depth + 0.28 * referral)),
            "graph_weight_cv_norm": (0.05 + 0.95 * (0.47 * gini + 0.31 * payout + 0.22 * central)),
        },
        index=features_df.index,
    )
    return matrix.clip(lower=0.0, upper=1.0)


def _validate_graph_matrix(graph_matrix: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in GRAPH_FEATURE_NAMES if name not in graph_matrix.columns]
    if missing:
        raise ValueError(f"Missing graph features: {missing}")
    matrix = graph_matrix[GRAPH_FEATURE_NAMES].astype(float).copy()
    if matrix.isna().any().any():
        raise ValueError("Graph feature matrix contains NaN values.")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Graph feature matrix contains non-finite values.")
    return matrix.clip(lower=0.0, upper=1.0)


def _validate_graph_artifact(artifact: dict[str, Any]) -> tuple[Any, IsotonicRegression | None]:
    if artifact.get("feature_names") != GRAPH_FEATURE_NAMES:
        raise ValueError("Cheops graph artifact has unexpected feature_names.")
    model = artifact.get("model")
    if model is None:
        raise ValueError("Cheops graph artifact missing model.")
    calibrator = artifact.get("calibrator")
    if calibrator is not None and not isinstance(calibrator, IsotonicRegression):
        raise ValueError("Cheops graph artifact calibrator has invalid type.")
    return model, calibrator


def predict_graph_probabilities(
    graph_matrix: pd.DataFrame,
    artifact: dict[str, Any],
) -> np.ndarray:
    matrix = _validate_graph_matrix(graph_matrix)
    model, calibrator = _validate_graph_artifact(artifact)
    raw = np.asarray(model.predict_proba(matrix), dtype=float)[:, 1]
    return _apply_calibration(raw, calibrator)


def train_graph_artifact(
    df: pd.DataFrame,
    *,
    random_state: int = SEED,
    model_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_tabular = df[FEATURE_COLUMNS].astype(float).copy()
    y = df["label"].astype(int).copy()
    graph_matrix = build_graph_matrix_from_tabular(x_tabular)

    x_train, x_temp, y_train, y_temp = train_test_split(
        graph_matrix,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    params = dict(DEFAULT_GRAPH_MODEL_PARAMS)
    params["random_state"] = random_state
    if model_params:
        params.update(model_params)

    model = lgb.LGBMClassifier(**params)
    model.fit(x_train[GRAPH_FEATURE_NAMES], y_train.to_numpy(dtype=int))

    val_raw = np.asarray(model.predict_proba(x_val[GRAPH_FEATURE_NAMES]), dtype=float)[:, 1]
    calibrator = _fit_isotonic(y_val, val_raw)

    test_raw = np.asarray(model.predict_proba(x_test[GRAPH_FEATURE_NAMES]), dtype=float)[:, 1]
    test_cal = _apply_calibration(test_raw, calibrator)
    test_pred = (test_cal >= 0.5).astype(int)

    heuristic = (
        0.34 * x_test["graph_hub_share"].to_numpy(dtype=float)
        + 0.30 * x_test["graph_density"].to_numpy(dtype=float)
        + 0.18 * x_test["graph_component_compactness"].to_numpy(dtype=float)
        + 0.10 * x_test["graph_transitivity"].to_numpy(dtype=float)
        + 0.08 * x_test["graph_weight_cv_norm"].to_numpy(dtype=float)
    )
    y_test_float = y_test.to_numpy(dtype=float)

    metrics = {
        "artifact_version": "cheops-graph-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "splits": {
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
        },
        "graph_head": {
            "roc_auc": _safe_roc_auc(y_test_float, test_cal),
            "accuracy": float(accuracy_score(y_test.to_numpy(dtype=int), test_pred)),
            "brier": float(brier_score_loss(y_test_float, test_cal)),
            "ece": _expected_calibration_error(y_test_float, test_cal),
        },
        "heuristic_fallback": {
            "roc_auc": _safe_roc_auc(y_test_float, heuristic),
            "brier": float(brier_score_loss(y_test_float, heuristic)),
            "ece": _expected_calibration_error(y_test_float, heuristic),
        },
    }

    artifact = {
        "artifact_version": "cheops-graph-v2",
        "feature_names": list(GRAPH_FEATURE_NAMES),
        "model": model,
        "calibrator": calibrator,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_state": random_state,
        },
    }
    return artifact, metrics


def save_graph_artifact(
    artifact: dict[str, Any],
    path: str | Path = GRAPH_V2_ARTIFACT_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target)
    return target


def save_graph_metrics(
    metrics: dict[str, Any],
    path: str | Path = GRAPH_V2_METRICS_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return target


def load_graph_artifact(path: str | Path = GRAPH_V2_ARTIFACT_PATH) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Cheops graph artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Cheops graph artifact has invalid format.")
    _validate_graph_artifact(artifact)
    return artifact


def extract_graph_features_from_case_window(case_window: CaseWindow) -> dict[str, float]:
    graph = nx.DiGraph()
    for event in case_window.events:
        graph.add_edge(event.sender_id, event.receiver_id, weight=float(event.amount))

    if graph.number_of_nodes() == 0:
        return {name: 0.0 for name in GRAPH_FEATURE_NAMES}

    density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0

    in_degrees = dict(graph.in_degree())
    max_in = max(in_degrees.values()) if in_degrees else 0.0
    sum_in = float(sum(in_degrees.values()))
    hub_share = 0.0 if sum_in <= 0 else max_in / sum_in

    components = nx.number_weakly_connected_components(graph)
    compactness = 1.0 / float(max(components, 1))

    undirected = graph.to_undirected()
    if undirected.number_of_nodes() >= 3:
        transitivity = float(nx.transitivity(undirected))
    else:
        transitivity = 0.0

    weights = np.asarray([float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)], dtype=float)
    mean_weight = float(np.mean(weights)) if weights.size > 0 else 0.0
    std_weight = float(np.std(weights, ddof=0)) if weights.size > 0 else 0.0
    weight_cv = 0.0 if mean_weight <= 0 else std_weight / mean_weight
    weight_cv_norm = min(weight_cv / 2.0, 1.0)

    return {
        "graph_density": _clip01(density),
        "graph_hub_share": _clip01(hub_share),
        "graph_component_compactness": _clip01(compactness),
        "graph_transitivity": _clip01(transitivity),
        "graph_weight_cv_norm": _clip01(weight_cv_norm),
    }


def predict_graph_from_case_window(case_window: CaseWindow, artifact: dict[str, Any]) -> float:
    features = extract_graph_features_from_case_window(case_window)
    matrix = pd.DataFrame([features], columns=GRAPH_FEATURE_NAMES)
    score = float(predict_graph_probabilities(matrix, artifact)[0])
    return _clip01(score)


def heuristic_graph_from_case_window(case_window: CaseWindow) -> float:
    features = extract_graph_features_from_case_window(case_window)
    raw = (
        0.34 * features["graph_hub_share"]
        + 0.30 * features["graph_density"]
        + 0.18 * features["graph_component_compactness"]
        + 0.10 * features["graph_transitivity"]
        + 0.08 * features["graph_weight_cv_norm"]
    )
    return _clip01(raw)
