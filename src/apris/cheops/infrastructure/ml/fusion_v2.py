from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from apris.cheops.infrastructure.ml.graph_v2 import (
    build_graph_matrix_from_tabular,
    predict_graph_probabilities,
)
from apris.cheops.infrastructure.ml.sequence_v2 import (
    build_sequence_matrix_from_tabular,
    predict_sequence_probabilities,
)
from apris.cheops.infrastructure.ml.tabular_v2 import predict_tabular_bundle
from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, SEED

FUSION_V2_ARTIFACT_PATH = Path("artifacts") / "cheops_v2_fusion_meta.joblib"
FUSION_V2_METRICS_PATH = Path("artifacts") / "cheops_v2_fusion_metrics.json"
FUSION_FEATURE_NAMES = ["tabular_prob", "sequence_prob", "graph_prob"]


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


def _proxy_sequence_prob(features_df: pd.DataFrame) -> np.ndarray:
    sequence_matrix = build_sequence_matrix_from_tabular(features_df)
    raw = (
        0.39 * sequence_matrix["event_rate_hour"].to_numpy(dtype=float)
        + 0.29 * sequence_matrix["burst_ratio_90s"].to_numpy(dtype=float)
        + 0.20 * sequence_matrix["median_delta_inverse"].to_numpy(dtype=float)
        + 0.07 * sequence_matrix["amount_cv_norm"].to_numpy(dtype=float)
        + 0.05 * sequence_matrix["unique_sender_ratio"].to_numpy(dtype=float)
    )
    return np.clip(raw, 0.0, 1.0)


def _proxy_graph_prob(features_df: pd.DataFrame) -> np.ndarray:
    central = _normalize_feature(features_df["centralization_index"], "centralization_index")
    depth = _normalize_feature(features_df["structural_depth"], "structural_depth")
    gini = _normalize_feature(features_df["gini_coefficient"], "gini_coefficient")
    payout = _normalize_feature(features_df["payout_dependency"], "payout_dependency")
    raw = 0.36 * central + 0.28 * depth + 0.22 * gini + 0.14 * payout
    return np.clip(raw.to_numpy(dtype=float), 0.0, 1.0)


def _tabular_probs(features_df: pd.DataFrame, tabular_bundle: dict[str, Any]) -> np.ndarray:
    probs: list[float] = []
    for _, row in features_df.iterrows():
        features = {name: float(row[name]) for name in FEATURE_COLUMNS}
        score, _ = predict_tabular_bundle(features, tabular_bundle)
        probs.append(_clip01(score))
    return np.asarray(probs, dtype=float)


def _build_fusion_matrix(
    features_df: pd.DataFrame,
    tabular_bundle: dict[str, Any],
    *,
    sequence_artifact: dict[str, Any] | None = None,
    graph_artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    tabular_prob = _tabular_probs(features_df, tabular_bundle)
    if sequence_artifact is None:
        sequence_prob = _proxy_sequence_prob(features_df)
    else:
        sequence_matrix = build_sequence_matrix_from_tabular(features_df)
        sequence_prob = np.asarray(
            predict_sequence_probabilities(sequence_matrix, sequence_artifact),
            dtype=float,
        )
    if graph_artifact is None:
        graph_prob = _proxy_graph_prob(features_df)
    else:
        graph_matrix = build_graph_matrix_from_tabular(features_df)
        graph_prob = np.asarray(
            predict_graph_probabilities(graph_matrix, graph_artifact),
            dtype=float,
        )
    return pd.DataFrame(
        {
            "tabular_prob": tabular_prob,
            "sequence_prob": sequence_prob,
            "graph_prob": graph_prob,
        },
        index=features_df.index,
    )


def train_fusion_meta(
    df: pd.DataFrame,
    tabular_bundle: dict[str, Any],
    *,
    random_state: int = SEED,
    sequence_artifact: dict[str, Any] | None = None,
    graph_artifact: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x = df[FEATURE_COLUMNS].astype(float).copy()
    y = df["label"].astype(int).copy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
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

    meta_train = _build_fusion_matrix(
        x_train,
        tabular_bundle,
        sequence_artifact=sequence_artifact,
        graph_artifact=graph_artifact,
    )
    meta_val = _build_fusion_matrix(
        x_val,
        tabular_bundle,
        sequence_artifact=sequence_artifact,
        graph_artifact=graph_artifact,
    )
    meta_test = _build_fusion_matrix(
        x_test,
        tabular_bundle,
        sequence_artifact=sequence_artifact,
        graph_artifact=graph_artifact,
    )

    meta_model = LogisticRegression(
        random_state=random_state,
        max_iter=400,
        class_weight="balanced",
    )
    meta_model.fit(meta_train[FUSION_FEATURE_NAMES], y_train.to_numpy(dtype=int))

    val_raw = np.asarray(meta_model.predict_proba(meta_val[FUSION_FEATURE_NAMES]), dtype=float)[:, 1]
    calibrator = _fit_isotonic(y_val, val_raw)

    test_raw = np.asarray(meta_model.predict_proba(meta_test[FUSION_FEATURE_NAMES]), dtype=float)[:, 1]
    test_cal = _apply_calibration(test_raw, calibrator)
    test_pred = (test_cal >= 0.5).astype(int)

    y_test_float = y_test.to_numpy(dtype=float)
    fallback_weighted = (
        0.58 * meta_test["tabular_prob"].to_numpy(dtype=float)
        + 0.22 * meta_test["sequence_prob"].to_numpy(dtype=float)
        + 0.20 * meta_test["graph_prob"].to_numpy(dtype=float)
    )

    metrics = {
        "artifact_version": "cheops-fusion-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "splits": {
            "train_rows": int(len(meta_train)),
            "val_rows": int(len(meta_val)),
            "test_rows": int(len(meta_test)),
        },
        "sequence_source": "trained_sequence_v2" if sequence_artifact is not None else "proxy",
        "graph_source": "trained_graph_v2" if graph_artifact is not None else "proxy",
        "meta_head": {
            "roc_auc": _safe_roc_auc(y_test_float, test_cal),
            "accuracy": float(accuracy_score(y_test.to_numpy(dtype=int), test_pred)),
            "brier": float(brier_score_loss(y_test_float, test_cal)),
            "ece": _expected_calibration_error(y_test_float, test_cal),
        },
        "fallback_weighted": {
            "roc_auc": _safe_roc_auc(y_test_float, fallback_weighted),
            "brier": float(brier_score_loss(y_test_float, fallback_weighted)),
            "ece": _expected_calibration_error(y_test_float, fallback_weighted),
        },
    }

    artifact = {
        "artifact_version": "cheops-fusion-v2",
        "feature_names": list(FUSION_FEATURE_NAMES),
        "meta_model": meta_model,
        "calibrator": calibrator,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_state": random_state,
        },
    }
    return artifact, metrics


def save_fusion_artifact(
    artifact: dict[str, Any],
    path: str | Path = FUSION_V2_ARTIFACT_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target)
    return target


def save_fusion_metrics(
    metrics: dict[str, Any],
    path: str | Path = FUSION_V2_METRICS_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return target


def load_fusion_artifact(path: str | Path = FUSION_V2_ARTIFACT_PATH) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Cheops fusion artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Cheops fusion artifact has invalid format.")
    feature_names = artifact.get("feature_names")
    if feature_names != FUSION_FEATURE_NAMES:
        raise ValueError("Cheops fusion artifact has unexpected feature_names.")
    if "meta_model" not in artifact:
        raise ValueError("Cheops fusion artifact missing meta_model.")
    return artifact


def _to_finite_probability(name: str, value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return _clip01(numeric)


def predict_fusion_meta(
    tabular_prob: float,
    sequence_prob: float,
    graph_prob: float,
    artifact: dict[str, Any],
) -> float:
    feature_names = artifact.get("feature_names")
    if feature_names != FUSION_FEATURE_NAMES:
        raise ValueError("Cheops fusion artifact has unexpected feature_names.")

    meta_model = artifact.get("meta_model")
    if meta_model is None:
        raise ValueError("Cheops fusion artifact missing meta_model.")

    calibrator = artifact.get("calibrator")
    payload = {
        "tabular_prob": _to_finite_probability("tabular_prob", tabular_prob),
        "sequence_prob": _to_finite_probability("sequence_prob", sequence_prob),
        "graph_prob": _to_finite_probability("graph_prob", graph_prob),
    }
    x = pd.DataFrame([[payload[name] for name in FUSION_FEATURE_NAMES]], columns=FUSION_FEATURE_NAMES)
    raw = float(meta_model.predict_proba(x)[0, 1])
    calibrated = float(_apply_calibration(np.array([raw], dtype=float), calibrator)[0])
    return _clip01(calibrated)
