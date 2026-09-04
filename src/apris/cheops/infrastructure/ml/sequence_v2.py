from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from apris.cheops.domain.models import CaseWindow
from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, SEED

SEQUENCE_V2_ARTIFACT_PATH = Path("artifacts") / "cheops_v2_sequence.joblib"
SEQUENCE_V2_METRICS_PATH = Path("artifacts") / "cheops_v2_sequence_metrics.json"
SEQUENCE_FEATURE_NAMES = [
    "event_rate_hour",
    "burst_ratio_90s",
    "median_delta_inverse",
    "amount_cv_norm",
    "unique_sender_ratio",
]

DEFAULT_SEQUENCE_MODEL_PARAMS: dict[str, Any] = {
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


def build_sequence_matrix_from_tabular(features_df: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in FEATURE_COLUMNS if name not in features_df.columns]
    if missing:
        raise ValueError(f"Missing tabular features for sequence branch: {missing}")

    growth = _normalize_feature(features_df["growth_rate"], "growth_rate")
    referral = _normalize_feature(features_df["referral_ratio"], "referral_ratio")
    holding_short = 1.0 - _normalize_feature(features_df["avg_holding_time"], "avg_holding_time")
    entropy_low = 1.0 - _normalize_feature(features_df["transaction_entropy"], "transaction_entropy")
    depth = _normalize_feature(features_df["structural_depth"], "structural_depth")
    central = _normalize_feature(features_df["centralization_index"], "centralization_index")
    payout = _normalize_feature(features_df["payout_dependency"], "payout_dependency")
    gini = _normalize_feature(features_df["gini_coefficient"], "gini_coefficient")
    reinvest = _normalize_feature(features_df["reinvestment_rate"], "reinvestment_rate")

    sequence_matrix = pd.DataFrame(
        {
            "event_rate_hour": (0.08 + 0.92 * (0.42 * growth + 0.19 * referral + 0.19 * depth + 0.20 * reinvest)),
            "burst_ratio_90s": (0.05 + 0.95 * (0.46 * growth + 0.31 * holding_short + 0.23 * entropy_low)),
            "median_delta_inverse": (0.06 + 0.94 * (0.44 * holding_short + 0.30 * entropy_low + 0.26 * growth)),
            "amount_cv_norm": (0.05 + 0.95 * (0.37 * central + 0.34 * payout + 0.29 * gini)),
            "unique_sender_ratio": (
                0.04 + 0.96 * (0.43 * referral + 0.30 * depth + 0.27 * (1.0 - central))
            ),
        },
        index=features_df.index,
    )
    return sequence_matrix.clip(lower=0.0, upper=1.0)


def _validate_sequence_matrix(sequence_matrix: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in SEQUENCE_FEATURE_NAMES if name not in sequence_matrix.columns]
    if missing:
        raise ValueError(f"Missing sequence features: {missing}")
    matrix = sequence_matrix[SEQUENCE_FEATURE_NAMES].astype(float).copy()
    if matrix.isna().any().any():
        raise ValueError("Sequence feature matrix contains NaN values.")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Sequence feature matrix contains non-finite values.")
    return matrix.clip(lower=0.0, upper=1.0)


def _validate_sequence_artifact(artifact: dict[str, Any]) -> tuple[Any, IsotonicRegression | None]:
    if artifact.get("feature_names") != SEQUENCE_FEATURE_NAMES:
        raise ValueError("Cheops sequence artifact has unexpected feature_names.")
    model = artifact.get("model")
    if model is None:
        raise ValueError("Cheops sequence artifact missing model.")
    calibrator = artifact.get("calibrator")
    if calibrator is not None and not isinstance(calibrator, IsotonicRegression):
        raise ValueError("Cheops sequence artifact calibrator has invalid type.")
    return model, calibrator


def predict_sequence_probabilities(
    sequence_matrix: pd.DataFrame,
    artifact: dict[str, Any],
) -> np.ndarray:
    matrix = _validate_sequence_matrix(sequence_matrix)
    model, calibrator = _validate_sequence_artifact(artifact)
    raw = np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)
    return _apply_calibration(raw, calibrator)


def train_sequence_artifact(
    df: pd.DataFrame,
    *,
    random_state: int = SEED,
    model_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_tabular = df[FEATURE_COLUMNS].astype(float).copy()
    y = df["label"].astype(int).copy()
    sequence_matrix = build_sequence_matrix_from_tabular(x_tabular)

    x_train, x_temp, y_train, y_temp = train_test_split(
        sequence_matrix,
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

    params = dict(DEFAULT_SEQUENCE_MODEL_PARAMS)
    params["random_state"] = random_state
    if model_params:
        params.update(model_params)

    model = lgb.LGBMClassifier(**params)
    model.fit(x_train[SEQUENCE_FEATURE_NAMES], y_train.to_numpy(dtype=int))

    val_raw = np.asarray(model.predict_proba(x_val[SEQUENCE_FEATURE_NAMES])[:, 1], dtype=float)
    calibrator = _fit_isotonic(y_val, val_raw)

    test_raw = np.asarray(model.predict_proba(x_test[SEQUENCE_FEATURE_NAMES])[:, 1], dtype=float)
    test_cal = _apply_calibration(test_raw, calibrator)
    test_pred = (test_cal >= 0.5).astype(int)

    heuristic = (
        0.39 * x_test["event_rate_hour"].to_numpy(dtype=float)
        + 0.29 * x_test["burst_ratio_90s"].to_numpy(dtype=float)
        + 0.20 * x_test["median_delta_inverse"].to_numpy(dtype=float)
        + 0.07 * x_test["amount_cv_norm"].to_numpy(dtype=float)
        + 0.05 * x_test["unique_sender_ratio"].to_numpy(dtype=float)
    )
    y_test_float = y_test.to_numpy(dtype=float)

    metrics = {
        "artifact_version": "cheops-sequence-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "splits": {
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
        },
        "sequence_head": {
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
        "artifact_version": "cheops-sequence-v2",
        "feature_names": list(SEQUENCE_FEATURE_NAMES),
        "model": model,
        "calibrator": calibrator,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_state": random_state,
        },
    }
    return artifact, metrics


def save_sequence_artifact(
    artifact: dict[str, Any],
    path: str | Path = SEQUENCE_V2_ARTIFACT_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, target)
    return target


def save_sequence_metrics(
    metrics: dict[str, Any],
    path: str | Path = SEQUENCE_V2_METRICS_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return target


def load_sequence_artifact(path: str | Path = SEQUENCE_V2_ARTIFACT_PATH) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Cheops sequence artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("Cheops sequence artifact has invalid format.")
    _validate_sequence_artifact(artifact)
    return artifact


def extract_sequence_features_from_case_window(case_window: CaseWindow) -> dict[str, float]:
    events = case_window.events
    if len(events) == 0:
        raise ValueError("Case window contains no events.")

    span_seconds = max((case_window.end_ts - case_window.start_ts).total_seconds(), 60.0)
    span_hours = span_seconds / 3600.0
    rate = len(events) / span_hours
    event_rate_hour = 1.0 - math.exp(-rate / 4.0)

    jumps = 0
    deltas: list[float] = []
    for idx in range(1, len(events)):
        delta_sec = max((events[idx].ts - events[idx - 1].ts).total_seconds(), 0.0)
        deltas.append(delta_sec)
        if delta_sec <= 90.0:
            jumps += 1
    burst_ratio_90s = jumps / max(len(events) - 1, 1)

    if deltas:
        median_delta_minutes = float(np.median(np.asarray(deltas, dtype=float)) / 60.0)
    else:
        median_delta_minutes = 60.0
    median_delta_inverse = 1.0 / (1.0 + (median_delta_minutes / 60.0))

    amounts = np.asarray([float(event.amount) for event in events], dtype=float)
    mean_amount = float(np.mean(amounts)) if amounts.size > 0 else 0.0
    std_amount = float(np.std(amounts, ddof=0)) if amounts.size > 0 else 0.0
    amount_cv = 0.0 if mean_amount <= 0 else std_amount / mean_amount
    amount_cv_norm = min(amount_cv / 2.0, 1.0)

    unique_sender_ratio = len({event.sender_id for event in events}) / max(len(events), 1)

    return {
        "event_rate_hour": _clip01(event_rate_hour),
        "burst_ratio_90s": _clip01(burst_ratio_90s),
        "median_delta_inverse": _clip01(median_delta_inverse),
        "amount_cv_norm": _clip01(amount_cv_norm),
        "unique_sender_ratio": _clip01(unique_sender_ratio),
    }


def predict_sequence_from_case_window(case_window: CaseWindow, artifact: dict[str, Any]) -> float:
    features = extract_sequence_features_from_case_window(case_window)
    matrix = pd.DataFrame([features], columns=SEQUENCE_FEATURE_NAMES)
    score = float(predict_sequence_probabilities(matrix, artifact)[0])
    return _clip01(score)


def heuristic_sequence_from_case_window(case_window: CaseWindow) -> float:
    features = extract_sequence_features_from_case_window(case_window)
    raw = (
        0.39 * features["event_rate_hour"]
        + 0.29 * features["burst_ratio_90s"]
        + 0.20 * features["median_delta_inverse"]
        + 0.07 * features["amount_cv_norm"]
        + 0.05 * features["unique_sender_ratio"]
    )
    return _clip01(raw)
