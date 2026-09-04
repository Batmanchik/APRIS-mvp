from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from apris.cheops.domain.typologies import FraudTypology, TYPOLOGY_NAMES
from apris.data_generator import FEATURE_COLUMNS, SEED

TABULAR_V2_BUNDLE_PATH = Path("artifacts") / "cheops_v2_tabular.joblib"
TABULAR_V2_METRICS_PATH = Path("artifacts") / "cheops_v2_metrics.json"

DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 220,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_samples": 8,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    uniques = np.unique(y_true.to_numpy())
    if len(uniques) < 2:
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
        prob_mean = float(np.mean(y_score[mask]))
        true_mean = float(np.mean(y_true[mask]))
        ece += abs(true_mean - prob_mean) * (float(np.sum(mask)) / total)
    return float(ece)


def _binarize_with_min_positives(
    mask: pd.Series,
    fallback_signal: pd.Series,
    *,
    min_positive_rate: float = 0.10,
) -> pd.Series:
    base = mask.astype(int)
    if float(base.mean()) >= min_positive_rate:
        return base

    candidate_idx = fallback_signal.sort_values(ascending=False).index
    min_pos = max(1, int(round(len(base) * min_positive_rate)))
    boosted = base.copy()
    boosted.loc[candidate_idx[:min_pos]] = 1
    return boosted.astype(int)


def derive_typology_targets(features: pd.DataFrame, global_label: pd.Series) -> pd.DataFrame:
    """Create deterministic multi-label typology targets from tabular risk signals."""
    growth = features["growth_rate"].astype(float)
    referral = features["referral_ratio"].astype(float)
    payout = features["payout_dependency"].astype(float)
    central = features["centralization_index"].astype(float)
    holding = features["avg_holding_time"].astype(float)
    reinvest = features["reinvestment_rate"].astype(float)
    gini = features["gini_coefficient"].astype(float)
    entropy = features["transaction_entropy"].astype(float)
    depth = features["structural_depth"].astype(float)
    high_risk = global_label.astype(float)

    legal_layering_raw = ((depth >= 8.0) & (central >= 0.45)) | ((gini >= 0.62) & (payout >= 0.85))
    bridge_raw = ((payout >= 0.92) & (entropy <= 2.35)) | ((growth >= 0.23) & (holding <= 40.0))
    mixing_raw = ((entropy <= 2.2) & (reinvest >= 0.56)) | ((central >= 0.58) & (depth >= 7.0))
    splitting_raw = ((referral >= 0.52) & (depth >= 7.0)) | ((growth >= 0.28) & (entropy <= 2.6))
    cash_out_raw = ((payout >= 1.00) & (holding <= 32.0)) | ((central >= 0.56) & (gini >= 0.68))

    fallback = 0.55 * high_risk + 0.25 * payout + 0.20 * central
    targets = pd.DataFrame(
        {
            FraudTypology.LEGAL_LAYERING.value: _binarize_with_min_positives(
                legal_layering_raw, fallback + 0.10 * depth
            ),
            FraudTypology.LEGAL_TO_CRYPTO_BRIDGE.value: _binarize_with_min_positives(
                bridge_raw, fallback + 0.10 * growth
            ),
            FraudTypology.CRYPTO_MIXING.value: _binarize_with_min_positives(
                mixing_raw, fallback + 0.10 * reinvest
            ),
            FraudTypology.STRUCTURED_SPLITTING.value: _binarize_with_min_positives(
                splitting_raw, fallback + 0.10 * referral
            ),
            FraudTypology.CASH_OUT.value: _binarize_with_min_positives(
                cash_out_raw, fallback + 0.10 * gini
            ),
        }
    )
    return targets[sorted(TYPOLOGY_NAMES)]


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


def train_tabular_bundle(
    df: pd.DataFrame,
    *,
    random_state: int = SEED,
    model_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train tabular v2 branch: global risk + typology probabilities."""
    x = df[FEATURE_COLUMNS].astype(float).copy()
    y_global = df["label"].astype(int).copy()
    y_typologies = derive_typology_targets(x, y_global)

    x_train, x_temp, y_train, y_temp, y_typ_train, y_typ_temp = train_test_split(
        x,
        y_global,
        y_typologies,
        test_size=0.30,
        random_state=random_state,
        stratify=y_global,
    )
    x_val, x_test, y_val, y_test, y_typ_val, y_typ_test = train_test_split(
        x_temp,
        y_temp,
        y_typ_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    params = dict(DEFAULT_MODEL_PARAMS)
    params["random_state"] = random_state
    if model_params:
        params.update(model_params)

    global_model = lgb.LGBMClassifier(**params)
    global_model.fit(x_train, y_train)
    global_val_raw = np.asarray(global_model.predict_proba(x_val), dtype=float)[:, 1]
    global_calibrator = _fit_isotonic(y_val, global_val_raw)
    global_test_raw = np.asarray(global_model.predict_proba(x_test), dtype=float)[:, 1]
    global_test_cal = _apply_calibration(global_test_raw, global_calibrator)

    typology_models: dict[str, Any] = {}
    typology_calibrators: dict[str, IsotonicRegression | None] = {}
    typology_test_scores: dict[str, np.ndarray] = {}

    for typology_name in sorted(TYPOLOGY_NAMES):
        y_train_bin = y_typ_train[typology_name].astype(int)
        y_val_bin = y_typ_val[typology_name].astype(int)

        model = lgb.LGBMClassifier(**params)
        model.fit(x_train, y_train_bin)

        val_raw = np.asarray(model.predict_proba(x_val), dtype=float)[:, 1]
        calibrator = _fit_isotonic(y_val_bin, val_raw)
        test_raw = np.asarray(model.predict_proba(x_test), dtype=float)[:, 1]
        test_cal = _apply_calibration(test_raw, calibrator)

        typology_models[typology_name] = model
        typology_calibrators[typology_name] = calibrator
        typology_test_scores[typology_name] = test_cal

    metrics: dict[str, Any] = {
        "bundle_version": "cheops-tabular-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "splits": {
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
        },
        "global": {
            "roc_auc": _safe_roc_auc(y_test, global_test_cal),
            "brier": float(brier_score_loss(y_test.to_numpy(dtype=float), global_test_cal)),
            "ece": _expected_calibration_error(y_test.to_numpy(dtype=float), global_test_cal),
        },
        "typologies": {},
    }

    for typology_name in sorted(TYPOLOGY_NAMES):
        y_true = y_typ_test[typology_name].astype(int)
        y_score = typology_test_scores[typology_name]
        metrics["typologies"][typology_name] = {
            "positive_rate": float(y_true.mean()),
            "roc_auc": _safe_roc_auc(y_true, y_score),
            "brier": float(brier_score_loss(y_true.to_numpy(dtype=float), y_score)),
            "ece": _expected_calibration_error(y_true.to_numpy(dtype=float), y_score),
        }

    bundle = {
        "bundle_version": "cheops-tabular-v2",
        "feature_names": list(FEATURE_COLUMNS),
        "typologies": sorted(TYPOLOGY_NAMES),
        "global_model": global_model,
        "global_calibrator": global_calibrator,
        "typology_models": typology_models,
        "typology_calibrators": typology_calibrators,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_state": random_state,
        },
    }
    return bundle, metrics


def save_tabular_bundle(bundle: dict[str, Any], path: str | Path = TABULAR_V2_BUNDLE_PATH) -> Path:
    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    return bundle_path


def save_tabular_metrics(metrics: dict[str, Any], path: str | Path = TABULAR_V2_METRICS_PATH) -> Path:
    metrics_path = Path(path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def load_tabular_bundle(path: str | Path = TABULAR_V2_BUNDLE_PATH) -> dict[str, Any]:
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Cheops tabular bundle not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError("Cheops tabular bundle has invalid format.")
    return bundle


def _validated_feature_names(bundle: dict[str, Any]) -> list[str]:
    feature_names_raw = bundle.get("feature_names")
    if not isinstance(feature_names_raw, list) or len(feature_names_raw) == 0:
        raise ValueError("Cheops tabular bundle is missing non-empty 'feature_names'.")
    return [str(name) for name in feature_names_raw]


def _build_prediction_row(features: dict[str, float], feature_names: list[str]) -> dict[str, float]:
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"Missing tabular features: {missing}")

    row: dict[str, float] = {}
    for name in feature_names:
        raw_value = features[name]
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{name}' must be numeric.") from exc
        if not np.isfinite(value):
            raise ValueError(f"Feature '{name}' must be finite.")
        row[name] = value
    return row


def predict_tabular_bundle(
    features: dict[str, float],
    bundle: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    feature_names = _validated_feature_names(bundle)
    row = _build_prediction_row(features, feature_names)
    x = pd.DataFrame([row], columns=feature_names)

    global_model = bundle["global_model"]
    global_calibrator = bundle.get("global_calibrator")
    global_raw = float(global_model.predict_proba(x)[0, 1])
    global_score = float(_apply_calibration(np.array([global_raw], dtype=float), global_calibrator)[0])

    typology_models: dict[str, Any] = bundle["typology_models"]
    typology_calibrators: dict[str, IsotonicRegression | None] = bundle["typology_calibrators"]
    missing_typology_models = [name for name in sorted(TYPOLOGY_NAMES) if name not in typology_models]
    if missing_typology_models:
        raise ValueError(f"Cheops tabular bundle missing typology models: {missing_typology_models}")
    typology_scores: dict[str, float] = {}

    for typology_name in sorted(TYPOLOGY_NAMES):
        model = typology_models[typology_name]
        calibrator = typology_calibrators.get(typology_name)
        raw = float(model.predict_proba(x)[0, 1])
        calibrated = float(_apply_calibration(np.array([raw], dtype=float), calibrator)[0])
        typology_scores[typology_name] = _clip01(calibrated)

    return _clip01(global_score), typology_scores
