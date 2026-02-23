from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, RISK_THRESHOLDS


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"

OPERATIONAL_INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "tx_count_total": (10.0, 2_000_000.0),
    "unique_counterparties": (2.0, 300_000.0),
    "new_clients_current": (1.0, 1_000_000.0),
    "new_clients_previous": (1.0, 1_000_000.0),
    "referred_clients_current": (0.0, 1_000_000.0),
    "incoming_funds": (1.0, 10_000_000_000.0),
    "payouts_total": (0.0, 10_000_000_000.0),
    "top1_wallet_share": (0.0, 1.0),
    "top10_wallet_share": (0.0, 1.0),
    "avg_holding_days": (1.0, 365.0),
    "repeat_investor_share": (0.0, 1.0),
    "max_referral_depth": (1.0, 30.0),
}


def load_artifacts(
    model_path: str | Path = MODEL_PATH,
    feature_names_path: str | Path = FEATURE_NAMES_PATH,
) -> tuple[Any, list[str]]:
    model_file = Path(model_path)
    feature_file = Path(feature_names_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature names not found: {feature_file}")

    model = joblib.load(model_file)
    feature_names = json.loads(feature_file.read_text(encoding="utf-8"))
    return model, feature_names


def _validate_inputs(features: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    prepared: dict[str, float] = {}
    for name in feature_names:
        try:
            value = float(features[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{name}' must be numeric.") from exc

        low, high = FEATURE_BOUNDS[name]
        if not (low <= value <= high):
            raise ValueError(f"Feature '{name}' out of range [{low}, {high}]: {value}")
        prepared[name] = value

    return pd.DataFrame([prepared], columns=feature_names)


def _risk_label(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def _clip_feature(name: str, value: float) -> float:
    low, high = FEATURE_BOUNDS[name]
    return float(max(low, min(high, value)))


def _validate_operational_inputs(raw: dict[str, Any]) -> dict[str, float]:
    missing = [name for name in OPERATIONAL_INPUT_BOUNDS if name not in raw]
    if missing:
        raise ValueError(f"Missing operational fields: {missing}")

    validated: dict[str, float] = {}
    for key, (low, high) in OPERATIONAL_INPUT_BOUNDS.items():
        try:
            value = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Operational field '{key}' must be numeric.") from exc
        if not (low <= value <= high):
            raise ValueError(f"Operational field '{key}' out of range [{low}, {high}]: {value}")
        validated[key] = value
    if validated["referred_clients_current"] > validated["new_clients_current"]:
        raise ValueError("referred_clients_current cannot exceed new_clients_current.")
    if validated["top1_wallet_share"] > validated["top10_wallet_share"]:
        raise ValueError("top1_wallet_share cannot exceed top10_wallet_share.")
    return validated


def operational_to_features(raw: dict[str, Any]) -> dict[str, float]:
    data = _validate_operational_inputs(raw)

    # Growth as relative change of new clients vs previous period.
    growth_rate = (data["new_clients_current"] - data["new_clients_previous"]) / max(
        data["new_clients_previous"], 1.0
    )
    referral_ratio = data["referred_clients_current"] / max(data["new_clients_current"], 1.0)
    payout_dependency = data["payouts_total"] / max(data["incoming_funds"], 1.0)
    centralization_index = data["top1_wallet_share"]
    avg_holding_time = data["avg_holding_days"]
    reinvestment_rate = data["repeat_investor_share"]

    # Approximate inequality from concentration profile.
    gini_est = 0.12 + 0.72 * data["top10_wallet_share"] + 0.22 * data["top1_wallet_share"]

    # Approximate entropy from counterpart diversity and concentration.
    entropy_ratio = math.log1p(data["unique_counterparties"]) / math.log1p(
        max(data["tx_count_total"], data["unique_counterparties"] + 1.0)
    )
    entropy_est = 0.3 + 4.7 * entropy_ratio * (1.0 - 0.55 * data["top1_wallet_share"])

    structural_depth = data["max_referral_depth"]

    features = {
        "growth_rate": _clip_feature("growth_rate", growth_rate),
        "referral_ratio": _clip_feature("referral_ratio", referral_ratio),
        "payout_dependency": _clip_feature("payout_dependency", payout_dependency),
        "centralization_index": _clip_feature("centralization_index", centralization_index),
        "avg_holding_time": _clip_feature("avg_holding_time", avg_holding_time),
        "reinvestment_rate": _clip_feature("reinvestment_rate", reinvestment_rate),
        "gini_coefficient": _clip_feature("gini_coefficient", gini_est),
        "transaction_entropy": _clip_feature("transaction_entropy", entropy_est),
        "structural_depth": _clip_feature("structural_depth", structural_depth),
    }
    return features


def predict_risk(
    features_dict: dict[str, Any],
    model: Any | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    if model is None or feature_names is None:
        model, feature_names = load_artifacts()

    input_df = _validate_inputs(features_dict, feature_names)
    probability = float(model.predict_proba(input_df)[0, 1])
    return {
        "prob": probability,
        "probability": probability,
        "label_text": _risk_label(probability),
        "threshold_policy": "fixed_0.4_0.7",
        "threshold_values": dict(RISK_THRESHOLDS),
    }


def explain(
    features_dict: dict[str, Any],
    top_k: int = 5,
    model: Any | None = None,
    feature_names: list[str] | None = None,
) -> list[dict[str, float | str]]:
    if model is None or feature_names is None:
        model, feature_names = load_artifacts()

    _validate_inputs(features_dict, feature_names)
    importances = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    return [{"feature": name, "importance": float(score)} for name, score in ranked]


def main() -> None:
    model, feature_names = load_artifacts()
    sample = {
        "growth_rate": 0.22,
        "referral_ratio": 0.55,
        "payout_dependency": 0.9,
        "centralization_index": 0.5,
        "avg_holding_time": 38.0,
        "reinvestment_rate": 0.56,
        "gini_coefficient": 0.55,
        "transaction_entropy": 2.6,
        "structural_depth": 8.0,
    }
    risk = predict_risk(sample, model=model, feature_names=feature_names)
    top = explain(sample, model=model, feature_names=feature_names)
    print("Risk engine: OK")
    print(risk)
    print(top)


if __name__ == "__main__":
    main()
