from __future__ import annotations

import pytest

from apris.data_generator import FEATURE_COLUMNS
from apris.risk_engine import explain, operational_to_features, predict_risk


def test_operational_to_features_returns_expected_keys(sample_operational: dict[str, float]) -> None:
    features = operational_to_features(sample_operational)

    assert set(features.keys()) == set(FEATURE_COLUMNS)
    for value in features.values():
        assert isinstance(value, float)


def test_operational_to_features_rejects_top1_above_top10(sample_operational: dict[str, float]) -> None:
    payload = dict(sample_operational)
    payload["top1_wallet_share"] = 0.9
    payload["top10_wallet_share"] = 0.5

    with pytest.raises(ValueError, match="top1_wallet_share cannot exceed top10_wallet_share"):
        operational_to_features(payload)


def test_predict_risk_valid_payload_returns_probability(
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    result = predict_risk(sample_features, model=dummy_model, feature_names=FEATURE_COLUMNS)

    assert 0.0 <= result["probability"] <= 1.0
    assert result["label_text"] in {"Low", "Medium", "High"}
    assert result["threshold_policy"] == "fixed_0.4_0.7"


def test_predict_risk_invalid_payload_raises(
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    payload = dict(sample_features)
    payload["growth_rate"] = -0.01

    with pytest.raises(ValueError, match="out of range"):
        predict_risk(payload, model=dummy_model, feature_names=FEATURE_COLUMNS)


def test_explain_returns_top_k_sorted(
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    result = explain(sample_features, top_k=3, model=dummy_model, feature_names=FEATURE_COLUMNS)

    assert len(result) == 3
    assert result[0]["importance"] >= result[1]["importance"] >= result[2]["importance"]


def test_explain_raises_when_feature_missing(
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    payload = dict(sample_features)
    payload.pop("transaction_entropy")

    with pytest.raises(ValueError, match="Missing features"):
        explain(payload, top_k=3, model=dummy_model, feature_names=FEATURE_COLUMNS)
