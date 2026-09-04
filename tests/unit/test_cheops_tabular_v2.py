from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from apris.cheops.domain.contracts import build_case_window
from apris.cheops.domain.typologies import TYPOLOGY_NAMES
from apris.cheops.infrastructure.ml.engine_v2 import MultiBranchRiskEngine
from apris.cheops.infrastructure.ml.tabular_v2 import (
    derive_typology_targets,
    predict_tabular_bundle,
    train_tabular_bundle,
)
from apris.data_generator import FEATURE_COLUMNS, build_dataset


def _event(idx: int, *, ts: datetime, channel: str, sender: str, receiver: str, amount: float) -> dict[str, object]:
    return {
        "event_id": f"ev-{idx}",
        "ts": ts.isoformat(),
        "amount": amount,
        "currency": "USD",
        "sender_id": sender,
        "receiver_id": receiver,
        "sender_type": "company",
        "receiver_type": "wallet",
        "channel": channel,
        "jurisdiction": "KZ",
        "asset_type": "token" if channel == "crypto" else "fiat",
    }


def test_derive_typology_targets_has_all_columns_and_binary_values() -> None:
    df = build_dataset(total_n=400, enforce_training_size=False)
    x = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int)

    targets = derive_typology_targets(x, y)

    assert list(targets.columns) == sorted(TYPOLOGY_NAMES)
    assert len(targets) == len(df)
    for name in TYPOLOGY_NAMES:
        uniques = set(targets[name].unique().tolist())
        assert uniques.issubset({0, 1})


def test_train_tabular_bundle_predict_and_engine_integration() -> None:
    df = build_dataset(total_n=500, enforce_training_size=False)
    bundle, metrics = train_tabular_bundle(
        df,
        random_state=42,
        model_params={"n_estimators": 40, "learning_rate": 0.08},
    )

    features = {name: float(df.iloc[0][name]) for name in FEATURE_COLUMNS}
    global_score, typology_scores = predict_tabular_bundle(features, bundle)

    assert 0.0 <= global_score <= 1.0
    assert set(typology_scores.keys()) == set(TYPOLOGY_NAMES)
    assert metrics["global"]["brier"] >= 0.0
    assert 0.0 <= metrics["global"]["ece"] <= 1.0

    now = datetime(2026, 3, 21, 12, 0, 0)
    events = [
        _event(1, ts=now - timedelta(minutes=18), channel="legal", sender="A", receiver="B", amount=500.0),
        _event(2, ts=now - timedelta(minutes=8), channel="crypto", sender="B", receiver="X", amount=480.0),
    ]
    case_window = build_case_window(events, case_id="case-bundle", window_hours=24)

    engine = MultiBranchRiskEngine(tabular_bundle=bundle, auto_load_artifacts=False)
    score = engine.score_case(case_window, tabular_features=features)

    assert 0.0 <= score.global_risk <= 1.0
    assert set(score.typology_probs.keys()) == set(TYPOLOGY_NAMES)


def test_predict_tabular_bundle_validates_missing_and_non_numeric_features() -> None:
    df = build_dataset(total_n=320, enforce_training_size=False)
    bundle, _ = train_tabular_bundle(
        df,
        random_state=42,
        model_params={"n_estimators": 30, "learning_rate": 0.1},
    )
    features = {name: float(df.iloc[0][name]) for name in FEATURE_COLUMNS}

    missing = dict(features)
    missing.pop("growth_rate")
    with pytest.raises(ValueError, match="Missing tabular features"):
        predict_tabular_bundle(missing, bundle)

    invalid = dict(features)
    invalid["growth_rate"] = "not-a-number"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="growth_rate"):
        predict_tabular_bundle(invalid, bundle)
