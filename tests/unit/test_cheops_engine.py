from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from apris.cheops.domain.contracts import build_case_window
from apris.cheops.domain.typologies import TYPOLOGY_NAMES
from apris.cheops.infrastructure.ml.engine_v2 import MultiBranchRiskEngine


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


def test_v2_engine_score_and_explain_with_fallback() -> None:
    now = datetime(2026, 3, 21, 12, 0, 0)
    events = [
        _event(1, ts=now - timedelta(minutes=25), channel="legal", sender="A", receiver="B", amount=1000.0),
        _event(2, ts=now - timedelta(minutes=16), channel="legal", sender="B", receiver="C", amount=820.0),
        _event(3, ts=now - timedelta(minutes=8), channel="crypto", sender="C", receiver="X", amount=780.0),
        _event(4, ts=now - timedelta(minutes=2), channel="crypto", sender="X", receiver="Y", amount=760.0),
    ]
    case_window = build_case_window(events, case_id="case-v2", window_hours=24)

    engine = MultiBranchRiskEngine(model=None, feature_names=None, auto_load_artifacts=False)
    score = engine.score_case(case_window)
    explanation = engine.explain_case(case_window)

    assert score.case_id == "case-v2"
    assert 0.0 <= score.global_risk <= 1.0
    assert set(score.typology_probs.keys()) == set(TYPOLOGY_NAMES)
    assert score.risk_band in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert explanation.summary
    assert 0.0 <= explanation.confidence <= 1.0
    assert len(explanation.tabular_factors) > 0


class _FixedProbModel:
    def __init__(self, prob: float) -> None:
        self.prob = float(prob)

    def predict_proba(self, _: object) -> np.ndarray:
        return np.array([[1.0 - self.prob, self.prob]], dtype=float)


def test_v2_engine_uses_logistic_fusion_head_when_available() -> None:
    now = datetime(2026, 3, 21, 12, 0, 0)
    events = [
        _event(1, ts=now - timedelta(minutes=20), channel="legal", sender="A", receiver="B", amount=700.0),
        _event(2, ts=now - timedelta(minutes=10), channel="crypto", sender="B", receiver="C", amount=650.0),
    ]
    case_window = build_case_window(events, case_id="case-fusion", window_hours=24)

    fusion_meta = {
        "feature_names": ["tabular_prob", "sequence_prob", "graph_prob"],
        "meta_model": _FixedProbModel(0.91),
        "calibrator": None,
    }
    engine = MultiBranchRiskEngine(
        model=None,
        feature_names=None,
        tabular_bundle=None,
        fusion_meta=fusion_meta,
        auto_load_artifacts=False,
    )
    health = engine.health()
    score = engine.score_case(
        case_window,
        tabular_features={
            "growth_rate": 0.31,
            "payout_dependency": 1.05,
            "centralization_index": 0.62,
        },
    )

    assert health["fusion"] == "logistic_meta_head"
    assert abs(score.global_risk - 0.91) < 1e-9


def test_v2_engine_uses_trained_sequence_branch_when_available() -> None:
    now = datetime(2026, 3, 21, 12, 0, 0)
    events = [
        _event(1, ts=now - timedelta(minutes=20), channel="legal", sender="A", receiver="B", amount=700.0),
        _event(2, ts=now - timedelta(minutes=12), channel="crypto", sender="B", receiver="C", amount=650.0),
        _event(3, ts=now - timedelta(minutes=4), channel="crypto", sender="C", receiver="D", amount=630.0),
    ]
    case_window = build_case_window(events, case_id="case-sequence", window_hours=24)

    sequence_model = {
        "feature_names": [
            "event_rate_hour",
            "burst_ratio_90s",
            "median_delta_inverse",
            "amount_cv_norm",
            "unique_sender_ratio",
        ],
        "model": _FixedProbModel(0.82),
        "calibrator": None,
    }
    engine = MultiBranchRiskEngine(
        model=None,
        feature_names=None,
        tabular_bundle=None,
        sequence_model=sequence_model,
        auto_load_artifacts=False,
    )

    sequence_score = engine._score_sequence(case_window)
    health = engine.health()

    assert abs(sequence_score - 0.82) < 1e-9
    assert health["sequence_branch"] == "trained_tcn_surrogate"


def test_v2_engine_uses_trained_graph_branch_when_available() -> None:
    now = datetime(2026, 3, 21, 12, 0, 0)
    events = [
        _event(1, ts=now - timedelta(minutes=20), channel="legal", sender="A", receiver="B", amount=700.0),
        _event(2, ts=now - timedelta(minutes=12), channel="crypto", sender="B", receiver="C", amount=650.0),
        _event(3, ts=now - timedelta(minutes=4), channel="crypto", sender="C", receiver="D", amount=630.0),
    ]
    case_window = build_case_window(events, case_id="case-graph", window_hours=24)

    graph_model = {
        "feature_names": [
            "graph_density",
            "graph_hub_share",
            "graph_component_compactness",
            "graph_transitivity",
            "graph_weight_cv_norm",
        ],
        "model": _FixedProbModel(0.79),
        "calibrator": None,
    }
    engine = MultiBranchRiskEngine(
        model=None,
        feature_names=None,
        tabular_bundle=None,
        graph_model=graph_model,
        auto_load_artifacts=False,
    )

    graph_score = engine._score_graph(case_window)
    health = engine.health()

    assert abs(graph_score - 0.79) < 1e-9
    assert health["graph_branch"] == "trained_graphsage_surrogate"
