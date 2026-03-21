from __future__ import annotations

from datetime import datetime, timedelta

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
