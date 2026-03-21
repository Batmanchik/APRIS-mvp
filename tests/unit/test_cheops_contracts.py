from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from apris.cheops.domain.contracts import (
    build_case_window,
    map_events_to_typology_labels,
    normalize_event,
    validate_event_schema,
)
from apris.cheops.domain.typologies import TYPOLOGY_NAMES


def _event(
    idx: int,
    *,
    ts: datetime,
    channel: str = "legal",
    sender: str = "S1",
    receiver: str = "R1",
    amount: float = 100.0,
    sender_type: str = "company",
    receiver_type: str = "wallet",
) -> dict[str, object]:
    return {
        "event_id": f"ev-{idx}",
        "ts": ts.isoformat(),
        "amount": amount,
        "currency": "usd",
        "sender_id": sender,
        "receiver_id": receiver,
        "sender_type": sender_type,
        "receiver_type": receiver_type,
        "channel": channel,
        "jurisdiction": "KZ",
        "asset_type": "fiat" if channel == "legal" else "token",
    }


def test_normalize_event_and_validate() -> None:
    raw = _event(1, ts=datetime(2026, 1, 1, 12, 0, 0), channel="crypto")
    event = normalize_event(raw)
    validate_event_schema(event)
    assert event.channel == "crypto"
    assert event.currency == "USD"


def test_validate_event_rejects_bad_channel() -> None:
    raw = _event(1, ts=datetime(2026, 1, 1, 12, 0, 0), channel="wire")
    with pytest.raises(ValueError, match="channel must be one of"):
        normalize_event(raw)


def test_build_case_window_orders_and_filters() -> None:
    end_ts = datetime(2026, 1, 1, 12, 0, 0)
    events = [
        _event(1, ts=end_ts - timedelta(hours=10)),
        _event(2, ts=end_ts - timedelta(hours=1)),
        _event(3, ts=end_ts - timedelta(hours=30)),
    ]
    window = build_case_window(events, case_id="case-1", window_hours=24)
    assert window.case_id == "case-1"
    assert len(window.events) == 2
    assert window.events[0].ts <= window.events[1].ts


def test_map_events_to_typology_labels_bridge_detected() -> None:
    base_ts = datetime(2026, 1, 1, 12, 0, 0)
    events = [
        _event(1, ts=base_ts - timedelta(minutes=40), channel="legal", sender="L1", receiver="L2"),
        _event(2, ts=base_ts - timedelta(minutes=20), channel="crypto", sender="L2", receiver="C1"),
        _event(3, ts=base_ts - timedelta(minutes=10), channel="crypto", sender="C1", receiver="C2"),
    ]
    labels = map_events_to_typology_labels(events)
    assert set(labels.keys()) == set(TYPOLOGY_NAMES)
    assert labels["LEGAL_TO_CRYPTO_BRIDGE"] == 1
