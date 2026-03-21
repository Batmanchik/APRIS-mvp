from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Sequence

import networkx as nx

from apris.cheops.domain.models import CaseWindow, TransactionEvent
from apris.cheops.domain.typologies import FraudTypology, TYPOLOGY_NAMES


ALLOWED_CHANNELS = {"legal", "crypto"}

REQUIRED_EVENT_FIELDS = {
    "event_id",
    "ts",
    "amount",
    "currency",
    "sender_id",
    "receiver_id",
    "sender_type",
    "receiver_type",
    "channel",
    "jurisdiction",
    "asset_type",
}


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        iso = value.strip().replace("Z", "+00:00")
        return datetime.fromisoformat(iso)
    raise ValueError("Field 'ts' must be ISO datetime string or datetime object.")


def normalize_event(raw: TransactionEvent | dict[str, Any]) -> TransactionEvent:
    if isinstance(raw, TransactionEvent):
        validate_event_schema(raw)
        return raw

    missing = REQUIRED_EVENT_FIELDS - set(raw.keys())
    if missing:
        raise ValueError(f"Missing event fields: {sorted(missing)}")

    event = TransactionEvent(
        event_id=str(raw["event_id"]).strip(),
        ts=_parse_timestamp(raw["ts"]),
        amount=float(raw["amount"]),
        currency=str(raw["currency"]).strip().upper(),
        sender_id=str(raw["sender_id"]).strip(),
        receiver_id=str(raw["receiver_id"]).strip(),
        sender_type=str(raw["sender_type"]).strip().lower(),
        receiver_type=str(raw["receiver_type"]).strip().lower(),
        channel=str(raw["channel"]).strip().lower(),
        jurisdiction=str(raw["jurisdiction"]).strip().upper(),
        asset_type=str(raw["asset_type"]).strip().lower(),
        tx_hash=(str(raw["tx_hash"]).strip() if raw.get("tx_hash") is not None else None),
        case_id=(str(raw["case_id"]).strip() if raw.get("case_id") is not None else None),
        metadata=dict(raw.get("metadata", {})),
    )
    validate_event_schema(event)
    return event


def validate_event_schema(event: TransactionEvent) -> None:
    if not event.event_id:
        raise ValueError("event_id must be non-empty.")
    if event.amount <= 0:
        raise ValueError("amount must be > 0.")
    if not event.sender_id or not event.receiver_id:
        raise ValueError("sender_id and receiver_id must be non-empty.")
    if event.sender_id == event.receiver_id:
        raise ValueError("sender_id and receiver_id must be different.")
    if event.channel not in ALLOWED_CHANNELS:
        raise ValueError(f"channel must be one of {sorted(ALLOWED_CHANNELS)}.")
    if not event.currency:
        raise ValueError("currency must be non-empty.")
    if not event.jurisdiction:
        raise ValueError("jurisdiction must be non-empty.")
    if not event.asset_type:
        raise ValueError("asset_type must be non-empty.")


def build_case_window(
    events: Sequence[TransactionEvent | dict[str, Any]],
    *,
    case_id: str | None = None,
    window_hours: int = 24,
) -> CaseWindow:
    if not events:
        raise ValueError("events cannot be empty.")
    if window_hours <= 0:
        raise ValueError("window_hours must be > 0.")

    normalized = [normalize_event(event) for event in events]
    normalized.sort(key=lambda e: e.ts)

    end_ts = normalized[-1].ts
    start_border = end_ts - timedelta(hours=window_hours)
    filtered = [event for event in normalized if event.ts >= start_border]
    if not filtered:
        raise ValueError("No events remained after applying the case window.")

    resolved_case_id = (
        (case_id.strip() if case_id else None)
        or next((event.case_id for event in filtered if event.case_id), None)
        or f"case-{end_ts.strftime('%Y%m%d%H%M%S')}"
    )
    if not resolved_case_id:
        raise ValueError("Unable to resolve case_id.")

    return CaseWindow(
        case_id=resolved_case_id,
        events=tuple(filtered),
        start_ts=filtered[0].ts,
        end_ts=filtered[-1].ts,
        window_hours=window_hours,
    )


def _max_graph_depth(events: tuple[TransactionEvent, ...]) -> int:
    graph = nx.DiGraph()
    for event in events:
        graph.add_edge(event.sender_id, event.receiver_id)

    if graph.number_of_nodes() == 0:
        return 1

    max_depth = 1
    for source in graph.nodes:
        depths: dict[str, int] = {source: 0}
        queue = [source]
        while queue:
            node = queue.pop(0)
            for neighbor in graph.successors(node):
                next_depth = depths[node] + 1
                if next_depth > depths.get(neighbor, -1) and next_depth <= 8:
                    depths[neighbor] = next_depth
                    queue.append(neighbor)
                    if next_depth > max_depth:
                        max_depth = next_depth
    return max_depth


def map_events_to_typology_labels(
    events: Sequence[TransactionEvent | dict[str, Any]],
) -> dict[str, int]:
    normalized = [normalize_event(event) for event in events]
    if not normalized:
        return {name: 0 for name in TYPOLOGY_NAMES}

    channels = {event.channel for event in normalized}
    amounts = [event.amount for event in normalized]
    receiver_counts = Counter(event.receiver_id for event in normalized)
    unique_receivers = len(receiver_counts)
    crypto_events = [event for event in normalized if event.channel == "crypto"]
    legal_events = [event for event in normalized if event.channel == "legal"]

    total_amount = sum(amounts)
    mean_amount = total_amount / max(len(amounts), 1)
    small_parts = sum(1 for amount in amounts if amount < mean_amount * 0.35)

    legal_to_crypto_bridge = int("legal" in channels and "crypto" in channels)
    structured_splitting = int(unique_receivers >= 5 and small_parts >= max(4, len(amounts) // 3))
    crypto_mixing = int(len(crypto_events) >= 6 and len({e.receiver_id for e in crypto_events}) >= 4)
    graph_depth = _max_graph_depth(tuple(normalized))
    legal_layering = int(graph_depth >= 4 and len(legal_events) >= 3)

    outgoing = sum(
        event.amount for event in normalized if event.sender_type in {"company", "legal_entity", "merchant"}
    )
    cash_out = int(outgoing / max(total_amount, 1.0) >= 0.62 and len(normalized) >= 6)

    labels = {
        FraudTypology.LEGAL_LAYERING.value: legal_layering,
        FraudTypology.LEGAL_TO_CRYPTO_BRIDGE.value: legal_to_crypto_bridge,
        FraudTypology.CRYPTO_MIXING.value: crypto_mixing,
        FraudTypology.STRUCTURED_SPLITTING.value: structured_splitting,
        FraudTypology.CASH_OUT.value: cash_out,
    }
    return labels
