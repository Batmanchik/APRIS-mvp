from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class TransactionEvent:
    event_id: str
    ts: datetime
    amount: float
    currency: str
    sender_id: str
    receiver_id: str
    sender_type: str
    receiver_type: str
    channel: str
    jurisdiction: str
    asset_type: str
    tx_hash: str | None = None
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseWindow:
    case_id: str
    events: tuple[TransactionEvent, ...]
    start_ts: datetime
    end_ts: datetime
    window_hours: int

