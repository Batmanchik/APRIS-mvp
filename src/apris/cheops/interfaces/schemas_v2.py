from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from apris.cheops.domain.typologies import TYPOLOGY_NAMES


class TransactionEventIn(BaseModel):
    event_id: str
    ts: datetime
    amount: float = Field(..., gt=0)
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
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreV2Request(BaseModel):
    case_id: str
    events: list[TransactionEventIn] = Field(..., min_length=1)
    window_hours: int = Field(default=24, ge=1, le=720)
    tabular_features: dict[str, float] | None = None


class ScoreV2Response(BaseModel):
    case_id: str
    global_risk: float
    typology_probs: dict[str, float]
    risk_band: str
    model_version: str
    calibration_version: str
    explanation_ready: bool


class BatchScoreV2Request(BaseModel):
    cases: list[ScoreV2Request] = Field(..., min_length=1)


class BatchScoreFailure(BaseModel):
    case_id: str
    error: str


class BatchScoreV2Response(BaseModel):
    results: list[ScoreV2Response]
    failures: list[BatchScoreFailure]


class ExplainV2Request(BaseModel):
    case_id: str
    events: list[TransactionEventIn] = Field(..., min_length=1)
    window_hours: int = Field(default=24, ge=1, le=720)
    tabular_features: dict[str, float] | None = None


class ExplainV2Response(BaseModel):
    summary: str
    tabular_factors: list[dict[str, float | str]]
    sequence_factors: list[dict[str, float | str]]
    graph_factors: list[dict[str, float | str]]
    confidence: float


class V2TypologyMetaResponse(BaseModel):
    typologies: list[str]


class V2ModelHealthResponse(BaseModel):
    status: str
    tabular_model_loaded: bool
    sequence_branch: str
    graph_branch: str
    fusion: str
    model_version: str
    calibration_version: str


def ensure_typology_keys(typology_probs: dict[str, float]) -> dict[str, float]:
    """Stabilize typology response shape for clients."""
    return {name: float(typology_probs.get(name, 0.0)) for name in TYPOLOGY_NAMES}

