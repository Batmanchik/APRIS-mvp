from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreOutput:
    case_id: str
    global_risk: float
    typology_probs: dict[str, float]
    risk_band: str
    model_version: str
    calibration_version: str
    explanation_ready: bool


@dataclass(frozen=True)
class ExplainOutput:
    summary: str
    tabular_factors: list[dict[str, float | str]]
    sequence_factors: list[dict[str, float | str]]
    graph_factors: list[dict[str, float | str]]
    confidence: float


@dataclass(frozen=True)
class BatchFailure:
    case_id: str
    error: str

