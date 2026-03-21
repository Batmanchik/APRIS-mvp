from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from apris.cheops.application.dto import BatchFailure, ExplainOutput, ScoreOutput
from apris.cheops.domain.contracts import build_case_window
from apris.cheops.domain.models import CaseWindow, TransactionEvent


class V2Engine(Protocol):
    def score_case(
        self,
        case_window: CaseWindow,
        *,
        tabular_features: dict[str, float] | None = None,
    ) -> ScoreOutput:
        ...

    def explain_case(
        self,
        case_window: CaseWindow,
        *,
        tabular_features: dict[str, float] | None = None,
    ) -> ExplainOutput:
        ...

    def health(self) -> dict[str, Any]:
        ...


@dataclass
class IngestCase:
    def execute(
        self,
        events: Sequence[TransactionEvent | dict[str, Any]],
        *,
        case_id: str | None = None,
        window_hours: int = 24,
    ) -> CaseWindow:
        return build_case_window(events, case_id=case_id, window_hours=window_hours)


@dataclass
class ScoreCase:
    ingest_case: IngestCase
    engine: V2Engine

    def execute(
        self,
        *,
        case_id: str,
        events: list[dict[str, Any]],
        window_hours: int,
        tabular_features: dict[str, float] | None = None,
    ) -> ScoreOutput:
        case_window = self.ingest_case.execute(events, case_id=case_id, window_hours=window_hours)
        return self.engine.score_case(case_window, tabular_features=tabular_features)


@dataclass
class ExplainCase:
    ingest_case: IngestCase
    engine: V2Engine

    def execute(
        self,
        *,
        case_id: str,
        events: list[dict[str, Any]],
        window_hours: int,
        tabular_features: dict[str, float] | None = None,
    ) -> ExplainOutput:
        case_window = self.ingest_case.execute(events, case_id=case_id, window_hours=window_hours)
        return self.engine.explain_case(case_window, tabular_features=tabular_features)


@dataclass
class ScoreBatch:
    score_case: ScoreCase

    def execute(self, cases: list[dict[str, Any]]) -> tuple[list[ScoreOutput], list[BatchFailure]]:
        results: list[ScoreOutput] = []
        failures: list[BatchFailure] = []
        for case in cases:
            case_id = str(case.get("case_id", "")).strip() or "unknown-case"
            try:
                result = self.score_case.execute(
                    case_id=case_id,
                    events=list(case["events"]),
                    window_hours=int(case.get("window_hours", 24)),
                    tabular_features=case.get("tabular_features"),
                )
                results.append(result)
            except Exception as exc:
                failures.append(BatchFailure(case_id=case_id, error=str(exc)))
        return results, failures
