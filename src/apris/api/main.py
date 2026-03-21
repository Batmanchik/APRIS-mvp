"""FastAPI REST API for Cheops AI risk scoring."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, NoReturn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apris.cheops.application.use_cases import ExplainCase, IngestCase, ScoreBatch, ScoreCase
from apris.cheops.domain.typologies import TYPOLOGY_NAMES
from apris.cheops.infrastructure.ml.engine_v2 import MultiBranchRiskEngine
from apris.cheops.interfaces.schemas_v2 import (
    BatchScoreFailure,
    BatchScoreV2Request,
    BatchScoreV2Response,
    ExplainV2Request,
    ExplainV2Response,
    ScoreV2Request,
    ScoreV2Response,
    V2ModelHealthResponse,
    V2TypologyMetaResponse,
    ensure_typology_keys,
)
from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, RISK_THRESHOLDS
from apris.risk_engine import (
    OPERATIONAL_INPUT_BOUNDS,
    explain,
    load_artifacts,
    operational_to_features,
    predict_risk,
)


_model: Any = None
_feature_names: list[str] | None = None
_v2_engine: MultiBranchRiskEngine | None = None
_ingest_case: IngestCase | None = None
_score_case: ScoreCase | None = None
_score_batch: ScoreBatch | None = None
_explain_case: ExplainCase | None = None


def _initialize_state() -> None:
    global _model, _feature_names, _v2_engine, _ingest_case, _score_case, _score_batch, _explain_case
    _model = None
    _feature_names = None
    try:
        _model, _feature_names = load_artifacts()
    except FileNotFoundError:
        pass

    _v2_engine = MultiBranchRiskEngine(model=_model, feature_names=_feature_names)
    _ingest_case = IngestCase()
    _score_case = ScoreCase(ingest_case=_ingest_case, engine=_v2_engine)
    _score_batch = ScoreBatch(score_case=_score_case)
    _explain_case = ExplainCase(ingest_case=_ingest_case, engine=_v2_engine)


def _load_model() -> None:
    """Backwards-compatible initializer used in tests."""
    _initialize_state()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _initialize_state()
    yield


app = FastAPI(
    title="Cheops AI Risk API",
    version="2.1.0",
    description="REST API for multi-channel fraud scoring (legal + crypto).",
    lifespan=_lifespan,
)


class FeaturesRequest(BaseModel):
    growth_rate: float = Field(..., ge=0.0, le=1.2)
    referral_ratio: float = Field(..., ge=0.0, le=1.0)
    payout_dependency: float = Field(..., ge=0.1, le=1.9)
    centralization_index: float = Field(..., ge=0.0, le=1.0)
    avg_holding_time: float = Field(..., ge=3.0, le=120.0)
    reinvestment_rate: float = Field(..., ge=0.0, le=1.0)
    gini_coefficient: float = Field(..., ge=0.1, le=1.0)
    transaction_entropy: float = Field(..., ge=0.3, le=5.0)
    structural_depth: float = Field(..., ge=2.0, le=16.0)


class OperationalRequest(BaseModel):
    tx_count_total: float
    unique_counterparties: float
    new_clients_current: float
    new_clients_previous: float
    referred_clients_current: float
    incoming_funds: float
    payouts_total: float
    top1_wallet_share: float
    top10_wallet_share: float
    avg_holding_days: float
    repeat_investor_share: float
    max_referral_depth: float


class ExplainRequest(BaseModel):
    features: dict[str, float]
    top_k: int = Field(default=5, ge=1, le=9)


class RiskResponse(BaseModel):
    probability: float
    label_text: str
    threshold_policy: str
    threshold_values: dict[str, float]
    derived_features: dict[str, float] | None = None


class ExplainResponse(BaseModel):
    explanations: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def _raise_unavailable(message: str) -> NoReturn:
    raise HTTPException(status_code=503, detail=message)


def _raise_unprocessable(exc: Exception) -> NoReturn:
    raise HTTPException(status_code=422, detail=str(exc))


def _ensure_model() -> tuple[Any, list[str]]:
    if _model is None or _feature_names is None:
        _raise_unavailable(
            "Model is not loaded. Train first: python -m apris.train_model"
        )
    return _model, _feature_names


def _ensure_v2_use_cases() -> tuple[ScoreCase, ScoreBatch, ExplainCase, MultiBranchRiskEngine]:
    if _score_case is None or _score_batch is None or _explain_case is None or _v2_engine is None:
        _raise_unavailable("Cheops v2 engine is not initialized.")
    return _score_case, _score_batch, _explain_case, _v2_engine


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/api/v1/predict", response_model=RiskResponse)
def api_predict(body: FeaturesRequest) -> RiskResponse:
    model, feature_names = _ensure_model()
    features = body.model_dump()
    try:
        result = predict_risk(features, model=model, feature_names=feature_names)
    except ValueError as exc:
        _raise_unprocessable(exc)
    return RiskResponse(
        probability=result["probability"],
        label_text=result["label_text"],
        threshold_policy=result["threshold_policy"],
        threshold_values=result["threshold_values"],
    )


@app.post("/api/v1/predict/ops", response_model=RiskResponse)
def api_predict_operational(body: OperationalRequest) -> RiskResponse:
    model, feature_names = _ensure_model()
    raw = body.model_dump()
    try:
        derived = operational_to_features(raw)
        result = predict_risk(derived, model=model, feature_names=feature_names)
    except ValueError as exc:
        _raise_unprocessable(exc)
    return RiskResponse(
        probability=result["probability"],
        label_text=result["label_text"],
        threshold_policy=result["threshold_policy"],
        threshold_values=result["threshold_values"],
        derived_features=derived,
    )


@app.post("/api/v1/explain", response_model=ExplainResponse)
def api_explain(body: ExplainRequest) -> ExplainResponse:
    model, feature_names = _ensure_model()
    try:
        result = explain(body.features, top_k=body.top_k, model=model, feature_names=feature_names)
    except ValueError as exc:
        _raise_unprocessable(exc)
    return ExplainResponse(explanations=result)


@app.get("/api/v1/meta/features")
def api_features_meta() -> dict[str, Any]:
    return {
        "feature_columns": FEATURE_COLUMNS,
        "feature_bounds": {k: {"min": v[0], "max": v[1]} for k, v in FEATURE_BOUNDS.items()},
        "operational_bounds": {k: {"min": v[0], "max": v[1]} for k, v in OPERATIONAL_INPUT_BOUNDS.items()},
        "risk_thresholds": dict(RISK_THRESHOLDS),
    }


@app.get("/api/v2/meta/typologies", response_model=V2TypologyMetaResponse)
def api_v2_typologies() -> V2TypologyMetaResponse:
    return V2TypologyMetaResponse(typologies=list(TYPOLOGY_NAMES))


@app.get("/api/v2/health/model", response_model=V2ModelHealthResponse)
def api_v2_health_model() -> V2ModelHealthResponse:
    _, _, _, engine = _ensure_v2_use_cases()
    return V2ModelHealthResponse(**engine.health())


@app.post("/api/v2/score", response_model=ScoreV2Response)
def api_v2_score(body: ScoreV2Request) -> ScoreV2Response:
    score_case, _, _, _ = _ensure_v2_use_cases()
    try:
        result = score_case.execute(
            case_id=body.case_id,
            events=[event.model_dump() for event in body.events],
            window_hours=body.window_hours,
            tabular_features=body.tabular_features,
        )
    except ValueError as exc:
        _raise_unprocessable(exc)

    return ScoreV2Response(
        case_id=result.case_id,
        global_risk=result.global_risk,
        typology_probs=ensure_typology_keys(result.typology_probs),
        risk_band=result.risk_band,
        model_version=result.model_version,
        calibration_version=result.calibration_version,
        explanation_ready=result.explanation_ready,
    )


@app.post("/api/v2/score/batch", response_model=BatchScoreV2Response)
def api_v2_score_batch(body: BatchScoreV2Request) -> BatchScoreV2Response:
    _, score_batch, _, _ = _ensure_v2_use_cases()
    results, failures = score_batch.execute(
        [
            {
                "case_id": case.case_id,
                "events": [event.model_dump() for event in case.events],
                "window_hours": case.window_hours,
                "tabular_features": case.tabular_features,
            }
            for case in body.cases
        ]
    )
    return BatchScoreV2Response(
        results=[
            ScoreV2Response(
                case_id=result.case_id,
                global_risk=result.global_risk,
                typology_probs=ensure_typology_keys(result.typology_probs),
                risk_band=result.risk_band,
                model_version=result.model_version,
                calibration_version=result.calibration_version,
                explanation_ready=result.explanation_ready,
            )
            for result in results
        ],
        failures=[BatchScoreFailure(case_id=fail.case_id, error=fail.error) for fail in failures],
    )


@app.post("/api/v2/explain", response_model=ExplainV2Response)
def api_v2_explain(body: ExplainV2Request) -> ExplainV2Response:
    _, _, explain_case, _ = _ensure_v2_use_cases()
    try:
        result = explain_case.execute(
            case_id=body.case_id,
            events=[event.model_dump() for event in body.events],
            window_hours=body.window_hours,
            tabular_features=body.tabular_features,
        )
    except ValueError as exc:
        _raise_unprocessable(exc)

    return ExplainV2Response(
        summary=result.summary,
        tabular_factors=result.tabular_factors,
        sequence_factors=result.sequence_factors,
        graph_factors=result.graph_factors,
        confidence=result.confidence,
    )
