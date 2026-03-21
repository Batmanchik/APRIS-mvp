"""
Thin HTTP client for Streamlit -> FastAPI communication.

All frontend modules must call API endpoints through this client and
must not import model inference functions directly.
"""
from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_API_TIMEOUT = 10.0


def _api_base_url() -> str:
    return os.getenv("CHEOPS_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def _api_timeout(default: float) -> float:
    raw = os.getenv("CHEOPS_API_TIMEOUT")
    if raw is None:
        return default
    try:
        timeout = float(raw)
    except ValueError:
        return default
    if timeout <= 0:
        return default
    return timeout


def _url(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    return f"{_api_base_url()}{normalized}"


def _request(
    method: str,
    path: str,
    *,
    json_payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    resp = requests.request(method=method, url=_url(path), json=json_payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def health_check() -> dict[str, Any]:
    return _request("GET", "/api/v1/health", timeout=_api_timeout(5.0))


def predict_from_features(features: dict[str, float]) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/v1/predict",
        json_payload=features,
        timeout=_api_timeout(DEFAULT_API_TIMEOUT),
    )


def predict_from_ops(operational: dict[str, float]) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/v1/predict/ops",
        json_payload=operational,
        timeout=_api_timeout(DEFAULT_API_TIMEOUT),
    )


def explain_features(features: dict[str, float], top_k: int = 5) -> list[dict[str, Any]]:
    payload = _request(
        "POST",
        "/api/v1/explain",
        json_payload={"features": features, "top_k": top_k},
        timeout=_api_timeout(DEFAULT_API_TIMEOUT),
    )
    return payload["explanations"]


def get_features_meta() -> dict[str, Any]:
    return _request("GET", "/api/v1/meta/features", timeout=_api_timeout(5.0))


def get_v2_typologies() -> dict[str, Any]:
    return _request("GET", "/api/v2/meta/typologies", timeout=_api_timeout(5.0))


def health_check_v2_model() -> dict[str, Any]:
    return _request("GET", "/api/v2/health/model", timeout=_api_timeout(5.0))


def score_case_v2(payload: dict[str, Any]) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/v2/score",
        json_payload=payload,
        timeout=_api_timeout(15.0),
    )


def score_batch_v2(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/v2/score/batch",
        json_payload={"cases": cases},
        timeout=_api_timeout(30.0),
    )


def explain_case_v2(payload: dict[str, Any]) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/v2/explain",
        json_payload=payload,
        timeout=_api_timeout(20.0),
    )
