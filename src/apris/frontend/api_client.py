"""
Легковесный HTTP-клиент для взаимодействия Streamlit и FastAPI.

Все frontend-модули вызывают API через этот клиент и
не импортируют функции инференса напрямую.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_API_TIMEOUT = 10.0
DEFAULT_API_RETRIES = 1
DEFAULT_API_RETRY_BACKOFF = 0.2


class ApiClientError(RuntimeError):
    """Исключение для пользовательских ошибок при вызове API из интерфейса."""


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


def _api_retries(default: int = DEFAULT_API_RETRIES) -> int:
    raw = os.getenv("CHEOPS_API_RETRIES")
    if raw is None:
        return default
    try:
        retries = int(raw)
    except ValueError:
        return default
    if retries < 0:
        return default
    return retries


def _api_retry_backoff(default: float = DEFAULT_API_RETRY_BACKOFF) -> float:
    raw = os.getenv("CHEOPS_API_RETRY_BACKOFF")
    if raw is None:
        return default
    try:
        backoff = float(raw)
    except ValueError:
        return default
    if backoff < 0:
        return default
    return backoff


def _url(path: str) -> str:
    normalized = path if path.startswith("/") else f"/{path}"
    return f"{_api_base_url()}{normalized}"


def _format_http_error(method: str, path: str, response: requests.Response) -> ApiClientError:
    body_preview = getattr(response, "text", "")
    if isinstance(body_preview, str):
        body_preview = body_preview.strip()
    else:
        body_preview = ""
    if len(body_preview) > 240:
        body_preview = f"{body_preview[:237]}..."
    detail = f"HTTP {response.status_code}"
    if body_preview:
        detail = f"{detail}: {body_preview}"
    return ApiClientError(f"{method.upper()} {path}: ошибка запроса ({detail})")


def _request(
    method: str,
    path: str,
    *,
    json_payload: dict[str, Any] | None = None,
    timeout: float,
    retryable: bool = False,
) -> dict[str, Any]:
    retries = _api_retries() if retryable else 0
    backoff = _api_retry_backoff()
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            resp = requests.request(method=method, url=_url(path), json=json_payload, timeout=timeout)
            resp.raise_for_status()
            try:
                return resp.json()
            except ValueError as exc:
                raise ApiClientError(f"{method.upper()} {path}: сервер вернул ответ не в формате JSON") from exc
        except requests.HTTPError as exc:
            response = exc.response
            if response is None:
                raise ApiClientError(f"{method.upper()} {path}: HTTP-ошибка без ответа сервера") from exc
            status_code = int(response.status_code)
            should_retry = retryable and status_code >= 500 and attempt < attempts
            if should_retry:
                if backoff > 0:
                    time.sleep(backoff * attempt)
                continue
            raise _format_http_error(method, path, response) from exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            should_retry = retryable and attempt < attempts
            if should_retry:
                if backoff > 0:
                    time.sleep(backoff * attempt)
                continue
            raise ApiClientError(
                f"{method.upper()} {path}: ошибка после {attempt} попыток ({exc.__class__.__name__})"
            ) from exc

    raise ApiClientError(f"{method.upper()} {path}: ошибка после {attempts} попыток")


def health_check() -> dict[str, Any]:
    return _request("GET", "/api/v1/health", timeout=_api_timeout(5.0), retryable=True)


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
    return _request("GET", "/api/v1/meta/features", timeout=_api_timeout(5.0), retryable=True)


def get_v2_typologies() -> dict[str, Any]:
    return _request("GET", "/api/v2/meta/typologies", timeout=_api_timeout(5.0), retryable=True)


def health_check_v2_model() -> dict[str, Any]:
    return _request("GET", "/api/v2/health/model", timeout=_api_timeout(5.0), retryable=True)


def health_check_v2_model_details() -> dict[str, Any]:
    return _request("GET", "/api/v2/health/model/details", timeout=_api_timeout(5.0), retryable=True)


def health_check_v2_runtime() -> dict[str, Any]:
    return _request("GET", "/api/v2/health/runtime", timeout=_api_timeout(5.0), retryable=True)


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
