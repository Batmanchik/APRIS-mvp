from __future__ import annotations

from typing import Any

import pytest
import requests

from apris.frontend import api_client


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _HttpErrorResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        raise requests.HTTPError(response=self)

    def json(self) -> dict[str, Any]:
        return {}


class _BadJsonResponse:
    status_code = 200
    text = "not-json"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        raise ValueError("invalid json")


def test_health_check_uses_env_base_url(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = timeout
        captured["json"] = json
        return _DummyResponse({"status": "ok"})

    monkeypatch.setenv("CHEOPS_API_BASE_URL", "http://127.0.0.1:9100/")
    monkeypatch.setattr(api_client.requests, "request", _fake_request)

    payload = api_client.health_check()
    assert payload["status"] == "ok"
    assert captured["method"] == "GET"
    assert captured["url"] == "http://127.0.0.1:9100/api/v1/health"
    assert captured["timeout"] == 5.0
    assert captured["json"] is None


def test_score_batch_uses_timeout_from_env(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = timeout
        captured["json"] = json
        return _DummyResponse({"results": [], "failures": []})

    monkeypatch.setenv("CHEOPS_API_TIMEOUT", "12")
    monkeypatch.setattr(api_client.requests, "request", _fake_request)

    result = api_client.score_batch_v2([{"case_id": "c1", "events": [{"event_id": "x"}], "window_hours": 24}])
    assert result == {"results": [], "failures": []}
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/v2/score/batch")
    assert captured["timeout"] == 12.0
    assert "cases" in captured["json"]


def test_invalid_timeout_env_falls_back_to_default(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        captured["timeout"] = timeout
        return _DummyResponse({"typologies": []})

    monkeypatch.setenv("CHEOPS_API_TIMEOUT", "-1")
    monkeypatch.setattr(api_client.requests, "request", _fake_request)

    api_client.get_v2_typologies()
    assert captured["timeout"] == 5.0


def test_health_check_v2_model_details_uses_expected_path(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = timeout
        captured["json"] = json
        return _DummyResponse({"status": "ok"})

    monkeypatch.setattr(api_client.requests, "request", _fake_request)
    payload = api_client.health_check_v2_model_details()

    assert payload == {"status": "ok"}
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/api/v2/health/model/details")
    assert captured["timeout"] == 5.0
    assert captured["json"] is None


def test_health_check_v2_runtime_uses_expected_path(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        captured["method"] = method
        captured["url"] = url
        captured["timeout"] = timeout
        captured["json"] = json
        return _DummyResponse({"status": "ok", "runtime": {}})

    monkeypatch.setattr(api_client.requests, "request", _fake_request)
    payload = api_client.health_check_v2_runtime()

    assert payload["status"] == "ok"
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/api/v2/health/runtime")
    assert captured["timeout"] == 5.0
    assert captured["json"] is None


def test_health_check_retries_once_on_connection_error(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.ConnectionError("temporary down")
        return _DummyResponse({"status": "ok"})

    monkeypatch.setenv("CHEOPS_API_RETRIES", "1")
    monkeypatch.setattr(api_client.time, "sleep", lambda _: None)
    monkeypatch.setattr(api_client.requests, "request", _fake_request)

    payload = api_client.health_check()
    assert payload["status"] == "ok"
    assert calls["count"] == 2


def test_get_retries_on_http_503_and_then_succeeds(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        calls["count"] += 1
        if calls["count"] == 1:
            return _HttpErrorResponse(status_code=503, text="service unavailable")
        return _DummyResponse({"status": "ok"})

    monkeypatch.setenv("CHEOPS_API_RETRIES", "1")
    monkeypatch.setattr(api_client.time, "sleep", lambda _: None)
    monkeypatch.setattr(api_client.requests, "request", _fake_request)

    payload = api_client.health_check_v2_model()
    assert payload["status"] == "ok"
    assert calls["count"] == 2


def test_post_does_not_retry_on_connection_error(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        calls["count"] += 1
        raise requests.ConnectionError("network down")

    monkeypatch.setenv("CHEOPS_API_RETRIES", "3")
    monkeypatch.setattr(api_client.time, "sleep", lambda _: None)
    monkeypatch.setattr(api_client.requests, "request", _fake_request)

    with pytest.raises(api_client.ApiClientError):
        api_client.score_batch_v2([{"case_id": "c1", "events": [{"event_id": "e1"}], "window_hours": 24}])

    assert calls["count"] == 1


def test_non_json_response_raises_api_client_error(monkeypatch) -> None:
    def _fake_request(*, method: str, url: str, json: dict[str, Any] | None, timeout: float):
        return _BadJsonResponse()

    monkeypatch.setattr(api_client.requests, "request", _fake_request)

    with pytest.raises(api_client.ApiClientError):
        api_client.health_check_v2_model()
