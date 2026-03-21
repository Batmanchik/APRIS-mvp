from __future__ import annotations

from typing import Any

from apris.frontend import api_client


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


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
