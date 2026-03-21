from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from apris.api import main as api_main
from apris.cheops.domain.typologies import TYPOLOGY_NAMES


def _event(
    idx: int,
    *,
    ts: datetime,
    channel: str = "legal",
    sender: str = "S1",
    receiver: str = "R1",
    amount: float = 120.0,
) -> dict[str, object]:
    return {
        "event_id": f"ev-{idx}",
        "ts": ts.isoformat(),
        "amount": amount,
        "currency": "USD",
        "sender_id": sender,
        "receiver_id": receiver,
        "sender_type": "company",
        "receiver_type": "wallet",
        "channel": channel,
        "jurisdiction": "KZ",
        "asset_type": "token" if channel == "crypto" else "fiat",
    }


def _score_payload(case_id: str = "case-v2") -> dict[str, object]:
    now = datetime(2026, 3, 21, 12, 0, 0)
    return {
        "case_id": case_id,
        "window_hours": 24,
        "events": [
            _event(1, ts=now - timedelta(minutes=40), channel="legal", sender="A", receiver="B", amount=900.0),
            _event(2, ts=now - timedelta(minutes=25), channel="legal", sender="B", receiver="C", amount=780.0),
            _event(3, ts=now - timedelta(minutes=10), channel="crypto", sender="C", receiver="X", amount=760.0),
            _event(4, ts=now - timedelta(minutes=3), channel="crypto", sender="X", receiver="Y", amount=740.0),
        ],
    }


def _broken_case(case_id: str = "case-broken") -> dict[str, object]:
    payload = _score_payload(case_id=case_id)
    events = payload["events"]
    if isinstance(events, list):
        events[0]["sender_id"] = "SAME"
        events[0]["receiver_id"] = "SAME"
    return payload


def _set_v2_none() -> None:
    api_main._v2_engine = None
    api_main._ingest_case = None
    api_main._score_case = None
    api_main._score_batch = None
    api_main._explain_case = None


def _restore_v2() -> None:
    api_main._load_model()


class TestApiV2Endpoints:
    def setup_method(self) -> None:
        _restore_v2()

    def teardown_method(self) -> None:
        _restore_v2()

    def test_meta_typologies(self) -> None:
        with TestClient(api_main.app) as client:
            response = client.get("/api/v2/meta/typologies")
            assert response.status_code == 200
            body = response.json()
            assert body["typologies"] == list(TYPOLOGY_NAMES)

    def test_health_model_returns_status(self) -> None:
        with TestClient(api_main.app) as client:
            response = client.get("/api/v2/health/model")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "ok"
            assert "model_version" in body
            assert "calibration_version" in body

    def test_score_happy_path(self) -> None:
        with TestClient(api_main.app) as client:
            response = client.post("/api/v2/score", json=_score_payload())
            assert response.status_code == 200
            body = response.json()
            assert body["case_id"] == "case-v2"
            assert 0.0 <= body["global_risk"] <= 1.0
            assert set(body["typology_probs"].keys()) == set(TYPOLOGY_NAMES)
            assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_score_returns_422_on_invalid_schema(self) -> None:
        payload = _score_payload()
        events = payload["events"]
        if isinstance(events, list):
            events[0]["amount"] = -1

        with TestClient(api_main.app) as client:
            response = client.post("/api/v2/score", json=payload)
            assert response.status_code == 422

    def test_explain_happy_path(self) -> None:
        with TestClient(api_main.app) as client:
            response = client.post("/api/v2/explain", json=_score_payload())
            assert response.status_code == 200
            body = response.json()
            assert isinstance(body["summary"], str)
            assert len(body["tabular_factors"]) > 0
            assert len(body["sequence_factors"]) > 0
            assert len(body["graph_factors"]) > 0
            assert 0.0 <= body["confidence"] <= 1.0

    def test_score_batch_partial_failure(self) -> None:
        payload = {
            "cases": [
                _score_payload(case_id="case-ok"),
                _broken_case(case_id="case-fail"),
            ]
        }
        with TestClient(api_main.app) as client:
            response = client.post("/api/v2/score/batch", json=payload)
            assert response.status_code == 200
            body = response.json()
            assert len(body["results"]) == 1
            assert body["results"][0]["case_id"] == "case-ok"
            assert len(body["failures"]) == 1
            assert body["failures"][0]["case_id"] == "case-fail"
            assert "sender_id and receiver_id must be different" in body["failures"][0]["error"]

    def test_v2_returns_503_when_engine_not_initialized(self) -> None:
        with TestClient(api_main.app) as client:
            _set_v2_none()
            response = client.get("/api/v2/health/model")
            assert response.status_code == 503
