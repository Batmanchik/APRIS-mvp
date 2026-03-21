from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from apris.data_generator import FEATURE_COLUMNS
from apris.api import main as api_main


@pytest.fixture
def client() -> Iterator[TestClient]:
    api_main._model = None
    api_main._feature_names = None
    with TestClient(api_main.app) as test_client:
        api_main._model = None
        api_main._feature_names = None
        yield test_client
    api_main._model = None
    api_main._feature_names = None


def test_health_endpoint_returns_status(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_returns_503_when_model_not_loaded(
    client: TestClient,
    sample_features: dict[str, float],
) -> None:
    response = client.post("/api/v1/predict", json=sample_features)
    assert response.status_code == 503


def test_predict_returns_200_when_model_loaded(
    client: TestClient,
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    api_main._model = dummy_model
    api_main._feature_names = FEATURE_COLUMNS

    response = client.post("/api/v1/predict", json=sample_features)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label_text"] in {"Low", "Medium", "High"}


def test_predict_returns_422_for_out_of_range_payload(
    client: TestClient,
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    api_main._model = dummy_model
    api_main._feature_names = FEATURE_COLUMNS
    payload = dict(sample_features)
    payload["growth_rate"] = -0.1

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_ops_returns_200_and_derived_features(
    client: TestClient,
    sample_operational: dict[str, float],
    dummy_model: object,
) -> None:
    api_main._model = dummy_model
    api_main._feature_names = FEATURE_COLUMNS

    response = client.post("/api/v1/predict/ops", json=sample_operational)
    assert response.status_code == 200
    body = response.json()
    assert "derived_features" in body
    assert set(body["derived_features"].keys()) == set(FEATURE_COLUMNS)


def test_predict_ops_returns_422_for_domain_validation(
    client: TestClient,
    sample_operational: dict[str, float],
    dummy_model: object,
) -> None:
    api_main._model = dummy_model
    api_main._feature_names = FEATURE_COLUMNS
    payload = dict(sample_operational)
    payload["top1_wallet_share"] = 0.8
    payload["top10_wallet_share"] = 0.5

    response = client.post("/api/v1/predict/ops", json=payload)
    assert response.status_code == 422


def test_explain_returns_200(
    client: TestClient,
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    api_main._model = dummy_model
    api_main._feature_names = FEATURE_COLUMNS

    response = client.post("/api/v1/explain", json={"features": sample_features, "top_k": 4})
    assert response.status_code == 200
    body = response.json()
    assert len(body["explanations"]) == 4


def test_explain_returns_422_when_features_incomplete(
    client: TestClient,
    sample_features: dict[str, float],
    dummy_model: object,
) -> None:
    api_main._model = dummy_model
    api_main._feature_names = FEATURE_COLUMNS
    payload = dict(sample_features)
    payload.pop("transaction_entropy")

    response = client.post("/api/v1/explain", json={"features": payload, "top_k": 3})
    assert response.status_code == 422
