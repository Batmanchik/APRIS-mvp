from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from apris.cheops.domain.contracts import build_case_window
from apris.cheops.infrastructure.ml.sequence_v2 import (
    SEQUENCE_FEATURE_NAMES,
    SEQUENCE_V2_ARTIFACT_PATH,
    SEQUENCE_V2_METRICS_PATH,
    build_sequence_matrix_from_tabular,
    extract_sequence_features_from_case_window,
    load_sequence_artifact,
    predict_sequence_from_case_window,
    save_sequence_artifact,
    save_sequence_metrics,
    train_sequence_artifact,
)
from apris.data_generator import FEATURE_COLUMNS, build_dataset


def _event(idx: int, *, ts: datetime, sender: str, receiver: str, amount: float) -> dict[str, object]:
    return {
        "event_id": f"ev-{idx}",
        "ts": ts.isoformat(),
        "amount": amount,
        "currency": "USD",
        "sender_id": sender,
        "receiver_id": receiver,
        "sender_type": "company",
        "receiver_type": "wallet",
        "channel": "crypto" if idx % 2 == 0 else "legal",
        "jurisdiction": "KZ",
        "asset_type": "token" if idx % 2 == 0 else "fiat",
    }


def test_train_sequence_artifact_and_predict_from_case_window() -> None:
    df = build_dataset(total_n=360, enforce_training_size=False)
    artifact, metrics = train_sequence_artifact(
        df,
        random_state=42,
        model_params={"n_estimators": 45, "learning_rate": 0.08},
    )

    assert artifact["artifact_version"] == "cheops-sequence-v2"
    assert artifact["feature_names"] == SEQUENCE_FEATURE_NAMES
    assert metrics["artifact_version"] == "cheops-sequence-v2"
    assert 0.0 <= metrics["sequence_head"]["ece"] <= 1.0
    assert 0.0 <= metrics["heuristic_fallback"]["ece"] <= 1.0

    now = datetime(2026, 3, 21, 13, 0, 0)
    events = [
        _event(1, ts=now - timedelta(minutes=14), sender="A", receiver="B", amount=500.0),
        _event(2, ts=now - timedelta(minutes=8), sender="B", receiver="C", amount=480.0),
        _event(3, ts=now - timedelta(minutes=3), sender="C", receiver="D", amount=460.0),
    ]
    case_window = build_case_window(events, case_id="case-seq", window_hours=24)
    score = predict_sequence_from_case_window(case_window, artifact)
    features = extract_sequence_features_from_case_window(case_window)

    assert 0.0 <= score <= 1.0
    assert set(features.keys()) == set(SEQUENCE_FEATURE_NAMES)


def test_save_and_load_sequence_artifact_and_metrics(tmp_path: Path) -> None:
    df = build_dataset(total_n=280, enforce_training_size=False)
    artifact, metrics = train_sequence_artifact(
        df,
        random_state=42,
        model_params={"n_estimators": 35, "learning_rate": 0.08},
    )
    artifact_path = tmp_path / "sequence.joblib"
    metrics_path = tmp_path / "sequence_metrics.json"
    save_sequence_artifact(artifact, artifact_path)
    save_sequence_metrics(metrics, metrics_path)

    loaded = load_sequence_artifact(artifact_path)
    assert loaded["artifact_version"] == "cheops-sequence-v2"
    assert metrics_path.exists()

    features_df = df[FEATURE_COLUMNS].astype(float).copy()
    matrix = build_sequence_matrix_from_tabular(features_df)
    assert list(matrix.columns) == SEQUENCE_FEATURE_NAMES


def test_sequence_paths_constants() -> None:
    artifact_norm = str(SEQUENCE_V2_ARTIFACT_PATH).replace("\\", "/")
    metrics_norm = str(SEQUENCE_V2_METRICS_PATH).replace("\\", "/")
    assert artifact_norm.endswith("artifacts/cheops_v2_sequence.joblib")
    assert metrics_norm.endswith("artifacts/cheops_v2_sequence_metrics.json")
