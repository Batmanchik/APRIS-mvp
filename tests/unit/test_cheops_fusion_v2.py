from __future__ import annotations

from pathlib import Path

from apris.cheops.infrastructure.ml.fusion_v2 import (
    FUSION_FEATURE_NAMES,
    FUSION_V2_ARTIFACT_PATH,
    FUSION_V2_METRICS_PATH,
    load_fusion_artifact,
    predict_fusion_meta,
    save_fusion_artifact,
    save_fusion_metrics,
    train_fusion_meta,
)
from apris.cheops.infrastructure.ml.graph_v2 import train_graph_artifact
from apris.cheops.infrastructure.ml.sequence_v2 import train_sequence_artifact
from apris.cheops.infrastructure.ml.tabular_v2 import train_tabular_bundle
from apris.data_generator import build_dataset


def test_train_fusion_meta_and_predict() -> None:
    df = build_dataset(total_n=320, enforce_training_size=False)
    tabular_bundle, _ = train_tabular_bundle(
        df,
        random_state=42,
        model_params={"n_estimators": 35, "learning_rate": 0.08},
    )
    sequence_artifact, _ = train_sequence_artifact(
        df,
        random_state=42,
        model_params={"n_estimators": 45, "learning_rate": 0.08},
    )
    graph_artifact, _ = train_graph_artifact(
        df,
        random_state=42,
        model_params={"n_estimators": 45, "learning_rate": 0.08},
    )

    artifact, metrics = train_fusion_meta(
        df,
        tabular_bundle,
        random_state=42,
        sequence_artifact=sequence_artifact,
        graph_artifact=graph_artifact,
    )

    assert artifact["artifact_version"] == "cheops-fusion-v2"
    assert artifact["feature_names"] == FUSION_FEATURE_NAMES
    assert metrics["artifact_version"] == "cheops-fusion-v2"
    assert metrics["sequence_source"] == "trained_sequence_v2"
    assert metrics["graph_source"] == "trained_graph_v2"
    assert 0.0 <= metrics["meta_head"]["ece"] <= 1.0
    assert 0.0 <= metrics["fallback_weighted"]["ece"] <= 1.0

    prob = predict_fusion_meta(0.81, 0.73, 0.66, artifact)
    assert 0.0 <= prob <= 1.0


def test_save_and_load_fusion_artifact_and_metrics(tmp_path: Path) -> None:
    df = build_dataset(total_n=260, enforce_training_size=False)
    tabular_bundle, _ = train_tabular_bundle(
        df,
        random_state=42,
        model_params={"n_estimators": 25, "learning_rate": 0.08},
    )
    artifact, metrics = train_fusion_meta(df, tabular_bundle, random_state=42)

    artifact_path = tmp_path / "fusion.joblib"
    metrics_path = tmp_path / "fusion_metrics.json"
    save_fusion_artifact(artifact, artifact_path)
    save_fusion_metrics(metrics, metrics_path)

    loaded = load_fusion_artifact(artifact_path)
    original_prob = predict_fusion_meta(0.72, 0.63, 0.58, artifact)
    loaded_prob = predict_fusion_meta(0.72, 0.63, 0.58, loaded)

    assert abs(original_prob - loaded_prob) < 1e-9
    assert metrics_path.exists()


def test_fusion_artifact_paths_constants() -> None:
    artifact_norm = str(FUSION_V2_ARTIFACT_PATH).replace("\\", "/")
    metrics_norm = str(FUSION_V2_METRICS_PATH).replace("\\", "/")
    assert artifact_norm.endswith("artifacts/cheops_v2_fusion_meta.joblib")
    assert metrics_norm.endswith("artifacts/cheops_v2_fusion_metrics.json")
