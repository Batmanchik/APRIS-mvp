from __future__ import annotations

from pathlib import Path

from apris.cheops.infrastructure.ml.model_registry_v2 import (
    attach_benchmark_report,
    build_model_registry,
    load_model_registry,
    save_model_registry,
)


def _legacy_metrics() -> dict[str, object]:
    return {
        "roc_auc": 0.99,
        "accuracy": 0.95,
        "recall_pyramid": 0.93,
        "precision_pyramid": 0.94,
        "random_state": 42,
        "dataset_rows": 4000,
        "feature_count": 9,
        "threshold_policy": "fixed_0.4_0.7",
        "threshold_values": {"medium": 0.4, "high": 0.7},
        "model": {"name": "LightGBMClassifier"},
    }


def _tabular_metrics() -> dict[str, object]:
    return {"global": {"roc_auc": 0.98, "brier": 0.04, "ece": 0.02}}


def _sequence_metrics() -> dict[str, object]:
    return {"sequence_head": {"roc_auc": 0.96, "brier": 0.06, "ece": 0.03}}


def _graph_metrics() -> dict[str, object]:
    return {"graph_head": {"roc_auc": 0.95, "brier": 0.07, "ece": 0.04}}


def _fusion_metrics() -> dict[str, object]:
    return {
        "meta_head": {"roc_auc": 0.97, "brier": 0.05, "ece": 0.03},
        "fallback_weighted": {"roc_auc": 0.98, "brier": 0.04, "ece": 0.02},
    }


def test_build_model_registry_without_benchmark_defaults_to_lightgbm() -> None:
    registry = build_model_registry(
        legacy_metrics=_legacy_metrics(),
        tabular_metrics=_tabular_metrics(),
        sequence_metrics=_sequence_metrics(),
        graph_metrics=_graph_metrics(),
        fusion_metrics=_fusion_metrics(),
        benchmark_report=None,
    )

    assert registry["registry_version"] == "cheops-model-registry-v2"
    assert registry["selected_tabular_candidate"] == "lightgbm"
    assert "metrics" in registry
    assert registry["benchmark"] is None


def test_attach_benchmark_report_updates_selection() -> None:
    registry = build_model_registry(
        legacy_metrics=_legacy_metrics(),
        tabular_metrics=_tabular_metrics(),
        sequence_metrics=_sequence_metrics(),
        graph_metrics=_graph_metrics(),
        fusion_metrics=_fusion_metrics(),
        benchmark_report=None,
    )
    benchmark_report = {
        "winner": "catboost",
        "winner_reason": "catboost selected by score policy",
    }
    updated = attach_benchmark_report(registry, benchmark_report)

    assert updated["selected_tabular_candidate"] == "catboost"
    assert updated["selection_reason"] == "catboost selected by score policy"
    assert updated["benchmark"]["winner"] == "catboost"
    assert "updated_at" in updated


def test_save_and_load_model_registry(tmp_path: Path) -> None:
    registry = build_model_registry(
        legacy_metrics=_legacy_metrics(),
        tabular_metrics=_tabular_metrics(),
        sequence_metrics=_sequence_metrics(),
        graph_metrics=_graph_metrics(),
        fusion_metrics=_fusion_metrics(),
        benchmark_report=None,
    )
    path = tmp_path / "registry.json"
    saved = save_model_registry(registry, path)

    assert saved == path
    loaded = load_model_registry(path)
    assert loaded["registry_version"] == "cheops-model-registry-v2"
    assert loaded["selected_tabular_candidate"] == "lightgbm"
