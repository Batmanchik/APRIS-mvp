from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_REGISTRY_V2_PATH = Path("artifacts") / "cheops_v2_model_registry.json"


def build_model_registry(
    *,
    legacy_metrics: dict[str, Any],
    tabular_metrics: dict[str, Any],
    sequence_metrics: dict[str, Any],
    graph_metrics: dict[str, Any],
    fusion_metrics: dict[str, Any],
    benchmark_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_tabular_candidate = "lightgbm"
    winner_reason: str | None = "Default baseline candidate (no benchmark report attached)."
    if benchmark_report is not None and benchmark_report.get("winner") is not None:
        selected_tabular_candidate = str(benchmark_report["winner"])
        winner_reason = (
            benchmark_report.get("winner_reason")
            or f"Selected by benchmark winner: {selected_tabular_candidate}"
        )

    return {
        "registry_version": "cheops-model-registry-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "random_state": legacy_metrics.get("random_state"),
            "dataset_rows": legacy_metrics.get("dataset_rows"),
            "feature_count": legacy_metrics.get("feature_count"),
            "threshold_policy": legacy_metrics.get("threshold_policy"),
            "threshold_values": legacy_metrics.get("threshold_values"),
            "legacy_model_name": legacy_metrics.get("model", {}).get("name"),
        },
        "selected_tabular_candidate": selected_tabular_candidate,
        "selection_reason": winner_reason,
        "artifacts": {
            "legacy_model": "artifacts/model.joblib",
            "tabular_v2_bundle": "artifacts/cheops_v2_tabular.joblib",
            "sequence_v2_model": "artifacts/cheops_v2_sequence.joblib",
            "graph_v2_model": "artifacts/cheops_v2_graph.joblib",
            "fusion_v2_meta": "artifacts/cheops_v2_fusion_meta.joblib",
            "feature_profile_v2": "artifacts/cheops_v2_feature_profile.json",
        },
        "metrics": {
            "legacy": {
                "roc_auc": legacy_metrics.get("roc_auc"),
                "accuracy": legacy_metrics.get("accuracy"),
                "recall_pyramid": legacy_metrics.get("recall_pyramid"),
                "precision_pyramid": legacy_metrics.get("precision_pyramid"),
            },
            "tabular_v2": {
                "global_roc_auc": tabular_metrics.get("global", {}).get("roc_auc"),
                "global_brier": tabular_metrics.get("global", {}).get("brier"),
                "global_ece": tabular_metrics.get("global", {}).get("ece"),
            },
            "sequence_v2": {
                "roc_auc": sequence_metrics.get("sequence_head", {}).get("roc_auc"),
                "brier": sequence_metrics.get("sequence_head", {}).get("brier"),
                "ece": sequence_metrics.get("sequence_head", {}).get("ece"),
            },
            "graph_v2": {
                "roc_auc": graph_metrics.get("graph_head", {}).get("roc_auc"),
                "brier": graph_metrics.get("graph_head", {}).get("brier"),
                "ece": graph_metrics.get("graph_head", {}).get("ece"),
            },
            "fusion_v2": {
                "meta_roc_auc": fusion_metrics.get("meta_head", {}).get("roc_auc"),
                "meta_brier": fusion_metrics.get("meta_head", {}).get("brier"),
                "meta_ece": fusion_metrics.get("meta_head", {}).get("ece"),
                "fallback_roc_auc": fusion_metrics.get("fallback_weighted", {}).get("roc_auc"),
                "fallback_brier": fusion_metrics.get("fallback_weighted", {}).get("brier"),
                "fallback_ece": fusion_metrics.get("fallback_weighted", {}).get("ece"),
            },
        },
        "benchmark": benchmark_report,
    }


def attach_benchmark_report(
    registry: dict[str, Any],
    benchmark_report: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(registry)
    selected_tabular_candidate = str(benchmark_report.get("winner") or "lightgbm")
    updated["selected_tabular_candidate"] = selected_tabular_candidate
    updated["selection_reason"] = (
        benchmark_report.get("winner_reason")
        or f"Selected by benchmark winner: {selected_tabular_candidate}"
    )
    updated["benchmark"] = benchmark_report
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated


def save_model_registry(
    registry: dict[str, Any],
    path: str | Path = MODEL_REGISTRY_V2_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return target


def load_model_registry(path: str | Path = MODEL_REGISTRY_V2_PATH) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Cheops model registry not found: {target}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Cheops model registry has invalid format.")
    if payload.get("registry_version") != "cheops-model-registry-v2":
        raise ValueError("Cheops model registry has unexpected version.")
    return payload
