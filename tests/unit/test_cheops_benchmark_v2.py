from __future__ import annotations

import json
from pathlib import Path

from apris.cheops.infrastructure.ml.benchmark_v2 import (
    BENCHMARK_REPORT_PATH,
    run_tabular_benchmark,
    save_benchmark_report,
)
from apris.data_generator import build_dataset


def test_run_tabular_benchmark_lightgbm_only() -> None:
    df = build_dataset(total_n=400, enforce_training_size=False)
    report = run_tabular_benchmark(df, random_state=42, include_optional=False)

    assert report["winner"] == "lightgbm"
    assert "selection_policy" in report
    assert isinstance(report["ranking"], list)
    assert "lightgbm" in report["candidates"]
    assert report["candidates"]["lightgbm"]["status"] == "trained"
    assert "selection_score" in report["candidates"]["lightgbm"]
    assert 0.0 <= report["candidates"]["lightgbm"]["ece"] <= 1.0


def test_save_benchmark_report(tmp_path: Path) -> None:
    report = {
        "benchmark_version": "x",
        "winner": "lightgbm",
        "candidates": {
            "lightgbm": {"status": "trained", "roc_auc": 0.99},
        },
    }
    path = tmp_path / "benchmark.json"
    saved = save_benchmark_report(report, path)

    assert saved == path
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["winner"] == "lightgbm"


def test_benchmark_default_report_path_constant() -> None:
    normalized = str(BENCHMARK_REPORT_PATH).replace("\\", "/")
    assert normalized.endswith("artifacts/cheops_v2_benchmark.json")
