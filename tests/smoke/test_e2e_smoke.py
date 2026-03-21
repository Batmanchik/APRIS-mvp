from __future__ import annotations

from pathlib import Path

import pytest

from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, build_dataset
from apris.risk_engine import load_artifacts, predict_risk
from apris.train_model import train_and_save


@pytest.mark.smoke
def test_train_save_and_infer_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    df = build_dataset(total_n=600, enforce_training_size=False)
    result = train_and_save(df, random_state=42)

    assert "metrics" in result
    assert "importances" in result
    assert (tmp_path / "artifacts" / "model.joblib").exists()
    assert (tmp_path / "artifacts" / "feature_names.json").exists()

    model, feature_names = load_artifacts()
    features = {
        name: float((FEATURE_BOUNDS[name][0] + FEATURE_BOUNDS[name][1]) / 2.0)
        for name in FEATURE_COLUMNS
    }
    prediction = predict_risk(features, model=model, feature_names=feature_names)
    assert 0.0 <= prediction["probability"] <= 1.0
