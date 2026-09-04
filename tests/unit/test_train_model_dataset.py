from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from apris import train_model
from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS


def _midpoint(feature_name: str) -> float:
    low, high = FEATURE_BOUNDS[feature_name]
    return float((low + high) / 2.0)


def _external_feature_df() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for label in [0, 1]:
        row: dict[str, float | int] = {name: _midpoint(name) for name in FEATURE_COLUMNS}
        if label == 1:
            row["growth_rate"] = min(FEATURE_BOUNDS["growth_rate"][1], row["growth_rate"] + 0.05)
        row["label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def test_prepare_dataset_external_without_borderline_uses_training_validation(
    monkeypatch,
) -> None:
    external_df = _external_feature_df()

    monkeypatch.setattr(train_model, "process_external_dataset", lambda _: external_df)

    def _unexpected_validate_dataset(*_: object, **__: object) -> None:
        raise AssertionError("validate_dataset must not be called for external data without is_borderline")

    monkeypatch.setattr(train_model, "validate_dataset", _unexpected_validate_dataset)

    prepared = train_model._prepare_dataset("external.csv")
    assert len(prepared) == 2
    assert "is_borderline" not in prepared.columns
    assert set(prepared["label"].tolist()) == {0, 1}


def test_load_features_for_drift_requires_feature_columns(tmp_path: Path) -> None:
    valid_row = {name: _midpoint(name) for name in FEATURE_COLUMNS}
    valid_path = tmp_path / "drift_ok.csv"
    pd.DataFrame([valid_row]).to_csv(valid_path, index=False)

    features = train_model._load_features_for_drift(str(valid_path))
    assert list(features.columns) == FEATURE_COLUMNS

    broken_path = tmp_path / "drift_broken.csv"
    pd.DataFrame([{"growth_rate": 0.2}]).to_csv(broken_path, index=False)
    with pytest.raises(ValueError, match="missing"):
        train_model._load_features_for_drift(str(broken_path))
