from __future__ import annotations

from pathlib import Path

from apris.cheops.infrastructure.ml.drift_v2 import (
    DRIFT_REPORT_V2_PATH,
    FEATURE_PROFILE_V2_PATH,
    build_drift_report,
    build_feature_profile,
    load_feature_profile,
    save_drift_report,
    save_feature_profile,
)
from apris.data_generator import FEATURE_COLUMNS, build_dataset


def test_feature_profile_and_drift_report_basics() -> None:
    df = build_dataset(total_n=500, enforce_training_size=False)
    features = df[FEATURE_COLUMNS].copy()
    profile = build_feature_profile(features, dataset_name="train.csv", random_state=42)

    same_report = build_drift_report(profile, features, dataset_name="same.csv")
    shifted = features.copy()
    shifted["centralization_index"] = 1.0
    shifted_report = build_drift_report(profile, shifted, dataset_name="shifted.csv")

    assert profile["profile_version"] == "cheops-feature-profile-v2"
    assert same_report["drift_version"] == "cheops-drift-v2"
    assert same_report["rows_current"] == len(features)
    assert shifted_report["overall_psi"] >= same_report["overall_psi"]


def test_save_and_load_feature_profile_and_drift_report(tmp_path: Path) -> None:
    df = build_dataset(total_n=300, enforce_training_size=False)
    features = df[FEATURE_COLUMNS].copy()
    profile = build_feature_profile(features, dataset_name="train.csv", random_state=42)
    report = build_drift_report(profile, features, dataset_name="test.csv")

    profile_path = tmp_path / "feature_profile.json"
    report_path = tmp_path / "drift_report.json"
    save_feature_profile(profile, profile_path)
    save_drift_report(report, report_path)
    loaded = load_feature_profile(profile_path)

    assert loaded["profile_version"] == "cheops-feature-profile-v2"
    assert report_path.exists()


def test_drift_v2_default_paths_constants() -> None:
    profile_norm = str(FEATURE_PROFILE_V2_PATH).replace("\\", "/")
    drift_norm = str(DRIFT_REPORT_V2_PATH).replace("\\", "/")
    assert profile_norm.endswith("artifacts/cheops_v2_feature_profile.json")
    assert drift_norm.endswith("artifacts/cheops_v2_drift_report.json")
