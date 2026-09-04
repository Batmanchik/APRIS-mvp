from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apris.data_generator import FEATURE_COLUMNS

FEATURE_PROFILE_V2_PATH = Path("artifacts") / "cheops_v2_feature_profile.json"
DRIFT_REPORT_V2_PATH = Path("artifacts") / "cheops_v2_drift_report.json"


def _to_float_list(values: np.ndarray) -> list[float]:
    return [float(x) for x in values.tolist()]


def _expected_bins(values: pd.Series, *, bins: int = 10) -> np.ndarray:
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(clean, quantiles)
    edges = np.asarray(edges, dtype=float)
    for idx in range(1, len(edges)):
        if edges[idx] <= edges[idx - 1]:
            edges[idx] = edges[idx - 1] + 1e-9
    return edges


def _psi_from_edges(base_values: np.ndarray, current_values: np.ndarray, edges: np.ndarray) -> float:
    eps = 1e-9
    if base_values.size == 0 or current_values.size == 0:
        return 0.0
    base_hist, _ = np.histogram(base_values, bins=edges)
    current_hist, _ = np.histogram(current_values, bins=edges)
    base_rate = base_hist / max(base_hist.sum(), 1)
    current_rate = current_hist / max(current_hist.sum(), 1)
    base_rate = np.clip(base_rate.astype(float), eps, 1.0)
    current_rate = np.clip(current_rate.astype(float), eps, 1.0)
    psi = np.sum((base_rate - current_rate) * np.log(base_rate / current_rate))
    return float(psi)


def _psi_from_rates(base_rate: np.ndarray, current_rate: np.ndarray) -> float:
    eps = 1e-9
    base = np.clip(base_rate.astype(float), eps, 1.0)
    current = np.clip(current_rate.astype(float), eps, 1.0)
    psi = np.sum((base - current) * np.log(base / current))
    return float(psi)


def _drift_level(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "high"


def build_feature_profile(
    features_df: pd.DataFrame,
    *,
    dataset_name: str,
    random_state: int,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "profile_version": "cheops-feature-profile-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "random_state": random_state,
        "rows": int(len(features_df)),
        "features": {},
    }
    for name in FEATURE_COLUMNS:
        series = features_df[name].astype(float)
        edges = _expected_bins(series, bins=10)
        baseline_hist, _ = np.histogram(series.to_numpy(dtype=float), bins=edges)
        baseline_rate = baseline_hist / max(baseline_hist.sum(), 1)
        profile["features"][name] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "p05": float(series.quantile(0.05)),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "bin_edges": _to_float_list(edges),
            "bin_rates": _to_float_list(np.asarray(baseline_rate, dtype=float)),
        }
    return profile


def save_feature_profile(
    profile: dict[str, Any],
    path: str | Path = FEATURE_PROFILE_V2_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return target


def load_feature_profile(path: str | Path = FEATURE_PROFILE_V2_PATH) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Cheops feature profile not found: {source}")
    profile = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError("Cheops feature profile has invalid format.")
    if profile.get("profile_version") != "cheops-feature-profile-v2":
        raise ValueError("Cheops feature profile has unexpected version.")
    return profile


def build_drift_report(
    baseline_profile: dict[str, Any],
    current_features_df: pd.DataFrame,
    *,
    dataset_name: str,
) -> dict[str, Any]:
    baseline_features = baseline_profile.get("features")
    if not isinstance(baseline_features, dict):
        raise ValueError("Baseline profile missing 'features'.")

    feature_reports: dict[str, Any] = {}
    psi_values: list[float] = []
    for name in FEATURE_COLUMNS:
        if name not in current_features_df.columns:
            raise ValueError(f"Current dataset missing feature '{name}'.")
        feature_base = baseline_features.get(name)
        if not isinstance(feature_base, dict):
            raise ValueError(f"Baseline profile missing feature stats for '{name}'.")
        edges = np.asarray(feature_base.get("bin_edges", []), dtype=float)
        if edges.size < 2:
            raise ValueError(f"Baseline profile for '{name}' has invalid bin_edges.")
        base_rates = np.asarray(feature_base.get("bin_rates", []), dtype=float)
        if base_rates.size != edges.size - 1:
            raise ValueError(f"Baseline profile for '{name}' has invalid bin_rates.")

        base_mean = float(feature_base["mean"])
        base_std = float(feature_base["std"])
        current_values = current_features_df[name].astype(float).to_numpy(dtype=float)
        current_hist, _ = np.histogram(current_values, bins=edges)
        current_rates = current_hist / max(current_hist.sum(), 1)
        psi = _psi_from_rates(base_rates, np.asarray(current_rates, dtype=float))
        psi_values.append(psi)

        current_mean = float(np.mean(current_values))
        current_std = float(np.std(current_values, ddof=0))
        feature_reports[name] = {
            "psi": float(psi),
            "level": _drift_level(psi),
            "baseline_mean": base_mean,
            "current_mean": current_mean,
            "baseline_std": base_std,
            "current_std": current_std,
            "mean_shift_abs": float(abs(current_mean - base_mean)),
        }

    overall_psi = float(np.mean(psi_values)) if psi_values else 0.0
    high_count = sum(1 for value in psi_values if value >= 0.25)
    moderate_count = sum(1 for value in psi_values if 0.10 <= value < 0.25)

    return {
        "drift_version": "cheops-drift-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_dataset": baseline_profile.get("dataset_name", "unknown"),
        "current_dataset": dataset_name,
        "rows_current": int(len(current_features_df)),
        "overall_psi": overall_psi,
        "overall_level": _drift_level(overall_psi),
        "features_high_drift": int(high_count),
        "features_moderate_drift": int(moderate_count),
        "feature_reports": feature_reports,
    }


def save_drift_report(
    report: dict[str, Any],
    path: str | Path = DRIFT_REPORT_V2_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
