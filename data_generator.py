from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd


SEED = 42
np.random.seed(SEED)

FEATURE_COLUMNS = [
    "growth_rate",
    "referral_ratio",
    "payout_dependency",
    "centralization_index",
    "avg_holding_time",
    "reinvestment_rate",
    "gini_coefficient",
    "transaction_entropy",
    "structural_depth",
]

FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "growth_rate": (0.0, 1.2),
    "referral_ratio": (0.0, 1.0),
    "payout_dependency": (0.1, 1.9),
    "centralization_index": (0.0, 1.0),
    "avg_holding_time": (3.0, 120.0),
    "reinvestment_rate": (0.0, 1.0),
    "gini_coefficient": (0.1, 1.0),
    "transaction_entropy": (0.3, 5.0),
    "structural_depth": (2.0, 16.0),
}

RISK_THRESHOLDS = {"medium": 0.4, "high": 0.7}
TARGET_BORDERLINE_SHARE = 0.15
TARGET_BORDERLINE_TOL = 0.02
MIN_CORRELATION = 0.2
MIN_OVERLAP_RATIO = 0.05


def _clip(values: np.ndarray, feature: str) -> np.ndarray:
    low, high = FEATURE_BOUNDS[feature]
    return np.clip(values, low, high)


def _base_legitimate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "growth_rate": _clip(rng.normal(0.11, 0.09, n), "growth_rate"),
            "referral_ratio": _clip(rng.beta(2.1, 4.2, n), "referral_ratio"),
            "payout_dependency": _clip(rng.normal(0.66, 0.22, n), "payout_dependency"),
            "centralization_index": _clip(rng.beta(2.2, 3.8, n), "centralization_index"),
            "avg_holding_time": _clip(rng.normal(55.0, 22.0, n), "avg_holding_time"),
            "reinvestment_rate": _clip(rng.beta(2.8, 3.0, n), "reinvestment_rate"),
            "gini_coefficient": _clip(rng.normal(0.41, 0.14, n), "gini_coefficient"),
            "transaction_entropy": _clip(rng.normal(3.3, 0.7, n), "transaction_entropy"),
            "structural_depth": _clip(rng.integers(2, 11, size=n).astype(float), "structural_depth"),
        }
    )
    return df


def _base_pyramid(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "growth_rate": _clip(rng.normal(0.33, 0.20, n), "growth_rate"),
            "referral_ratio": _clip(rng.beta(4.5, 2.5, n), "referral_ratio"),
            "payout_dependency": _clip(rng.normal(1.02, 0.30, n), "payout_dependency"),
            "centralization_index": _clip(rng.beta(4.2, 2.4, n), "centralization_index"),
            "avg_holding_time": _clip(rng.normal(30.0, 15.0, n), "avg_holding_time"),
            "reinvestment_rate": _clip(rng.beta(4.0, 2.5, n), "reinvestment_rate"),
            "gini_coefficient": _clip(rng.normal(0.67, 0.16, n), "gini_coefficient"),
            "transaction_entropy": _clip(rng.normal(2.0, 0.8, n), "transaction_entropy"),
            "structural_depth": _clip(rng.integers(4, 17, size=n).astype(float), "structural_depth"),
        }
    )
    return df


def _apply_correlations(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    growth_adjust = 0.38 * (df["referral_ratio"] - 0.5) + rng.normal(0.0, 0.03, len(df))
    df["growth_rate"] = _clip(df["growth_rate"] + growth_adjust, "growth_rate")

    gini_adjust = 0.48 * (df["centralization_index"] - 0.5) + rng.normal(0.0, 0.03, len(df))
    df["gini_coefficient"] = _clip(df["gini_coefficient"] + gini_adjust, "gini_coefficient")
    return df


def _inject_borderline(
    df: pd.DataFrame,
    label: int,
    rng: np.random.Generator,
    share: float,
) -> pd.DataFrame:
    result = df.copy()
    result["is_borderline"] = False
    border_count = int(round(len(result) * share))
    if border_count <= 0:
        return result

    borderline_idx = rng.choice(result.index.to_numpy(), size=border_count, replace=False)
    result.loc[borderline_idx, "is_borderline"] = True

    if label == 0:
        result.loc[borderline_idx, "referral_ratio"] = _clip(
            result.loc[borderline_idx, "referral_ratio"] + rng.normal(0.22, 0.07, border_count),
            "referral_ratio",
        )
        result.loc[borderline_idx, "growth_rate"] = _clip(
            result.loc[borderline_idx, "growth_rate"] + rng.normal(0.14, 0.08, border_count),
            "growth_rate",
        )
        result.loc[borderline_idx, "payout_dependency"] = _clip(
            result.loc[borderline_idx, "payout_dependency"] + rng.normal(0.18, 0.08, border_count),
            "payout_dependency",
        )
        result.loc[borderline_idx, "centralization_index"] = _clip(
            result.loc[borderline_idx, "centralization_index"] + rng.normal(0.16, 0.07, border_count),
            "centralization_index",
        )
        result.loc[borderline_idx, "avg_holding_time"] = _clip(
            result.loc[borderline_idx, "avg_holding_time"] - rng.normal(10.0, 4.0, border_count),
            "avg_holding_time",
        )
        result.loc[borderline_idx, "structural_depth"] = _clip(
            result.loc[borderline_idx, "structural_depth"] + rng.integers(1, 4, size=border_count),
            "structural_depth",
        )
    else:
        result.loc[borderline_idx, "referral_ratio"] = _clip(
            result.loc[borderline_idx, "referral_ratio"] - rng.normal(0.20, 0.08, border_count),
            "referral_ratio",
        )
        result.loc[borderline_idx, "growth_rate"] = _clip(
            result.loc[borderline_idx, "growth_rate"] - rng.normal(0.12, 0.08, border_count),
            "growth_rate",
        )
        result.loc[borderline_idx, "payout_dependency"] = _clip(
            result.loc[borderline_idx, "payout_dependency"] - rng.normal(0.20, 0.10, border_count),
            "payout_dependency",
        )
        result.loc[borderline_idx, "centralization_index"] = _clip(
            result.loc[borderline_idx, "centralization_index"] - rng.normal(0.18, 0.07, border_count),
            "centralization_index",
        )
        result.loc[borderline_idx, "avg_holding_time"] = _clip(
            result.loc[borderline_idx, "avg_holding_time"] + rng.normal(12.0, 5.0, border_count),
            "avg_holding_time",
        )
        result.loc[borderline_idx, "structural_depth"] = _clip(
            result.loc[borderline_idx, "structural_depth"] - rng.integers(1, 4, size=border_count),
            "structural_depth",
        )

    return _apply_correlations(result, rng)


def generate_legitimate(
    n: int,
    seed: int = SEED,
    borderline_share: float = TARGET_BORDERLINE_SHARE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = _base_legitimate(n, rng)
    df = _inject_borderline(base, label=0, rng=rng, share=borderline_share)
    df["label"] = 0
    return df


def generate_pyramid(
    n: int,
    seed: int = SEED + 1,
    borderline_share: float = TARGET_BORDERLINE_SHARE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = _base_pyramid(n, rng)
    df = _inject_borderline(base, label=1, rng=rng, share=borderline_share)
    df["label"] = 1
    return df


def build_dataset(
    total_n: int = 4000,
    seed: int = SEED,
    borderline_share: float = TARGET_BORDERLINE_SHARE,
    enforce_training_size: bool = True,
) -> pd.DataFrame:
    if enforce_training_size:
        assert 3000 <= total_n <= 5000, "Dataset size must be in range [3000, 5000]."
    else:
        assert 200 <= total_n <= 5000, "Dataset size must be in range [200, 5000]."
    half = total_n // 2
    legit_n = half
    pyramid_n = total_n - half

    legit = generate_legitimate(legit_n, seed=seed, borderline_share=borderline_share)
    pyramid = generate_pyramid(pyramid_n, seed=seed + 1, borderline_share=borderline_share)
    df = pd.concat([legit, pyramid], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validate_dataset(df, borderline_target=borderline_share)
    return df


def generate_live_batch(
    total_n: int = 800,
    seed: int | None = None,
    borderline_share: float = TARGET_BORDERLINE_SHARE,
) -> tuple[pd.DataFrame, int]:
    if seed is None:
        seed = int(time.time_ns() % (2**32 - 1))

    dataset = build_dataset(
        total_n=total_n,
        seed=seed,
        borderline_share=borderline_share,
        enforce_training_size=False,
    )
    return dataset, seed


def validate_dataset(
    df: pd.DataFrame,
    borderline_target: float = TARGET_BORDERLINE_SHARE,
    borderline_tol: float = TARGET_BORDERLINE_TOL,
    min_correlation: float = MIN_CORRELATION,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
) -> None:
    required = FEATURE_COLUMNS + ["label", "is_borderline"]
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    finite = np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float))
    assert finite.all(), "Dataset contains NaN or Inf values."

    for feature, (low, high) in FEATURE_BOUNDS.items():
        values = df[feature].astype(float)
        assert (values >= low).all(), f"{feature} has values below {low}"
        assert (values <= high).all(), f"{feature} has values above {high}"

    corr_ref_growth = float(df["referral_ratio"].corr(df["growth_rate"]))
    corr_cent_gini = float(df["centralization_index"].corr(df["gini_coefficient"]))
    assert corr_ref_growth > min_correlation, (
        f"referral_ratio-growth_rate corr too low: {corr_ref_growth:.3f}"
    )
    assert corr_cent_gini > min_correlation, (
        f"centralization_index-gini_coefficient corr too low: {corr_cent_gini:.3f}"
    )

    borderline_share = float(df["is_borderline"].mean())
    assert abs(borderline_share - borderline_target) <= borderline_tol, (
        f"borderline share {borderline_share:.3f} out of range "
        f"{borderline_target:.3f} +/- {borderline_tol:.3f}"
    )

    pyramid_share = float(df["label"].mean())
    assert 0.48 <= pyramid_share <= 0.52, "Label distribution is not close to 50/50."

    for feature in FEATURE_COLUMNS:
        q0_low = float(df[df["label"] == 0][feature].quantile(0.05))
        q0_high = float(df[df["label"] == 0][feature].quantile(0.95))
        q1_low = float(df[df["label"] == 1][feature].quantile(0.05))
        q1_high = float(df[df["label"] == 1][feature].quantile(0.95))
        overlap = max(0.0, min(q0_high, q1_high) - max(q0_low, q1_low))
        span = max(q0_high, q1_high) - min(q0_low, q1_low)
        overlap_ratio = 0.0 if span == 0 else overlap / span
        assert overlap_ratio > min_overlap_ratio, (
            f"Insufficient class overlap for {feature}: overlap_ratio={overlap_ratio:.3f}"
        )


def dataset_report(df: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": float(len(df)),
        "pyramid_share": float(df["label"].mean()),
        "borderline_share": float(df["is_borderline"].mean()),
        "corr_referral_growth": float(df["referral_ratio"].corr(df["growth_rate"])),
        "corr_centralization_gini": float(df["centralization_index"].corr(df["gini_coefficient"])),
    }


def main() -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(total_n=4000, seed=SEED, borderline_share=TARGET_BORDERLINE_SHARE)
    csv_path = artifacts_dir / "synthetic_dataset.csv"
    dataset.to_csv(csv_path, index=False)

    report = dataset_report(dataset)
    print("Data generation: OK")
    for key, value in report.items():
        if key == "rows":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"saved_csv: {csv_path}")


if __name__ == "__main__":
    main()
