"""
Risk scorer for Company-Level Crypto-Ponzi Detection.

Implements weighted formula to compute Crypto_Ponzi_Probability (0-1)
and assign risk level: Low / Medium / High / Critical.
"""
from __future__ import annotations

from typing import Any


# ── Weights ────────────────────────────────────────────────────────
FACTOR_WEIGHTS = {
    "dependency_ratio": 0.25,
    "crypto_exposure_ratio": 0.20,
    "concentration_index": 0.20,
    "inverse_holding_time": 0.20,
    "inverse_entropy": 0.15,
}

# ── Normalization bounds ───────────────────────────────────────────
# Used to normalize raw metrics into [0, 1] before scoring.
NORM_BOUNDS = {
    "dependency_ratio": (0.0, 1.5),
    "crypto_exposure_ratio": (0.0, 0.8),
    "concentration_index": (0.0, 0.7),
    "avg_holding_time": (1.0, 50.0),
    "entropy_of_flows": (0.0, 1.0),
}

# ── Risk level thresholds ─────────────────────────────────────────
RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.50,
    "high": 0.70,
}

RISK_LABELS = {
    "Low": {"ru": "Низкий", "emoji": "✅", "css": "risk-banner-low"},
    "Medium": {"ru": "Средний", "emoji": "⚠️", "css": "risk-banner-medium"},
    "High": {"ru": "Высокий", "emoji": "🔶", "css": "risk-banner-high"},
    "Critical": {"ru": "Критический", "emoji": "🚨", "css": "risk-banner-critical"},
}


def _normalize(value: float, low: float, high: float) -> float:
    """Normalize value to [0, 1] range using min-max scaling."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _risk_level(probability: float) -> str:
    """Map probability to risk level string."""
    if probability >= RISK_THRESHOLDS["high"]:
        return "Critical"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "High"
    if probability >= RISK_THRESHOLDS["low"]:
        return "Medium"
    return "Low"


def compute_crypto_ponzi_score(metrics: dict[str, float]) -> dict[str, Any]:
    """
    Compute Crypto_Ponzi_Probability and risk level.

    Formula (weighted):
        0.25 * dependency_ratio_norm
      + 0.20 * crypto_exposure_ratio_norm
      + 0.20 * concentration_index_norm
      + 0.20 * (1 - avg_holding_time_norm)   [inverse: short holding = risky]
      + 0.15 * (1 - entropy_norm)             [inverse: low entropy = risky]

    Returns:
        {
            probability: float,
            risk_level: str,
            risk_level_ru: str,
            risk_emoji: str,
            risk_css: str,
            factor_contributions: dict[str, float],
            normalized_factors: dict[str, float],
        }
    """
    # Normalize raw metrics
    dep_norm = _normalize(
        metrics.get("dependency_ratio", 0.0),
        *NORM_BOUNDS["dependency_ratio"],
    )
    crypto_norm = _normalize(
        metrics.get("crypto_exposure_ratio", 0.0),
        *NORM_BOUNDS["crypto_exposure_ratio"],
    )
    conc_norm = _normalize(
        metrics.get("concentration_index", 0.0),
        *NORM_BOUNDS["concentration_index"],
    )
    holding_norm = _normalize(
        metrics.get("avg_holding_time", 90.0),
        *NORM_BOUNDS["avg_holding_time"],
    )
    entropy_norm = _normalize(
        metrics.get("entropy_of_flows", 0.5),
        *NORM_BOUNDS["entropy_of_flows"],
    )

    # Inverse metrics (lower = riskier)
    inv_holding = 1.0 - holding_norm
    inv_entropy = 1.0 - entropy_norm

    # Factor contributions
    contributions = {
        "dependency_ratio": 0.25 * dep_norm,
        "crypto_exposure_ratio": 0.20 * crypto_norm,
        "concentration_index": 0.20 * conc_norm,
        "inverse_holding_time": 0.20 * inv_holding,
        "inverse_entropy": 0.15 * inv_entropy,
    }

    probability = sum(contributions.values())
    probability = max(0.0, min(1.0, probability))

    level = _risk_level(probability)
    label_info = RISK_LABELS[level]

    return {
        "probability": round(probability, 4),
        "risk_level": level,
        "risk_level_ru": label_info["ru"],
        "risk_emoji": label_info["emoji"],
        "risk_css": label_info["css"],
        "factor_contributions": contributions,
        "normalized_factors": {
            "dependency_ratio": round(dep_norm, 4),
            "crypto_exposure_ratio": round(crypto_norm, 4),
            "concentration_index": round(conc_norm, 4),
            "inverse_holding_time": round(inv_holding, 4),
            "inverse_entropy": round(inv_entropy, 4),
        },
    }


# ── Factor labels for display ─────────────────────────────────────
FACTOR_LABELS_RU = {
    "dependency_ratio": "Зависимость выплат от притока",
    "crypto_exposure_ratio": "Крипто-экспозиция",
    "concentration_index": "Концентрация контрагентов",
    "inverse_holding_time": "Короткий срок удержания",
    "inverse_entropy": "Неравномерность потоков",
}
