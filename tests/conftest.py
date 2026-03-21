from __future__ import annotations

import numpy as np
import pytest

from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS


class DummyModel:
    def __init__(self, prob: float = 0.73, importances: list[float] | None = None) -> None:
        self._prob = float(prob)
        self.feature_importances_ = np.array(
            importances if importances is not None else list(range(1, len(FEATURE_COLUMNS) + 1)),
            dtype=float,
        )

    def predict_proba(self, x: object) -> np.ndarray:
        try:
            rows = len(x)  # type: ignore[arg-type]
        except TypeError:
            rows = 1
        probs = np.full(rows, self._prob, dtype=float)
        return np.column_stack([1.0 - probs, probs])


@pytest.fixture
def sample_features() -> dict[str, float]:
    features: dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        low, high = FEATURE_BOUNDS[name]
        features[name] = float((low + high) / 2.0)
    return features


@pytest.fixture
def sample_operational() -> dict[str, float]:
    return {
        "tx_count_total": 1200.0,
        "unique_counterparties": 320.0,
        "new_clients_current": 90.0,
        "new_clients_previous": 70.0,
        "referred_clients_current": 22.0,
        "incoming_funds": 2_100_000.0,
        "payouts_total": 1_300_000.0,
        "top1_wallet_share": 0.18,
        "top10_wallet_share": 0.55,
        "avg_holding_days": 42.0,
        "repeat_investor_share": 0.38,
        "max_referral_depth": 5.0,
    }


@pytest.fixture
def dummy_model() -> DummyModel:
    return DummyModel()
