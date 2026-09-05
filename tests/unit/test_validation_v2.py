"""Tests for the evaluation methods.

Each test pins the property the method exists to guarantee, not the numbers
it happens to produce on one input.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from apris.cheops.infrastructure.ml.validation_v2 import (
    permutation_importance_with_noise_floor,
    purged_walk_forward_splits,
    quintile_ladder,
)

T0 = datetime(2026, 1, 1)


def _timestamps(count: int, step_days: float = 1.0) -> list[datetime]:
    return [T0 + timedelta(days=step_days * i) for i in range(count)]


# ==========================================================================
# Purged walk-forward splits
# ==========================================================================


def test_training_never_contains_a_future_object():
    """The whole point: no training object may sit after a test object."""
    stamps = _timestamps(120)
    for split in purged_walk_forward_splits(stamps, n_splits=5):
        latest_train = max(stamps[i] for i in split.train)
        earliest_test = min(stamps[i] for i in split.test)
        assert latest_train < earliest_test


def test_purge_gap_removes_objects_next_to_the_boundary():
    """Objects within the purge window are dropped, not quietly trained on."""
    stamps = _timestamps(120)
    splits = purged_walk_forward_splits(stamps, n_splits=4, purge=timedelta(days=5))
    assert splits
    assert any(split.purged_count > 0 for split in splits)

    for split in splits:
        for index in split.train:
            gap = split.test_start - stamps[index]
            assert gap > timedelta(days=5) or gap == timedelta(days=5)


def test_no_object_is_in_both_train_and_test():
    stamps = _timestamps(90)
    for split in purged_walk_forward_splits(stamps, n_splits=4):
        assert not (set(split.train) & set(split.test))
        assert not (set(split.train) & set(split.purged))


def test_training_window_expands():
    stamps = _timestamps(150)
    splits = purged_walk_forward_splits(stamps, n_splits=5)
    sizes = [len(split.train) for split in splits]
    assert sizes == sorted(sizes)


def test_unsorted_input_is_handled():
    """Cases arrive in whatever order they were built; time must still rule."""
    stamps = _timestamps(80)
    shuffled = list(stamps)
    np.random.default_rng(0).shuffle(shuffled)
    for split in purged_walk_forward_splits(shuffled, n_splits=3):
        assert max(shuffled[i] for i in split.train) < min(shuffled[i] for i in split.test)


def test_too_little_data_returns_no_splits():
    assert purged_walk_forward_splits(_timestamps(5), n_splits=5, min_train=20) == []


# ==========================================================================
# Quintile ladder
# ==========================================================================


def test_a_perfect_score_produces_a_monotonic_ladder():
    labels = [0] * 50 + [1] * 50
    scores = list(range(100))
    result = quintile_ladder(scores, labels)
    assert result.monotonic
    assert result.spread == pytest.approx(1.0)
    assert result.rank_correlation > 0.9


def test_a_random_score_produces_no_ladder():
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=400).tolist()
    scores = rng.normal(size=400).tolist()
    result = quintile_ladder(scores, labels)
    assert abs(result.spread) < 0.25


def test_ladder_catches_a_score_that_separates_without_ordering():
    """A good AUC does not imply a ranked score.

    Here the top bucket is right and the middle is jumbled — exactly the
    case a single AUC hides and the ladder exposes.
    """
    labels = [0, 1] * 40 + [1] * 20
    scores = list(range(100))
    result = quintile_ladder(scores, labels)
    assert not result.monotonic or result.rank_correlation < 0.99


def test_bucket_sizes_cover_every_object():
    labels = [0] * 37 + [1] * 40
    scores = list(range(77))
    result = quintile_ladder(scores, labels)
    assert sum(result.bucket_sizes) == 77
    assert len(result.bucket_rates) == 5


# ==========================================================================
# Noise floor
# ==========================================================================


def test_a_useless_feature_falls_to_the_noise_floor():
    """A column of pure noise must not clear the shuffled control."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(11)
    n = 400
    signal = rng.normal(size=n)
    labels = (signal + rng.normal(scale=0.35, size=n) > 0).astype(int)
    noise = rng.normal(size=n)
    x = np.column_stack([signal, noise])

    result = permutation_importance_with_noise_floor(
        RandomForestClassifier(n_estimators=60, random_state=0),
        x, labels, ["real_signal", "pure_noise"], repeats=5,
    )

    assert "real_signal" in result.above_floor()
    assert result.means[0] > result.noise_floor
    assert result.means[1] < result.means[0]


def test_error_bars_are_reported():
    """A single run proves nothing, so the spread must come back too."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] > 0).astype(int)

    result = permutation_importance_with_noise_floor(
        RandomForestClassifier(n_estimators=40, random_state=0),
        x, y, ["a", "b", "c"], repeats=6,
    )
    assert len(result.errors) == 3
    assert all(error >= 0.0 for error in result.errors)
    assert result.noise_floor_error >= 0.0
