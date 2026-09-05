"""Evaluation methods that stop a result from being an accident.

Three techniques, each answering a way a detection result can look good and
mean nothing.

1. **Purged walk-forward splits.** Ordinary k-fold puts future data in the
   training set and past data in the test set. On temporal data that is
   training on the future to predict the past, and it inflates every score
   quietly. Splits here are time-ordered, and a purge gap sits between train
   and test so cases whose event windows overlap the boundary cannot leak
   across it.

2. **Quintile ladder.** A single AUC says a score separates *somewhere*. It
   does not say the score is ordered. Sorting objects by the score into
   buckets and reading the fraud rate per bucket does: a real signal produces
   a monotonic ladder, a lucky one produces a jumble with the same AUC.

3. **Noise floor from a shuffled control.** Importance rankings are read as
   if every listed feature matters. Adding a deliberately random column and
   measuring it the same way gives an explicit floor: whatever cannot beat a
   shuffled column is noise, however good the story around it sounds.

All three come from quantitative finance practice, where the same failures
appear as inflated backtests. The transfer is deliberate and partial — the
techniques that apply to any temporal, imbalanced detection problem are here,
and the ones specific to price series (stationarity of returns, ADF on
prices) are not, because this project has no price series and applying them
would be decoration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np


# ==========================================================================
# 1. Purged walk-forward splits
# ==========================================================================


@dataclass(frozen=True)
class WalkForwardSplit:
    index: int
    train: tuple[int, ...]
    test: tuple[int, ...]
    purged: tuple[int, ...]
    train_end: datetime
    test_start: datetime

    @property
    def purged_count(self) -> int:
        return len(self.purged)


def purged_walk_forward_splits(
    timestamps: Sequence[datetime],
    *,
    n_splits: int = 5,
    purge: timedelta = timedelta(days=3),
    min_train: int = 20,
) -> list[WalkForwardSplit]:
    """Expanding-window splits with a purge gap between train and test.

    Objects are ordered by time. Each split trains on everything up to a
    boundary and tests on the block after it; anything inside ``purge`` of
    the boundary is dropped from training rather than silently included.

    The purge is what makes this different from a plain time split. A case
    spans a window of events, so a case whose window straddles the boundary
    carries information from the test period into training. Dropping it costs
    a few rows and removes the leak.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if not timestamps:
        return []

    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    total = len(order)
    if total <= min_train:
        return []

    usable = total - min_train
    block = max(1, usable // n_splits)

    splits: list[WalkForwardSplit] = []
    for split_index in range(n_splits):
        train_size = min_train + split_index * block
        test_start_pos = train_size
        test_end_pos = min(total, test_start_pos + block)
        if test_start_pos >= total or test_end_pos <= test_start_pos:
            break

        test_positions = order[test_start_pos:test_end_pos]
        test_start_ts = timestamps[test_positions[0]]
        boundary = test_start_ts - purge

        train_positions: list[int] = []
        purged_positions: list[int] = []
        for position in order[:test_start_pos]:
            if timestamps[position] > boundary:
                purged_positions.append(position)
            else:
                train_positions.append(position)

        if not train_positions:
            continue

        splits.append(
            WalkForwardSplit(
                index=split_index,
                train=tuple(train_positions),
                test=tuple(test_positions),
                purged=tuple(purged_positions),
                train_end=timestamps[train_positions[-1]],
                test_start=test_start_ts,
            )
        )
    return splits


# ==========================================================================
# 2. Quintile ladder
# ==========================================================================


@dataclass(frozen=True)
class LadderResult:
    bucket_rates: tuple[float, ...]
    bucket_sizes: tuple[int, ...]
    spread: float
    monotonic: bool
    rank_correlation: float

    def describe(self) -> str:
        shape = "monotonic" if self.monotonic else "NOT monotonic"
        return (
            f"{shape}, top-bottom spread {self.spread:+.3f}, "
            f"rank correlation {self.rank_correlation:+.3f}"
        )


def quintile_ladder(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    buckets: int = 5,
) -> LadderResult:
    """Sort by score, bucket, and read the positive rate per bucket.

    A real signal ranks: the rate rises from the bottom bucket to the top.
    A score with a good AUC and a jumbled ladder is separating on a subset
    rather than ordering the population, and it will not survive a threshold
    being moved.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if len(scores) < buckets:
        raise ValueError(f"need at least {buckets} objects")

    order = np.argsort(np.asarray(scores, dtype=float), kind="stable")
    label_array = np.asarray(labels, dtype=float)[order]
    chunks = np.array_split(label_array, buckets)

    rates = tuple(float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks)
    sizes = tuple(int(chunk.size) for chunk in chunks)
    spread = rates[-1] - rates[0]
    monotonic = all(rates[i] <= rates[i + 1] + 1e-12 for i in range(len(rates) - 1))

    positions = np.arange(len(rates), dtype=float)
    if np.std(rates) > 0:
        correlation = float(np.corrcoef(positions, np.asarray(rates))[0, 1])
    else:
        correlation = 0.0

    return LadderResult(
        bucket_rates=rates,
        bucket_sizes=sizes,
        spread=float(spread),
        monotonic=bool(monotonic),
        rank_correlation=correlation,
    )


# ==========================================================================
# 3. Noise floor from a shuffled control
# ==========================================================================


@dataclass(frozen=True)
class ImportanceResult:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    errors: tuple[float, ...]
    noise_floor: float
    noise_floor_error: float

    def above_floor(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, mean in zip(self.feature_names, self.means)
            if mean > self.noise_floor
        )

    def below_floor(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, mean in zip(self.feature_names, self.means)
            if mean <= self.noise_floor
        )


def permutation_importance_with_noise_floor(
    fit_predict: object,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    repeats: int = 10,
    seed: int = 42,
) -> ImportanceResult:
    """Permutation importance measured against an explicit noise floor.

    A shuffled control column is appended to the matrix and treated exactly
    like a real feature. Whatever scores no better than that column carries
    no information, and the comparison is on the same axis rather than an
    eyeballed threshold.

    ``fit_predict`` must expose scikit-learn's ``fit`` and ``predict_proba``.
    Error bars are the standard deviation over repeats: a single run proves
    nothing, since the whole quantity is an average over random shuffles.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    control = rng.permutation(np.asarray(y, dtype=float)).reshape(-1, 1)
    augmented = np.hstack([np.asarray(x, dtype=float), control])
    names = (*feature_names, "shuffled_control")

    model = fit_predict
    model.fit(augmented, y)  # type: ignore[attr-defined]
    baseline = roc_auc_score(y, np.asarray(model.predict_proba(augmented))[:, 1])  # type: ignore[attr-defined]

    means: list[float] = []
    errors: list[float] = []
    for column in range(augmented.shape[1]):
        drops: list[float] = []
        for _ in range(repeats):
            shuffled = augmented.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            score = roc_auc_score(
                y, np.asarray(model.predict_proba(shuffled))[:, 1]  # type: ignore[attr-defined]
            )
            drops.append(float(baseline - score))
        means.append(float(np.mean(drops)))
        errors.append(float(np.std(drops)))

    return ImportanceResult(
        feature_names=names[:-1],
        means=tuple(means[:-1]),
        errors=tuple(errors[:-1]),
        noise_floor=means[-1],
        noise_floor_error=errors[-1],
    )
