"""Tests for the prevalence sweep.

Two things need pinning here and they fail in opposite directions.

The thinning is the sweep's only claim on the word "measured": if it drops
honest rows, or lands somewhere other than the prevalence it was asked for,
every cell below the natural share is a number about a world nobody
specified.

The projection is arithmetic, so it can be checked exactly rather than
plausibly. It is also the half most likely to be believed without checking,
because it produces a smooth curve whatever it is fed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from apris.cheops.infrastructure.experiments.ladder import Row
from apris.cheops.infrastructure.experiments.prevalence import (
    HONEST_SCALE,
    PREVALENCE_TARGETS,
    project_at_budget,
    project_at_recall,
    scaled_world_config,
    thin_positives,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _rows(positives: int, negatives: int) -> list[Row]:
    total = positives + negatives
    return [
        Row(
            key=str(i),
            ts=START + timedelta(hours=i),
            features={"a": float(i)},
            label=1 if i < positives else 0,
            events=(),
            members=(str(i),),
        )
        for i in range(total)
    ]


# ==========================================================================
# Thinning: the only thing that may change is how rare fraud is
# ==========================================================================


@pytest.mark.parametrize("target", PREVALENCE_TARGETS)
def test_thinning_lands_on_the_prevalence_it_was_asked_for(target: float):
    rows = _rows(positives=2000, negatives=20000)
    thinned = thin_positives(rows, target, np.random.default_rng(1))
    achieved = sum(r.label for r in thinned) / len(thinned)
    # One positive is the granularity, so the tolerance is one row's worth.
    assert abs(achieved - target) <= 1.0 / len(thinned) + 1e-9


def test_thinning_never_touches_an_honest_row():
    """Dropping negatives would raise prevalence while claiming to lower it.

    It would also delete exactly the population the difficulty of the world
    is made of, and the metric would improve for the wrong reason.
    """
    rows = _rows(positives=500, negatives=5000)
    thinned = thin_positives(rows, 0.01, np.random.default_rng(2))
    kept_honest = [r.key for r in thinned if r.label == 0]
    assert kept_honest == [r.key for r in rows if r.label == 0]


def test_thinning_keeps_the_rows_in_time_order():
    """The split downstream is by time; a reshuffle here would silently
    change which fold a row lands in and nothing would complain."""
    rows = _rows(positives=300, negatives=3000)
    thinned = thin_positives(rows, 0.01, np.random.default_rng(3))
    stamps = [r.ts for r in thinned]
    assert stamps == sorted(stamps)


def test_thinning_is_reproducible_from_the_seed():
    rows = _rows(positives=300, negatives=3000)
    first = thin_positives(rows, 0.01, np.random.default_rng(7))
    second = thin_positives(rows, 0.01, np.random.default_rng(7))
    assert [r.key for r in first] == [r.key for r in second]


def test_asking_for_more_fraud_than_the_world_holds_is_refused():
    """Reaching a higher prevalence means deleting honest people, which is a
    different experiment. Refusing beats silently returning the rows."""
    rows = _rows(positives=10, negatives=1000)
    with pytest.raises(ValueError, match="above the natural share"):
        thin_positives(rows, 0.5, np.random.default_rng(0))


# ==========================================================================
# Projection: exact arithmetic, checked exactly
# ==========================================================================


def test_precision_at_a_recall_matches_the_formula_by_hand():
    """TPR 0.8 at FPR 0.01, one fraud in a thousand.

    alerted = 0.001*0.8 + 0.999*0.01 = 0.010790
    caught  = 0.000800
    precision = 0.0008 / 0.01079 = 0.074143
    """
    fpr = np.array([0.0, 0.01, 0.10, 1.0])
    tpr = np.array([0.0, 0.80, 0.95, 1.0])
    point = project_at_recall(fpr, tpr, prevalence=0.001, recall=0.8)

    assert point.recall == pytest.approx(0.80)
    assert point.precision == pytest.approx(0.0008 / 0.01079, rel=1e-9)
    assert point.alerts_per_1000_accounts == pytest.approx(10.79, rel=1e-9)
    assert point.reviews_per_catch == pytest.approx(1.0 / point.precision)


def test_the_same_detector_loses_precision_as_fraud_gets_rarer():
    """The whole point of the sweep, as a monotonicity rather than a value."""
    fpr = np.array([0.0, 0.01, 0.10, 1.0])
    tpr = np.array([0.0, 0.80, 0.95, 1.0])
    precisions = [
        project_at_recall(fpr, tpr, prevalence=p, recall=0.8).precision
        for p in (0.25, 0.0687, 0.01, 0.001)
    ]
    assert precisions == sorted(precisions, reverse=True)


def test_a_budget_is_spent_on_honest_accounts_when_fraud_is_rare():
    """At a ten-per-cent budget the alert rate is capped, so the recall the
    budget buys is whatever the ROC gives at that false-positive rate."""
    fpr = np.array([0.0, 0.01, 0.10, 1.0])
    tpr = np.array([0.0, 0.80, 0.95, 1.0])
    point = project_at_budget(fpr, tpr, prevalence=0.001, budget=0.10)

    assert point.alert_rate <= 0.10 + 1e-12
    assert point.recall == pytest.approx(0.80)
    assert point.precision < 0.10


def test_a_perfect_separation_still_prices_correctly():
    """A detector with no false positives keeps precision 1.0 at any
    prevalence — the arithmetic must not manufacture a loss that is not there."""
    fpr = np.array([0.0, 0.0, 1.0])
    tpr = np.array([0.0, 1.0, 1.0])
    point = project_at_recall(fpr, tpr, prevalence=0.0001, recall=1.0)
    assert point.precision == pytest.approx(1.0)
    assert point.reviews_per_catch == pytest.approx(1.0)


# ==========================================================================
# The world the sweep runs in
# ==========================================================================


def test_scaling_multiplies_the_honest_side_and_leaves_the_fraud_alone():
    """A sweep that also added rings would change two things at once, and the
    fall in precision could then be read as either."""
    base = scaled_world_config(seed=1, scale=1)
    scaled = scaled_world_config(seed=1, scale=HONEST_SCALE)

    assert scaled.mule_networks == base.mule_networks
    assert scaled.pyramids == base.pyramids
    assert scaled.terminals == base.terminals
    assert scaled.salary_earners == base.salary_earners * HONEST_SCALE
    assert scaled.fast_spenders == base.fast_spenders * HONEST_SCALE
    assert scaled.crowd_collections == base.crowd_collections * HONEST_SCALE


def test_the_targets_stay_below_the_share_the_world_can_reach():
    """Every target must be reachable by thinning alone. The AFC's 6.87 % is
    the world's own natural share and is reported unthinned, so the swept
    targets sit strictly below it."""
    assert max(PREVALENCE_TARGETS) < 0.0687
    assert min(PREVALENCE_TARGETS) > 0.0
