"""Tests for the nine legacy features computed from raw events.

Two of these pin defects that were found the expensive way: a feature coming
out as a flat zero, and a window taken from the data rather than from the
analysis period.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.legacy_features_v2 import (
    LEGACY_BOUNDS,
    LEGACY_NAMES,
    fifo_match,
    infer_referrals,
    legacy_features,
    referral_depth,
)

T0 = datetime(2026, 1, 1)
T1 = T0 + timedelta(days=120)


def _event(sender: str, receiver: str, amount: float, day: float, *, cash: bool = False):
    return TransactionEvent(
        event_id=f"E{sender}{receiver}{day}",
        ts=T0 + timedelta(days=day),
        amount=amount,
        currency="KZT",
        sender_id=sender,
        receiver_id=receiver,
        sender_type="person",
        receiver_type="atm" if cash else "company",
        channel="legal",
        jurisdiction="KZ",
        asset_type="cash" if cash else "fiat",
    )


# ==========================================================================
# FIFO matching
# ==========================================================================


def test_fifo_matches_first_in_against_first_out():
    events = [
        _event("A", "CORE", 100.0, 0),
        _event("B", "CORE", 100.0, 10),
        _event("CORE", "X", 100.0, 20),
    ]
    matches, total_in, total_out, unmatched = fifo_match("CORE", events)
    assert total_in == pytest.approx(200.0)
    assert total_out == pytest.approx(100.0)
    assert unmatched == pytest.approx(0.0)
    # The parcel that left is the one that arrived first: 20 days held.
    assert len(matches) == 1
    assert matches[0].held_seconds == pytest.approx(timedelta(days=20).total_seconds())


def test_fifo_splits_a_parcel_across_two_outflows():
    events = [
        _event("A", "CORE", 100.0, 0),
        _event("CORE", "X", 40.0, 5),
        _event("CORE", "Y", 60.0, 9),
    ]
    matches, _, _, unmatched = fifo_match("CORE", events)
    assert len(matches) == 2
    assert sum(m.amount for m in matches) == pytest.approx(100.0)
    assert unmatched == pytest.approx(0.0)


def test_spending_a_pre_existing_balance_is_reported_as_unmatched():
    """Money leaving with no matching arrival is what an ordinary account
    does and a pass-through does not."""
    events = [_event("CORE", "X", 500.0, 3)]
    _, total_in, total_out, unmatched = fifo_match("CORE", events)
    assert total_in == pytest.approx(0.0)
    assert total_out == pytest.approx(500.0)
    assert unmatched == pytest.approx(500.0)


# ==========================================================================
# Referral reconstruction
# ==========================================================================


def test_referrals_are_reconstructed_from_the_bonus_payment():
    """A deposit followed by a percentage paid to an existing participant."""
    events = [
        _event("P1", "CORE", 100_000.0, 0),
        _event("P2", "CORE", 200_000.0, 1),
        _event("CORE", "P1", 20_000.0, 1.5),   # 10 % of P2's deposit
    ]
    referrals = infer_referrals("CORE", events)
    assert referrals.get("P2") == "P1"


def test_a_payout_outside_the_window_is_not_a_referral_bonus():
    events = [
        _event("P1", "CORE", 100_000.0, 0),
        _event("P2", "CORE", 200_000.0, 1),
        _event("CORE", "P1", 20_000.0, 40),   # far too late
    ]
    assert infer_referrals("CORE", events) == {}


def test_referral_depth_survives_a_cycle():
    """The tree is inferred from noisy evidence and can contain a cycle a
    real chain never would. Walking one without a guard hangs."""
    assert referral_depth({"A": "B", "B": "C", "C": "A"}, root="CORE") <= 64


# ==========================================================================
# The nine features
# ==========================================================================


def _pyramid_events(depositors: int = 40) -> list[TransactionEvent]:
    """Recruitment that accelerates, which is what makes it a pyramid.

    A linear arrival rate is the one shape ``growth_rate`` is entitled to
    score at zero, so a fixture built that way tests nothing.
    """
    events: list[TransactionEvent] = []
    for i in range(depositors):
        # Concave in the index, so the gap between arrivals SHRINKS: sparse
        # at first, crowded by the end. Eleven join before the midpoint of
        # the analysis window and twenty-nine after it.
        day = 100.0 * (i / depositors) ** 0.4
        events.append(_event(f"INV{i}", "CORE", 100_000.0, day))
        if i > 0:
            events.append(_event("CORE", f"INV{i - 1}", 12_000.0, day + 0.5))
    return events


def test_all_nine_are_returned_and_inside_their_bounds():
    features = legacy_features("CORE", _pyramid_events(), T0, T1)
    assert set(features) == set(LEGACY_NAMES)
    for name, value in features.items():
        low, high = LEGACY_BOUNDS[name]
        assert low <= value <= high, f"{name}={value} outside {LEGACY_BOUNDS[name]}"


def test_growth_is_not_flat_zero_when_the_entity_is_growing():
    """Pinned because it was exactly 0.000 for every pyramid once.

    The analysis window had been taken as the range of the events rather than
    the analysis period, and ordinary spending drags the tail far past the
    horizon, moving the midpoint past all the growth.
    """
    features = legacy_features("CORE", _pyramid_events(), T0, T1)
    assert features["growth_rate"] > 0.1


def test_repeat_depositors_move_the_reinvestment_rate():
    """Also pinned as a flat zero once: investors deposited exactly once."""
    once = legacy_features("CORE", _pyramid_events(), T0, T1)["reinvestment_rate"]

    repeated = _pyramid_events()
    repeated += [_event(f"INV{i}", "CORE", 50_000.0, 90 + i) for i in range(10)]
    twice = legacy_features("CORE", repeated, T0, T1)["reinvestment_rate"]

    assert twice > once


def test_payout_dependency_reads_the_ponzi_definition():
    """Paying out more than comes in is the literal definition, and the
    feature must move with it."""
    modest = legacy_features("CORE", _pyramid_events(), T0, T1)["payout_dependency"]

    heavy = _pyramid_events()
    heavy += [_event("CORE", f"INV{i}", 80_000.0, 100 + i * 0.1) for i in range(30)]
    heavier = legacy_features("CORE", heavy, T0, T1)["payout_dependency"]

    assert heavier > modest


def test_the_window_is_the_analysis_period_not_the_data_range():
    """Events outside the requested window must be ignored entirely."""
    events = _pyramid_events()
    events.append(_event("LATE", "CORE", 5_000_000.0, 500))
    inside = legacy_features("CORE", events, T0, T1)
    without = legacy_features("CORE", _pyramid_events(), T0, T1)
    assert inside == without
