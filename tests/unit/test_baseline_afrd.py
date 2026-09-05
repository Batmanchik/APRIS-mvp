"""Tests for the published-rule baseline.

The point of a baseline is that it is not a straw man, so most of these pin
the criteria FIRING. The two that pin an abstention are the argument of the
whole project and matter more: a rule that cannot fire on a fresh ring is
what leaves the gap this work measures.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.baseline_afrd import (
    APPLICABLE_CRITERIA,
    UNAVAILABLE_CRITERIA,
    afrd_verdict,
    criteria_for_scope,
)

T0 = datetime(2026, 4, 1, 9, 0, 0)


def _event(sender: str, receiver: str, amount: float, minutes: float, *, cash: bool = False):
    return TransactionEvent(
        event_id=f"E{sender}{receiver}{minutes}",
        ts=T0 + timedelta(minutes=minutes),
        amount=amount,
        currency="KZT",
        sender_id=sender,
        receiver_id=receiver,
        sender_type="person",
        receiver_type="atm" if cash else "person",
        channel="legal",
        jurisdiction="KZ",
        asset_type="cash" if cash else "fiat",
    )


# ==========================================================================
# The criteria fire
# ==========================================================================


def test_shared_terminal_trips_the_device_criterion():
    events = [
        _event("M1", "ATM1", 200_000.0, 0, cash=True),
        _event("M2", "ATM1", 200_000.0, 7, cash=True),
    ]
    verdict = afrd_verdict(["M1", "M2"], events)
    assert verdict.shared_device


def test_one_member_using_a_terminal_twice_is_not_a_shared_device():
    """An ordinary person making two withdrawals must not trip a rule about
    two people sharing hardware."""
    events = [
        _event("M1", "ATM1", 200_000.0, 0, cash=True),
        _event("M1", "ATM1", 150_000.0, 9, cash=True),
    ]
    assert not afrd_verdict(["M1", "M2"], events).shared_device


def test_the_same_terminal_a_week_later_is_not_shared_use():
    events = [
        _event("M1", "ATM1", 200_000.0, 0, cash=True),
        _event("M2", "ATM1", 200_000.0, 60 * 24 * 7, cash=True),
    ]
    assert not afrd_verdict(["M1", "M2"], events).shared_device


def test_paying_a_listed_account_trips_the_list_criterion():
    events = [_event("M1", "KNOWN", 500_000.0, 0)]
    verdict = afrd_verdict(["M1", "M2"], events, listed=frozenset({"KNOWN"}))
    assert verdict.listed


def test_being_paid_by_a_listed_account_also_trips_it():
    events = [_event("KNOWN", "M1", 500_000.0, 0)]
    assert afrd_verdict(["M1"], events, listed=frozenset({"KNOWN"})).listed


def test_a_listed_member_of_the_candidate_itself_does_not_count():
    """Otherwise the baseline is scored on an answer handed in with the
    question — the same leak the case builder was rewritten to remove."""
    events = [_event("M1", "M2", 500_000.0, 0)]
    verdict = afrd_verdict(["M1", "M2"], events, listed=frozenset({"M2"}))
    assert not verdict.listed


def test_a_spike_against_an_account_s_own_baseline_trips_deviation():
    events = [_event("P", "SHOP", 10_000.0, day * 24 * 60) for day in range(30)]
    events.append(_event("P", "SHOP", 900_000.0, 31 * 24 * 60))
    assert afrd_verdict(["P"], events).profile_deviation


def test_a_consistently_large_account_does_not_trip_deviation():
    """The comparison is against the account's own median, so being richer
    than average is not a rule violation."""
    events = [_event("RICH", "SHOP", 900_000.0, day * 24 * 60) for day in range(30)]
    assert not afrd_verdict(["RICH"], events).profile_deviation


# ==========================================================================
# The criteria abstain — the argument of the work
# ==========================================================================


def test_deviation_cannot_fire_on_an_account_with_no_history():
    """A ring of freshly opened accounts has no usual profile to deviate
    from, so criterion 4 is unavailable exactly where it is needed."""
    events = [_event("FRESH", "ATM1", 900_000.0, minute, cash=True) for minute in (0, 5, 9)]
    assert not afrd_verdict(["FRESH"], events).profile_deviation


def test_a_clean_ring_off_the_list_scores_zero_on_everything_but_the_exit():
    """Nothing on the list, no history, and — if the organiser uses a
    different terminal per mule — no shared device either."""
    events = [
        _event("SRC", "M1", 400_000.0, 0),
        _event("SRC", "M2", 400_000.0, 1),
        _event("M1", "ATM1", 390_000.0, 4, cash=True),
        _event("M2", "ATM2", 390_000.0, 6, cash=True),
    ]
    verdict = afrd_verdict(["M1", "M2"], events)
    assert verdict.fired == 0
    assert verdict.score == 0.0
    assert not verdict.flagged_any


def test_the_missing_criterion_is_reported_not_silently_dropped():
    verdict = afrd_verdict(["M1"], [])
    assert verdict.unavailable == UNAVAILABLE_CRITERIA
    assert "shared_phone" in verdict.unavailable
    assert len(APPLICABLE_CRITERIA) == 3


# ==========================================================================
# The score is rankable
# ==========================================================================


def test_score_is_the_share_of_expressible_criteria_that_fired():
    events = [
        _event("M1", "ATM1", 200_000.0, 0, cash=True),
        _event("M2", "ATM1", 200_000.0, 7, cash=True),
        _event("M1", "KNOWN", 100_000.0, 20),
    ]
    verdict = afrd_verdict(["M1", "M2"], events, listed=frozenset({"KNOWN"}))
    assert verdict.fired == 2
    assert verdict.score == 2 / 3
    assert verdict.flagged_any and verdict.flagged_two


# ==========================================================================
# Which criteria the score is taken over
# ==========================================================================


def test_shared_device_is_excluded_from_the_score_at_both_units():
    """It abstains at one unit and is circular at the other.

    Account: one account is not two people sharing hardware, measured 0.000
    on both classes. Network: it fires on 100% of fraudulent candidates and
    4.2% of honest ones because discovery links accounts BY the shared
    terminal, so it reports which link built the candidate.
    """
    for scope in ("account", "network"):
        assert "shared_device" not in criteria_for_scope(scope)
        assert criteria_for_scope(scope) == ("listed", "profile_deviation")


def test_an_unknown_scope_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        criteria_for_scope("cluster")


def test_an_excluded_criterion_is_still_evaluated_and_reported():
    """Excluding it from the score and hiding it are different acts. The
    breakdown is the evidence for the exclusion."""
    events = [
        _event("M1", "ATM1", 200_000.0, 0, cash=True),
        _event("M2", "ATM1", 200_000.0, 7, cash=True),
    ]
    verdict = afrd_verdict(["M1", "M2"], events, counted=criteria_for_scope("network"))
    assert verdict.shared_device, "the criterion must still be visible"
    assert verdict.fired == 0, "but it must not move the score"
    assert verdict.score == 0.0


def test_the_score_is_over_the_counted_criteria_only():
    events = [
        _event("M1", "ATM1", 200_000.0, 0, cash=True),
        _event("M2", "ATM1", 200_000.0, 7, cash=True),
        _event("M1", "KNOWN", 100_000.0, 20),
    ]
    verdict = afrd_verdict(
        ["M1", "M2"],
        events,
        listed=frozenset({"KNOWN"}),
        counted=criteria_for_scope("account"),
    )
    # listed fired, profile_deviation did not: one of two.
    assert verdict.fired == 1
    assert verdict.score == 0.5
