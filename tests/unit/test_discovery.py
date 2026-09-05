"""Tests for candidate discovery.

The guarantee under test is not "discovery finds a lot". It is that discovery
never consults the answers — and that the recall ceiling it imposes is
reported rather than hidden.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta

import pytest

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.simulation.config import SimulationConfig
from apris.cheops.infrastructure.simulation.discovery import (
    discover_candidates,
    label_candidates,
)
from apris.cheops.infrastructure.simulation.generator import generate_world

T0 = datetime(2026, 3, 1, 12, 0, 0)

SMALL = SimulationConfig(
    seed=13,
    days=40,
    salary_earners=120,
    freelancers=20,
    traders=12,
    fast_spenders=60,
    family_circles=6,
    crowd_collections=10,
    marketplace_sellers=30,
    employers=3,
    mule_networks=10,
    pyramids=3,
    terminals=10,
    merchants=50,
)


@pytest.fixture(scope="module")
def world():
    return generate_world(SMALL)


def _event(sender: str, receiver: str, offset_minutes: float, *, cash: bool = False):
    return TransactionEvent(
        event_id=f"E{sender}{receiver}{int(offset_minutes * 60)}",
        ts=T0 + timedelta(minutes=offset_minutes),
        amount=100_000.0,
        currency="KZT",
        sender_id=sender,
        receiver_id=receiver,
        sender_type="person",
        receiver_type="atm" if cash else "person",
        channel="legal",
        jurisdiction="KZ",
        asset_type="cash" if cash else "fiat",
    )


class _FakeWorld:
    """Minimal stand-in carrying events and nothing else."""

    def __init__(self, events):
        self.events = events
        self.networks = []
        self.populations = {}
        self.accounts = {}


# ==========================================================================
# The guarantee
# ==========================================================================


def test_discovery_ignores_the_answer_file(world):
    """Removing every network must not change what discovery proposes.

    This is the whole point. The previous case builder grouped accounts from
    ``world.networks``, which handed the detector the hardest half of the
    problem already solved.
    """
    with_answers = discover_candidates(world)
    blinded = dataclasses.replace(world, networks=[])
    without_answers = discover_candidates(blinded)

    assert [c.member_ids for c in with_answers] == [c.member_ids for c in without_answers]
    assert with_answers, "discovery must propose something"


def test_shared_terminal_links_accounts():
    """Two accounts cashing out at one ATM minutes apart form one candidate."""
    events = [
        _event("SRC", "A", 0),
        _event("SRC", "B", 1),
        _event("A", "ATM1", 3, cash=True),
        _event("B", "ATM1", 5, cash=True),
    ]
    candidates = discover_candidates(_FakeWorld(events), min_size=2)
    assert candidates
    members = set(candidates[0].member_ids)
    assert {"A", "B"} <= members
    assert "shared_terminal" in candidates[0].link_reasons


def test_common_ancestor_links_accounts_without_a_direct_edge():
    """Accounts that never pay each other are still linked by a shared origin.

    This is what survives an evasion that randomises the direct sender:
    the shared origin moves one hop up rather than disappearing.
    """
    events = [
        _event("ORIGIN", "F1", 0),
        _event("ORIGIN", "F2", 1),
        _event("F1", "M1", 2),
        _event("F2", "M2", 3),
    ]
    candidates = discover_candidates(_FakeWorld(events), min_size=2)
    assert candidates
    members = set(candidates[0].member_ids)
    assert {"M1", "M2"} <= members
    assert "common_ancestor" in candidates[0].link_reasons


def test_accounts_far_apart_in_time_are_not_linked():
    """The window is real: the same ATM a week later is not the same event."""
    events = [
        _event("SRC", "A", 0),
        _event("A", "ATM1", 2, cash=True),
        _event("SRC2", "B", 10_000),
        _event("B", "ATM1", 10_002, cash=True),
    ]
    candidates = discover_candidates(
        _FakeWorld(events), min_size=2, terminal_window=timedelta(minutes=30)
    )
    for candidate in candidates:
        members = set(candidate.member_ids)
        assert not ({"A", "B"} <= members)


def test_candidate_sizes_are_bounded(world):
    candidates = discover_candidates(world, min_size=3, max_size=50)
    assert candidates
    assert all(3 <= c.size <= 50 for c in candidates)


# ==========================================================================
# The ceiling
# ==========================================================================


def test_coverage_is_reported_and_is_a_real_ceiling(world):
    """Networks discovery never proposed cannot be found by any model.

    The previous design could not produce this number at all, because the
    grouping came from the answers and coverage was one by construction.
    """
    candidates = discover_candidates(world)
    _, report = label_candidates(world, candidates)

    assert report.networks_total > 0
    assert 0.0 <= report.coverage <= 1.0
    assert report.networks_covered + len(report.missed_network_ids) == report.networks_total


def test_labels_are_attached_after_discovery_not_before(world):
    """Labelling must not change the candidates it labels."""
    candidates = discover_candidates(world)
    before = [c.member_ids for c in candidates]
    labels, _ = label_candidates(world, candidates)
    after = [c.member_ids for c in candidates]

    assert before == after
    assert len(labels) == len(candidates)
    assert set(labels) <= {0, 1}


def test_both_classes_are_present(world):
    """Discovery must propose honest clusters too, or there is no task."""
    candidates = discover_candidates(world)
    labels, _ = label_candidates(world, candidates)
    assert 0 in labels, "discovery proposed no honest candidates"
    assert 1 in labels, "discovery proposed no fraudulent candidates"
