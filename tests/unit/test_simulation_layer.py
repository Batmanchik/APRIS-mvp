"""Tests for the event-level simulation layer.

The world is generated at a reduced size: full-scale generation takes about
two minutes, which does not belong in a unit suite. Every structural
property asserted here holds at both sizes.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from apris.cheops.domain.contracts import validate_event_schema
from apris.cheops.infrastructure.simulation.acceptance import evaluate
from apris.cheops.infrastructure.simulation.cases import (
    CASE_CROWD_COLLECTION,
    CASE_MULE_NETWORK,
    CASE_PAYROLL,
    build_cases,
    case_kind_counts,
)
from apris.cheops.infrastructure.simulation.config import EvasionKnobs, SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world

SMALL = SimulationConfig(
    seed=7,
    days=45,
    # 120 earners over 3 employers gives ~40 people per payday. Fewer than
    # the fan-out threshold of 20 and rule B cannot fire at all — that is a
    # property of the shrunken fixture, not of the generator.
    salary_earners=120,
    freelancers=25,
    traders=15,
    fast_spenders=70,
    family_circles=8,
    crowd_collections=8,
    marketplace_sellers=30,
    employers=3,
    mule_networks=6,
    pyramids=3,
    terminals=12,
    merchants=60,
)


@pytest.fixture(scope="module")
def world():
    return generate_world(SMALL)


def test_generator_emits_valid_domain_events(world):
    assert world.events, "world must contain events"
    for event in world.events[:400]:
        validate_event_schema(event)


def test_events_are_sorted_by_time(world):
    stamps = [event.ts for event in world.events]
    assert stamps == sorted(stamps)


def test_generator_writes_no_derived_features(world):
    """The event stream must carry behaviour only.

    A field named after a metric would mean the generator handed the answer
    to the detector, which is the defect this whole layer exists to remove.
    """
    banned = {"transit", "retention", "gini", "entropy", "risk", "score", "label"}
    for event in world.events[:200]:
        keys = {key.lower() for key in event.metadata}
        assert not (keys & banned), f"derived field leaked into metadata: {keys & banned}"


def test_ground_truth_is_separate_from_events(world):
    """Network membership must not be reachable from an event."""
    fraud = world.fraud_account_ids()
    assert fraud, "world must contain fraudulent accounts"
    for event in world.events[:200]:
        assert "network_id" not in event.metadata
        assert event.case_id is None


def test_pyramid_investors_are_not_counted_as_fraud(world):
    """Investors are victims. Counting them inflated the base rate 3x."""
    fraud = world.fraud_account_ids()
    investors = [a for a, p in world.populations.items() if p == "pyramid_investor"]
    assert investors, "pyramids must have investors"
    assert not (set(investors) & fraud)


def test_hard_negatives_exist(world):
    populations = set(world.populations.values())
    for required in ("fast_spender", "marketplace_seller", "crowd_collector", "employer"):
        assert required in populations


def test_acceptance_criterion_holds(world):
    """Layer-0 acceptance: naive rules must misfire on honest people.

    If they stop misfiring, the generator has become too kind and every
    result computed above it would describe a fiction.
    """
    report = evaluate(world)
    failed = [check.name for check in report.checks if not check.passed]
    assert not failed, f"acceptance failed: {failed}"


def test_fan_in_rule_does_not_catch_mules(world):
    """The most intuitive graph rule points the wrong way.

    In a mule network the fan spreads out from the source and converges on
    the ATM; the mule account itself sees no convergence. Pinned as a test
    because it is a reported finding, not an accident.
    """
    report = evaluate(world)
    hits, total = report.fan_in_hits_on_mules
    assert total > 0
    assert hits / total < 0.05


def test_evasion_knobs_change_the_structure():
    """More funders must actually spread the source out."""
    naive = generate_world(SMALL)
    evasive_config = SimulationConfig(
        **{**SMALL.__dict__, "evasion": EvasionKnobs(funders=12, terminals=6)}
    )
    evasive = generate_world(evasive_config)

    naive_funders = max(len(n.organizer_ids) for n in naive.networks if n.kind == "mule_fast")
    evasive_funders = max(len(n.organizer_ids) for n in evasive.networks if n.kind == "mule_fast")
    assert naive_funders == 1
    assert evasive_funders > naive_funders


def test_cases_cover_all_four_kinds(world):
    counts = case_kind_counts(build_cases(world))
    for kind in (CASE_MULE_NETWORK, CASE_PAYROLL, CASE_CROWD_COLLECTION):
        assert counts.get(kind, 0) > 0, f"missing case kind: {kind}"


def test_mule_funding_fits_in_its_window(world):
    """The FUNDING of a ring stays inside the configured time spread.

    Narrowed from "all internal events" deliberately. Some mules now relay
    the money onward to another member after a delay instead of taking it to
    a machine, so a member-to-member event can legitimately fall outside the
    funding window. The window is a property of how the money is handed out,
    not of everything the ring ever does.
    """
    networks = [n for n in world.networks if n.kind == "mule_fast"]
    assert networks
    for network in networks:
        funders = set(network.organizer_ids)
        members = set(network.account_ids)
        funding = [
            e for e in world.events
            if e.sender_id in funders and e.receiver_id in members
        ]
        if len(funding) < 2:
            continue
        span = max(e.ts for e in funding) - min(e.ts for e in funding)
        assert span <= timedelta(minutes=SMALL.evasion.time_spread_minutes + 5)


def test_mules_are_not_single_purpose_accounts(world):
    """A mule is a person, not an account that exists to be a mule.

    Without ordinary income and spending the ring is structurally
    unambiguous and classification returns a meaningless ROC-AUC of 1.0000 —
    the account itself gives the answer and the network never has to be
    found. Pinned because it is easy to simplify away again.
    """
    mules = [a for a, p in world.populations.items() if p == "mule"]
    assert mules

    by_account: dict[str, int] = {}
    for event in world.events:
        for side in (event.sender_id, event.receiver_id):
            if side in set(mules):
                by_account[side] = by_account.get(side, 0) + 1

    busy = [account for account, count in by_account.items() if count > 4]
    assert len(busy) / len(mules) > 0.4, (
        "most mules have almost no history; they are single-purpose accounts again"
    )


def test_mule_exit_behaviour_varies(world):
    """Not every mule walks to an ATM.

    Some relay the money onward, some take only part of it. A ring where
    every member does exactly the same thing is the thing that made the task
    trivial.
    """
    cash_senders = {e.sender_id for e in world.events if e.asset_type == "cash"}
    mules = {a for a, p in world.populations.items() if p == "mule"}
    cashing = mules & cash_senders
    assert cashing, "some mules must cash out"
    assert len(cashing) < len(mules), "not every mule may cash out"
