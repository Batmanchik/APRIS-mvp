"""Tests for the evasion curve.

The curve's whole value is that one dial moves at a time and the price of
each dial is written down. Both are properties of the declared points, so
they are pinned here rather than trusted to review.

The overlap profile is the other half: it is what tells blindness from
fragmentation, and it is the diagnostic a coverage of exactly 0.000 needs
before anybody reports it as a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from apris.cheops.infrastructure.experiments.evasion_curve import (
    COVERAGE_THRESHOLD,
    CURVE,
    GRIP_THRESHOLD,
    overlap_profile,
)


@dataclass
class _Network:
    network_id: str
    kind: str
    account_ids: tuple[str, ...]


@dataclass
class _World:
    networks: list[_Network]


@dataclass
class _Candidate:
    member_ids: tuple[str, ...]


# ==========================================================================
# The shape of the curve
# ==========================================================================


def test_the_curve_starts_from_the_naive_organiser():
    """Without the (1, 1) point nothing on the curve has a baseline and the
    fall cannot be attributed to the dials at all."""
    assert CURVE[0].funders == 1
    assert CURVE[0].terminals == 1


def test_every_point_but_the_joint_one_moves_a_single_dial():
    """Two dials moving at once cannot be read as the price of either. The
    joint point exists only because it is the W5 rung of the ladder and has
    to stay comparable with it."""
    joint = [p for p in CURVE if p.funders > 1 and p.terminals > 1]
    assert [p.key for p in joint] == ["f6t4"]
    assert (joint[0].funders, joint[0].terminals) == (6, 4)


def test_both_dials_are_swept_over_more_than_two_points():
    """Two points name an effect; they do not say where it breaks, which is
    the entire reason this module exists beside the ladder."""
    funder_values = sorted({p.funders for p in CURVE if p.terminals == 1})
    terminal_values = sorted({p.terminals for p in CURVE if p.funders == 1})
    assert len(funder_values) >= 4
    assert len(terminal_values) >= 3


def test_every_point_says_what_it_costs_the_organiser():
    """A detectability curve with no price axis reads as 'evasion is free'."""
    for point in CURVE:
        assert point.cost.strip()
        assert len(point.cost) > 10, point.key


def test_point_keys_are_unique():
    assert len({p.key for p in CURVE}) == len(CURVE)


# ==========================================================================
# The overlap profile: blindness is not fragmentation
# ==========================================================================


def test_a_ring_held_whole_scores_one():
    world = _World([_Network("n1", "mule_fast", ("a", "b", "c", "d"))])
    profile = overlap_profile(world, [_Candidate(("a", "b", "c", "d", "x"))])

    assert profile.networks == 1
    assert profile.median == pytest.approx(1.0)
    assert profile.reaching_coverage == 1
    assert profile.grip_share == pytest.approx(1.0)


def test_a_ring_split_across_candidates_scores_its_largest_piece():
    """Half the ring in one candidate and half in another is fragmentation:
    the profile shows a grip, and coverage — thresholded at 0.5 — does not
    necessarily fail. Reporting only coverage loses this distinction."""
    world = _World([_Network("n1", "mule_fast", ("a", "b", "c", "d"))])
    profile = overlap_profile(
        world, [_Candidate(("a", "b")), _Candidate(("c",)), _Candidate(("d",))]
    )

    assert profile.median == pytest.approx(0.5)
    assert profile.reaching_grip == 1
    assert profile.mean == pytest.approx(0.5)


def test_a_ring_nobody_grouped_scores_zero_and_reports_no_grip():
    """The W5 shape: every member in a different candidate. Coverage 0.000
    and grip 0.0 together are what make 'blindness, not fragmentation' a
    measured statement rather than an interpretation."""
    world = _World([_Network("n1", "mule_fast", ("a", "b", "c", "d"))])
    profile = overlap_profile(
        world,
        [_Candidate((letter, f"honest_{letter}")) for letter in "abcd"],
    )

    assert profile.median == pytest.approx(0.25)
    assert profile.reaching_coverage == 0


def test_no_candidates_at_all_is_zero_overlap_not_an_error():
    world = _World([_Network("n1", "mule_fast", ("a", "b"))])
    profile = overlap_profile(world, [])

    assert profile.median == pytest.approx(0.0)
    assert profile.best == pytest.approx(0.0)
    assert profile.reaching_grip == 0


def test_only_mule_rings_are_counted():
    """Pyramids are fraud but not the object discovery is trying to recover;
    counting them would make coverage move when the pyramid count changes."""
    world = _World(
        [
            _Network("n1", "mule_fast", ("a", "b")),
            _Network("p1", "pyramid", ("x", "y", "z")),
        ]
    )
    profile = overlap_profile(world, [_Candidate(("a", "b"))])

    assert profile.networks == 1
    assert profile.median == pytest.approx(1.0)


def test_a_world_with_no_rings_reports_zero_networks_not_a_perfect_score():
    profile = overlap_profile(_World([]), [_Candidate(("a",))])
    assert profile.networks == 0
    assert profile.grip_share == 0.0


def test_the_grip_threshold_sits_below_the_coverage_threshold():
    """Grip answers 'does discovery have any hold on this ring at all'. At or
    above the coverage threshold it would answer the same question twice."""
    assert 0.0 < GRIP_THRESHOLD < COVERAGE_THRESHOLD
