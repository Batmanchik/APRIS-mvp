"""Tests for graph and sequence features computed from raw events.

The point of these tests is not that the numbers are pretty. It is that
each feature measures what its name says — the property the legacy
``*_from_tabular`` builders could not have, since they derived structural
and temporal features from nine period aggregates.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.event_features_v2 import (
    BURST_WINDOW_SECONDS,
    GRAPH_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
    build_graph_matrix_from_events,
    build_sequence_matrix_from_events,
    graph_features_from_events,
    sequence_features_from_events,
)

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _event(sender: str, receiver: str, amount: float, offset_seconds: float) -> TransactionEvent:
    return TransactionEvent(
        event_id=f"EV{sender}{receiver}{int(offset_seconds)}",
        ts=T0 + timedelta(seconds=offset_seconds),
        amount=amount,
        currency="KZT",
        sender_id=sender,
        receiver_id=receiver,
        sender_type="person",
        receiver_type="person",
        channel="legal",
        jurisdiction="KZ",
        asset_type="fiat",
    )


def _relay_network(mules: int = 10, spread_seconds: float = 600.0) -> list[TransactionEvent]:
    """source -> mules -> single ATM, all inside one window."""
    events: list[TransactionEvent] = []
    for i in range(mules):
        at = spread_seconds * i / mules
        events.append(_event("SRC", f"MUL{i}", 100_000.0, at))
        events.append(_event(f"MUL{i}", "ATM", 98_000.0, at + 60))
    return events


def _payroll(employees: int = 10) -> list[TransactionEvent]:
    """One company pays many people. Fan-out, no convergence."""
    return [_event("EMP", f"P{i}", 300_000.0, i * 30) for i in range(employees)]


def _whip_round(donors: int = 10) -> list[TransactionEvent]:
    """Many people pay one. Convergence, no relay."""
    return [_event(f"D{i}", "COL", 20_000.0, i * 600) for i in range(donors)]


# ==========================================================================
# Graph branch
# ==========================================================================


def test_graph_features_return_full_contract():
    features = graph_features_from_events(_relay_network())
    assert set(features) == set(GRAPH_FEATURE_COLUMNS)
    assert all(0.0 <= v <= 1.0 for v in features.values())


def test_relay_share_is_high_only_for_a_relay_structure():
    """The headline structural feature.

    Layering means value leaves an origin and exits somewhere else through
    intermediaries. A payroll fans out but nothing reconverges; a whip-round
    converges but nothing was relayed to it from a dominant source.
    """
    relay = graph_features_from_events(_relay_network())["graph_relay_share"]
    payroll = graph_features_from_events(_payroll())["graph_relay_share"]
    whip = graph_features_from_events(_whip_round())["graph_relay_share"]

    assert relay > 0.3
    assert payroll == pytest.approx(0.0)
    assert whip == pytest.approx(0.0)


def test_relay_share_is_zero_when_source_equals_sink():
    """A single account dominating both directions relays nothing.

    This is the case that broke the first version of the feature: a
    whip-round collector is both the biggest sender and the biggest
    receiver, and a degree-based score rated it as high as a mule network.
    """
    events = [
        _event("D1", "COL", 50_000.0, 0),
        _event("D2", "COL", 50_000.0, 10),
        _event("COL", "SHOP1", 40_000.0, 100),
        _event("COL", "SHOP2", 40_000.0, 200),
    ]
    assert graph_features_from_events(events)["graph_relay_share"] == pytest.approx(0.0)


def test_fanout_and_hub_shares_point_in_opposite_directions():
    payroll = graph_features_from_events(_payroll())
    whip = graph_features_from_events(_whip_round())

    assert payroll["graph_fanout_share"] > 0.9
    assert payroll["graph_hub_share"] < 0.3
    assert whip["graph_hub_share"] > 0.9
    assert whip["graph_fanout_share"] < 0.3


def test_graph_features_degrade_gracefully_on_tiny_input():
    assert graph_features_from_events([]) == {k: 0.0 for k in GRAPH_FEATURE_COLUMNS}
    single = [_event("A", "B", 1000.0, 0)]
    assert graph_features_from_events(single) == {k: 0.0 for k in GRAPH_FEATURE_COLUMNS}


# ==========================================================================
# Sequence branch
# ==========================================================================


def test_sequence_features_return_full_contract():
    features = sequence_features_from_events(_relay_network())
    assert set(features) == set(SEQUENCE_FEATURE_COLUMNS)
    assert all(0.0 <= v <= 1.0 for v in features.values())


def test_burst_ratio_measures_an_actual_90_second_window():
    """The name promises 90 seconds; the code must deliver 90 seconds.

    Six events packed inside the window, four spread far outside it.
    """
    packed = [_event("A", f"B{i}", 1000.0, i * 10) for i in range(6)]
    spread = [_event("A", f"C{i}", 1000.0, 3600.0 * (i + 1)) for i in range(4)]
    ratio = sequence_features_from_events(packed + spread)["burst_ratio_90s"]
    assert ratio == pytest.approx(6 / 10)

    # widening the pack past the window must lower the ratio
    wider = [_event("A", f"B{i}", 1000.0, i * (BURST_WINDOW_SECONDS / 2)) for i in range(6)]
    assert sequence_features_from_events(wider + spread)["burst_ratio_90s"] < ratio


def test_fast_operation_reads_faster_than_a_slow_one():
    fast = sequence_features_from_events(_relay_network(spread_seconds=300.0))
    slow = sequence_features_from_events(_whip_round())
    assert fast["event_rate_hour"] > slow["event_rate_hour"]
    assert fast["median_delta_inverse"] > slow["median_delta_inverse"]


def test_unique_sender_ratio_distinguishes_one_payer_from_many():
    payroll = sequence_features_from_events(_payroll())
    whip = sequence_features_from_events(_whip_round())
    assert payroll["unique_sender_ratio"] < 0.2
    assert whip["unique_sender_ratio"] > 0.9


# ==========================================================================
# Matrix builders
# ==========================================================================


def test_matrix_builders_produce_one_row_per_case():
    groups = [_relay_network(), _payroll(), _whip_round()]
    graph_matrix = build_graph_matrix_from_events(groups)
    sequence_matrix = build_sequence_matrix_from_events(groups)

    assert list(graph_matrix.columns) == list(GRAPH_FEATURE_COLUMNS)
    assert list(sequence_matrix.columns) == list(SEQUENCE_FEATURE_COLUMNS)
    assert len(graph_matrix) == len(sequence_matrix) == 3
    assert graph_matrix.to_numpy().min() >= 0.0
    assert graph_matrix.to_numpy().max() <= 1.0
