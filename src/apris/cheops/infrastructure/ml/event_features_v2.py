"""Graph and sequence features computed from RAW EVENTS.

Why this module exists
----------------------
``graph_v2.build_graph_matrix_from_tabular`` and
``sequence_v2.build_sequence_matrix_from_tabular`` produce their matrices as
hand-written linear combinations of the same nine aggregate features, e.g.::

    graph_density   = 0.05 + 0.95 * (0.38*central + 0.34*depth + 0.28*entropy_low)
    burst_ratio_90s = 0.05 + 0.95 * (0.46*growth + 0.31*holding_short + ...)

Two consequences follow. First, the "graph branch" never reads a graph and
the "sequence branch" never reads a sequence, so fusing the three branches
adds no information: all of them are deterministic transformations of one
vector. Second, ``burst_ratio_90s`` promises a 90-second window while being
derived from ``avg_holding_time``, which is measured in days — the name
asserts a time resolution the input does not carry.

This module computes the SAME five columns per branch, with the same
[0, 1] contract, from an actual event stream. Column names are preserved so
the existing training and inference code is a drop-in caller. After the
switch every column measures what its name says.

The tabular branch keeps reading aggregates; that is legitimate — it is a
tabular branch. Only the two branches that claim structure are moved.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import median

import networkx as nx
import numpy as np
import pandas as pd

from apris.cheops.domain.models import TransactionEvent

# Column names differ from graph_v2's legacy matrix on two slots, and
# deliberately. Computed honestly on this data, ``graph_component_compactness``
# is 1.000 for every case (each case is one connected component by
# construction) and ``graph_transitivity`` is 0.000 for every case (star and
# chain shapes contain no triangles). Both are correct measurements of
# quantities that carry no information here, so keeping the names would mean
# shipping two dead columns. They are replaced by the pair that does carry
# the structure: divergence at the top and convergence at the bottom.
GRAPH_FEATURE_COLUMNS: tuple[str, ...] = (
    "graph_density",
    "graph_hub_share",
    "graph_fanout_share",
    "graph_relay_share",
    "graph_weight_cv_norm",
)

SEQUENCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "event_rate_hour",
    "burst_ratio_90s",
    "median_delta_inverse",
    "amount_cv_norm",
    "unique_sender_ratio",
)

BURST_WINDOW_SECONDS = 90.0

# Saturation constants map an unbounded quantity onto [0, 1] without a hard
# cut: x / (x + k). Each k is the value at which the feature reads 0.5, so
# the constant states what counts as "a lot" for that quantity.
_RATE_HALF_POINT_PER_HOUR = 12.0
_INTERVAL_HALF_POINT_MINUTES = 6.0
_CV_HALF_POINT = 1.0


def _saturate(value: float, half_point: float) -> float:
    if value <= 0.0 or half_point <= 0.0:
        return 0.0
    return float(value / (value + half_point))


def _clip01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _coefficient_of_variation(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if mean <= 0.0:
        return 0.0
    return float(array.std(ddof=0) / mean)


# ==========================================================================
# Graph branch
# ==========================================================================


def graph_features_from_events(events: Iterable[TransactionEvent]) -> dict[str, float]:
    """Five graph features computed on the real transaction graph."""
    materialized = list(events)
    empty = {name: 0.0 for name in GRAPH_FEATURE_COLUMNS}
    if len(materialized) < 2:
        return empty

    graph = nx.DiGraph()
    for event in materialized:
        if graph.has_edge(event.sender_id, event.receiver_id):
            graph[event.sender_id][event.receiver_id]["weight"] += event.amount
        else:
            graph.add_edge(event.sender_id, event.receiver_id, weight=event.amount)

    node_count = graph.number_of_nodes()
    if node_count < 2:
        return empty

    density = nx.density(graph)

    # Hub share: how much of all incoming value lands on the single biggest
    # receiver. A mule network converges on one cash-out point, so this is
    # the structural signature the branch is supposed to carry.
    incoming: dict[str, float] = {}
    for sender, receiver, data in graph.edges(data=True):
        incoming[receiver] = incoming.get(receiver, 0.0) + float(data["weight"])
    total_incoming = sum(incoming.values())
    hub_share = (max(incoming.values()) / total_incoming) if total_incoming > 0 else 0.0

    # Fan-out share: how much of all outgoing value leaves the single
    # biggest sender. Together with hub_share this is the hourglass test.
    # A payroll has high fan-out and low hub share; a whip-round has the
    # reverse; only a mule network has BOTH — money spreads from one source
    # and reconverges on one exit. No honest structure does that.
    outgoing: dict[str, float] = {}
    for sender, _receiver, data in graph.edges(data=True):
        outgoing[sender] = outgoing.get(sender, 0.0) + float(data["weight"])
    total_outgoing = sum(outgoing.values())
    fanout_share = (max(outgoing.values()) / total_outgoing) if total_outgoing > 0 else 0.0

    relay = relay_share(graph, incoming, outgoing)

    weights = [float(data["weight"]) for _, _, data in graph.edges(data=True)]
    weight_cv = _saturate(_coefficient_of_variation(weights), _CV_HALF_POINT)

    return {
        "graph_density": _clip01(density),
        "graph_hub_share": _clip01(hub_share),
        "graph_fanout_share": _clip01(fanout_share),
        "graph_relay_share": _clip01(relay),
        "graph_weight_cv_norm": _clip01(weight_cv),
    }


def relay_share(
    graph: nx.DiGraph,
    incoming: dict[str, float],
    outgoing: dict[str, float],
) -> float:
    """Share of value relayed from a dominant source to a dominant sink
    through intermediaries that keep nothing.

    This is the textbook definition of layering: funds leave an origin,
    pass through accounts that hold them briefly, and exit at a different
    point. Encoding the definition — not fitting a curve to the data.

    Degree alone cannot express it. A whip-round's collector is both the
    largest sender and the largest receiver, so convergence and divergence
    are simultaneously high while nothing is being relayed anywhere: the
    first version of this feature scored a whip-round 0.492 against a mule
    network's 0.500 for exactly that reason. Requiring source != sink and
    an actual two-hop path is what separates them.
    """
    if not incoming or not outgoing:
        return 0.0

    sink = max(incoming, key=lambda node: incoming[node])
    source = max(outgoing, key=lambda node: outgoing[node])
    if source == sink:
        return 0.0

    relayed = 0.0
    for intermediary in graph.successors(source):
        if intermediary == sink:
            continue
        if graph.has_edge(intermediary, sink):
            relayed += min(
                float(graph[source][intermediary]["weight"]),
                float(graph[intermediary][sink]["weight"]),
            )

    total = sum(float(data["weight"]) for _, _, data in graph.edges(data=True))
    return (relayed / total) if total > 0 else 0.0


# ==========================================================================
# Sequence branch
# ==========================================================================


def sequence_features_from_events(events: Iterable[TransactionEvent]) -> dict[str, float]:
    """Five sequence features computed on the real event timeline."""
    materialized = sorted(events, key=lambda e: e.ts)
    empty = {name: 0.0 for name in SEQUENCE_FEATURE_COLUMNS}
    if len(materialized) < 2:
        return empty

    stamps = [event.ts for event in materialized]
    span_seconds = (stamps[-1] - stamps[0]).total_seconds()
    span_hours = max(span_seconds / 3600.0, 1.0 / 60.0)
    rate_per_hour = len(materialized) / span_hours

    # burst_ratio_90s now actually measures a 90-second window: the largest
    # share of events falling inside any such window, found by two pointers.
    left = 0
    largest_burst = 1
    for right in range(len(stamps)):
        while (stamps[right] - stamps[left]).total_seconds() > BURST_WINDOW_SECONDS:
            left += 1
        largest_burst = max(largest_burst, right - left + 1)
    burst_ratio = largest_burst / len(stamps)

    gaps_minutes = [
        (stamps[i + 1] - stamps[i]).total_seconds() / 60.0 for i in range(len(stamps) - 1)
    ]
    median_gap = median(gaps_minutes) if gaps_minutes else 0.0
    # Inverse: short intervals -> value near one.
    median_delta_inverse = 1.0 - _saturate(median_gap, _INTERVAL_HALF_POINT_MINUTES)

    amount_cv = _saturate(
        _coefficient_of_variation([event.amount for event in materialized]), _CV_HALF_POINT
    )

    unique_senders = len({event.sender_id for event in materialized})
    unique_sender_ratio = unique_senders / len(materialized)

    return {
        "event_rate_hour": _clip01(_saturate(rate_per_hour, _RATE_HALF_POINT_PER_HOUR)),
        "burst_ratio_90s": _clip01(burst_ratio),
        "median_delta_inverse": _clip01(median_delta_inverse),
        "amount_cv_norm": _clip01(amount_cv),
        "unique_sender_ratio": _clip01(unique_sender_ratio),
    }


# ==========================================================================
# Matrix builders — drop-in replacements for the *_from_tabular builders
# ==========================================================================


def build_graph_matrix_from_events(
    event_groups: Sequence[Sequence[TransactionEvent]],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Graph matrix for a list of cases, each case being a list of events."""
    rows = [graph_features_from_events(group) for group in event_groups]
    frame = pd.DataFrame(rows, columns=list(GRAPH_FEATURE_COLUMNS))
    if index is not None:
        frame.index = index
    return frame.clip(lower=0.0, upper=1.0)


def build_sequence_matrix_from_events(
    event_groups: Sequence[Sequence[TransactionEvent]],
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Sequence matrix for a list of cases, each case being a list of events."""
    rows = [sequence_features_from_events(group) for group in event_groups]
    frame = pd.DataFrame(rows, columns=list(SEQUENCE_FEATURE_COLUMNS))
    if index is not None:
        frame.index = index
    return frame.clip(lower=0.0, upper=1.0)
