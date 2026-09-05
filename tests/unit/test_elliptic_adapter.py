"""Tests for the Elliptic adapter.

No download happens here. The dataset is ~690 MB and lives outside the
repository; these tests pin the logic on hand-built graphs whose answers are
known by construction.
"""

from __future__ import annotations

import networkx as nx
import pytest

from apris.cheops.infrastructure.external.elliptic import (
    STRUCTURAL_FEATURE_NAMES,
    neighbourhood,
    structural_features,
)


def _relay_graph(intermediaries: int = 8) -> nx.DiGraph:
    """source -> many -> single sink."""
    graph = nx.DiGraph()
    for i in range(intermediaries):
        graph.add_edge("SRC", f"M{i}")
        graph.add_edge(f"M{i}", "SINK")
    return graph


def _fan_out_graph(receivers: int = 8) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i in range(receivers):
        graph.add_edge("SRC", f"R{i}")
    return graph


def _fan_in_graph(senders: int = 8) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i in range(senders):
        graph.add_edge(f"S{i}", "COL")
    return graph


def test_structural_features_return_the_full_contract():
    features = structural_features(_relay_graph())
    assert set(features) == set(STRUCTURAL_FEATURE_NAMES)
    assert all(0.0 <= value <= 1.0 for value in features.values())


def test_relay_share_detects_a_relay_without_amounts():
    """The unweighted counterpart still recognises the shape.

    This is what makes the Elliptic result interpretable: if the counting
    version could not see a relay at all, a null result there would say
    nothing about the domain.
    """
    assert structural_features(_relay_graph())["relay_share"] > 0.3
    assert structural_features(_fan_out_graph())["relay_share"] == pytest.approx(0.0)
    assert structural_features(_fan_in_graph())["relay_share"] == pytest.approx(0.0)


def test_hub_and_fanout_point_in_opposite_directions():
    fan_out = structural_features(_fan_out_graph())
    fan_in = structural_features(_fan_in_graph())
    assert fan_out["fanout_share"] > 0.9
    assert fan_out["hub_share"] < 0.3
    assert fan_in["hub_share"] > 0.9
    assert fan_in["fanout_share"] < 0.3


def test_tiny_graphs_return_zeros_rather_than_raising():
    assert structural_features(nx.DiGraph()) == {n: 0.0 for n in STRUCTURAL_FEATURE_NAMES}
    single = nx.DiGraph()
    single.add_edge("A", "B")
    assert structural_features(single) == {n: 0.0 for n in STRUCTURAL_FEATURE_NAMES}


def test_neighbourhood_ignores_edge_direction():
    """A relay is invisible unless both feeders and fed nodes are included."""
    graph = _relay_graph(intermediaries=5)
    around_sink = neighbourhood(graph, "SINK", hops=2)
    assert "SRC" in around_sink
    assert any(node.startswith("M") for node in around_sink)


def test_neighbourhood_respects_the_cap():
    hub = nx.DiGraph()
    for i in range(2000):
        hub.add_edge("HUB", f"N{i}")
    assert neighbourhood(hub, "HUB", hops=2, cap=150).number_of_nodes() <= 150
