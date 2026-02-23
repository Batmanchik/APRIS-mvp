from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from data_generator import SEED


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_transaction_graph(
    features: dict[str, Any],
    seed: int = SEED,
) -> nx.DiGraph:
    rng = np.random.default_rng(seed)

    centralization = _clamp(float(features.get("centralization_index", 0.5)), 0.0, 1.0)
    structural_depth = int(round(float(features.get("structural_depth", 6))))
    depth = int(_clamp(float(structural_depth), 2.0, 16.0))

    # Keep graph compact for fast UI rendering.
    node_count = int(_clamp(30 + depth * 3, 30, 80))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))

    # Level-based hierarchy controlled by structural_depth.
    levels: list[list[int]] = [[0]]
    remaining_nodes = list(range(1, node_count))
    for _ in range(1, depth):
        if not remaining_nodes:
            break
        per_level = max(1, int(np.ceil(len(remaining_nodes) / (depth - len(levels) + 1))))
        current_level = [remaining_nodes.pop(0) for _ in range(min(per_level, len(remaining_nodes)))]
        levels.append(current_level)
    if remaining_nodes:
        levels[-1].extend(remaining_nodes)

    for i in range(1, len(levels)):
        prev_level = levels[i - 1]
        current_level = levels[i]
        for node in current_level:
            parent = int(rng.choice(prev_level))
            graph.add_edge(parent, node)

    # Higher centralization forces more edges into one hub.
    hub = 0
    possible_sources = list(range(1, node_count))
    rng.shuffle(possible_sources)
    extra_hub_edges = int(centralization * (node_count - 1))
    for src in possible_sources[:extra_hub_edges]:
        graph.add_edge(src, hub)

    # Small amount of random traffic edges for realism.
    extra_edges = int(node_count * 0.1)
    for _ in range(extra_edges):
        src = int(rng.integers(0, node_count))
        dst = int(rng.integers(0, node_count))
        if src != dst:
            graph.add_edge(src, dst)

    return graph


def compute_hub_metrics(graph: nx.DiGraph) -> dict[str, float | int]:
    if graph.number_of_nodes() == 0:
        return {"hub_node": -1, "hub_in_degree_share": 0.0, "hub_betweenness": 0.0}

    in_degrees = dict(graph.in_degree())
    hub_node = max(in_degrees, key=in_degrees.get)
    total_in = float(sum(in_degrees.values()))
    hub_in = float(in_degrees[hub_node])
    hub_share = 0.0 if total_in == 0 else hub_in / total_in

    betweenness = nx.betweenness_centrality(graph, normalized=True)
    return {
        "hub_node": int(hub_node),
        "hub_in_degree_share": float(hub_share),
        "hub_betweenness": float(betweenness.get(hub_node, 0.0)),
    }


def plot_graph(graph: nx.DiGraph, hub_node: int | None = None) -> plt.Figure:
    if hub_node is None:
        in_degrees = dict(graph.in_degree())
        hub_node = max(in_degrees, key=in_degrees.get) if in_degrees else -1

    pos = nx.spring_layout(graph, seed=SEED)
    colors = ["#d62728" if node == hub_node else "#1f77b4" for node in graph.nodes]
    sizes = [220 if node == hub_node else 85 for node in graph.nodes]

    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color="#8a8a8a", width=0.6, alpha=0.6, ax=ax, arrows=False)
    ax.set_title("Synthetic Transaction Graph")
    ax.axis("off")
    return fig
