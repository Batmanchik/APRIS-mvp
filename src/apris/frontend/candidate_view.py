"""Presentation helpers for a discovered candidate.

Two rules hold everywhere in this module.

**Nothing is invented.** The earlier dashboard drew a "transaction network"
for each object by picking a preset from the risk score and generating fake
transactions from a hash of the object's id — a picture that agreed with the
verdict because it was drawn from the verdict. Every drawing here is made
from the candidate's own events and from nothing else.

**Scoring belongs to the API.** The interface never loads a model. It turns
a candidate into a request and renders what comes back, so what an analyst
sees is what the service actually returns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from apris.cheops.domain.models import TransactionEvent

INK = "#131A24"
INK_SOFT = "#4C596A"
RULE = "#D8DEE5"
ACCENT = "#1B5B66"
FRAUD = "#A8261F"
EXIT = "#B4741C"

# Payload size guard. A candidate can hold thousands of events, and the whole
# stream does not need to cross the wire for a demo: the newest events are
# the ones a window-based model reads anyway.
MAX_EVENTS_PER_REQUEST = 600


def event_to_payload(event: TransactionEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "ts": event.ts.isoformat(),
        "amount": float(event.amount),
        "currency": event.currency,
        "sender_id": event.sender_id,
        "receiver_id": event.receiver_id,
        "sender_type": event.sender_type,
        "receiver_type": event.receiver_type,
        "channel": event.channel,
        "jurisdiction": event.jurisdiction,
        "asset_type": event.asset_type,
        "tx_hash": event.tx_hash,
        "case_id": event.case_id,
        "metadata": dict(event.metadata),
    }


def candidate_payload(
    case_id: str,
    events: Sequence[TransactionEvent],
    *,
    window_hours: int = 24,
    max_events: int = MAX_EVENTS_PER_REQUEST,
) -> dict[str, Any]:
    """Turn a candidate into a `/api/v2/score` request body.

    When the stream is trimmed, the tail is kept rather than the head: the
    request declares a window, and the most recent events are the ones inside
    it.
    """
    ordered = sorted(events, key=lambda event: event.ts)
    trimmed = ordered[-max_events:] if len(ordered) > max_events else ordered
    return {
        "case_id": case_id,
        "window_hours": window_hours,
        "events": [event_to_payload(event) for event in trimmed],
    }


def window_hours_for(events: Sequence[TransactionEvent], *, cap: int = 720) -> int:
    """The candidate's own span, in hours, clamped to what the API accepts."""
    if len(events) < 2:
        return 1
    stamps = [event.ts for event in events]
    span_hours = (max(stamps) - min(stamps)).total_seconds() / 3600.0
    return int(max(1, min(cap, round(span_hours) or 1)))


# ==========================================================================
# Tables
# ==========================================================================


def events_table(events: Iterable[TransactionEvent], *, limit: int = 200) -> pd.DataFrame:
    rows = [
        {
            "Время": event.ts,
            "Отправитель": event.sender_id,
            "Получатель": event.receiver_id,
            "Сумма": float(event.amount),
            "Тип получателя": event.receiver_type,
            "Актив": event.asset_type,
        }
        for event in sorted(events, key=lambda e: e.ts)[:limit]
    ]
    return pd.DataFrame(rows)


def features_table(features: dict[str, float], labels: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Признак": name,
                "Что измеряет": labels.get(name, ""),
                "Значение": float(value),
            }
            for name, value in features.items()
        ]
    )


# ==========================================================================
# The graph, drawn from the events themselves
# ==========================================================================


def _roles(graph: nx.DiGraph) -> dict[str, str]:
    """Sort nodes into source, relay and exit by their own degrees."""
    roles: dict[str, str] = {}
    for node in graph.nodes:
        incoming = graph.in_degree(node)
        outgoing = graph.out_degree(node)
        if incoming == 0:
            roles[node] = "source"
        elif outgoing == 0:
            roles[node] = "exit"
        else:
            roles[node] = "relay"
    return roles


def plot_candidate_graph(
    events: Sequence[TransactionEvent],
    *,
    max_nodes: int = 90,
) -> plt.Figure:
    """Draw the candidate's real transfer graph in three levels.

    The layout is explicit rather than force-directed for the same reason the
    report's figures are: a spring layout produces a handsome cloud that says
    nothing, while sources on top, relays in the middle and exits at the
    bottom make convergence or fan-out readable at a glance. When a candidate
    is too large to draw, the busiest accounts are kept and the caption says
    how many were dropped — a truncated picture that admits it beats a
    tidy one that does not.
    """
    graph = nx.DiGraph()
    for event in events:
        if graph.has_edge(event.sender_id, event.receiver_id):
            graph[event.sender_id][event.receiver_id]["weight"] += float(event.amount)
        else:
            graph.add_edge(event.sender_id, event.receiver_id, weight=float(event.amount))

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "нет событий", ha="center", va="center", color=INK_SOFT)
        return fig

    dropped = 0
    if graph.number_of_nodes() > max_nodes:
        # Keep the heaviest flows, not the busiest accounts. Ranking by
        # degree and taking a subgraph of the winners was the first attempt
        # and it drew a row of unconnected dots: in a structure that
        # converges on one exit, the top accounts rarely pay each other, so
        # every edge fell outside the selection. Choosing edges first
        # guarantees the picture still shows movement.
        by_weight = sorted(
            graph.edges(data=True), key=lambda edge: float(edge[2]["weight"]), reverse=True
        )
        kept: set[str] = set()
        trimmed = nx.DiGraph()
        for sender, receiver, data in by_weight:
            newcomers = {sender, receiver} - kept
            if len(kept) + len(newcomers) > max_nodes:
                continue
            kept |= newcomers
            trimmed.add_edge(sender, receiver, weight=float(data["weight"]))
        dropped = graph.number_of_nodes() - trimmed.number_of_nodes()
        graph = trimmed

    roles = _roles(graph)
    levels = {"source": 3.0, "relay": 0.0, "exit": -3.0}
    colors = {"source": ACCENT, "relay": INK_SOFT, "exit": EXIT}

    positions: dict[str, tuple[float, float]] = {}
    for role, height in levels.items():
        members = sorted(node for node, kind in roles.items() if kind == role)
        for index, node in enumerate(members):
            x = ((index + 0.5) / len(members)) * 10.0 - 5.0 if members else 0.0
            positions[node] = (x, height)

    weights = [float(data["weight"]) for _, _, data in graph.edges(data=True)] or [1.0]
    widest = max(weights)
    for sender, receiver, data in graph.edges(data=True):
        x1, y1 = positions[sender]
        x2, y2 = positions[receiver]
        width = 0.4 + 2.6 * (float(data["weight"]) / widest)
        ax.plot([x1, x2], [y1, y2], color=FRAUD, alpha=0.28, linewidth=width, zorder=1)

    for node, (x, y) in positions.items():
        ax.scatter(
            [x],
            [y],
            s=140 if roles[node] == "relay" else 190,
            color=colors[roles[node]],
            edgecolors="#FFFFFF",
            linewidths=1.2,
            zorder=2,
        )

    counts = {role: sum(1 for kind in roles.values() if kind == role) for role in levels}
    for role, height in levels.items():
        ax.text(
            -5.9,
            height,
            {"source": "источники", "relay": "посредники", "exit": "выходы"}[role]
            + f"\n{counts[role]}",
            ha="right",
            va="center",
            fontsize=9,
            color=colors[role],
            fontweight="bold",
        )

    ax.set_xlim(-7.6, 5.8)
    ax.set_ylim(-4.4, 4.4)
    note = "Граф построен по событиям кандидата. Толщина ребра — сумма переводов."
    if dropped:
        note += (
            f" Показаны самые крупные потоки ({graph.number_of_nodes()} счетов), "
            f"скрыто {dropped}."
        )
    ax.text(-0.9, -4.15, note, ha="center", va="top", fontsize=8.5, color=INK_SOFT)
    return fig


def plot_feature_bars(features: dict[str, float]) -> plt.Figure:
    """The ten event features as they came out, ordered by value."""
    ordered = sorted(features.items(), key=lambda item: item[1])
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.barh(names, values, color=ACCENT, height=0.62, edgecolor="none")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("значение признака (все нормированы в 0..1)", color=INK_SOFT)
    ax.grid(axis="x", color=RULE, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=INK_SOFT)
    for index, value in enumerate(values):
        ax.text(value + 0.015, index, f"{value:.2f}", va="center", fontsize=9, color=INK)
    return fig
