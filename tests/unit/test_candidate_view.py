"""Tests for what the interface draws and sends.

The picture on screen is evidence an analyst acts on, so the tests here are
about honesty rather than aesthetics: a drawing must come from the data, a
trimmed drawing must still show the structure it claims to show, and the
request must carry what it says it carries.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

from apris.cheops.domain.models import TransactionEvent
from apris.frontend.candidate_view import (
    candidate_payload,
    plot_candidate_graph,
    window_hours_for,
)

T0 = datetime(2026, 3, 1, 9, 0, 0)


def _event(sender: str, receiver: str, minutes: float, amount: float = 100_000.0):
    return TransactionEvent(
        event_id=f"E{sender}-{receiver}-{minutes}",
        ts=T0 + timedelta(minutes=minutes),
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


def _star(sources: int, exits: int) -> list[TransactionEvent]:
    """One relay collecting from many and paying out to many.

    This is the shape that broke the first version of the drawing: the
    busiest accounts by degree are the relay and nothing else, and a subgraph
    of the top-degree nodes therefore contained no edges at all.
    """
    events = []
    for index in range(sources):
        events.append(_event(f"SRC{index:03d}", "RELAY", index, 50_000.0 + index))
    for index in range(exits):
        events.append(_event("RELAY", f"OUT{index:03d}", sources + index, 40_000.0 + index))
    return events


# ==========================================================================
# The drawing
# ==========================================================================


def test_a_trimmed_graph_still_has_edges_to_show():
    """Defect found the expensive way: the picture came out as a bare row.

    Trimming by node degree kept the accounts and threw away every edge
    between them, so a large candidate rendered as unconnected dots labelled
    "sources 90" — a drawing that showed nothing while looking deliberate.
    """
    events = _star(sources=120, exits=120)
    figure = plot_candidate_graph(events, max_nodes=40)
    axes = figure.axes[0]

    drawn_edges = [line for line in axes.lines if len(line.get_xdata()) == 2]
    assert drawn_edges, "a trimmed graph must still draw the flows it kept"

    drawn_nodes = sum(collection.get_offsets().shape[0] for collection in axes.collections)
    assert 0 < drawn_nodes <= 40


def test_the_caption_admits_what_was_hidden():
    events = _star(sources=120, exits=120)
    figure = plot_candidate_graph(events, max_nodes=30)
    captions = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "скрыто" in captions


def test_a_small_graph_is_drawn_whole_without_a_caveat():
    events = _star(sources=3, exits=3)
    figure = plot_candidate_graph(events, max_nodes=90)
    captions = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "скрыто" not in captions


def test_an_empty_candidate_says_so_instead_of_raising():
    figure = plot_candidate_graph([])
    captions = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "нет событий" in captions


def test_roles_are_read_off_the_graph_not_assumed():
    """Sources only send, exits only receive, relays do both."""
    figure = plot_candidate_graph(_star(sources=4, exits=5))
    captions = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "источники\n4" in captions
    assert "посредники\n1" in captions
    assert "выходы\n5" in captions


# ==========================================================================
# The request
# ==========================================================================


def test_a_long_stream_is_trimmed_to_its_tail():
    events = _star(sources=400, exits=400)
    payload = candidate_payload("CAND1", events, max_events=50)
    assert len(payload["events"]) == 50
    stamps = [event["ts"] for event in payload["events"]]
    assert stamps == sorted(stamps)
    # The tail, not the head: the window a model reads is the recent one.
    newest = max(event.ts for event in events).isoformat()
    assert stamps[-1] == newest


def test_the_window_reflects_the_candidates_own_span():
    events = [_event("A", "B", 0.0), _event("B", "C", 60.0 * 5)]
    assert window_hours_for(events) == 5
    assert window_hours_for(events[:1]) == 1


def test_the_window_stays_inside_what_the_api_accepts():
    events = [_event("A", "B", 0.0), _event("B", "C", 60.0 * 5000)]
    assert window_hours_for(events) == 720
