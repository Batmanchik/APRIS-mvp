"""Figures for the research write-up and the defence.

One rule: **every figure shows a real measurement**. Nothing is drawn for
decoration. A beautiful chart with invented numbers is caught; a beautiful
chart with measured ones is not, and it does the arguing for you.

One visual system across every figure — one palette, one type treatment,
one grid. Inconsistent styling reads as carelessness faster than any error
in the numbers does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.simulation.cases import LabelledCase

DEFAULT_FIGURE_DIR = Path("artifacts") / "figures"

# ==========================================================================
# Palette — the same colours the project documents use
# ==========================================================================

INK = "#131A24"
INK_SOFT = "#4C596A"
INK_FAINT = "#8A96A5"
RULE = "#D8DEE5"
PAPER = "#FFFFFF"

ACCENT = "#1B5B66"        # the instrument
FRAUD = "#A8261F"         # fraud
FRAUD_LIGHT = "#D8695F"
# Deliberately differs from FRAUD in lightness, not only hue: close tones
# merge on a projector, and hard negatives are the class a reader must be
# able to tell apart at a glance.
HARD_NEGATIVE = "#D18700"
HONEST = "#1F6B4A"
CASH = "#6D3A8C"          # exit point, ATM

CASE_COLORS = {
    "mule_network": FRAUD,
    "pyramid": FRAUD_LIGHT,
    "payroll": HARD_NEGATIVE,
    "crowd_collection": HONEST,
}

CASE_LABELS = {
    "mule_network": "mule network",
    "pyramid": "pyramid",
    "payroll": "payroll",
    "crowd_collection": "whip-round",
}


def use_project_style() -> None:
    """Apply the single visual system used by every figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": RULE,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "xtick.color": INK_FAINT,
            "ytick.color": INK_FAINT,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
        }
    )


def _save(
    fig: plt.Figure,
    name: str,
    source_note: str,
    directory: Path,
    note_y: float = -0.02,
) -> Path:
    """Save with a source line. The note is not optional: it says what the
    figure was computed from, which is half of why a reader believes it.

    ``note_y`` is figure-relative; panels with rotated tick labels need a
    lower value or the note lands on top of them.
    """
    fig.text(0.5, note_y, source_note, ha="center", va="top", fontsize=8.5, color=INK_FAINT)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==========================================================================
# Figure 1 — anatomy of a real network
# ==========================================================================


def plot_network_anatomy(
    case: LabelledCase,
    *,
    name: str = "network_anatomy",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """Source -> mules -> ATM, laid out by role.

    The layout is set explicitly rather than by a force-directed algorithm.
    A spring layout draws a handsome cloud that says nothing; the scheme is
    three levels, so it is drawn as three levels. The hourglass shape then
    reads in about a second, which is the entire point of the figure.
    """
    members = set(case.member_ids)
    inbound = [e for e in case.events if e.sender_id in members and e.receiver_id in members]
    cash = [e for e in case.events if e.asset_type == "cash" and e.sender_id in members]

    sources = sorted({e.sender_id for e in inbound})
    relays = sorted({e.receiver_id for e in inbound})
    exits = sorted({e.receiver_id for e in cash})
    if not sources or not relays or not exits:
        raise ValueError("case does not have a source -> relay -> exit structure")

    def _row(items: Sequence[str], height: float) -> dict[str, tuple[float, float]]:
        return {
            item: ((i + 0.5) / len(items) * 10 - 5, height) for i, item in enumerate(items)
        }

    pos = {**_row(sources, 3.2), **_row(relays, 0.0), **_row(exits, -3.2)}

    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    amounts = [e.amount for e in inbound + cash] or [1.0]
    widest = max(amounts)

    for event in inbound:
        x1, y1 = pos[event.sender_id]
        x2, y2 = pos[event.receiver_id]
        ax.plot([x1, x2], [y1, y2], color=FRAUD, alpha=0.32,
                linewidth=0.6 + 2.2 * event.amount / widest, zorder=1)
    for event in cash:
        x1, y1 = pos[event.sender_id]
        x2, y2 = pos[event.receiver_id]
        ax.plot([x1, x2], [y1, y2], color=CASH, alpha=0.38,
                linewidth=0.6 + 2.2 * event.amount / widest, zorder=1)

    for items, size, color, marker, label in (
        (sources, 520, FRAUD, "o", f"source ({len(sources)})"),
        (relays, 190, FRAUD_LIGHT, "o", f"mules ({len(relays)})"),
        (exits, 560, CASH, "s", f"ATM ({len(exits)})"),
    ):
        ax.scatter(
            [pos[i][0] for i in items], [pos[i][1] for i in items],
            s=size, c=color, marker=marker, edgecolors=PAPER, linewidths=2,
            zorder=3, label=label,
        )

    for text, height, color in (
        ("SOURCE", 3.2, FRAUD), ("MULES", 0.0, FRAUD_LIGHT), ("EXIT", -3.2, CASH)
    ):
        ax.text(-5.4, height, text, ha="right", va="center",
                fontsize=10, color=color, weight="bold")

    total_in = sum(e.amount for e in inbound)
    total_out = sum(e.amount for e in cash)
    retained = 100.0 * (1.0 - total_out / total_in) if total_in else 0.0
    stamps = [e.ts for e in inbound + cash]
    span_minutes = (max(stamps) - min(stamps)).total_seconds() / 60.0 if stamps else 0.0

    ax.set_title(
        f"Anatomy of {case.case_id}: "
        f"{len(sources)} source -> {len(relays)} mules -> {len(exits)} ATM"
    )
    ax.set_xlim(-7.8, 5.8)
    ax.set_ylim(-6.2, 4.6)
    ax.axis("off")
    ax.legend(loc="upper right", ncol=3)

    box = (
        f"in       {total_in:>14,.0f} KZT\n"
        f"out      {total_out:>14,.0f} KZT\n"
        f"retained {total_in - total_out:>14,.0f} KZT  ({retained:.1f} %)\n"
        f"whole operation fits in {span_minutes:.0f} min"
    ).replace(",", " ")
    ax.text(-7.6, -6.0, box, ha="left", va="bottom", fontsize=9.5, color=INK,
            family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F4F6F8",
                      edgecolor=RULE, linewidth=1))

    return _save(
        fig, name,
        "Real network from the simulator. Line width is the transferred amount. "
        "Layout set by role, not by a force-directed algorithm.",
        directory,
    )


# ==========================================================================
# Figure 2 — feature separation across case kinds
# ==========================================================================


def plot_feature_separation(
    rows: Sequence[dict[str, float]],
    kinds: Sequence[str],
    features: Sequence[str],
    *,
    name: str = "feature_separation",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """One panel per feature, one bar per case kind.

    Each panel states its own separation, so the figure carries its
    measurement rather than inviting the reader to trust the shape.
    """
    order = [k for k in ("mule_network", "pyramid", "payroll", "crowd_collection") if k in kinds]
    fig, axes = plt.subplots(1, len(features), figsize=(3.6 * len(features), 4.4))
    if len(features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, features):
        means, colors = [], []
        for kind in order:
            values = [r[feature] for r, k in zip(rows, kinds) if k == kind]
            means.append(float(np.mean(values)) if values else 0.0)
            colors.append(CASE_COLORS.get(kind, ACCENT))
        ax.bar(range(len(order)), means, color=colors, width=0.62)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CASE_LABELS.get(k, k) for k in order], rotation=30, ha="right")
        ax.set_ylim(0, max(1.0, max(means) * 1.25))
        ax.set_title(feature, fontsize=11.5)

        fraud_mean = means[0] if order and order[0] == "mule_network" else max(means)
        rest = [m for m, k in zip(means, order) if k != "mule_network"]
        gap = fraud_mean - (max(rest) if rest else 0.0)
        ax.text(0.97, 0.95, f"gap {gap:+.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10.5, weight="bold",
                color=FRAUD if gap > 0.2 else INK_FAINT,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=PAPER,
                          edgecolor=RULE, linewidth=0.9))

    fig.suptitle(
        "Features computed from raw events, by case kind",
        fontsize=15, fontweight="bold", y=1.03,
    )
    return _save(
        fig, name,
        "Means over every case in the simulated world. "
        "Gap = mule network minus the strongest other kind.",
        directory,
        note_y=-0.16,
    )


# ==========================================================================
# Figure 3 — detectability curve
# ==========================================================================


def plot_detectability_curve(
    knob_values: Sequence[float],
    recall_by_detector: dict[str, Sequence[float]],
    *,
    knob_label: str = "independent funders",
    name: str = "detectability_curve",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """Detection against evasion effort — the main figure of the study.

    The knee is the result: the point past which a detector loses the
    network, and therefore the price the organiser has to pay to get there.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    palette = [ACCENT, FRAUD, HARD_NEGATIVE, HONEST, CASH]

    for (label, recalls), color in zip(recall_by_detector.items(), palette):
        ax.plot(knob_values, recalls, marker="o", markersize=5, color=color, label=label)
        knee = _find_knee(list(knob_values), list(recalls))
        if knee is not None:
            ax.axvline(knee, color=color, alpha=0.25, linestyle="--", linewidth=1.4)

    ax.set_xlabel(knob_label)
    ax.set_ylabel("share of networks detected")
    ax.set_ylim(0, 1.02)
    ax.set_title("Detectability against evasion effort")
    ax.legend(loc="lower left")
    return _save(
        fig, name,
        "Dashed lines mark the knee: where a detector starts losing networks.",
        directory,
    )


def _find_knee(xs: list[float], ys: list[float]) -> float | None:
    """First x where the curve falls fastest. Simple by design: a knee is
    read off a plot by eye, and the marker only has to agree with the eye."""
    if len(xs) < 3:
        return None
    drops = [(ys[i] - ys[i + 1], xs[i]) for i in range(len(xs) - 1)]
    steepest, position = max(drops, key=lambda pair: pair[0])
    return position if steepest > 0.05 else None


# ==========================================================================
# Figure 4 — value already gone at the moment of the alert
# ==========================================================================


def plot_value_lost(
    detector_names: Sequence[str],
    recalls: Sequence[float],
    value_lost_shares: Sequence[float],
    *,
    name: str = "value_lost",
    directory: Path = DEFAULT_FIGURE_DIR,
) -> Path:
    """Recall against money not saved.

    The point of the figure is that these two rankings need not agree: a
    detector with lower recall but a faster alert can leave less money gone.
    Money is the quantity the regulator reports, so it is the quantity worth
    optimising.
    """
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    saved = [1.0 - share for share in value_lost_shares]

    ax.scatter(recalls, saved, s=180, c=ACCENT, edgecolors=PAPER, linewidths=2, zorder=3)
    for label, x, y in zip(detector_names, recalls, saved):
        ax.annotate(label, (x, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=10, color=INK_SOFT)

    limit = max([*recalls, *saved, 0.1]) * 1.15
    ax.plot([0, limit], [0, limit], color=INK_FAINT, alpha=0.4,
            linestyle=":", linewidth=1.4, zorder=1)
    ax.text(limit * 0.96, limit * 0.9, "rankings agree", ha="right",
            fontsize=9.5, color=INK_FAINT)

    ax.set_xlabel("share of networks detected")
    ax.set_ylabel("share of value saved")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_title("Detection is not the same as money saved")
    return _save(
        fig, name,
        "A point above the diagonal saves more money than its recall suggests: "
        "it alerts earlier.",
        directory,
    )


def sequence_span_minutes(events: Sequence[TransactionEvent]) -> float:
    """Helper shared by figures and captions."""
    if len(events) < 2:
        return 0.0
    stamps = sorted(event.ts for event in events)
    return float(math.ceil((stamps[-1] - stamps[0]).total_seconds() / 60.0))
