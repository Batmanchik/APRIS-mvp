"""The nine legacy features, computed from RAW EVENTS.

Why this matters
----------------
The nine features the original model was trained on were written directly by
that model's own generator: it put ``gini = 0.67`` into a pyramid's row, and
the model read the answer instead of finding it. Its reported ROC-AUC of
0.993 therefore measured nothing at all, and no external validation was
structurally possible.

Computing the same nine from an independent event stream is what makes the
old model testable for the first time.

The first such run found it flagging none of thirty pyramids at its own
high-risk threshold. That result did not survive: two of the nine were coming
out as flat zeros because of defects in our generator, not in the model, and
once those were fixed the same model found forty of forty. The episode is
recorded in Finding 2 because measuring a model through a broken instrument
and blaming the model is a mistake worth not repeating.

The referral tree is inferred, not read
---------------------------------------
Two of the nine — ``referral_ratio`` and ``structural_depth`` — need to know
who recruited whom. The ground-truth file has that tree, and reading it would
be cheating. It is reconstructed from money instead: a depositor pays the
core, and shortly afterwards the core pays a percentage of that amount to
someone who is already a participant. That someone recruited them. This is
how a referral network is rebuilt from a bank statement in a real
investigation.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from apris.cheops.domain.models import TransactionEvent

# Bounds the original model was trained against. Values must land inside them
# or that model refuses the input.
LEGACY_BOUNDS: dict[str, tuple[float, float]] = {
    "growth_rate": (0.0, 1.2),
    "referral_ratio": (0.0, 1.0),
    "payout_dependency": (0.1, 1.9),
    "centralization_index": (0.0, 1.0),
    "avg_holding_time": (3.0, 120.0),
    "reinvestment_rate": (0.0, 1.0),
    "gini_coefficient": (0.1, 1.0),
    "transaction_entropy": (0.3, 5.0),
    "structural_depth": (2.0, 16.0),
}

LEGACY_NAMES: list[str] = list(LEGACY_BOUNDS)

ASSET_CASH = "cash"

# A referral bonus follows the deposit that earned it within this window and
# is a percentage of it. Both are the mechanics of the scheme, not thresholds
# tuned to make the reconstruction work.
BONUS_WINDOW = timedelta(hours=72)
BONUS_SHARE_RANGE = (0.03, 0.25)

MAX_TREE_DEPTH = 64


@dataclass(frozen=True)
class FlowMatch:
    """One matched parcel of money: how much, and how long it sat."""

    amount: float
    held_seconds: float


def _clip(name: str, value: float) -> float:
    low, high = LEGACY_BOUNDS[name]
    return float(max(low, min(high, value)))


def _gini(values: Sequence[float]) -> float:
    """Inequality of amounts, by the standard sorted-series formula."""
    ordered = sorted(v for v in values if v > 0)
    count = len(ordered)
    if count == 0:
        return 0.0
    total = sum(ordered)
    if total <= 0:
        return 0.0
    cumulative = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * cumulative) / (count * total) - (count + 1.0) / count


def _entropy(counts: Sequence[float]) -> float:
    """Shannon entropy over how value is spread across counterparties."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


# ==========================================================================
# FIFO matching — the financial core
# ==========================================================================


def fifo_match(
    account: str, events: Sequence[TransactionEvent]
) -> tuple[list[FlowMatch], float, float, float]:
    """Run an account's flow through a first-in-first-out queue.

    Money is fungible: when three payments arrive and one leaves, there is no
    fact of the matter about which tenge left. FIFO is a design decision, it
    is recorded as one in the methodology, and its ambiguity is acknowledged
    rather than hidden. Everything downstream that speaks of holding time
    rests on it.

    Returns the matched parcels, total in, total out, and the outflow that
    found no matching inflow — money spent from a balance that was already
    there, which is what an ordinary account does and a pass-through does not.
    """
    # (remaining amount, when it arrived). A partially consumed parcel is
    # popped and its remainder pushed back, which keeps the queue a plain
    # tuple rather than a mutable union the type checker cannot follow.
    queue: deque[tuple[float, datetime]] = deque()
    matches: list[FlowMatch] = []
    total_in = total_out = unmatched_out = 0.0

    for event in sorted(events, key=lambda e: e.ts):
        if event.receiver_id == account:
            queue.append((event.amount, event.ts))
            total_in += event.amount
        elif event.sender_id == account:
            total_out += event.amount
            remaining = event.amount
            while remaining > 1e-9 and queue:
                available, arrived = queue.popleft()
                taken = min(available, remaining)
                matches.append(
                    FlowMatch(
                        amount=taken,
                        held_seconds=(event.ts - arrived).total_seconds(),
                    )
                )
                remaining -= taken
                if available - taken > 1e-9:
                    queue.appendleft((available - taken, arrived))
            if remaining > 1e-9:
                unmatched_out += remaining

    return matches, total_in, total_out, unmatched_out


# ==========================================================================
# Referral reconstruction from money
# ==========================================================================


def infer_referrals(entity: str, events: Sequence[TransactionEvent]) -> dict[str, str]:
    """Who recruited whom, derived from payments rather than read from answers.

    A depositor pays the core at time t. If within 72 hours the core pays an
    existing participant between 3 % and 25 % of that amount, that participant
    is taken to have recruited them.
    """
    deposits = [
        e for e in events if e.receiver_id == entity and e.asset_type != ASSET_CASH
    ]
    payouts = sorted(
        (e for e in events if e.sender_id == entity and e.asset_type != ASSET_CASH),
        key=lambda e: e.ts,
    )

    known: set[str] = set()
    referrals: dict[str, str] = {}
    low, high = BONUS_SHARE_RANGE

    for deposit in sorted(deposits, key=lambda e: e.ts):
        for payout in payouts:
            if payout.ts < deposit.ts:
                continue
            if payout.ts - deposit.ts > BONUS_WINDOW:
                break
            share = payout.amount / deposit.amount if deposit.amount > 0 else 0.0
            if low <= share <= high and payout.receiver_id in known:
                if payout.receiver_id != deposit.sender_id:
                    referrals[deposit.sender_id] = payout.receiver_id
                    break
        known.add(deposit.sender_id)
    return referrals


def referral_depth(referrals: dict[str, str], root: str) -> int:
    """Depth of the reconstructed tree, defended against cycles.

    The tree is inferred from noisy evidence, so it can contain a cycle that
    a real recruitment chain never would. Walking one without a guard hangs.
    """
    deepest = 1
    for node in referrals:
        depth, seen, current = 1, {node}, node
        while current in referrals:
            current = referrals[current]
            if current in seen or current == root:
                break
            seen.add(current)
            depth += 1
            if depth > MAX_TREE_DEPTH:
                break
        deepest = max(deepest, depth)
    return deepest


# ==========================================================================
# The nine features
# ==========================================================================


def legacy_features(
    entity: str,
    events: Sequence[TransactionEvent],
    start: datetime,
    end: datetime,
) -> dict[str, float]:
    """The original nine, for an entity that collects money from many people.

    ``start`` and ``end`` must be the analysis period, not the range of the
    events themselves. Ordinary spending drags an account's tail far past the
    horizon, and taking the range from the data moved the midpoint so far
    that growth came out as exactly 0.000 for every pyramid.
    """
    window = [e for e in events if start <= e.ts <= end]
    inflows = [
        e for e in window if e.receiver_id == entity and e.asset_type != ASSET_CASH
    ]
    outflows = [e for e in window if e.sender_id == entity]

    total_in = sum(e.amount for e in inflows)
    total_out = sum(e.amount for e in outflows)

    midpoint = start + (end - start) / 2
    early = {e.sender_id for e in inflows if e.ts < midpoint}
    late = {e.sender_id for e in inflows if e.ts >= midpoint} - early
    growth = (len(late) - len(early)) / max(len(early), 1)

    referrals = infer_referrals(entity, window)
    depositors = {e.sender_id for e in inflows}
    referral_ratio = len(referrals) / max(len(depositors), 1)

    payout_dependency = total_out / total_in if total_in > 0 else 0.0

    by_sender: dict[str, float] = defaultdict(float)
    for event in inflows:
        by_sender[event.sender_id] += event.amount
    centralization = (
        max(by_sender.values()) / total_in if total_in > 0 and by_sender else 0.0
    )

    matches, _, _, _ = fifo_match(entity, window)
    if matches:
        weight = sum(m.amount for m in matches)
        held = sum(m.amount * m.held_seconds for m in matches) / weight
        holding_days = held / 86400.0
    else:
        holding_days = float((end - start).days)

    repeat_counts: dict[str, int] = defaultdict(int)
    for event in inflows:
        repeat_counts[event.sender_id] += 1
    repeats = sum(1 for count in repeat_counts.values() if count > 1)
    reinvestment = repeats / max(len(repeat_counts), 1)

    return {
        "growth_rate": _clip("growth_rate", growth),
        "referral_ratio": _clip("referral_ratio", referral_ratio),
        "payout_dependency": _clip("payout_dependency", payout_dependency),
        "centralization_index": _clip("centralization_index", centralization),
        "avg_holding_time": _clip("avg_holding_time", holding_days),
        "reinvestment_rate": _clip("reinvestment_rate", reinvestment),
        "gini_coefficient": _clip("gini_coefficient", _gini(list(by_sender.values()))),
        "transaction_entropy": _clip(
            "transaction_entropy", _entropy(list(by_sender.values()))
        ),
        "structural_depth": _clip(
            "structural_depth", float(referral_depth(referrals, entity))
        ),
    }
