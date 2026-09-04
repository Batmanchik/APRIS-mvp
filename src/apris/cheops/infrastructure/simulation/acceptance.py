"""Acceptance criterion for the simulation layer.

The single check that must not be skipped. It answers the question that
invalidated the previous generation of this project: is the generator too
kind to the detector?

Three naive rules — the three ideas that come to mind first — must each
misfire on honest people:

    A. "cashed out within 5 minutes of money arriving"
       must flag the student who withdraws everything at once.
       If it does not, account-level detection would be solving a fiction.

    B. "sent money to 20+ distinct people within an hour"
       must flag an employer on payday.
       If it does not, there is no honest fan-out and any disbursement
       would look criminal.

    C. "received money from 20+ distinct people within a day"
       must flag a whip-round for a common cause.
       If it does not, there is no honest convergence and a graph detector
       would be handed a task where any convergence means fraud.

Rule C also carries a finding worth reporting on its own: the most intuitive
graph idea — "many senders into one account" — POINTS THE WRONG WAY for a
mule network. There the fan spreads out from the source and converges on the
ATM; the mule account itself sees no convergence at all.

A second gate exists because the first version of this criterion was too
weak: it checked that naive RULES misfire, and they did, while a gradient
boosting model on the same account-level features still separated the
classes perfectly. Rules failing is not evidence that the task is hard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.simulation.config import (
    ASSET_CASH,
    DEFAULT_NAIVE_RULES,
    NaiveRuleThresholds,
    SimulationConfig,
)
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld, generate_world

HONEST_POPULATIONS = frozenset(
    {
        "salary",
        "freelancer",
        "trader",
        "fast_spender",
        "marketplace_seller",
        "family_circle",
        "crowd_collector",
        "crowd_donor",
        "employer",
        "pyramid_investor",
    }
)


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class AcceptanceReport:
    checks: tuple[AcceptanceCheck, ...]
    breakdowns: dict[str, dict[str, tuple[int, int]]]
    fan_in_hits_on_mules: tuple[int, int]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


# ==========================================================================
# Naive rules — raw events only, exactly what a detector would see
# ==========================================================================


def rule_fast_cashout(
    events: list[TransactionEvent], rules: NaiveRuleThresholds = DEFAULT_NAIVE_RULES
) -> set[str]:
    incoming: dict[str, list[TransactionEvent]] = defaultdict(list)
    cashouts: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type == ASSET_CASH:
            cashouts[event.sender_id].append(event)
        else:
            incoming[event.receiver_id].append(event)

    window = timedelta(minutes=rules.fast_cashout_minutes)
    flagged: set[str] = set()
    for account, outgoing in cashouts.items():
        arrivals = incoming.get(account, [])
        if not arrivals:
            continue
        for out_event in outgoing:
            for in_event in arrivals:
                gap = out_event.ts - in_event.ts
                if timedelta(0) <= gap <= window and out_event.amount >= in_event.amount * 0.8:
                    flagged.add(account)
                    break
            if account in flagged:
                break
    return flagged


def _distinct_partners_in_window(
    events: list[TransactionEvent], partner: str, threshold: int, window: timedelta
) -> bool:
    events.sort(key=lambda e: e.ts)
    left = 0
    seen: dict[str, int] = defaultdict(int)
    for right in range(len(events)):
        seen[getattr(events[right], partner)] += 1
        while events[right].ts - events[left].ts > window:
            key = getattr(events[left], partner)
            seen[key] -= 1
            if seen[key] == 0:
                del seen[key]
            left += 1
        if len(seen) >= threshold:
            return True
    return False


def rule_fan_out(
    events: list[TransactionEvent], rules: NaiveRuleThresholds = DEFAULT_NAIVE_RULES
) -> set[str]:
    by_sender: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type != ASSET_CASH:
            by_sender[event.sender_id].append(event)
    window = timedelta(hours=rules.fan_out_window_hours)
    return {
        account
        for account, group in by_sender.items()
        if _distinct_partners_in_window(group, "receiver_id", rules.fan_out_count, window)
    }


def rule_fan_in(
    events: list[TransactionEvent], rules: NaiveRuleThresholds = DEFAULT_NAIVE_RULES
) -> set[str]:
    by_receiver: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type != ASSET_CASH:
            by_receiver[event.receiver_id].append(event)
    window = timedelta(hours=rules.fan_in_window_hours)
    return {
        account
        for account, group in by_receiver.items()
        if _distinct_partners_in_window(group, "sender_id", rules.fan_in_count, window)
    }


# ==========================================================================
# Report
# ==========================================================================


def _breakdown(world: SimulatedWorld, flagged: set[str]) -> dict[str, tuple[int, int]]:
    total: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)
    for account, population in world.populations.items():
        total[population] += 1
        if account in flagged:
            hit[population] += 1
    return {population: (hit[population], total[population]) for population in sorted(total)}


def _share(breakdown: dict[str, tuple[int, int]], population: str) -> tuple[float, str]:
    hits, total = breakdown.get(population, (0, 0))
    return ((hits / total) if total else 0.0), f"{hits}/{total}"


def evaluate(world: SimulatedWorld) -> AcceptanceReport:
    events = world.events
    flagged_a = rule_fast_cashout(events)
    flagged_b = rule_fan_out(events)
    flagged_c = rule_fan_in(events)

    breakdowns = {
        "fast_cashout": _breakdown(world, flagged_a),
        "fan_out": _breakdown(world, flagged_b),
        "fan_in": _breakdown(world, flagged_c),
    }

    student, student_detail = _share(breakdowns["fast_cashout"], "fast_spender")
    employer, employer_detail = _share(breakdowns["fan_out"], "employer")
    collector, collector_detail = _share(breakdowns["fan_in"], "crowd_collector")
    mule_a, mule_a_detail = _share(breakdowns["fast_cashout"], "mule")
    funder, funder_detail = _share(breakdowns["fan_out"], "mule_funder")
    mule_c_hits, mule_c_total = breakdowns["fan_in"].get("mule", (0, 0))

    checks = (
        AcceptanceCheck("A misfires on the fast-spending student", student >= 0.20, student_detail),
        AcceptanceCheck("B misfires on the payday employer", employer >= 0.50, employer_detail),
        AcceptanceCheck("C misfires on the whip-round", collector >= 0.50, collector_detail),
        AcceptanceCheck("A still catches mules", mule_a >= 0.50, mule_a_detail),
        # 0.35 rather than 0.50: networks smaller than the fan-out threshold
        # cannot produce a fan at all. That is a property of the size
        # distribution, not a defect.
        AcceptanceCheck("B still catches the network source", funder >= 0.35, funder_detail),
    )

    return AcceptanceReport(
        checks=checks,
        breakdowns=breakdowns,
        fan_in_hits_on_mules=(mule_c_hits, mule_c_total),
    )


def run(config: SimulationConfig | None = None) -> AcceptanceReport:
    return evaluate(generate_world(config or SimulationConfig()))
