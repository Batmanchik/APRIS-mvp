"""Turn a simulated world into labelled cases.

A *case* is the event neighbourhood of one entity — the unit a detector
actually scores. Four kinds are produced, and the pairing matters: each
fraudulent kind sits opposite an honest structure with the same shape.

    mule_network      fan-out then convergence on an ATM, minutes
    payroll           fan-out from one company, honest, minutes   <- pairs with the above
    crowd_collection  convergence into one account, honest, a day
    pyramid           convergence into one account, months        <- pairs with the above

Scoring these four is what tests whether structure carries information that
single-account features do not. If a detector cannot tell a mule network
from a payroll, or a pyramid from a whip-round, then the network level adds
nothing and the project's main hypothesis is false.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld

CASE_MULE_NETWORK = "mule_network"
CASE_PAYROLL = "payroll"
CASE_CROWD_COLLECTION = "crowd_collection"
CASE_PYRAMID = "pyramid"

FRAUDULENT_CASE_KINDS = frozenset({CASE_MULE_NETWORK, CASE_PYRAMID})


@dataclass(frozen=True)
class LabelledCase:
    case_id: str
    kind: str
    events: tuple[TransactionEvent, ...]
    member_ids: tuple[str, ...]

    @property
    def label(self) -> int:
        return 1 if self.kind in FRAUDULENT_CASE_KINDS else 0


def _events_by_account(world: SimulatedWorld) -> dict[str, list[TransactionEvent]]:
    index: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in world.events:
        index[event.sender_id].append(event)
        index[event.receiver_id].append(event)
    return index


def _collect(
    index: dict[str, list[TransactionEvent]], members: list[str]
) -> tuple[TransactionEvent, ...]:
    seen: dict[str, TransactionEvent] = {}
    for member in members:
        for event in index.get(member, []):
            seen[event.event_id] = event
    return tuple(sorted(seen.values(), key=lambda e: e.ts))


def build_cases(world: SimulatedWorld, *, min_events: int = 6) -> list[LabelledCase]:
    """Build every labelled case present in the world."""
    index = _events_by_account(world)
    cases: list[LabelledCase] = []

    for network in world.networks:
        members = list(network.account_ids) + list(network.organizer_ids)
        events = _collect(index, members)
        if len(events) < min_events:
            continue
        kind = CASE_MULE_NETWORK if network.kind == "mule_fast" else CASE_PYRAMID
        cases.append(
            LabelledCase(
                case_id=network.network_id,
                kind=kind,
                events=events,
                member_ids=tuple(members),
            )
        )

    for account, population in world.populations.items():
        if population == "employer":
            kind = CASE_PAYROLL
        elif population == "crowd_collector":
            kind = CASE_CROWD_COLLECTION
        else:
            continue
        events = _collect(index, [account])
        if len(events) < min_events:
            continue
        cases.append(
            LabelledCase(
                case_id=f"{kind}-{account}",
                kind=kind,
                events=events,
                member_ids=(account,),
            )
        )

    return cases


def case_kind_counts(cases: list[LabelledCase]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case.kind] += 1
    return dict(sorted(counts.items()))
