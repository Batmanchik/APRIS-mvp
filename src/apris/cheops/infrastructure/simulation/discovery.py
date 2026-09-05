"""Candidate discovery — finding cases without looking at the answers.

The problem this replaces
------------------------
``cases.build_cases`` assembles a mule-network case from ``world.networks``,
which is the ground-truth file. The twenty accounts that form a ring are
handed to the detector already grouped, while an honest case arrives as a
single account. A classifier then separates them on almost any column, every
cross-validation split returns ROC-AUC 1.0000, and permutation importance is
0.0000 for every feature including the shuffled control — the signature of a
saturated task rather than a good detector.

A real detector is never handed the grouping. It has to propose candidate
clusters itself and then decide which of them are networks. Defining the unit
of analysis from the answers deletes the harder half of the problem before
modelling starts.

How discovery works here
------------------------
Accounts are linked when they **share a resource**, never when they pay each
other. Three links, each chosen because it survives an evasion the payment
graph does not:

1. **Same exit point in a short window.** Two accounts that cashed out at one
   ATM within minutes. The organiser has to be physically present to collect
   the cash, so the exit is the one thing a ring cannot spread cheaply.
2. **Common funding ancestor within k hops.** Randomising the direct sender
   pushes the shared origin one hop up rather than removing it; forty
   genuinely independent funders cost forty real accounts holding real money.
3. **Same tight time window.** An operation run in one burst leaves its
   members co-occurring in time whatever else is randomised.

Connected components of that link graph are the candidates. A candidate may
turn out to be a ring, a payroll, a whip-round, or junk — telling them apart
is the measured task.

Ground truth enters exactly once, after discovery, to label a candidate that
has already been found. It also yields a number the previous design could not
produce: **the share of real networks discovery never proposed at all**, which
is a ceiling on recall no downstream model can lift.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.simulation.config import ASSET_CASH
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld

# Two accounts cashing out at one terminal inside this window are linked.
DEFAULT_TERMINAL_WINDOW = timedelta(minutes=30)

# Depth of the search for a shared funding ancestor.
DEFAULT_ANCESTOR_HOPS = 2

# A candidate smaller than this is not a structure worth scoring.
MIN_CANDIDATE_SIZE = 3

# Discovery must not return one giant blob; a component past this is split
# off as unusable rather than silently dominating every metric.
MAX_CANDIDATE_SIZE = 400


@dataclass(frozen=True)
class Candidate:
    """A cluster proposed without reference to the answers."""

    candidate_id: str
    member_ids: tuple[str, ...]
    events: tuple[TransactionEvent, ...]
    link_reasons: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.member_ids)


@dataclass(frozen=True)
class DiscoveryReport:
    """What discovery proposed, and what it missed."""

    candidates: tuple[Candidate, ...]
    networks_total: int
    networks_covered: int
    missed_network_ids: tuple[str, ...]

    @property
    def coverage(self) -> float:
        """Share of real networks that discovery proposed at all.

        This is a ceiling on recall: a network no candidate contains cannot
        be found by any model downstream, however good it is.
        """
        return self.networks_covered / self.networks_total if self.networks_total else 0.0


# ==========================================================================
# Union-find over accounts
# ==========================================================================


class _Union:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.reasons: dict[str, set[str]] = defaultdict(set)

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str, reason: str) -> None:
        a, b = self.find(left), self.find(right)
        self.reasons[a].add(reason)
        if a == b:
            return
        self.parent[b] = a
        self.reasons[a] |= self.reasons.pop(b, set())

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return out


# ==========================================================================
# The three links
# ==========================================================================


def _link_shared_terminal(
    events: list[TransactionEvent], union: _Union, window: timedelta
) -> None:
    """Accounts that cashed out at the same terminal within one window."""
    by_terminal: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type == ASSET_CASH:
            by_terminal[event.receiver_id].append(event)

    for withdrawals in by_terminal.values():
        withdrawals.sort(key=lambda e: e.ts)
        left = 0
        for right in range(len(withdrawals)):
            while withdrawals[right].ts - withdrawals[left].ts > window:
                left += 1
            for middle in range(left, right):
                if withdrawals[middle].sender_id != withdrawals[right].sender_id:
                    union.union(
                        withdrawals[middle].sender_id,
                        withdrawals[right].sender_id,
                        "shared_terminal",
                    )


def _link_common_ancestor(
    events: list[TransactionEvent], union: _Union, hops: int, window: timedelta
) -> None:
    """Accounts fed, directly or within k hops, from the same origin.

    Randomising the immediate sender moves the shared origin up a hop; it
    does not remove it, unless the organiser buys genuinely independent
    funding for every mule.
    """
    incoming: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type != ASSET_CASH:
            incoming[event.receiver_id].append(event)

    def ancestors(account: str) -> set[str]:
        seen: set[str] = set()
        frontier = {account}
        for _ in range(hops):
            nxt: set[str] = set()
            for node in frontier:
                for event in incoming.get(node, []):
                    if event.sender_id not in seen:
                        nxt.add(event.sender_id)
            seen |= nxt
            frontier = nxt
            if not frontier:
                break
        return seen

    fed_by: dict[str, list[tuple[str, TransactionEvent]]] = defaultdict(list)
    for account, arrivals in incoming.items():
        for ancestor in ancestors(account):
            latest = max(arrivals, key=lambda e: e.ts)
            fed_by[ancestor].append((account, latest))

    for children in fed_by.values():
        if len(children) < 2:
            continue
        children.sort(key=lambda pair: pair[1].ts)
        left = 0
        for right in range(len(children)):
            while children[right][1].ts - children[left][1].ts > window:
                left += 1
            for middle in range(left, right):
                if children[middle][0] != children[right][0]:
                    union.union(children[middle][0], children[right][0], "common_ancestor")


# ==========================================================================
# Discovery
# ==========================================================================


def discover_candidates(
    world: SimulatedWorld,
    *,
    terminal_window: timedelta = DEFAULT_TERMINAL_WINDOW,
    ancestor_hops: int = DEFAULT_ANCESTOR_HOPS,
    min_size: int = MIN_CANDIDATE_SIZE,
    max_size: int = MAX_CANDIDATE_SIZE,
) -> list[Candidate]:
    """Propose candidate clusters from events alone.

    ``world.networks`` is deliberately never read here. The only thing taken
    from the world is its event stream.
    """
    events = list(world.events)
    union = _Union()

    _link_shared_terminal(events, union, terminal_window)
    _link_common_ancestor(events, union, ancestor_hops, terminal_window)

    by_account: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        by_account[event.sender_id].append(event)
        by_account[event.receiver_id].append(event)

    candidates: list[Candidate] = []
    for index, (root, members) in enumerate(sorted(union.groups().items())):
        if not (min_size <= len(members) <= max_size):
            continue
        collected: dict[str, TransactionEvent] = {}
        for member in members:
            for event in by_account.get(member, []):
                collected[event.event_id] = event
        if len(collected) < min_size:
            continue
        candidates.append(
            Candidate(
                candidate_id=f"CAND{index:05d}",
                member_ids=tuple(sorted(members)),
                events=tuple(sorted(collected.values(), key=lambda e: e.ts)),
                link_reasons=tuple(sorted(union.reasons.get(root, set()))),
            )
        )
    return candidates


def label_candidates(
    world: SimulatedWorld,
    candidates: list[Candidate],
    *,
    overlap: float = 0.5,
) -> tuple[list[int], DiscoveryReport]:
    """Attach labels AFTER discovery, and report what was never proposed.

    A candidate counts as a network when at least ``overlap`` of one
    network's accounts are inside it. The threshold matters: discovery may
    split a ring or swallow it inside a larger blob, and both are partial
    successes that a strict rule would score as failures.
    """
    network_members = {
        network.network_id: set(network.account_ids)
        for network in world.networks
        if network.kind == "mule_fast"
    }

    labels: list[int] = []
    covered: set[str] = set()
    for candidate in candidates:
        members = set(candidate.member_ids)
        hit = False
        for network_id, accounts in network_members.items():
            if accounts and len(accounts & members) / len(accounts) >= overlap:
                hit = True
                covered.add(network_id)
        labels.append(1 if hit else 0)

    report = DiscoveryReport(
        candidates=tuple(candidates),
        networks_total=len(network_members),
        networks_covered=len(covered),
        missed_network_ids=tuple(sorted(set(network_members) - covered)),
    )
    return labels, report
