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
3. **Same recipient in a short window.** The mirror of the link above, and it
   exists so the honest side of the task is representable: a whip-round is
   forty people paying one collector, and without this link those forty share
   nothing the other rules can see.

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
from datetime import datetime, timedelta

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

# A receiver paid by more distinct accounts than this over the whole period
# is a hub by nature — a merchant, an exchange, a utility. Linking everyone
# who ever paid one would merge unrelated people into a single component.
DEFAULT_HUB_DEGREE_CAP = 120

# An intermediary paying more distinct receivers than this is not relaying,
# it is distributing. Expansion stops there so one shop cannot join every
# customer to every other.
DEFAULT_RELAY_FAN_CAP = 25


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
    events: list[TransactionEvent],
    union: _Union,
    hops: int,
    window: timedelta,
    relay_fan_cap: int,
) -> None:
    """Accounts fed, directly or within k hops, from the same origin.

    Randomising the immediate sender moves the shared origin up a hop; it
    does not remove it, unless the organiser buys genuinely independent
    funding for every mule.

    The time attached to a link is the moment money arrived ALONG THAT PATH,
    not the account's most recent activity. An earlier version used the
    latter, and it quietly broke the honest side of the task: two employees
    paid by one company on the same payday were not linked, because their
    "time" was whichever salary happened to be their last. Payrolls then
    never appeared as candidates at all, honest clusters came out at a
    median size of three against eleven for rings, and a classifier
    separated the two on size alone.
    """
    outgoing: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type != ASSET_CASH:
            outgoing[event.sender_id].append(event)

    # ancestor -> [(account, time money arrived along this path)]
    reach: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for sender, sent in outgoing.items():
        for event in sent:
            reach[sender].append((event.receiver_id, event.ts))

    # Expansion must not pass THROUGH a hub. An account paying hundreds of
    # distinct receivers — an employer, a pyramid core, a shop — links all of
    # its descendants to each other at the next hop, and one such node is
    # enough to merge the whole world.
    #
    # Measured before this guard: a single component of 5 398 accounts formed
    # and was then discarded for exceeding the size cap, taking with it every
    # salary earner, 243 fast spenders and 210 mules. The hard half of the
    # task was being deleted silently, which is why the remaining candidates
    # looked trivially separable.
    fan_out = {sender: len({e.receiver_id for e in sent}) for sender, sent in outgoing.items()}

    for _ in range(max(0, hops - 1)):
        extended: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
        for ancestor, children in reach.items():
            for child, _arrived in children:
                if fan_out.get(child, 0) > relay_fan_cap:
                    continue
                for event in outgoing.get(child, []):
                    extended[ancestor].append((event.receiver_id, event.ts))
        for ancestor, children in extended.items():
            reach[ancestor].extend(children)

    for children in reach.values():
        if len(children) < 2:
            continue
        ordered = sorted(children, key=lambda pair: pair[1])
        left = 0
        for right in range(len(ordered)):
            while ordered[right][1] - ordered[left][1] > window:
                left += 1
            for middle in range(left, right):
                if ordered[middle][0] != ordered[right][0]:
                    union.union(ordered[middle][0], ordered[right][0], "common_ancestor")


def _link_common_receiver(
    events: list[TransactionEvent],
    union: _Union,
    window: timedelta,
    hub_degree_cap: int,
) -> None:
    """Accounts that paid the same recipient inside one window.

    The mirror image of the ancestor link, and it exists so the honest side
    of the task is representable at all. A whip-round is forty people paying
    one collector; without this link those forty share no resource the
    other rules can see, the collector stands alone below the minimum size,
    and the honest population never reaches the candidate set.

    Receivers with very high lifetime in-degree are skipped. A merchant or
    an exchange is a hub by its nature, and linking everyone who ever paid
    one would merge unrelated people into a single blob.
    """
    by_receiver: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type != ASSET_CASH:
            by_receiver[event.receiver_id].append(event)

    for arrivals in by_receiver.values():
        if len({e.sender_id for e in arrivals}) > hub_degree_cap:
            continue
        arrivals.sort(key=lambda e: e.ts)
        left = 0
        for right in range(len(arrivals)):
            while arrivals[right].ts - arrivals[left].ts > window:
                left += 1
            for middle in range(left, right):
                if arrivals[middle].sender_id != arrivals[right].sender_id:
                    union.union(
                        arrivals[middle].sender_id,
                        arrivals[right].sender_id,
                        "common_receiver",
                    )


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
    hub_degree_cap: int = DEFAULT_HUB_DEGREE_CAP,
    relay_fan_cap: int = DEFAULT_RELAY_FAN_CAP,
) -> list[Candidate]:
    """Propose candidate clusters from events alone.

    ``world.networks`` is deliberately never read here. The only thing taken
    from the world is its event stream.
    """
    events = list(world.events)
    union = _Union()

    _link_shared_terminal(events, union, terminal_window)
    _link_common_ancestor(events, union, ancestor_hops, terminal_window, relay_fan_cap)
    _link_common_receiver(events, union, terminal_window, hub_degree_cap)

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
