"""The published rule baseline, and an honest account of what it cannot see.

Why a baseline at all
---------------------
"Machine learning beats nothing" is not a result. The question a judge asks
first is why the rules already approved for this exact job are not enough,
and the only answer that survives is a measured one — the same data, the same
unit of analysis, the same metric.

The rules
---------
Kazakhstan's dropper identification rules (approved April 2026) define a mule
by four criteria:

1. transfers to persons on a list,
2. a shared telephone number,
3. a shared IP address or device,
4. deviation from the customer's usual profile.

What this module implements
---------------------------
Three of the four, because this simulator emits payments and nothing else:

* **listed** — a member transacted with an account already on the list. The
  list is supplied by the caller and must be built from structures that
  closed BEFORE the scored window, which is how such a list is built in
  practice. Seeding it from the answers for the candidate being scored would
  be the same leak the case builder was rewritten to remove.
* **shared_device** — two members cashing out at the same terminal. An ATM is
  a device, so this is criterion 3 read literally rather than by analogy.
* **profile_deviation** — a member moving far more than its own established
  baseline. It requires a baseline, so it cannot fire on an account with no
  history, which is the property the rules inherit and this measures.

Criterion 2 is **not expressible on payment data at all**, and the result
object says so rather than quietly scoring out of three. That absence is not
a handicap invented to make the baseline lose: it is the argument. Three of
the four criteria are facts about identity and hardware, which a transactional
layer at the level of a national payment system does not hold — and the one
that is about behaviour needs a history a freshly opened account does not
have. A ring of clean accounts, one phone each, none on any list, is the
population these rules were not built to see.

The score
---------
``fired / applicable`` — a rankable number in [0, 1] so the baseline can be
put on the same ROC curve as every model above it. A rule set is normally
run as a threshold, so ``flagged_any`` and ``flagged_two`` are reported too;
which of the three is quoted is stated wherever the number appears.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta

from apris.cheops.domain.models import TransactionEvent

ASSET_CASH = "cash"

# Two members using one terminal inside this window are using it together.
# Same value as discovery's link, and deliberately: a baseline given a
# narrower window than the detector would be losing on a technicality.
DEFAULT_TERMINAL_WINDOW = timedelta(minutes=30)

# A profile needs a profile. Below this many days of prior activity the
# account has no usual behaviour to deviate from, and the criterion abstains
# rather than guessing.
DEFAULT_MIN_HISTORY_DAYS = 14

# How far past its own daily baseline an account has to move before the
# deviation is worth a rule firing.
DEFAULT_DEVIATION_FACTOR = 4.0

# The criteria this module can evaluate, and the one it cannot.
APPLICABLE_CRITERIA: tuple[str, ...] = ("listed", "shared_device", "profile_deviation")
UNAVAILABLE_CRITERIA: tuple[str, ...] = ("shared_phone",)


@dataclass(frozen=True)
class AfrdVerdict:
    """One candidate, judged by the rules that payment data can express."""

    listed: bool
    shared_device: bool
    profile_deviation: bool

    #: Criteria that could not be evaluated on this data at all. Reported so
    #: a reader can see the baseline is scored out of three, not four.
    unavailable: tuple[str, ...] = UNAVAILABLE_CRITERIA

    #: Criteria the score is actually taken over. Scope-dependent — see
    #: ``criteria_for_scope``.
    counted: tuple[str, ...] = APPLICABLE_CRITERIA

    @property
    def fired(self) -> int:
        return sum(int(bool(getattr(self, name))) for name in self.counted)

    @property
    def score(self) -> float:
        return self.fired / len(self.counted) if self.counted else 0.0

    @property
    def flagged_any(self) -> bool:
        return self.fired >= 1

    @property
    def flagged_two(self) -> bool:
        return self.fired >= 2


def criteria_for_scope(scope: str) -> tuple[str, ...]:
    """Which criteria may be counted at a given unit of analysis.

    ``shared_device`` is excluded everywhere, and for two different reasons
    that happen to arrive at one answer.

    At the ACCOUNT unit it cannot fire at all: one account is not two people
    sharing hardware. Measured over 25 539 accounts it was 0.000 on both
    classes. A criterion that cannot fire is abstaining, not passing.

    At the NETWORK unit it fires on 100 % of fraudulent candidates and 4.2 %
    of honest ones — and discovery LINKS accounts by their shared terminal in
    the first place. It is reporting which of the three link types built the
    candidate, not whether the candidate is a ring. Scoring the baseline on
    that is the same circularity the case builder was rewritten to remove; it
    would simply be pointing the other way, in the baseline's favour, which
    is worse than pointing at our own.

    Leaving the same two criteria at both units is also what makes the H1
    comparison legitimate: same rules, same columns, only the unit moves.
    """
    if scope not in {"account", "network"}:
        raise ValueError(f"unknown scope: {scope}")
    return ("listed", "profile_deviation")


def _criterion_listed(
    members: frozenset[str], events: Sequence[TransactionEvent], listed: frozenset[str]
) -> bool:
    """A member paid, or was paid by, someone already on the list.

    A member who is themselves on the list is ignored. Scoring a candidate as
    caught because the answer was handed in with it is the leak, not the rule.
    """
    outside = listed - members
    if not outside:
        return False
    for event in events:
        if event.sender_id in members and event.receiver_id in outside:
            return True
        if event.receiver_id in members and event.sender_id in outside:
            return True
    return False


def _criterion_shared_device(
    members: frozenset[str],
    events: Sequence[TransactionEvent],
    window: timedelta,
) -> bool:
    """Two different members cashed out at one terminal within the window."""
    by_terminal: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in events:
        if event.asset_type == ASSET_CASH and event.sender_id in members:
            by_terminal[event.receiver_id].append(event)

    for withdrawals in by_terminal.values():
        withdrawals.sort(key=lambda e: e.ts)
        for index, first in enumerate(withdrawals):
            for second in withdrawals[index + 1 :]:
                if second.ts - first.ts > window:
                    break
                if second.sender_id != first.sender_id:
                    return True
    return False


def _criterion_profile_deviation(
    members: frozenset[str],
    events: Sequence[TransactionEvent],
    min_history_days: int,
    factor: float,
) -> bool:
    """Some member moved far more in a day than that member usually does.

    The comparison is against the account's own median day, so a person who
    is simply richer than average never trips it. An account with fewer than
    ``min_history_days`` active days has no usual behaviour and abstains,
    which is exactly the gap the rules carry.
    """
    per_member: dict[str, dict[object, float]] = defaultdict(lambda: defaultdict(float))
    for event in events:
        if event.sender_id in members:
            per_member[event.sender_id][event.ts.date()] += event.amount

    for daily in per_member.values():
        if len(daily) < min_history_days:
            continue
        amounts = sorted(daily.values())
        baseline = statistics.median(amounts[:-1]) if len(amounts) > 1 else amounts[0]
        if baseline > 0 and amounts[-1] > factor * baseline:
            return True
    return False


def afrd_verdict(
    member_ids: Iterable[str],
    events: Sequence[TransactionEvent],
    *,
    listed: frozenset[str] = frozenset(),
    terminal_window: timedelta = DEFAULT_TERMINAL_WINDOW,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    deviation_factor: float = DEFAULT_DEVIATION_FACTOR,
    counted: tuple[str, ...] = APPLICABLE_CRITERIA,
) -> AfrdVerdict:
    """Score one candidate by the criteria this data can express.

    Every criterion is always evaluated, so the per-criterion breakdown stays
    available for inspection; ``counted`` decides which of them the composite
    score is taken over. Reporting a criterion and excluding it from the
    score are different acts, and hiding the first would delete the evidence
    for the second.
    """
    members = frozenset(member_ids)
    return AfrdVerdict(
        listed=_criterion_listed(members, events, listed),
        shared_device=_criterion_shared_device(members, events, terminal_window),
        profile_deviation=_criterion_profile_deviation(
            members, events, min_history_days, deviation_factor
        ),
        counted=counted,
    )
