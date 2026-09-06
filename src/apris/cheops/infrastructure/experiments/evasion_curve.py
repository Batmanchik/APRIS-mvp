"""What evasion costs, one dial at a time.

The gap this closes
-------------------
The ladder of worlds priced evasion at exactly two points: the naive
organiser (one funder, one terminal) at W4, and a hiding one (six funders,
four terminals) at W5. Between them the network unit went from seeing every
ring whole to seeing nothing, and the account unit did not move. Two points
name an effect; they do not say **where it breaks**, and "where" is the only
part an organiser and a defender both care about. Two funders may already be
enough, or it may take five: those are different worlds for anyone deciding
what to monitor.

So this sweeps each dial alone, holding the other at its naive value, and
adds the joint point that W5 uses. One dial at a time because the dials cost
different things — a funder is an account holding real money, a terminal is
people driven across a city — and a curve in which both moved at once cannot
be read as either price.

What is measured, and why it is not just a score
------------------------------------------------
A ROC-AUC on the network unit says nothing when discovery proposes nothing:
the metric is simply absent, which is how the W5 finding nearly got read as
"the model degrades". It does not degrade — it goes blind. So every point
carries three things that separate blindness from weakness:

``coverage``
    share of real rings discovery proposed at all, before any model. This is
    the unit's own ceiling, and it is what evasion actually attacks.
``overlap``
    for each real ring, the largest share of it sitting inside any single
    candidate. Coverage is this quantity thresholded at 0.5, so the
    distribution says whether a lost ring was fragmented (overlap around
    0.3, discovery half-saw it) or invisible (overlap 0.0, discovery never
    put two of its members together). A threshold artefact and real
    blindness look identical in coverage and nothing alike here.
``account unit``
    the same world scored per account. Evasion that buys the organiser
    nothing against the account unit is a fact about which defence survives
    an adversary who pays, and it is the half of the result that is useful
    to a bank.

Both units come from the same generated world, so a point is one world
scored two ways, never two worlds compared.

Held fixed
----------
The W4 population mix, the seeds, the detector, the split, the metrics. Only
the two evasion knobs move. A sweep that also changed the populations would
be a second ladder wearing the name of a curve.

The first run — 2026-09-06, forest-v1, three seeds
---------------------------------------------------
::

    point  fund term   cover  ovl med   grip   acc ROC  acc R@10%
    naive     1    1   1.000    1.000   1.00    0.9660     0.4143
    f2        2    1   0.956    0.836   0.99    0.9684     0.3792
    f3        3    1   0.650    0.534   0.90    0.9609     0.3492
    f4        4    1   0.333    0.390   0.79    0.9768     0.3629
    f6        6    1   0.128    0.262   0.71    0.9740     0.3303
    t2        1    2   0.983    1.000   0.99    0.9813     0.3399
    t3        1    3   0.750    1.000   0.93    0.9730     0.3586
    t4        1    4   0.750    1.000   0.87    0.9736     0.3491
    f6t4      6    4   0.000    0.000   0.32    0.9702     0.3777

**The funders dial is where it breaks, and it breaks at three.** Two funders
cost the network unit almost nothing (0.956 coverage). Three take it to
0.650, four to 0.333, six to 0.128. Splitting the money across three
accounts is the single cheapest thing an organiser can do to us.

**Terminals are the expensive dial and they buy much less.** Four ATMs leave
three rings in four still recoverable, and the median overlap stays at 1.000
throughout: losing a ring to the terminal dial is all-or-nothing per ring,
not a gradual erosion. Driving people across a city is the more visible cost
and the weaker attack.

**The two together are worse than either alone**, and not by addition. At six
funders alone, discovery still has a grip on 71 % of rings; at four terminals
alone, 87 %. At both, coverage is 0.000, the median overlap is 0.000, and the
grip is 0.32. The funders fragment the ring, the terminals remove what is
left to group it by, and only the combination is blindness rather than
fragmentation. W5 is that corner, which is why the ladder's W5 row was empty.

**The account unit does not move anywhere on this curve**: 0.9609 to 0.9813
ROC-AUC with no trend, against a network unit going from whole to nothing.
Recall inside the review budget drifts 0.41 to 0.33 — the only account-level
price of evasion measured so far, and it is small.

A trap in this table, stated because it is easy to misread
-----------------------------------------------------------
The network ROC-AUC column *rises* as coverage collapses (0.8514 at f3,
0.9887 at f6). That is survivorship, not improvement: it is computed only
over the candidates discovery still proposed, and the rings that survive
heavy evasion are the ones that hid worst. Coverage and the overlap profile
are the honest columns on this curve; the network score alone would say the
opposite of what happened, which is exactly why coverage is reported first
and the score last.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from apris.cheops.infrastructure.experiments.ladder import (
    build_account_rows,
    build_network_rows,
)
from apris.cheops.infrastructure.experiments.ladder_of_worlds import (
    DETECTOR_VERSION,
    LADDER,
    UnitResult,
    evaluate_unit,
)
from apris.cheops.infrastructure.simulation.config import EvasionKnobs, SimulationConfig
from apris.cheops.infrastructure.simulation.discovery import (
    Candidate,
    discover_candidates,
    label_candidates,
)
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld, generate_world

REPORT_PATH = Path("artifacts") / "evasion_curve.json"

#: The share of one ring that must sit inside a single candidate for
#: ``label_candidates`` to call the ring covered. Repeated here because the
#: overlap distribution is only readable against the threshold it feeds.
COVERAGE_THRESHOLD = 0.5

#: A ring is "seen at all" when at least a tenth of it lands in one
#: candidate. Well below the coverage threshold on purpose: the question this
#: answers is whether discovery has any grip on the ring, not whether it wins.
GRIP_THRESHOLD = 0.10


@dataclass(frozen=True)
class EvasionPoint:
    """One setting of the dials, and the price the organiser pays for it."""

    key: str
    funders: int
    terminals: int
    #: What the setting costs whoever runs the scheme, in plain words. Written
    #: down beside the number because a detectability curve without a price
    #: axis invites the reading "evasion is free, we lose".
    cost: str


#: One dial at a time from the naive organiser, then the joint W5 point.
#: Funders go to six because that is where W5 sits; terminals to four for the
#: same reason. Nothing beyond W5, because past it the sweep would be pricing
#: a scheme nobody has described.
CURVE: tuple[EvasionPoint, ...] = (
    EvasionPoint("naive", 1, 1, "nothing: one source, one ATM"),
    EvasionPoint("f2", 2, 1, "a second account holding real money"),
    EvasionPoint("f3", 3, 1, "three funded accounts"),
    EvasionPoint("f4", 4, 1, "four funded accounts"),
    EvasionPoint("f6", 6, 1, "six funded accounts"),
    EvasionPoint("t2", 1, 2, "a second ATM, people driven to it"),
    EvasionPoint("t3", 1, 3, "three ATMs across the city"),
    EvasionPoint("t4", 1, 4, "four ATMs across the city"),
    EvasionPoint("f6t4", 6, 4, "both, and this is the W5 rung"),
)


@dataclass(frozen=True)
class OverlapProfile:
    """How much of each real ring discovery managed to hold together."""

    networks: int
    median: float
    mean: float
    best: float
    reaching_grip: int
    reaching_coverage: int

    @property
    def grip_share(self) -> float:
        return self.reaching_grip / self.networks if self.networks else 0.0


def overlap_profile(world: SimulatedWorld, candidates: Sequence[Candidate]) -> OverlapProfile:
    """Best single-candidate overlap per real ring, summarised.

    The diagnostic that separates "discovery split the ring" from "discovery
    never saw it". Reported at every point of the curve rather than only
    where coverage collapses, because a profile is only readable against the
    profiles beside it.
    """
    rings = [
        set(network.account_ids)
        for network in world.networks
        if network.kind == "mule_fast" and network.account_ids
    ]
    if not rings:
        return OverlapProfile(0, 0.0, 0.0, 0.0, 0, 0)

    member_sets = [set(candidate.member_ids) for candidate in candidates]
    best_per_ring = [
        max((len(ring & members) / len(ring) for members in member_sets), default=0.0)
        for ring in rings
    ]
    values = np.asarray(best_per_ring, dtype=float)
    return OverlapProfile(
        networks=len(rings),
        median=float(np.median(values)),
        mean=float(values.mean()),
        best=float(values.max()),
        reaching_grip=int((values >= GRIP_THRESHOLD).sum()),
        reaching_coverage=int((values >= COVERAGE_THRESHOLD).sum()),
    )


@dataclass(frozen=True)
class EvasionResult:
    key: str
    funders: int
    terminals: int
    cost: str
    seed: int
    candidates: int
    overlap: OverlapProfile
    units: tuple[UnitResult, ...]


def run_point(point: EvasionPoint, seed: int) -> EvasionResult:
    """One world at one setting of the dials, scored at both units."""
    rung = next(r for r in LADDER if r.key == "W4")
    config = SimulationConfig(
        seed=seed,
        evasion=EvasionKnobs(funders=point.funders, terminals=point.terminals),
        **rung.config,
    )
    world = generate_world(config)
    candidates = discover_candidates(world)
    labels, discovery = label_candidates(world, candidates)

    account_rows, ceiling = build_account_rows(world)
    _, structural = build_network_rows(world, candidates, labels)

    return EvasionResult(
        key=point.key,
        funders=point.funders,
        terminals=point.terminals,
        cost=point.cost,
        seed=seed,
        candidates=len(candidates),
        overlap=overlap_profile(world, candidates),
        units=(
            evaluate_unit(account_rows, "account", ceiling.coverage),
            evaluate_unit(structural, "network", discovery.coverage),
        ),
    )


@dataclass(frozen=True)
class EvasionCurveReport:
    run_id: str
    detector: str
    generated_at: str
    seeds: tuple[int, ...]
    results: tuple[EvasionResult, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "detector": self.detector,
                "generated_at": self.generated_at,
                "seeds": list(self.seeds),
                "results": [asdict(r) for r in self.results],
            },
            indent=2,
            sort_keys=True,
        )

    def table(self) -> str:
        header = (
            f"{'point':<6}{'fund':>5}{'term':>5}{'cover':>8}{'ovl med':>9}"
            f"{'ovl mean':>10}{'grip':>7}{'net ROC':>9}{'acc ROC':>9}{'acc R@10%':>10}"
        )
        lines = [header, "-" * len(header)]
        for point in CURVE:
            group = [r for r in self.results if r.key == point.key]
            if not group:
                continue
            network = [u for r in group for u in r.units if u.unit == "network"]
            account = [u for r in group for u in r.units if u.unit == "account"]
            lines.append(
                f"{point.key:<6}{point.funders:>5}{point.terminals:>5}"
                f"{np.mean([u.coverage for u in network]):>8.3f}"
                f"{np.mean([r.overlap.median for r in group]):>9.3f}"
                f"{np.mean([r.overlap.mean for r in group]):>10.3f}"
                f"{np.mean([r.overlap.grip_share for r in group]):>7.2f}"
                f"{_mean_cell(network, 'roc_auc'):>9}"
                f"{_mean_cell(account, 'roc_auc'):>9}"
                f"{_mean_cell(account, 'recall_at_budget'):>10}"
            )
        return "\n".join(lines)


def _mean_cell(cells: Sequence[UnitResult], attribute: str) -> str:
    """Mean over seeds, or an em dash — never a zero standing in for absence."""
    values = [getattr(c, attribute) for c in cells if getattr(c, attribute) is not None]
    if not values:
        return "—"
    return f"{float(np.mean(values)):.4f}"


def run_evasion_curve(
    seeds: Sequence[int], points: Sequence[EvasionPoint] = CURVE
) -> EvasionCurveReport:
    results: list[EvasionResult] = []
    for point in points:
        for seed in seeds:
            results.append(run_point(point, seed))
    return EvasionCurveReport(
        run_id=uuid.uuid4().hex[:8],
        detector=DETECTOR_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seeds=tuple(seeds),
        results=tuple(results),
    )


def write_report(report: EvasionCurveReport, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    archive = path.with_name(f"{path.stem}_{report.detector}_{report.run_id}.json")
    archive.write_text(report.to_json(), encoding="utf-8")
    return path


__all__ = [
    "COVERAGE_THRESHOLD",
    "CURVE",
    "GRIP_THRESHOLD",
    "EvasionCurveReport",
    "EvasionPoint",
    "EvasionResult",
    "OverlapProfile",
    "overlap_profile",
    "run_evasion_curve",
    "run_point",
    "write_report",
]
