"""The ladder of worlds — declared before it is run, and reported in full.

The decision this encodes
-------------------------
Building the hardest possible world first and then spending days finding out
why nothing worked is how this project spent its first week. It is also
unsound: when everything fails in a difficult world you cannot tell whether
the method is bad or the world is too hard. A simple world where the signal
is known to exist is a check on the instrument, not a way of winning.

So the worlds are a ladder, from a clean schema to a noisy one under an
adversary, and the ladder is fixed HERE, before any run. Every rung is
reported, including the ones we fall off. That is the whole difference
between choosing the difficulty and naming it, which is ordinary science,
and building five worlds and showing the one that came out well.

Two axes, and they are not the same axis
----------------------------------------
The rung is the x-axis and it measures **how hard the world is**. A detector
scores lower as it climbs, and that is the correct shape: the interesting
number is where it stops holding.

Progress is the curve LIFTING over releases. Every report is stamped with
``detector`` and ``run_id``, so two runs can be drawn on one pair of axes and
the release is the distance between the lines. A single run is a diagnosis;
the difference between two is the result.

What is held fixed across rungs
-------------------------------
The same detector, the same features, the same purged walk-forward split, the
same metrics, the same seeds. Only the world changes. A rung that also
changed the model would measure the model.

Both units of analysis are carried up the ladder together, because the
question of which unit survives difficulty is more useful than which unit
wins in one world — and much harder to get by accident.

The first honest run — 2026-09-05, forest-v1, three seeds
----------------------------------------------------------
::

    rung  unit      base   ceil     ROC      AP   R@10%
    W1    account  0.202  0.767  0.9783  0.9360  0.5025
          network  0.577  0.994  0.9992  0.9993  0.1855
    W2    account  0.236  0.772  0.9691  0.9209  0.4204
          network  0.597  1.000  0.9943  0.9955  0.1640
    W3    account  0.302  0.768  0.9819  0.9612  0.3336
          network  0.537  1.000  0.9983  0.9987  0.1744
    W4    account  0.242  0.777  0.9648  0.9259  0.4143
          network  0.436  1.000  0.9653  0.9664  0.2261
    W5    account  0.264  0.770  0.9646  0.9280  0.3777
          network  0.000  0.000       —       —       —

**The network unit is better than the account unit on every rung it can see
at all, and then it does not degrade — it disappears.** W1 to W4 it runs at
0.965 to 0.999 against the account unit's 0.965 to 0.982. At W5 it has no
score, and the reason is not the model: discovery proposes 157 candidates and
not one of them holds half of any ring.

That was checked rather than assumed, because a coverage of exactly zero is
the shape a threshold artefact has. ``label_candidates`` needs 50 % of a
network inside one candidate, so a ring recovered in fragments would score as
a miss. Measured on one seed, best single-candidate overlap per network:

===============  =========  =========
                 W4         W5
===============  =========  =========
median           1.000      **0.000**
mean             0.994      0.058
best of all 60   1.000      0.429
reaching 0.10    60 / 60    21 / 60
===============  =========  =========

For more than half the rings at W5, discovery does not place even two members
in one candidate. It is blindness, not fragmentation.

Meanwhile the account unit does not notice: 0.9648 to 0.9646.

So the finding is about **what evasion buys**, and it prices both sides. Six
independent funders and four terminals are real money and real accounts to
the organiser. They take network-level discovery from seeing every ring whole
to seeing nothing, and leave account-level detection untouched. Neither unit
dominates: one is stronger, the other is the one that survives an adversary
who pays. That is a more useful sentence than either half of H1.

Every rung reports its own ceilings
-----------------------------------
The account unit cannot judge an account with no history; discovery does not
propose every network. Those are limits of the UNIT, not of any model above
it, and they move as the world changes. Reading a score without them is how
a number gets believed for the wrong reason.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from apris.cheops.infrastructure.experiments.ladder import (
    ACCOUNT_FEATURE_COLUMNS,
    Row,
    build_account_rows,
    build_network_rows,
)
from apris.cheops.infrastructure.ml.validation_v2 import purged_walk_forward_splits
from apris.cheops.infrastructure.simulation.config import EvasionKnobs, SimulationConfig
from apris.cheops.infrastructure.simulation.discovery import (
    discover_candidates,
    label_candidates,
)
from apris.cheops.infrastructure.simulation.generator import generate_world

REPORT_PATH = Path("artifacts") / "ladder_of_worlds.json"

#: Bumped by hand whenever the detector changes in a way that should move the
#: curve. It is what makes two runs comparable, so it is not derived from the
#: git hash — a documentation edit must not look like a new detector.
DETECTOR_VERSION = "forest-v1"

# A review budget, not a threshold. An analyst has a day, not a cutoff.
REVIEW_BUDGET = 0.10


@dataclass(frozen=True)
class Rung:
    """One world, and the single reason it exists.

    ``why`` is mandatory and is printed with the number. A rung whose purpose
    cannot be said in a line is a parameter sweep wearing a name.
    """

    key: str
    title: str
    why: str
    #: Population counts. Typed as int rather than object so a rung that
    #: sets a knob the config does not have fails at the type check
    #: instead of at generation time, three minutes into a run.
    config: dict[str, int] = field(default_factory=dict)
    evasion: dict[str, int] = field(default_factory=dict)


#: What every rung shares. The honest population is NOT in here: it is what
#: the ladder varies, and it is varied in KIND rather than in size.
_BASE: dict[str, int] = dict(
    days=90,
    mule_networks=60,
    pyramids=0,
    terminals=40,
    merchants=120,
)

#: Roughly how many honest accounts each rung carries. Held near-constant on
#: purpose, and the first run is why.
#:
#: The first version of this ladder simply zeroed every honest population at
#: W1 and switched them on going up. The result was a world that was 96.3 %
#: fraud: ROC-AUC 1.0000 because there was nothing to confuse, recall at a
#: ten-per-cent budget capped at 0.10 because the budget cannot hold the
#: class, and no network metric at all on four rungs out of five, since every
#: candidate discovery proposed was a real ring and one class has no ROC
#: curve.
#:
#: That ladder was measuring class imbalance and calling it difficulty. A rung
#: must change what the negatives ARE, not how many there are.
_HONEST_TARGET = 2000


LADDER: tuple[Rung, ...] = (
    Rung(
        key="W1",
        title="negatives that look nothing alike",
        why=(
            "A check on the instrument. Salaries arrive monthly and are spent "
            "over weeks; a ring arrives and leaves in minutes. If a detector "
            "cannot separate those it is broken rather than challenged."
        ),
        config=dict(_BASE, salary_earners=1400, employers=40, freelancers=600),
    ),
    Rung(
        key="W2",
        title="+ negatives that empty the account too",
        why=(
            "The first real hard negative, and it shares the behaviour the "
            "exit is recognised by. Separating a mule from a fast spender is "
            "the task; separating one from a salary is not."
        ),
        config=dict(_BASE, salary_earners=700, employers=40, freelancers=300,
                    fast_spenders=1000),
    ),
    Rung(
        key="W3",
        title="+ negatives with the same SHAPE",
        why=(
            "A whip-round is forty people paying one collector and a payday is "
            "one employer paying forty. Both are the shape discovery links on, "
            "so this is the rung where a structural feature has to earn itself."
        ),
        config=dict(_BASE, salary_earners=400, employers=60, freelancers=200,
                    fast_spenders=600, crowd_collections=180,
                    marketplace_sellers=400, family_circles=120),
    ),
    Rung(
        key="W4",
        title="+ a second fraud on a different clock",
        why=(
            "Pyramids run over months beside rings running over minutes. One "
            "detector now has to hold two time scales at once, which is what "
            "the whole flow-based idea claims it can do."
        ),
        config=dict(_BASE, salary_earners=400, employers=60, freelancers=200,
                    traders=90, fast_spenders=600, crowd_collections=180,
                    marketplace_sellers=400, family_circles=120, pyramids=20),
    ),
    Rung(
        key="W5",
        title="+ an organiser who is hiding",
        why=(
            "The same world with the evasion knobs up: money arrives from "
            "several independent funders and leaves through several terminals. "
            "This rung says what evasion costs — and who pays, the model or "
            "the unit of analysis."
        ),
        config=dict(_BASE, salary_earners=400, employers=60, freelancers=200,
                    traders=90, fast_spenders=600, crowd_collections=180,
                    marketplace_sellers=400, family_circles=120, pyramids=20),
        evasion=dict(funders=6, terminals=4),
    ),
)


@dataclass(frozen=True)
class UnitResult:
    unit: str
    rows: int
    positives: int
    base_rate: float
    coverage: float
    roc_auc: float | None
    average_precision: float | None
    recall_at_budget: float | None


@dataclass(frozen=True)
class RungResult:
    key: str
    title: str
    why: str
    seed: int
    world: dict[str, float]
    units: tuple[UnitResult, ...]


def detector_model() -> RandomForestClassifier:
    """The detector every experiment in this project has to use.

    Public and shared on purpose. A prevalence sweep or an evasion curve run
    with its own forest would measure the forest as well as the world, and
    the whole point of the ladder is that only the world changes.
    """
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=1,
        n_jobs=-1,
    )


def recall_at_budget(
    scores: np.ndarray, truth: np.ndarray, budget: float = REVIEW_BUDGET
) -> float:
    """Share of the fraud caught inside the slice a human can actually review.

    Reported beside ROC-AUC because it is the only one of the three an
    operator experiences. A model can rank well and still put nothing
    findable in the first ten per cent.
    """
    if truth.sum() == 0:
        return 0.0
    take = max(1, int(len(scores) * budget))
    top = np.argsort(-scores)[:take]
    return float(truth[top].sum() / truth.sum())


def pooled_out_of_fold(rows: Sequence[Row]) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold scores for one set of rows: purged walk-forward, one forest.

    Returned rather than scored here so that an experiment which needs the
    whole ranking — a precision at a budget, an ROC curve to project from —
    reads the same numbers the ladder reads, from the same split.
    """
    labels = np.array([r.label for r in rows], dtype=int)
    if len(rows) < 40 or len(np.unique(labels)) < 2:
        return np.asarray([]), np.asarray([])

    columns = sorted(rows[0].features)
    matrix = pd.DataFrame([r.features for r in rows], columns=columns).astype(float).to_numpy()
    splits = purged_walk_forward_splits([r.ts for r in rows], n_splits=5,
                                        purge=timedelta(days=2))

    pooled_scores: list[float] = []
    pooled_truth: list[int] = []
    for split in splits:
        train_labels = labels[list(split.train)]
        if len(np.unique(train_labels)) < 2:
            continue
        model = detector_model()
        model.fit(matrix[list(split.train)], train_labels)
        probability = np.asarray(model.predict_proba(matrix[list(split.test)]))[:, 1]
        pooled_scores.extend(float(p) for p in probability)
        pooled_truth.extend(int(t) for t in labels[list(split.test)])

    return np.asarray(pooled_scores), np.asarray(pooled_truth)


def evaluate_unit(rows: Sequence[Row], unit: str, coverage: float) -> UnitResult:
    """Score one unit of analysis, ceiling carried alongside the score.

    Public for the same reason as ``detector_model``: an experiment that
    scores a world its own way is comparing two things at once.
    """
    labels = np.array([r.label for r in rows], dtype=int)
    base_rate = float(labels.mean()) if len(labels) else 0.0
    blank = UnitResult(unit, len(rows), int(labels.sum()), base_rate, coverage,
                       None, None, None)

    scores, truth = pooled_out_of_fold(rows)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        return blank

    return UnitResult(
        unit=unit,
        rows=len(rows),
        positives=int(labels.sum()),
        base_rate=base_rate,
        coverage=coverage,
        roc_auc=float(roc_auc_score(truth, scores)),
        average_precision=float(average_precision_score(truth, scores)),
        recall_at_budget=recall_at_budget(scores, truth),
    )


def run_rung(rung: Rung, seed: int) -> RungResult:
    """One world, both units, the same detector."""
    config = SimulationConfig(
        seed=seed,
        evasion=EvasionKnobs(**rung.evasion) if rung.evasion else EvasionKnobs(),
        **rung.config,
    )
    world = generate_world(config)
    candidates = discover_candidates(world)
    labels, discovery = label_candidates(world, candidates)

    account_rows, ceiling = build_account_rows(world)
    _, structural = build_network_rows(world, candidates, labels)

    return RungResult(
        key=rung.key,
        title=rung.title,
        why=rung.why,
        seed=seed,
        world=world.summary(),
        units=(
            evaluate_unit(account_rows, "account", ceiling.coverage),
            evaluate_unit(structural, "network", discovery.coverage),
        ),
    )


@dataclass(frozen=True)
class LadderOfWorldsReport:
    run_id: str
    detector: str
    generated_at: str
    seeds: tuple[int, ...]
    results: tuple[RungResult, ...]

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
            f"{'rung':<5}{'world':<34}{'unit':<9}{'rows':>7}{'base':>7}"
            f"{'ceil':>7}{'ROC':>8}{'AP':>8}{'R@10%':>8}"
        )
        lines = [header, "-" * len(header)]
        for key in [r.key for r in LADDER]:
            group = [r for r in self.results if r.key == key]
            if not group:
                continue
            for unit in ("account", "network"):
                cells = [u for r in group for u in r.units if u.unit == unit]
                if not cells:
                    continue
                first = group[0]
                lines.append(
                    f"{key if unit == 'account' else '':<5}"
                    f"{first.title[:33] if unit == 'account' else '':<34}"
                    f"{unit:<9}"
                    f"{int(np.mean([c.rows for c in cells])):>7}"
                    f"{np.mean([c.base_rate for c in cells]):>7.3f}"
                    f"{np.mean([c.coverage for c in cells]):>7.3f}"
                    f"{_mean_cell(cells, 'roc_auc'):>8}"
                    f"{_mean_cell(cells, 'average_precision'):>8}"
                    f"{_mean_cell(cells, 'recall_at_budget'):>8}"
                )
        return "\n".join(lines)


def _mean_cell(cells: Sequence[UnitResult], attribute: str) -> str:
    """Mean over seeds, or an em dash — never a zero standing in for absence."""
    values = [getattr(c, attribute) for c in cells if getattr(c, attribute) is not None]
    if not values:
        return "—"
    return f"{float(np.mean(values)):.4f}"


def run_ladder_of_worlds(seeds: Sequence[int]) -> LadderOfWorldsReport:
    results: list[RungResult] = []
    for rung in LADDER:
        for seed in seeds:
            results.append(run_rung(rung, seed))
    return LadderOfWorldsReport(
        run_id=uuid.uuid4().hex[:8],
        detector=DETECTOR_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        seeds=tuple(seeds),
        results=tuple(results),
    )


def write_report(report: LadderOfWorldsReport, path: Path = REPORT_PATH) -> Path:
    """Append-friendly: each run gets its own file beside the latest.

    Two runs on one pair of axes is the point of the whole module, so a run
    must not overwrite the one it is meant to be compared with.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    archive = path.with_name(f"{path.stem}_{report.detector}_{report.run_id}.json")
    archive.write_text(report.to_json(), encoding="utf-8")
    return path


__all__ = [
    "DETECTOR_VERSION",
    "LADDER",
    "REVIEW_BUDGET",
    "LadderOfWorldsReport",
    "Rung",
    "RungResult",
    "UnitResult",
    "detector_model",
    "evaluate_unit",
    "pooled_out_of_fold",
    "recall_at_budget",
    "run_ladder_of_worlds",
    "run_rung",
    "write_report",
]
