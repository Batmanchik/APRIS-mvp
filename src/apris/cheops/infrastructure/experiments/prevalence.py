"""What the detector is worth when fraud is actually rare.

The hole this closes
--------------------
Every number this project has published so far was measured in a world that
is roughly a quarter fraud. The Anti-Fraud Centre's own figure is **6.87 %
of registered incidents**, and the share of *accounts* involved in a scheme
in a national payment system is far below even that — fractions of a per
cent. A banker's first question about "the top ten per cent of the list is
all real mules" is therefore not a hostile one: it is the only question.

So this module holds prevalence up as its own axis, exactly as the ladder of
worlds holds difficulty. It is the one thing that stood between the ladder
and a claim about the real world.

Two arms, and they answer different halves of the question
----------------------------------------------------------
**Arm A — projection.** ROC is class-conditional: TPR and FPR are computed
inside a class, so neither moves when the *mix* of classes changes. Every
operator-facing number can then be re-derived at any prevalence pi by

    precision = pi*TPR / (pi*TPR + (1-pi)*FPR)
    alert rate = pi*TPR + (1-pi)*FPR

This is arithmetic, not an experiment, and it is exact **under one
assumption**: that a rare fraud looks the same as a common one, i.e. the
class-conditional score distributions do not change with pi. It says what
rarity costs at the *operating point*, given a fixed detector.

**Arm B — measurement.** The assumption Arm A cannot check is the *other*
half: a detector trained where positives are rare has fewer of them to learn
from, and may rank worse. That is measured, not assumed — positives are
thinned to the target prevalence in the row set **before** the split, so the
forest trains and is tested at that prevalence.

Reading the two together is the point. Where Arm B matches Arm A, the loss
is pure arithmetic and no modelling work will get it back; where Arm B falls
below Arm A, the loss is learning under rarity, which is a fixable problem
with known names (resampling, cost-sensitive training, anomaly-first
pipelines).

Why positives are thinned rather than negatives multiplied
----------------------------------------------------------
The honest alternative — generating a world with two hundred times more
honest accounts — costs two hundred times the runtime and produces exactly
the same estimate, because what limits the estimate is the number of
positives, and that number is what we are shrinking anyway.

Thinning is not free of assumptions and they are stated with the result:

* the kept rings are a uniform random sample of the rings, so it assumes no
  systematic difference between the ring that stays and the ring that goes.
  With rings drawn from one generator that holds by construction; with real
  data it would not, and this is the first thing to re-check on real data.
* **the world is not re-generated.** The dropped mules still exist in the
  events, so the graph a kept mule sits in is unchanged. This measures
  detection of a rare fraud, not a world in which fraud is rare — the
  neighbourhood structure stays as dense as it was. It therefore gives the
  detector the benefit of the doubt, and that is the safe direction: a
  number that would only get worse if the assumption were dropped.
* below roughly a hundred positives the metrics are noisy, so every cell
  carries its positive count and cells are run over several seeds. At 0.1 %
  a world of this size holds about a dozen fraud rows: that cell is reported
  because leaving it out would be choosing the range in which we look good,
  and it is reported **with** its count so nobody reads it as precise.

The world used here
-------------------
The W4 rung of the ladder — the hardest one where both units still score —
with every honest population and the merchant infrastructure multiplied by
``HONEST_SCALE``. Fraud structures are held fixed. That does two things at
once: it gives enough honest rows for the thin cells to mean anything, and
at ``HONEST_SCALE = 6`` the world's *natural* account-level fraud share
lands at 6.8 %, within a rounding of the AFC's published 6.87 %. The
richest cell of this sweep is therefore not a synthetic construct at all.

The first run — 2026-09-06, forest-v1, three seeds
--------------------------------------------------
Measured (positives thinned before the split, so the forest learns at that
rarity too), means over three seeds::

    fraud     pos    ROC (range)        R@10%   P@10%   reviews per catch
    6.75 %    908   0.9771              0.885   0.597   1.7
    1.00 %    126   0.9553 (.927-.984)  0.808   0.075   13.4
    0.50 %     63   0.9456 (.881-.999)  0.789   0.034   29.2
    0.10 %     13   0.9183 (.761-.999)  0.852   0.008   125.0

**ROC-AUC barely notices. Precision falls sixty-fold.** From 6.75 % down to
0.1 % the ranking metric moves 0.977 to 0.918 — a number anyone would still
call excellent — while the analyst's experience goes from under two files
per fraud found to over a hundred. Any report of this project that quotes
ROC-AUC without a prevalence beside it is quoting the metric that does not
move.

So the headline of R1 has to be rewritten. "The top ten per cent of the list
is all real mules" is true at 25 % fraud and at 6.75 %; at 1 % the same list
is 7 % real, and at 0.1 % it is 1 %.

Against the projection, the arithmetic accounts for most but not all of it::

    fraud     measured P@10%   projected P@10%
    1.00 %    0.075            0.096
    0.50 %    0.034            0.049
    0.10 %    0.008            0.010

Measured sits a fifth to a third below the projection at every target. That
gap is the cost of learning where positives are scarce — the part that is
not arithmetic and therefore the part that can be worked on. It is also the
part measured with 13 to 126 positives, so it is a direction rather than a
coefficient.

The finding that changes what we would build
---------------------------------------------
The ten-per-cent review budget is the wrong policy when fraud is rare, and
the projection says so in the operator's own units. At 0.1 % fraud, holding
a fixed budget means 100 alerts per 1000 accounts at a precision of 0.010.
Holding a *threshold* instead:

* catch **half** the mules: 0.6 alerts per 1000 accounts, precision 0.83 —
  five reviews to find four real cases;
* catch **four in five**: 17.4 alerts per 1000, precision 0.075.

The ranking is steep at the top, so half the fraud sits above a threshold
almost nothing honest reaches. Rarity does not destroy the detector; it
destroys the budget-shaped way of using it. The first half of the queue is
nearly free, and the second half is where the money goes — which is a
statement a bank can act on, and it is the opposite of "we lose at low
prevalence".

What this does NOT show
-----------------------
* It is one world's generator, at one difficulty rung, with one detector.
* The 0.1 % cells rest on thirteen fraud rows each; the spread across seeds
  there (0.761 to 0.999 ROC-AUC) is the honest width of that estimate, and
  it is wider than any difference we would claim between two detectors.
* Rarity here is thinned, not generated: the world around a kept mule is as
  dense as it was. A world where fraud is genuinely rare would also be a
  world with fewer mule neighbours, which can only make it harder.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from apris.cheops.infrastructure.experiments.ladder import Row, build_account_rows
from apris.cheops.infrastructure.experiments.ladder_of_worlds import (
    DETECTOR_VERSION,
    LADDER,
    REVIEW_BUDGET,
    pooled_out_of_fold,
)
from apris.cheops.infrastructure.simulation.config import EvasionKnobs, SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world

REPORT_PATH = Path("artifacts") / "prevalence_sweep.json"

#: Multiplier on every honest population and on the merchant infrastructure.
#: Six, because that is where the world's own fraud share meets the published
#: 6.87 % without anything being thinned at all.
HONEST_SCALE = 6

#: Which populations scale. Fraud structures (``mule_networks``, ``pyramids``)
#: and the terminal count deliberately do not: the sweep varies how rare the
#: fraud is among honest people, not how much fraud there is.
_SCALED_POPULATIONS: tuple[str, ...] = (
    "salary_earners",
    "employers",
    "freelancers",
    "traders",
    "fast_spenders",
    "crowd_collections",
    "marketplace_sellers",
    "family_circles",
    "merchants",
)

#: 6.87 % is the AFC's published share of mule incidents; the rest is the
#: descent towards a national payment system, where a scheme account is a
#: fraction of a per cent of all accounts.
PREVALENCE_TARGETS: tuple[float, ...] = (0.01, 0.005, 0.001)

#: Recall levels the alert load is quoted at. An operator does not buy a
#: ROC-AUC, they buy "how many files to open to catch four fraudsters in five".
RECALL_TARGETS: tuple[float, ...] = (0.5, 0.8)


def scaled_world_config(seed: int, scale: int = HONEST_SCALE) -> SimulationConfig:
    """The W4 world with its honest side multiplied, fraud held fixed."""
    rung = next(r for r in LADDER if r.key == "W4")
    config = dict(rung.config)
    for name in _SCALED_POPULATIONS:
        if name in config:
            config[name] = config[name] * scale
    # ``evasion`` is passed explicitly, at its naive default: this sweep
    # varies rarity and nothing else, and the evasion curve is where those
    # dials move.
    return SimulationConfig(seed=seed, evasion=EvasionKnobs(), **config)


def thin_positives(
    rows: Sequence[Row], target: float, rng: np.random.Generator
) -> list[Row]:
    """Drop fraud rows at random until the set is ``target`` fraud.

    Negatives are never touched, so what changes between cells is the rarity
    of the positive class and nothing else. Row order is preserved, because
    the split downstream is by time and a reshuffle would silently change it.

    Raises if the target is above what the rows already hold: reaching a
    higher prevalence would mean deleting honest people, which is a different
    experiment wearing the same name.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"prevalence must lie in (0, 1), got {target}")

    positives = [i for i, row in enumerate(rows) if row.label == 1]
    negatives = len(rows) - len(positives)
    if negatives == 0:
        raise ValueError("no honest rows to dilute against")

    natural = len(positives) / len(rows)
    if target > natural:
        raise ValueError(
            f"target {target:.4f} is above the natural share {natural:.4f}; "
            "thinning cannot add fraud"
        )

    # k / (k + negatives) == target  ->  k = target * negatives / (1 - target)
    keep_count = int(round(target * negatives / (1.0 - target)))
    keep_count = max(1, min(keep_count, len(positives)))
    kept = set(rng.choice(np.asarray(positives), size=keep_count, replace=False).tolist())
    return [row for i, row in enumerate(rows) if row.label == 0 or i in kept]


@dataclass(frozen=True)
class OperatingPoint:
    """One threshold, described the way an operator experiences it."""

    prevalence: float
    recall: float
    precision: float
    alert_rate: float
    alerts_per_1000_accounts: float
    reviews_per_catch: float


def project_at_recall(
    fpr: np.ndarray, tpr: np.ndarray, prevalence: float, recall: float
) -> OperatingPoint:
    """What catching ``recall`` of the fraud costs at ``prevalence``.

    Reads the measured ROC curve at the first threshold that reaches the
    recall, then rescales to the target prevalence. Pure arithmetic on
    class-conditional rates: nothing here is fitted.
    """
    reached = np.flatnonzero(tpr >= recall)
    index = int(reached[0]) if len(reached) else int(len(tpr) - 1)
    return _point(float(fpr[index]), float(tpr[index]), prevalence)


def project_at_budget(
    fpr: np.ndarray, tpr: np.ndarray, prevalence: float, budget: float = REVIEW_BUDGET
) -> OperatingPoint:
    """What a fixed review budget buys at ``prevalence``.

    The budget is a share of the whole population, so the threshold moves
    with prevalence: when fraud is rare almost the entire budget is spent on
    honest accounts, and the recall it buys is read off the ROC at
    ``alert rate == budget``.
    """
    alert_rate = prevalence * tpr + (1.0 - prevalence) * fpr
    affordable = np.flatnonzero(alert_rate <= budget)
    index = int(affordable[-1]) if len(affordable) else 0
    return _point(float(fpr[index]), float(tpr[index]), prevalence)


def _point(fpr: float, tpr: float, prevalence: float) -> OperatingPoint:
    alerted = prevalence * tpr + (1.0 - prevalence) * fpr
    caught = prevalence * tpr
    precision = caught / alerted if alerted > 0 else 0.0
    return OperatingPoint(
        prevalence=prevalence,
        recall=tpr,
        precision=precision,
        alert_rate=alerted,
        alerts_per_1000_accounts=alerted * 1000.0,
        reviews_per_catch=(1.0 / precision) if precision > 0 else float("inf"),
    )


@dataclass(frozen=True)
class PrevalenceCell:
    """One measured world at one prevalence: Arm B."""

    target_prevalence: float
    achieved_prevalence: float
    seed: int
    rows: int
    positives: int
    roc_auc: float | None
    average_precision: float | None
    #: AP has a floor equal to the prevalence, so the raw number falls even
    #: for a perfect detector. The lift says whether the detector got worse.
    average_precision_lift: float | None
    recall_at_budget: float | None
    precision_at_budget: float | None
    projected_recall_at_budget: float | None
    projected_precision_at_budget: float | None
    operating_points: tuple[OperatingPoint, ...]


def _measured_budget_point(
    scores: np.ndarray, truth: np.ndarray, budget: float = REVIEW_BUDGET
) -> tuple[float, float]:
    """Recall and precision inside the top ``budget`` of the ranking."""
    if truth.sum() == 0:
        return 0.0, 0.0
    take = max(1, int(len(scores) * budget))
    top = np.argsort(-scores)[:take]
    caught = float(truth[top].sum())
    return caught / float(truth.sum()), caught / float(take)


def run_cell(rows: Sequence[Row], target: float, seed: int) -> PrevalenceCell:
    """Thin to ``target``, then train and test at that prevalence."""
    rng = np.random.default_rng(seed)
    thinned = thin_positives(rows, target, rng)
    labels = np.array([r.label for r in thinned], dtype=int)
    achieved = float(labels.mean())

    scores, truth = pooled_out_of_fold(thinned)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        return PrevalenceCell(
            target_prevalence=target,
            achieved_prevalence=achieved,
            seed=seed,
            rows=len(thinned),
            positives=int(labels.sum()),
            roc_auc=None,
            average_precision=None,
            average_precision_lift=None,
            recall_at_budget=None,
            precision_at_budget=None,
            projected_recall_at_budget=None,
            projected_precision_at_budget=None,
            operating_points=(),
        )

    observed = float(truth.mean())
    average_precision = float(average_precision_score(truth, scores))
    recall, precision = _measured_budget_point(scores, truth)
    fpr, tpr, _ = roc_curve(truth, scores)
    projected = project_at_budget(fpr, tpr, observed)

    return PrevalenceCell(
        target_prevalence=target,
        achieved_prevalence=achieved,
        seed=seed,
        rows=len(thinned),
        positives=int(labels.sum()),
        roc_auc=float(roc_auc_score(truth, scores)),
        average_precision=average_precision,
        average_precision_lift=average_precision / observed if observed > 0 else None,
        recall_at_budget=recall,
        precision_at_budget=precision,
        projected_recall_at_budget=projected.recall,
        projected_precision_at_budget=projected.precision,
        operating_points=tuple(
            project_at_recall(fpr, tpr, observed, r) for r in RECALL_TARGETS
        ),
    )


@dataclass(frozen=True)
class ProjectedCell:
    """Arm A: the rich-world detector, re-priced at a prevalence it never saw."""

    prevalence: float
    seed: int
    at_budget: OperatingPoint
    operating_points: tuple[OperatingPoint, ...]


@dataclass(frozen=True)
class PrevalenceReport:
    run_id: str
    detector: str
    generated_at: str
    honest_scale: int
    seeds: tuple[int, ...]
    natural: tuple[PrevalenceCell, ...]
    measured: tuple[PrevalenceCell, ...]
    projected: tuple[ProjectedCell, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "detector": self.detector,
                "generated_at": self.generated_at,
                "honest_scale": self.honest_scale,
                "seeds": list(self.seeds),
                "natural": [asdict(c) for c in self.natural],
                "measured": [asdict(c) for c in self.measured],
                "projected": [asdict(c) for c in self.projected],
            },
            indent=2,
            sort_keys=True,
        )

    def table(self) -> str:
        header = (
            f"{'fraud':>8}{'rows':>8}{'pos':>6}{'ROC':>8}{'AP':>8}{'AP/base':>9}"
            f"{'R@10%':>8}{'P@10%':>8}{'proj P':>8}{'per catch':>11}"
        )
        lines = [header, "-" * len(header)]
        for cell in self.natural + self.measured:
            lines.append(_row(cell))
        lines.append("")
        lines.append("projection from the 6.87 % detector, same ROC, rarer world:")
        lines.append(
            f"{'fraud':>8}{'R@10%':>8}{'P@10%':>8}"
            + "".join(f"{'R=' + f'{r:.0%}':>22}" for r in RECALL_TARGETS)
        )
        for projected in _mean_projected(self.projected):
            cells = "".join(
                f"{point.alerts_per_1000_accounts:>13.1f}/1k{point.precision:>7.4f}"
                for point in projected.operating_points
            )
            lines.append(
                f"{projected.prevalence:>8.4f}"
                f"{projected.at_budget.recall:>8.4f}"
                f"{projected.at_budget.precision:>8.4f}{cells}"
            )
        return "\n".join(lines)


def _row(cell: PrevalenceCell) -> str:
    def cell_text(value: float | None, width: int, digits: int = 4) -> str:
        return f"{'—':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    per_catch = (
        "—"
        if not cell.precision_at_budget
        else f"{1.0 / cell.precision_at_budget:.1f}"
    )
    return (
        f"{cell.achieved_prevalence:>8.4f}{cell.rows:>8}{cell.positives:>6}"
        f"{cell_text(cell.roc_auc, 8)}{cell_text(cell.average_precision, 8)}"
        f"{cell_text(cell.average_precision_lift, 9, 1)}"
        f"{cell_text(cell.recall_at_budget, 8)}{cell_text(cell.precision_at_budget, 8)}"
        f"{cell_text(cell.projected_precision_at_budget, 8)}{per_catch:>11}"
    )


def _mean_projected(cells: Sequence[ProjectedCell]) -> list[ProjectedCell]:
    """One line per prevalence, averaged over seeds."""
    out: list[ProjectedCell] = []
    for prevalence in sorted({c.prevalence for c in cells}, reverse=True):
        group = [c for c in cells if c.prevalence == prevalence]
        out.append(
            ProjectedCell(
                prevalence=prevalence,
                seed=-1,
                at_budget=_mean_point([c.at_budget for c in group]),
                operating_points=tuple(
                    _mean_point([c.operating_points[i] for c in group])
                    for i in range(len(group[0].operating_points))
                ),
            )
        )
    return out


def _mean_point(points: Sequence[OperatingPoint]) -> OperatingPoint:
    return OperatingPoint(
        prevalence=points[0].prevalence,
        recall=float(np.mean([p.recall for p in points])),
        precision=float(np.mean([p.precision for p in points])),
        alert_rate=float(np.mean([p.alert_rate for p in points])),
        alerts_per_1000_accounts=float(np.mean([p.alerts_per_1000_accounts for p in points])),
        reviews_per_catch=float(np.mean([p.reviews_per_catch for p in points])),
    )


def run_prevalence_sweep(
    seeds: Sequence[int],
    targets: Sequence[float] = PREVALENCE_TARGETS,
    scale: int = HONEST_SCALE,
) -> PrevalenceReport:
    natural: list[PrevalenceCell] = []
    measured: list[PrevalenceCell] = []
    projected: list[ProjectedCell] = []

    for seed in seeds:
        world = generate_world(scaled_world_config(seed, scale))
        rows, _ = build_account_rows(world)

        scores, truth = pooled_out_of_fold(rows)
        observed = float(truth.mean())
        fpr, tpr, _ = roc_curve(truth, scores)
        recall, precision = _measured_budget_point(scores, truth)
        average_precision = float(average_precision_score(truth, scores))
        budget_point = project_at_budget(fpr, tpr, observed)
        natural.append(
            PrevalenceCell(
                target_prevalence=observed,
                achieved_prevalence=observed,
                seed=seed,
                rows=len(rows),
                positives=int(sum(r.label for r in rows)),
                roc_auc=float(roc_auc_score(truth, scores)),
                average_precision=average_precision,
                average_precision_lift=average_precision / observed,
                recall_at_budget=recall,
                precision_at_budget=precision,
                projected_recall_at_budget=budget_point.recall,
                projected_precision_at_budget=budget_point.precision,
                operating_points=tuple(
                    project_at_recall(fpr, tpr, observed, r) for r in RECALL_TARGETS
                ),
            )
        )

        # Arm A: the SAME ROC, re-priced. No refit, by design — this is the
        # arithmetic half, and mixing a refit into it would hide which half
        # of the loss is which.
        for target in targets:
            projected.append(
                ProjectedCell(
                    prevalence=target,
                    seed=seed,
                    at_budget=project_at_budget(fpr, tpr, target),
                    operating_points=tuple(
                        project_at_recall(fpr, tpr, target, r) for r in RECALL_TARGETS
                    ),
                )
            )

        # Arm B: thin, then train and test where fraud is that rare.
        for target in targets:
            measured.append(run_cell(rows, target, seed))

    return PrevalenceReport(
        run_id=uuid.uuid4().hex[:8],
        detector=DETECTOR_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        honest_scale=scale,
        seeds=tuple(seeds),
        natural=tuple(natural),
        measured=tuple(measured),
        projected=tuple(projected),
    )


def write_report(report: PrevalenceReport, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    archive = path.with_name(f"{path.stem}_{report.detector}_{report.run_id}.json")
    archive.write_text(report.to_json(), encoding="utf-8")
    return path


__all__ = [
    "HONEST_SCALE",
    "PREVALENCE_TARGETS",
    "OperatingPoint",
    "PrevalenceCell",
    "PrevalenceReport",
    "ProjectedCell",
    "project_at_budget",
    "project_at_recall",
    "run_cell",
    "run_prevalence_sweep",
    "scaled_world_config",
    "thin_positives",
    "write_report",
]
