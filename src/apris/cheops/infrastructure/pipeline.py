"""One command from raw events to the queue on an analyst's desk.

Why this exists
---------------
Every piece of this already worked and none of it was joined up. A world was
built by the interface, candidates by a page, features by an experiment,
scores by a script, and the result of a run lived in whichever JSON file the
script happened to write. Nothing in the repository answered the one question
a bank actually asks — **what does the analyst see on Monday morning** — and
a demo that cannot answer it is a set of graphs, not a system.

So this module runs the whole chain in one call: world, discovery, features,
detector, and then the part that was missing, turning scores into a queue of
cases somebody is expected to work through.

The queue is where the rarity result lands
------------------------------------------
The prevalence sweep found that a fixed review budget — "look at the top ten
per cent" — is the wrong policy once fraud is rare, because the budget is
then spent almost entirely on honest accounts. A threshold is the right one:
at 0.1 % fraud, catching half the mules costs 0.6 alerts per 1000 accounts
at a precision of 0.83.

This module is that finding made operational. The queue is cut by a
threshold, the threshold is chosen for a stated recall target, and the
queue's length is an OUTPUT rather than a setting. A quiet week produces a
short queue, which is the correct behaviour and the one a budget cannot
express.

The threshold is chosen on the past, never on the block it is applied to
-------------------------------------------------------------------------
Choosing a cut-off on the same rows it is then measured on is the oldest way
to publish a number that cannot be reproduced on Monday. So the walk-forward
split is used the way production would: every fold but the last is the
calibration history, the threshold is read off it at the target recall, and
the last fold is the unseen block the queue is actually built from. The
numbers reported — precision, recall, reviews per catch — are that last block
alone.

Labels are carried on every item and are never used before the queue exists.
They are how the run reports what it caught; a production deployment has the
same queue and no labels, which is exactly the point of keeping them out of
everything upstream of the cut.

Both units, because they fail differently
-----------------------------------------
The network queue is stronger where it can see at all, and evasion takes it
to zero; the account queue is weaker and does not notice evasion. Producing
both, side by side, from one world is what makes the pair usable: the
evasion curve says the organiser can blind one of them, and an operator who
runs only that one has no fallback.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from apris.cheops.infrastructure.experiments.ladder import (
    Row,
    build_account_rows,
    build_network_rows,
)
from apris.cheops.infrastructure.experiments.ladder_of_worlds import (
    DETECTOR_VERSION,
    detector_model,
)
from apris.cheops.infrastructure.ml.validation_v2 import purged_walk_forward_splits
from apris.cheops.infrastructure.simulation.discovery import (
    discover_candidates,
    label_candidates,
)
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld, generate_world
from apris.cheops.infrastructure.simulation.presets import (
    DEFAULT_PRESET,
    DEFAULT_SEED,
    preset_config,
)

QUEUE_PATH = Path("artifacts") / "analyst_queue.json"

#: Share of the fraud the queue is cut to catch. Half rather than everything
#: because the prevalence sweep priced the difference: at 0.1 % fraud the
#: first half of the queue costs 0.6 alerts per 1000 accounts and the last
#: fifth costs thirty times that. Which half a bank buys is a business
#: decision, so it is an argument with a default, not a constant.
DEFAULT_TARGET_RECALL = 0.5

#: Folds in the walk-forward. The last one is the block the queue is built
#: from; the rest are the history the threshold is read off.
N_SPLITS = 5

#: Gap between training and test inside the split. Without it a case that
#: straddles the boundary teaches the model about its own test block.
PURGE = timedelta(days=2)


@dataclass(frozen=True)
class QueueItem:
    """One case an analyst is asked to open."""

    rank: int
    unit: str
    key: str
    score: float
    members: int
    events: int
    amount_total: float
    first_seen: str
    last_seen: str
    #: Ground truth. Present because this is a simulated world and the run
    #: has to report what it caught; nothing upstream of the cut reads it,
    #: and a real deployment simply has this field empty.
    truth: int


@dataclass(frozen=True)
class QueueOutcome:
    """What the cut produced on the block it had never seen."""

    unit: str
    #: Rows in the unseen block the queue was cut from.
    block_rows: int
    block_positives: int
    block_prevalence: float
    threshold: float
    target_recall: float
    queued: int
    caught: int
    precision: float
    recall: float
    reviews_per_catch: float | None
    #: Coverage of the unit itself, before any model: the share of real rings
    #: discovery proposed, or the share of fraud accounts with enough history
    #: to be judged. A queue cannot beat it, so it is reported beside it.
    unit_ceiling: float
    items: tuple[QueueItem, ...]


@dataclass(frozen=True)
class PipelineReport:
    run_id: str
    detector: str
    generated_at: str
    preset: str
    seed: int
    world: dict[str, float]
    seconds: float
    outcomes: tuple[QueueOutcome, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "detector": self.detector,
                "generated_at": self.generated_at,
                "preset": self.preset,
                "seed": self.seed,
                "world": self.world,
                "seconds": self.seconds,
                "outcomes": [asdict(o) for o in self.outcomes],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )

    def table(self) -> str:
        header = (
            f"{'очередь':<10}{'блок':>7}{'из них':>8}{'порог':>8}{'дел':>6}"
            f"{'нашли':>7}{'точность':>10}{'полнота':>9}{'на находку':>12}"
        )
        lines = [header, "-" * len(header)]
        for outcome in self.outcomes:
            per_catch = (
                "—" if outcome.reviews_per_catch is None
                else f"{outcome.reviews_per_catch:.1f}"
            )
            # An empty queue has no precision — nobody was asked to look at
            # anything. Printing 0.000 there reads as "every case was wrong",
            # which is the opposite of what happened.
            precision = "—" if outcome.queued == 0 else f"{outcome.precision:.3f}"
            lines.append(
                f"{outcome.unit:<10}{outcome.block_rows:>7}{outcome.block_positives:>8}"
                f"{outcome.threshold:>8.3f}{outcome.queued:>6}{outcome.caught:>7}"
                f"{precision:>10}{outcome.recall:>9.3f}{per_catch:>12}"
            )
        return "\n".join(lines)


def _matrix(rows: Sequence[Row]) -> np.ndarray:
    columns = sorted(rows[0].features)
    frame = pd.DataFrame([r.features for r in rows], columns=columns)
    return frame.astype(float).to_numpy()


def _walk_forward_scores(
    rows: Sequence[Row],
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], np.ndarray]:
    """Out-of-fold scores, kept fold by fold rather than pooled.

    The pooled version in ``ladder_of_worlds`` answers "how well does this
    rank"; here the fold boundary itself is load-bearing, because the last
    one is the block standing in for the week nobody has seen yet.
    """
    labels = np.array([r.label for r in rows], dtype=int)
    if len(rows) < 40 or len(np.unique(labels)) < 2:
        return [], labels

    matrix = _matrix(rows)
    splits = purged_walk_forward_splits([r.ts for r in rows], n_splits=N_SPLITS, purge=PURGE)

    scored: list[tuple[int, np.ndarray, np.ndarray]] = []
    for index, split in enumerate(splits):
        train_index = list(split.train)
        test_index = np.asarray(list(split.test), dtype=int)
        if len(np.unique(labels[train_index])) < 2 or len(test_index) == 0:
            continue
        model = detector_model()
        model.fit(matrix[train_index], labels[train_index])
        probability = np.asarray(model.predict_proba(matrix[test_index]))[:, 1]
        scored.append((index, test_index, probability))
    return scored, labels


def threshold_for_recall(
    scores: np.ndarray, truth: np.ndarray, target_recall: float
) -> float:
    """The lowest cut that still catches ``target_recall`` of the history.

    Read off the calibration folds only. Returns the highest possible cut
    when the history holds no fraud at all — an empty queue is the honest
    answer there, and inventing a low threshold to fill it would hand the
    analyst a list of everybody.
    """
    positives = scores[truth == 1]
    if len(positives) == 0:
        return float("inf")
    quantile = float(np.quantile(positives, 1.0 - target_recall))
    return quantile


def build_queue(
    rows: Sequence[Row],
    unit: str,
    ceiling: float,
    target_recall: float = DEFAULT_TARGET_RECALL,
) -> QueueOutcome:
    """Calibrate on the history, cut the unseen block, report what it caught."""
    scored, labels = _walk_forward_scores(rows)
    blank = QueueOutcome(
        unit=unit,
        block_rows=0,
        block_positives=0,
        block_prevalence=0.0,
        threshold=float("inf"),
        target_recall=target_recall,
        queued=0,
        caught=0,
        precision=0.0,
        recall=0.0,
        reviews_per_catch=None,
        unit_ceiling=ceiling,
        items=(),
    )
    if len(scored) < 2:
        return blank

    *history, latest = scored
    history_scores = np.concatenate([s for _, _, s in history])
    history_truth = np.concatenate([labels[i] for _, i, _ in history])
    threshold = threshold_for_recall(history_scores, history_truth, target_recall)

    _, block_index, block_scores = latest
    block_truth = labels[block_index]

    keep = np.flatnonzero(block_scores >= threshold)
    order = keep[np.argsort(-block_scores[keep])]

    items: list[QueueItem] = []
    for rank, position in enumerate(order, start=1):
        row = rows[int(block_index[position])]
        stamps = [e.ts for e in row.events]
        items.append(
            QueueItem(
                rank=rank,
                unit=unit,
                key=row.key,
                score=float(block_scores[position]),
                members=len(row.members),
                events=len(row.events),
                amount_total=float(sum(e.amount for e in row.events)),
                first_seen=min(stamps).isoformat() if stamps else "",
                last_seen=max(stamps).isoformat() if stamps else "",
                truth=int(row.label),
            )
        )

    caught = int(sum(item.truth for item in items))
    block_positives = int(block_truth.sum())
    precision = caught / len(items) if items else 0.0
    return QueueOutcome(
        unit=unit,
        block_rows=len(block_index),
        block_positives=block_positives,
        block_prevalence=float(block_truth.mean()) if len(block_truth) else 0.0,
        threshold=threshold,
        target_recall=target_recall,
        queued=len(items),
        caught=caught,
        precision=precision,
        recall=caught / block_positives if block_positives else 0.0,
        reviews_per_catch=(1.0 / precision) if precision > 0 else None,
        unit_ceiling=ceiling,
        items=tuple(items),
    )


def run_pipeline(
    preset: str = DEFAULT_PRESET,
    seed: int = DEFAULT_SEED,
    target_recall: float = DEFAULT_TARGET_RECALL,
    world: SimulatedWorld | None = None,
) -> PipelineReport:
    """World to queue, one call. ``world`` is injectable for tests."""
    started = datetime.now(timezone.utc)
    if world is None:
        world = generate_world(preset_config(preset, seed))

    candidates = discover_candidates(world)
    labels, discovery = label_candidates(world, candidates)
    account_rows, ceiling = build_account_rows(world)
    _, network_rows = build_network_rows(world, candidates, labels)

    outcomes = (
        build_queue(network_rows, "сети", discovery.coverage, target_recall),
        build_queue(account_rows, "счета", ceiling.coverage, target_recall),
    )
    finished = datetime.now(timezone.utc)

    return PipelineReport(
        run_id=uuid.uuid4().hex[:8],
        detector=DETECTOR_VERSION,
        generated_at=finished.isoformat(timespec="seconds"),
        preset=preset,
        seed=seed,
        world=world.summary(),
        seconds=(finished - started).total_seconds(),
        outcomes=outcomes,
    )


def write_queue(report: PipelineReport, path: Path = QUEUE_PATH) -> Path:
    """The queue the interface reads. Overwritten each run on purpose:
    yesterday's queue is not a result to preserve, it is yesterday's work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    return path


def read_queue(path: Path = QUEUE_PATH) -> dict[str, object] | None:
    """Whatever the last run left, or None. Missing is not an error: the
    interface has to say "конвейер ещё не запускался" rather than crash."""
    if not path.exists():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


__all__ = [
    "DEFAULT_TARGET_RECALL",
    "QUEUE_PATH",
    "PipelineReport",
    "QueueItem",
    "QueueOutcome",
    "build_queue",
    "read_queue",
    "run_pipeline",
    "threshold_for_recall",
    "write_queue",
]
