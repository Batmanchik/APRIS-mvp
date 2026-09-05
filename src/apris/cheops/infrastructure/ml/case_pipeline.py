"""From a world to a measured number, with nothing skipped in between.

The audit records a baseline of ROC-AUC 0.8111 for candidate classification,
but no committed code reproduced it: the run happened in a session and the
number survived only in prose. A figure nobody can re-run is an assertion,
not a result.

This module is the missing path, and it is deliberately short enough to read
in one sitting:

    world -> discover_candidates -> features from events -> purged
    walk-forward -> ROC-AUC, PR-AUC, quintile ladder

Three properties are load-bearing.

**Discovery never reads the answers.** ``discover_candidates`` proposes
clusters from the event stream alone, and labels are attached afterwards, so
coverage — the share of real networks any candidate contains — is a real
ceiling on recall rather than 1.0 by construction (Finding 4).

**Features are computed from events**, by the same two functions the graph
and sequence branches use, so nothing here is a renamed copy of the tabular
vector (Finding 1).

**Validation is purged and walk-forward.** Candidates are ordered by when
their last event happened, each fold trains on the past and tests on the
block after it, and anything within the purge gap of the boundary is dropped
from training rather than silently included.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from apris.cheops.infrastructure.ml.event_features_v2 import (
    GRAPH_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
    graph_features_from_events,
    sequence_features_from_events,
)
from apris.cheops.infrastructure.ml.validation_v2 import (
    LadderResult,
    purged_walk_forward_splits,
    quintile_ladder,
)
from apris.cheops.infrastructure.simulation.discovery import (
    Candidate,
    DiscoveryReport,
    discover_candidates,
    label_candidates,
)
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld

CASE_FEATURE_COLUMNS: tuple[str, ...] = tuple(GRAPH_FEATURE_COLUMNS) + tuple(
    SEQUENCE_FEATURE_COLUMNS
)

DEFAULT_PURGE = timedelta(days=3)
DEFAULT_SPLITS = 5
DEFAULT_MIN_TRAIN = 20


# ==========================================================================
# Dataset
# ==========================================================================


def case_features(candidate: Candidate) -> dict[str, float]:
    """The ten event-derived features for one candidate."""
    features = graph_features_from_events(candidate.events)
    features.update(sequence_features_from_events(candidate.events))
    return features


@dataclass(frozen=True)
class CaseDataset:
    """Candidates, their features, and the labels attached after the fact."""

    candidates: tuple[Candidate, ...]
    features: pd.DataFrame
    labels: np.ndarray
    timestamps: tuple[datetime, ...]
    discovery: DiscoveryReport

    @property
    def size(self) -> int:
        return len(self.candidates)

    @property
    def positives(self) -> int:
        return int(self.labels.sum())

    @property
    def base_rate(self) -> float:
        return float(self.labels.mean()) if self.size else 0.0

    @property
    def coverage(self) -> float:
        return self.discovery.coverage

    def member_counts(self) -> list[int]:
        return [candidate.size for candidate in self.candidates]


def build_case_dataset(world: SimulatedWorld, **discovery_kwargs: Any) -> CaseDataset:
    """Propose candidates from events, then describe and label them.

    Order matters and is the whole point: proposing comes first and cannot
    see ``world.networks``; labelling comes second and is used only to score
    what discovery already produced.
    """
    candidates = discover_candidates(world, **discovery_kwargs)
    labels, report = label_candidates(world, candidates)

    rows = [case_features(candidate) for candidate in candidates]
    frame = pd.DataFrame(rows, columns=list(CASE_FEATURE_COLUMNS))
    frame.index = pd.Index([candidate.candidate_id for candidate in candidates])

    # A candidate's position in time is when it finished. A ring that ran in
    # March must not be trained on to predict one that ran in February.
    timestamps = tuple(
        max(event.ts for event in candidate.events) for candidate in candidates
    )

    return CaseDataset(
        candidates=tuple(candidates),
        features=frame,
        labels=np.asarray(labels, dtype=int),
        timestamps=timestamps,
        discovery=report,
    )


# ==========================================================================
# Validation
# ==========================================================================


@dataclass(frozen=True)
class FoldResult:
    index: int
    train_size: int
    test_size: int
    purged: int
    positives_in_test: int
    roc_auc: float | None
    pr_auc: float | None

    @property
    def scorable(self) -> bool:
        """A test block holding one class alone cannot produce an AUC."""
        return self.roc_auc is not None


@dataclass(frozen=True)
class ValidationReport:
    """What the detector scores, and on what it was measured.

    ``note`` carries why a run produced no score. A blank AUC with no reason
    beside it invites the reader to assume the run failed technically, when
    the usual cause is that the evidence was too thin to measure — a
    different fact, and the more important one.
    """

    folds: tuple[FoldResult, ...]
    roc_auc: float | None
    pr_auc: float | None
    ladder: LadderResult | None
    single_feature_auc: dict[str, float]
    candidates: int
    positives: int
    base_rate: float
    coverage: float
    missed_networks: tuple[str, ...]
    note: str = ""

    @property
    def scored_folds(self) -> int:
        return sum(1 for fold in self.folds if fold.scorable)

    def describe(self) -> str:
        if self.roc_auc is None:
            return f"not scorable: {self.note or 'no score was produced'}"
        ladder = self.ladder.describe() if self.ladder is not None else "no ladder"
        return (
            f"ROC-AUC {self.roc_auc:.4f}, PR-AUC {self.pr_auc:.4f} "
            f"over {self.scored_folds} folds; ladder {ladder}"
        )


def _new_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def _single_feature_auc(features: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    """In-sample AUC per feature, reported as a caveat rather than a result.

    It is the number that tempts: one column separating the whole dataset
    looks like a strong feature. Read against the out-of-fold figure it says
    something different — how much of that separation survives a time split.
    """
    if len(set(labels.tolist())) < 2:
        return {}
    scores: dict[str, float] = {}
    for column in features.columns:
        values = features[column].to_numpy(dtype=float)
        if float(np.std(values)) == 0.0:
            scores[str(column)] = 0.5
            continue
        scores[str(column)] = float(roc_auc_score(labels, values))
    return scores


def run_case_validation(
    dataset: CaseDataset,
    *,
    n_splits: int = DEFAULT_SPLITS,
    purge: timedelta = DEFAULT_PURGE,
    min_train: int = DEFAULT_MIN_TRAIN,
    seed: int = 42,
) -> ValidationReport:
    """Purged walk-forward over the candidate set.

    Scores from every test block are pooled into one out-of-fold vector: the
    blocks are disjoint, so a single AUC over the pool measures the whole
    period rather than averaging folds of unequal size. The ladder is read
    off the same pool, because a score can separate without ordering and only
    the ladder shows it.
    """
    def unscorable(note: str, folds: tuple[FoldResult, ...] = ()) -> ValidationReport:
        return ValidationReport(
            folds=folds,
            roc_auc=None,
            pr_auc=None,
            ladder=None,
            single_feature_auc={},
            candidates=dataset.size,
            positives=dataset.positives,
            base_rate=dataset.base_rate,
            coverage=dataset.coverage,
            missed_networks=dataset.discovery.missed_network_ids,
            note=note,
        )

    if dataset.size == 0:
        return unscorable("discovery proposed no candidates")

    matrix = dataset.features.to_numpy(dtype=float)
    labels = dataset.labels

    splits = purged_walk_forward_splits(
        list(dataset.timestamps),
        n_splits=n_splits,
        purge=purge,
        min_train=min_train,
    )
    if not splits:
        # Two different causes, and telling them apart is the difference
        # between "gather more data" and "the purge is set too wide".
        if dataset.size <= min_train:
            return unscorable(
                f"{dataset.size} candidates is not more than min_train={min_train}"
            )
        return unscorable(
            f"the purge of {purge} left no training rows before any test block"
        )

    folds: list[FoldResult] = []
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []

    for split in splits:
        train_idx = list(split.train)
        test_idx = list(split.test)
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # A fold whose training block holds one class teaches nothing. It is
        # recorded rather than dropped silently, so the count of scorable
        # folds stays visible next to the score.
        if len(set(y_train.tolist())) < 2:
            folds.append(
                FoldResult(
                    index=split.index,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    purged=split.purged_count,
                    positives_in_test=int(y_test.sum()),
                    roc_auc=None,
                    pr_auc=None,
                )
            )
            continue

        model = _new_model(seed)
        model.fit(matrix[train_idx], y_train)
        probabilities = np.asarray(model.predict_proba(matrix[test_idx]), dtype=float)[:, 1]

        both_classes = len(set(y_test.tolist())) == 2
        folds.append(
            FoldResult(
                index=split.index,
                train_size=len(train_idx),
                test_size=len(test_idx),
                purged=split.purged_count,
                positives_in_test=int(y_test.sum()),
                roc_auc=float(roc_auc_score(y_test, probabilities)) if both_classes else None,
                pr_auc=(
                    float(average_precision_score(y_test, probabilities))
                    if both_classes
                    else None
                ),
            )
        )
        pooled_scores.extend(probabilities.tolist())
        pooled_labels.extend(y_test.tolist())

    single_feature = _single_feature_auc(dataset.features, labels)

    if len(set(pooled_labels)) < 2:
        starved = sum(1 for fold in folds if fold.train_size < 2 or not fold.scorable)
        return ValidationReport(
            folds=tuple(folds),
            roc_auc=None,
            pr_auc=None,
            ladder=None,
            single_feature_auc=single_feature,
            candidates=dataset.size,
            positives=dataset.positives,
            base_rate=dataset.base_rate,
            coverage=dataset.coverage,
            missed_networks=dataset.discovery.missed_network_ids,
            note=(
                f"no fold held both classes ({starved} of {len(folds)} folds "
                "produced no score)"
            ),
        )

    ladder = (
        quintile_ladder(pooled_scores, pooled_labels)
        if len(pooled_scores) >= 5
        else None
    )
    return ValidationReport(
        folds=tuple(folds),
        roc_auc=float(roc_auc_score(pooled_labels, pooled_scores)),
        pr_auc=float(average_precision_score(pooled_labels, pooled_scores)),
        ladder=ladder,
        single_feature_auc=single_feature,
        candidates=dataset.size,
        positives=dataset.positives,
        base_rate=dataset.base_rate,
        coverage=dataset.coverage,
        missed_networks=dataset.discovery.missed_network_ids,
    )


def fit_case_model(dataset: CaseDataset, *, seed: int = 42) -> RandomForestClassifier | None:
    """Fit on everything, for scoring candidates in the interface.

    Separate from validation on purpose. This model has seen every candidate
    it will be asked about, so its scores order the queue an analyst works
    through and must never be quoted as a measurement. The measurement is
    ``run_case_validation``.
    """
    if dataset.size == 0 or len(set(dataset.labels.tolist())) < 2:
        return None
    model = _new_model(seed)
    model.fit(dataset.features.to_numpy(dtype=float), dataset.labels)
    return model
