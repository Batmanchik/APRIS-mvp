"""E1 and E2 — the detector ladder, and whether the unit of analysis is what pays.

The question
------------
H1 says raising the unit of analysis from the account to the network buys
more than raising model complexity at the same unit. That is a comparison,
so it needs a grid rather than a number, and every cell has to be measured
the same way or the comparison is decoration.

The grid
--------
Four rungs of the ladder::

    rules       the published AFRD criteria, no fitting at all
    logistic    a linear model
    forest      a random forest
    boosting    gradient boosting

crossed with three scopes::

    account            one account per row, features of that account
    network_pooled     one discovered cluster per row, member features
                       averaged — the unit is raised and NOTHING else
    network_structural the same clusters, plus features that only exist
                       once there is a structure to measure

The middle scope is the whole point. Without it a gain from ``account`` to
``network_structural`` is ambiguous: it could be the unit, or it could be
five extra columns. ``network_pooled`` holds the feature kind fixed and moves
only the unit, so the two effects are separated instead of asserted.

What is fixed across every cell
-------------------------------
The same world, the same discovery output, the same purged walk-forward
splits, the same metrics. Models get no per-cell tuning, because a grid where
one cell was tuned and another was not measures the tuning.

The account scope cannot use the discovered clusters, so its rows are the
personal accounts of the world. Its base rate is therefore different, which
is why **average precision is reported beside ROC-AUC** — ROC-AUC is
insensitive to the base rate and would make the two scopes look comparable
in a way they are not.

Do not quote the network-scope rules cell
-----------------------------------------
Measured 2026-09-05: at the network scope the rule baseline reaches ROC-AUC
0.9895, above every fitted model, and the whole of it is one criterion.
``shared_device`` fires on **100 % of fraudulent candidates and 4.2 % of
honest ones** — because discovery LINKS accounts by their shared terminal in
the first place. The criterion is not detecting the ring, it is reporting
which of the three link types built the candidate. That is circular, and a
number produced that way does not survive being asked about.

``listed`` fires on nothing at all at that scope (0.000 on both classes): a
candidate's events span the members' whole ordinary lives, so no candidate
ever satisfies "this structure closed, and can therefore have been
investigated". A criterion that cannot fire is abstaining, not passing.

So the honest baseline is the **account** scope, which is also where the
published rules actually operate — a rule identifies a mule ACCOUNT. Both
defects are recorded in the plan with their fixes; until those land, the
network-scope rules row is diagnostic output and nothing more.

The honest ceiling
------------------
Discovery does not propose every network, and a network never proposed cannot
be found by anything downstream. Coverage is carried into the report next to
the scores, so the network-scope numbers are read against their own ceiling
rather than against one.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.baseline_afrd import afrd_verdict
from apris.cheops.infrastructure.ml.event_features_v2 import (
    GRAPH_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
    graph_features_from_events,
    sequence_features_from_events,
)
from apris.cheops.infrastructure.ml.validation_v2 import (
    purged_walk_forward_splits,
    quintile_ladder,
)
from apris.cheops.infrastructure.simulation.config import ASSET_CASH, SimulationConfig
from apris.cheops.infrastructure.simulation.discovery import (
    Candidate,
    discover_candidates,
    label_candidates,
)
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld, generate_world

REPORT_PATH = Path("artifacts") / "experiment_ladder.json"

SEED = 20261005

ACCOUNT_FEATURE_COLUMNS: tuple[str, ...] = (
    "out_in_ratio",
    "cash_out_share",
    "median_hold_hours_inverse",
    "counterparty_concentration",
    "active_day_share",
    *SEQUENCE_FEATURE_COLUMNS,
)

MODEL_NAMES: tuple[str, ...] = ("rules", "logistic", "forest", "boosting")
SCOPE_NAMES: tuple[str, ...] = ("account", "network_pooled", "network_structural")


# ==========================================================================
# Rows
# ==========================================================================


@dataclass(frozen=True)
class Row:
    """One object of analysis: its features, its label, and when it happened.

    ``ts`` is the row's own position in time and is what the walk-forward
    split orders on. A row with no events cannot be placed in time and is
    dropped rather than given a default date, which would silently push it
    into the training half of every split.
    """

    key: str
    ts: datetime
    features: dict[str, float]
    label: int
    events: tuple[TransactionEvent, ...]
    members: tuple[str, ...]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def account_features(account: str, events: Sequence[TransactionEvent]) -> dict[str, float]:
    """Everything measurable about one account from its own payments.

    These are the columns the account scope is judged on, and the same
    columns are averaged to build ``network_pooled``. Each is squashed into
    [0, 1] so a linear model is not decided by units.
    """
    inflow = [e for e in events if e.receiver_id == account]
    outflow = [e for e in events if e.sender_id == account]
    total_in = sum(e.amount for e in inflow)
    total_out = sum(e.amount for e in outflow)

    cash_out = sum(e.amount for e in outflow if e.asset_type == ASSET_CASH)

    # Time between money arriving and the next money leaving. Short is the
    # pass-through signature; the inverse is taken so larger always means
    # more suspicious, which keeps the sign of a linear coefficient readable.
    holds: list[float] = []
    arrivals = sorted(e.ts for e in inflow)
    for departure in sorted(e.ts for e in outflow):
        earlier = [a for a in arrivals if a <= departure]
        if earlier:
            holds.append((departure - earlier[-1]).total_seconds() / 3600.0)
    median_hold = statistics.median(holds) if holds else 720.0

    by_counterparty: dict[str, float] = defaultdict(float)
    for event in outflow:
        by_counterparty[event.receiver_id] += event.amount
    concentration = (
        max(by_counterparty.values()) / total_out if total_out > 0 and by_counterparty else 0.0
    )

    days = {e.ts.date() for e in events}
    span = 1
    if events:
        first = min(e.ts for e in events).date()
        last = max(e.ts for e in events).date()
        span = max(1, (last - first).days + 1)

    features = {
        "out_in_ratio": min(1.0, _safe_ratio(total_out, total_in)),
        "cash_out_share": min(1.0, _safe_ratio(cash_out, total_out)),
        "median_hold_hours_inverse": 1.0 / (1.0 + median_hold),
        "counterparty_concentration": float(concentration),
        "active_day_share": min(1.0, len(days) / span),
    }
    features.update(sequence_features_from_events(events))
    return {name: float(features.get(name, 0.0)) for name in ACCOUNT_FEATURE_COLUMNS}


def _events_by_account(world: SimulatedWorld) -> dict[str, list[TransactionEvent]]:
    grouped: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in world.events:
        grouped[event.sender_id].append(event)
        if event.receiver_id != event.sender_id:
            grouped[event.receiver_id].append(event)
    return grouped


def build_account_rows(world: SimulatedWorld) -> list[Row]:
    """One row per personal account, labelled by whether it is in a network.

    This is the unit the published rules work at, and the unit at which a
    mule is an ordinary student withdrawing money.
    """
    grouped = _events_by_account(world)
    fraud = world.fraud_account_ids()

    rows: list[Row] = []
    for account, account_type in world.populations.items():
        events = grouped.get(account, [])
        if not events:
            continue
        if account_type in {"terminal", "merchant", "employer"}:
            continue
        rows.append(
            Row(
                key=account,
                ts=max(e.ts for e in events),
                features=account_features(account, events),
                label=1 if account in fraud else 0,
                events=tuple(sorted(events, key=lambda e: e.ts)),
                members=(account,),
            )
        )
    return rows


def build_network_rows(
    world: SimulatedWorld, candidates: Sequence[Candidate], labels: Sequence[int]
) -> tuple[list[Row], list[Row]]:
    """Pooled and structural rows over the SAME discovered clusters.

    The two lists are returned together because they must be the same objects
    in the same order. Building them separately is how a comparison quietly
    becomes a comparison of two different datasets.
    """
    grouped = _events_by_account(world)

    pooled: list[Row] = []
    structural: list[Row] = []
    for candidate, label in zip(candidates, labels):
        if not candidate.events:
            continue

        per_member = [
            account_features(member, grouped.get(member, []))
            for member in candidate.member_ids
            if grouped.get(member)
        ]
        if not per_member:
            continue
        averaged = {
            name: float(np.mean([m[name] for m in per_member]))
            for name in ACCOUNT_FEATURE_COLUMNS
        }

        ts = max(e.ts for e in candidate.events)
        pooled.append(
            Row(
                key=candidate.candidate_id,
                ts=ts,
                features=averaged,
                label=int(label),
                events=candidate.events,
                members=candidate.member_ids,
            )
        )

        enriched = dict(averaged)
        enriched.update(graph_features_from_events(candidate.events))
        structural.append(
            Row(
                key=candidate.candidate_id,
                ts=ts,
                features=enriched,
                label=int(label),
                events=candidate.events,
                members=candidate.member_ids,
            )
        )
    return pooled, structural


# ==========================================================================
# The rule baseline
# ==========================================================================


@dataclass(frozen=True)
class CriterionBreakdown:
    """How often one criterion fires, split by the answer.

    A composite baseline score can look strong for a reason that does not
    survive being named. This exists so each criterion is quoted separately
    and the reader can see which one is carrying the number — in particular
    whether ``shared_device`` at the network scope is measuring the rule or
    measuring the fact that discovery linked those accounts BY their shared
    terminal in the first place.
    """

    criterion: str
    fire_rate_fraud: float
    fire_rate_honest: float

    @property
    def lift(self) -> float:
        return self.fire_rate_fraud - self.fire_rate_honest


def rule_breakdown(rows: Sequence[Row]) -> tuple[CriterionBreakdown, ...]:
    """Per-criterion firing rates, using the same forward-built list."""
    verdicts = _rule_verdicts(rows)
    fraud = [v for v, r in zip(verdicts, rows) if r.label == 1]
    honest = [v for v, r in zip(verdicts, rows) if r.label == 0]

    def rate(group: Sequence[object], name: str) -> float:
        if not group:
            return 0.0
        return float(np.mean([bool(getattr(v, name)) for v in group]))

    return tuple(
        CriterionBreakdown(
            criterion=name,
            fire_rate_fraud=rate(fraud, name),
            fire_rate_honest=rate(honest, name),
        )
        for name in ("listed", "shared_device", "profile_deviation")
    )


def _rule_verdicts(rows: Sequence[Row]) -> list[object]:
    ordered = sorted(range(len(rows)), key=lambda i: rows[i].ts)
    listed: set[str] = set()
    verdicts: list[object] = [None] * len(rows)

    for position in ordered:
        row = rows[position]
        window_start = min(e.ts for e in row.events)
        available = frozenset(listed - set(row.members))
        verdicts[position] = afrd_verdict(row.members, row.events, listed=available)
        if row.label == 1 and max(e.ts for e in row.events) <= window_start + timedelta(days=30):
            listed.update(row.members)
    return verdicts


def rule_scores(rows: Sequence[Row]) -> np.ndarray:
    """AFRD criteria, with the list built from structures that closed earlier.

    A real list of known mule accounts is the residue of past investigations,
    so it is assembled here the same way: for each row, the members of every
    fraudulent row whose last event predates this row's window. Nothing about
    the row being scored enters its own list.
    """
    ordered = sorted(range(len(rows)), key=lambda i: rows[i].ts)
    listed: set[str] = set()
    scores = np.zeros(len(rows), dtype=float)

    for position in ordered:
        row = rows[position]
        window_start = min(e.ts for e in row.events)
        available = frozenset(listed - set(row.members))
        scores[position] = afrd_verdict(row.members, row.events, listed=available).score
        # Only structures that have finished by now can have been investigated.
        if row.label == 1 and max(e.ts for e in row.events) <= window_start + timedelta(days=30):
            listed.update(row.members)
    return scores


# ==========================================================================
# Fitted models
# ==========================================================================


def _make_model(name: str):
    if name == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
                ),
            ]
        )
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "boosting":
        return GradientBoostingClassifier(random_state=SEED)
    raise ValueError(f"unknown model: {name}")


@dataclass(frozen=True)
class CellResult:
    scope: str
    model: str
    rows: int
    positives: int
    base_rate: float
    roc_auc: float | None
    average_precision: float | None
    folds: int
    ladder: str | None
    note: str = ""


def _pooled_out_of_fold(
    rows: Sequence[Row], columns: Sequence[str], model_name: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """Out-of-fold predictions across purged walk-forward splits.

    Scores from different folds are pooled and scored once rather than
    averaged per fold. Averaging per-fold AUCs weights a fold of eleven rows
    the same as a fold of four hundred, and the early folds here are small.
    """
    frame = pd.DataFrame([r.features for r in rows], columns=list(columns)).astype(float)
    x = frame.to_numpy()
    y = np.array([r.label for r in rows], dtype=int)
    timestamps = [r.ts for r in rows]

    splits = purged_walk_forward_splits(timestamps, n_splits=5, purge=timedelta(days=2))

    predictions: list[float] = []
    truths: list[int] = []
    used = 0
    for split in splits:
        train_y = y[list(split.train)]
        if len(np.unique(train_y)) < 2:
            continue
        model = _make_model(model_name)
        model.fit(x[list(split.train)], train_y)
        probability = np.asarray(model.predict_proba(x[list(split.test)]))[:, 1]
        predictions.extend(float(p) for p in probability)
        truths.extend(int(t) for t in y[list(split.test)])
        used += 1

    return np.array(predictions), np.array(truths), used


def evaluate_cell(rows: Sequence[Row], scope: str, model_name: str) -> CellResult:
    columns = sorted(rows[0].features) if rows else []
    y_all = np.array([r.label for r in rows], dtype=int)
    base_rate = float(y_all.mean()) if len(y_all) else 0.0

    if model_name == "rules":
        # No fitting, so there is nothing to hold out from. The rules are
        # scored on every row, with the list built forward in time.
        scores = rule_scores(rows)
        truths = y_all
        folds = 0
        note = "no fitting; scored on all rows with a forward-built list"
    else:
        scores, truths, folds = _pooled_out_of_fold(rows, columns, model_name)
        note = "out-of-fold, purged walk-forward"

    if len(truths) == 0 or len(np.unique(truths)) < 2:
        return CellResult(
            scope=scope,
            model=model_name,
            rows=len(rows),
            positives=int(y_all.sum()),
            base_rate=base_rate,
            roc_auc=None,
            average_precision=None,
            folds=folds,
            ladder=None,
            note="one class only in the evaluated rows",
        )

    # (scores, labels), in that order — the ladder sorts BY the score.
    ladder = quintile_ladder([float(s) for s in scores], [int(t) for t in truths])
    return CellResult(
        scope=scope,
        model=model_name,
        rows=len(rows),
        positives=int(y_all.sum()),
        base_rate=base_rate,
        roc_auc=float(roc_auc_score(truths, scores)),
        average_precision=float(average_precision_score(truths, scores)),
        folds=folds,
        ladder=ladder.describe(),
        note=note,
    )


# ==========================================================================
# The run
# ==========================================================================


@dataclass(frozen=True)
class LadderReport:
    seed: int
    world: dict[str, float]
    coverage: float
    networks_total: int
    networks_covered: int
    cells: tuple[CellResult, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "seed": self.seed,
                "world": self.world,
                "discovery": {
                    "coverage": self.coverage,
                    "networks_total": self.networks_total,
                    "networks_covered": self.networks_covered,
                },
                "cells": [asdict(cell) for cell in self.cells],
            },
            indent=2,
            sort_keys=True,
        )

    def table(self) -> str:
        header = (
            f"{'scope':<20}{'model':<10}{'rows':>7}{'pos':>6}{'base':>8}"
            f"{'ROC-AUC':>10}{'AP':>9}  ladder"
        )
        lines = [header, "-" * len(header)]
        for cell in self.cells:
            auc = f"{cell.roc_auc:.4f}" if cell.roc_auc is not None else "—"
            ap = f"{cell.average_precision:.4f}" if cell.average_precision is not None else "—"
            lines.append(
                f"{cell.scope:<20}{cell.model:<10}{cell.rows:>7}{cell.positives:>6}"
                f"{cell.base_rate:>8.3f}{auc:>10}{ap:>9}  {cell.ladder or cell.note}"
            )
        return "\n".join(lines)


def run_ladder(config: SimulationConfig) -> LadderReport:
    """Generate one world and fill the whole grid from it."""
    world = generate_world(config)
    candidates = discover_candidates(world)
    labels, discovery = label_candidates(world, candidates)

    account_rows = build_account_rows(world)
    pooled_rows, structural_rows = build_network_rows(world, candidates, labels)

    scopes: dict[str, list[Row]] = {
        "account": account_rows,
        "network_pooled": pooled_rows,
        "network_structural": structural_rows,
    }

    cells: list[CellResult] = []
    for scope in SCOPE_NAMES:
        rows = scopes[scope]
        if not rows:
            continue
        for model_name in MODEL_NAMES:
            cells.append(evaluate_cell(rows, scope, model_name))

    return LadderReport(
        seed=config.seed,
        world=world.summary(),
        coverage=discovery.coverage,
        networks_total=discovery.networks_total,
        networks_covered=discovery.networks_covered,
        cells=tuple(cells),
    )


def write_report(report: LadderReport, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
    return path


__all__ = [
    "ACCOUNT_FEATURE_COLUMNS",
    "GRAPH_FEATURE_COLUMNS",
    "CellResult",
    "LadderReport",
    "Row",
    "account_features",
    "build_account_rows",
    "build_network_rows",
    "evaluate_cell",
    "rule_scores",
    "run_ladder",
    "write_report",
]
