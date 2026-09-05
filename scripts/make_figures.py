"""Draw every figure from the real run, and print where each one went.

    python scripts/make_figures.py

The figures module has existed for a while and nothing called it, so the
project had numbers nobody could look at. Everything here is computed from a
generated world and a real fit — no figure takes a hand-typed value, because
a picture drawn from a number somebody remembered is a picture that will
disagree with the table on the next slide.
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from apris.cheops.infrastructure.experiments.ladder import (
    ACCOUNT_FEATURE_COLUMNS,
    build_account_rows,
    build_network_rows,
    rule_scores,
)
from apris.cheops.infrastructure.reporting.figures import (
    plot_detectability_curve,
    plot_feature_separation,
    plot_network_anatomy,
    plot_value_lost,
    use_project_style,
)
from apris.cheops.infrastructure.simulation.cases import LabelledCase, build_cases
from apris.cheops.infrastructure.simulation.config import EvasionKnobs, SimulationConfig
from apris.cheops.infrastructure.simulation.discovery import (
    discover_candidates,
    label_candidates,
)
from apris.cheops.infrastructure.simulation.generator import generate_world

BASE = dict(
    seed=20261005,
    days=120,
    mule_networks=90,
    pyramids=30,
    crowd_collections=220,
    family_circles=300,
    employers=80,
    terminals=120,
)

# The evasion sweep runs a whole world per point, so it gets a smaller one.
SWEEP = dict(BASE, days=60, mule_networks=40, pyramids=10, crowd_collections=80,
             family_circles=100, employers=30, terminals=60)

FUNDER_VALUES = (1, 2, 4, 8, 16)


def _fit_scores(rows, columns, model):
    """Out-of-sample enough for a picture: fit on the first half by time.

    The grid in ``experiments.ladder`` is the authority on the numbers; this
    is here so a figure is drawn from a fit rather than from a memory.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i].ts)
    cut = len(order) // 2
    frame = pd.DataFrame([r.features for r in rows], columns=list(columns)).astype(float)
    x = frame.to_numpy()
    y = np.array([r.label for r in rows], dtype=int)

    train, test = order[:cut], order[cut:]
    if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
        return None, None
    model.fit(x[train], y[train])
    return np.asarray(model.predict_proba(x[test]))[:, 1], y[test]


def _forest():
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, class_weight="balanced",
        random_state=1, n_jobs=-1,
    )


def _logistic():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=1)),
    ])


def figure_anatomy(world) -> str:
    """One real mule network, drawn as source -> mules -> ATM."""
    cases = build_cases(world)
    mule_cases = [c for c in cases if c.kind == "mule_network"]
    if not mule_cases:
        return "skipped: no mule case in this world"
    # The median case by size, so the picture is typical rather than extreme.
    chosen = sorted(mule_cases, key=lambda c: len(c.member_ids))[len(mule_cases) // 2]
    burst = _burst_only(chosen)
    path = plot_network_anatomy(burst)
    return (
        f"{path}  ({len(burst.member_ids)} accounts, {len(burst.events)} events "
        f"of the case's {len(chosen.events)})"
    )


def _burst_only(case: LabelledCase, within: timedelta = timedelta(hours=24)) -> LabelledCase:
    """Keep the operation, drop the mules' ordinary lives.

    Since mules were given wages, shopping and rent, a case's events run for
    months, and the first version of this figure said the operation "fits in
    166 521 min" and that the network RETAINED minus 1.4 million tenge. Both
    were arithmetic over a student's grocery shopping.

    The operation is: what the source paid out, and what each recipient took
    to an ATM within a day of being paid. Everything else is that person's
    life, and drawing it hides the shape the figure exists to show.
    """
    members = set(case.member_ids)
    paid_out: dict[str, float] = defaultdict(float)
    for event in case.events:
        if event.sender_id in members:
            paid_out[event.sender_id] += 1

    # The source is the member that pays the most distinct others.
    fan: dict[str, set[str]] = defaultdict(set)
    for event in case.events:
        if event.sender_id in members and event.receiver_id in members:
            fan[event.sender_id].add(event.receiver_id)
    if not fan:
        return case
    source = max(fan, key=lambda a: len(fan[a]))

    transfers = [e for e in case.events if e.sender_id == source and e.receiver_id in members]
    if not transfers:
        return case
    arrival = {e.receiver_id: e.ts for e in transfers}

    exits = [
        e
        for e in case.events
        if e.asset_type == "cash"
        and e.sender_id in arrival
        and timedelta(0) <= e.ts - arrival[e.sender_id] <= within
    ]

    kept = tuple(sorted(transfers + exits, key=lambda e: e.ts))
    return LabelledCase(
        case_id=case.case_id,
        kind=case.kind,
        events=kept,
        member_ids=tuple(sorted({source} | set(arrival))),
    )


def figure_separation(world) -> str:
    """Where the honest populations sit against the fraudulent ones."""
    cases = build_cases(world)
    grouped: dict[str, list[list]] = {}
    for case in cases:
        grouped.setdefault(case.kind, []).append(case)

    rows: list[dict[str, float]] = []
    kinds: list[str] = []
    features = (
        "cash_out_share",
        "median_hold_hours_inverse",
        "counterparty_concentration",
        "amount_cv_norm",
    )
    account_rows, _ = build_account_rows(world)
    by_account = {r.key: r.features for r in account_rows}

    for kind, members in grouped.items():
        for case in members:
            present = [by_account[m] for m in case.member_ids if m in by_account]
            if not present:
                continue
            rows.append({f: float(np.mean([p[f] for p in present])) for f in features})
            kinds.append(kind)

    if not rows:
        return "skipped: no case had a scoreable member"
    path = plot_feature_separation(rows, kinds, list(features))
    return f"{path}  ({len(rows)} cases over {len(set(kinds))} kinds)"


def figure_detectability(verbose: bool = True) -> str:
    """Recall against how hard the organiser works to hide — the main figure.

    A whole world is generated per point, per detector, because the knob is a
    property of the generator. That is why it is slow, and why the sweep uses
    a smaller world than the rest.
    """
    recall: dict[str, list[float]] = {"AFRD rules": [], "account · forest": [],
                                      "network · logistic": []}

    for funders in FUNDER_VALUES:
        config = SimulationConfig(**SWEEP, evasion=EvasionKnobs(funders=funders))
        world = generate_world(config)
        candidates = discover_candidates(world)
        labels, _ = label_candidates(world, candidates)
        account_rows, _ = build_account_rows(world)
        _, structural = build_network_rows(world, candidates, labels)

        scores = rule_scores(account_rows, "account")
        truth = np.array([r.label for r in account_rows])
        recall["AFRD rules"].append(_recall_at_top(scores, truth))

        s, t = _fit_scores(account_rows, ACCOUNT_FEATURE_COLUMNS, _forest())
        recall["account · forest"].append(_measured(s, t))

        columns = sorted(structural[0].features) if structural else []
        s, t = (_fit_scores(structural, columns, _logistic()) if structural else (None, None))
        recall["network · logistic"].append(_measured(s, t))

        if verbose:
            print(f"    funders={funders:>2}: " + "  ".join(
                f"{k} {v[-1]:.3f}" for k, v in recall.items()))

    path = plot_detectability_curve(FUNDER_VALUES, recall)
    return str(path)


def _measured(scores, truth) -> float:
    """NaN where the fit was impossible, never 0.0.

    A point that could not be measured and a detector that found nothing are
    different facts, and matplotlib breaks the line at NaN instead of drawing
    a collapse that did not happen. Writing 0.0 here is the same mistake as a
    feature returning zero where it should abstain.
    """
    if scores is None:
        return float("nan")
    return _recall_at_top(scores, truth)


def _recall_at_top(scores, truth, budget: float = 0.10) -> float:
    """Share of the fraud caught inside the top 10 % an analyst can review.

    Recall at a review budget, not recall at a threshold. An analyst has a
    day, not a cutoff, and a metric that ignores that is measuring something
    nobody experiences.
    """
    if scores is None or len(scores) == 0 or truth.sum() == 0:
        return 0.0
    take = max(1, int(len(scores) * budget))
    top = np.argsort(-np.asarray(scores))[:take]
    return float(truth[top].sum() / truth.sum())


def figure_value_lost(world) -> str:
    """Recall against the money already gone when the alert fires.

    The two rankings need not agree, and money is what the Anti-Fraud Centre
    reports, so it is the quantity worth optimising.
    """
    account_rows, _ = build_account_rows(world)
    truth = np.array([r.label for r in account_rows])

    names, recalls, lost = [], [], []
    for label, scorer in (
        ("AFRD rules", lambda: (rule_scores(account_rows, "account"), truth)),
        ("logistic", lambda: _fit_scores(account_rows, ACCOUNT_FEATURE_COLUMNS, _logistic())),
        ("forest", lambda: _fit_scores(account_rows, ACCOUNT_FEATURE_COLUMNS, _forest())),
    ):
        scores, t = scorer()
        if scores is None:
            continue
        names.append(label)
        recalls.append(_recall_at_top(scores, t))
        lost.append(_value_lost_share(account_rows, scores, t))

    path = plot_value_lost(names, recalls, lost)
    return f"{path}  ({', '.join(f'{n} {r:.2f}/{v:.2f}' for n, r, v in zip(names, recalls, lost))})"


def _value_lost_share(rows, scores, truth, budget: float = 0.10) -> float:
    """Of all the cash mules withdrew, what share belonged to one we missed.

    Cash out is the exit: once it is at the ATM it is gone. Everything still
    inside the banking system at the moment of the alert is money that could
    in principle have been held.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i].ts)
    tail = order[len(order) // 2:]
    if len(tail) != len(truth):
        tail = tail[: len(truth)]

    take = max(1, int(len(scores) * budget))
    flagged = set(np.argsort(-np.asarray(scores))[:take].tolist())

    gone = missed = 0.0
    for position, row_index in enumerate(tail):
        if truth[position] != 1:
            continue
        cash = sum(
            e.amount for e in rows[row_index].events
            if e.sender_id == rows[row_index].key and e.asset_type == "cash"
        )
        gone += cash
        if position not in flagged:
            missed += cash
    return missed / gone if gone > 0 else 0.0


def main() -> int:
    use_project_style()
    started = time.time()

    print("generating the world...")
    world = generate_world(SimulationConfig(**BASE))

    print("\n1/4  anatomy of one real network")
    print("    " + figure_anatomy(world))

    print("\n2/4  where the honest populations sit")
    print("    " + figure_separation(world))

    print("\n3/4  recall against money already gone")
    print("    " + figure_value_lost(world))

    print("\n4/4  detectability curve (one world per point, slow)")
    print("    " + figure_detectability())

    print(f"\nartifacts/figures/ — {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
