from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from apris.data_generator import FEATURE_COLUMNS, SEED

BENCHMARK_REPORT_PATH = Path("artifacts") / "cheops_v2_benchmark.json"
SELECTION_POLICY = {
    "score_formula": "0.65*roc_auc + 0.20*accuracy + 0.10*(1-brier) + 0.05*(1-ece)",
    "fallback_when_roc_auc_missing": "accuracy",
    "weights": {
        "roc_auc": 0.65,
        "accuracy": 0.20,
        "one_minus_brier": 0.10,
        "one_minus_ece": 0.05,
    },
}


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, *, bins: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y_true))
    ece = 0.0
    for idx in range(bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == bins - 1:
            mask = (y_score >= low) & (y_score <= high)
        else:
            mask = (y_score >= low) & (y_score < high)
        if not np.any(mask):
            continue
        predicted = float(np.mean(y_score[mask]))
        observed = float(np.mean(y_true[mask]))
        ece += abs(observed - predicted) * (float(np.sum(mask)) / total)
    return float(ece)


def _lightgbm_factory(random_state: int) -> Any:
    return lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _catboost_factory(random_state: int) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=220,
        learning_rate=0.05,
        depth=6,
        random_seed=random_state,
        verbose=False,
    )


def _xgboost_factory(random_state: int) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )


def _candidate_factories(include_optional: bool) -> dict[str, Callable[[int], Any]]:
    factories: dict[str, Callable[[int], Any]] = {"lightgbm": _lightgbm_factory}
    if include_optional:
        factories["catboost"] = _catboost_factory
        factories["xgboost"] = _xgboost_factory
    return factories


def _selection_score(
    *,
    roc_auc: float | None,
    accuracy: float,
    brier: float,
    ece: float,
) -> float:
    if roc_auc is None:
        # Conservative fallback when ROC-AUC is undefined (single-class test slice).
        return float(accuracy)
    return float(
        0.65 * roc_auc
        + 0.20 * accuracy
        + 0.10 * (1.0 - min(max(brier, 0.0), 1.0))
        + 0.05 * (1.0 - min(max(ece, 0.0), 1.0))
    )


def run_tabular_benchmark(
    df: pd.DataFrame,
    *,
    random_state: int = SEED,
    include_optional: bool = True,
) -> dict[str, Any]:
    x = df[FEATURE_COLUMNS].astype(float).copy()
    y = df["label"].astype(int).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    report: dict[str, Any] = {
        "benchmark_version": "cheops-tabular-benchmark-v2",
        "random_state": random_state,
        "rows_total": int(len(df)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "selection_policy": dict(SELECTION_POLICY),
        "candidates": {},
        "ranking": [],
        "winner": None,
        "winner_reason": None,
    }

    best_model: str | None = None
    best_score = float("-inf")

    for name, factory in _candidate_factories(include_optional).items():
        try:
            model = factory(random_state)
        except Exception as exc:
            report["candidates"][name] = {"status": "skipped", "reason": str(exc)}
            continue

        try:
            model.fit(x_train, y_train)
            proba = np.asarray(model.predict_proba(x_test), dtype=float)[:, 1]
            pred = (proba >= 0.5).astype(int)
            roc_auc = _safe_roc_auc(y_test, proba)
            accuracy = float(accuracy_score(y_test, pred))
            brier = float(brier_score_loss(y_test.astype(float), proba))
            ece = _expected_calibration_error(y_test.astype(float), proba)
            selection_score = _selection_score(
                roc_auc=roc_auc,
                accuracy=accuracy,
                brier=brier,
                ece=ece,
            )

            entry = {
                "status": "trained",
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "brier": brier,
                "ece": ece,
                "selection_score": selection_score,
            }
            report["candidates"][name] = entry

            score_for_selection = selection_score
            if score_for_selection > best_score:
                best_score = score_for_selection
                best_model = name
        except Exception as exc:
            report["candidates"][name] = {"status": "failed", "reason": str(exc)}

    trained = [
        {"name": name, "selection_score": float(values["selection_score"])}
        for name, values in report["candidates"].items()
        if isinstance(values, dict) and values.get("status") == "trained"
    ]
    trained.sort(key=lambda item: item["selection_score"], reverse=True)
    report["ranking"] = trained

    report["winner"] = best_model
    if best_model is not None:
        winner_metrics = report["candidates"][best_model]
        report["winner_reason"] = (
            f"{best_model} selected by policy "
            f"{SELECTION_POLICY['score_formula']}; "
            f"selection_score={winner_metrics['selection_score']:.6f}"
        )
    return report


def save_benchmark_report(
    report: dict[str, Any],
    path: str | Path = BENCHMARK_REPORT_PATH,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
