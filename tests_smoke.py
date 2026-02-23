from __future__ import annotations

import argparse
from pathlib import Path

from data_generator import FEATURE_COLUMNS, build_dataset, validate_dataset
from graph_module import build_transaction_graph, compute_hub_metrics
from risk_engine import FEATURE_NAMES_PATH, MODEL_PATH, explain, predict_risk
from train_model import main as train_main


PRESETS = {
    "legit": {
        "growth_rate": 0.08,
        "referral_ratio": 0.20,
        "payout_dependency": 0.55,
        "centralization_index": 0.25,
        "avg_holding_time": 60.0,
        "reinvestment_rate": 0.42,
        "gini_coefficient": 0.34,
        "transaction_entropy": 3.8,
        "structural_depth": 4.0,
    },
    "suspicious": {
        "growth_rate": 0.22,
        "referral_ratio": 0.56,
        "payout_dependency": 0.92,
        "centralization_index": 0.54,
        "avg_holding_time": 34.0,
        "reinvestment_rate": 0.60,
        "gini_coefficient": 0.56,
        "transaction_entropy": 2.4,
        "structural_depth": 8.0,
    },
    "pyramid": {
        "growth_rate": 0.62,
        "referral_ratio": 0.86,
        "payout_dependency": 1.35,
        "centralization_index": 0.86,
        "avg_holding_time": 14.0,
        "reinvestment_rate": 0.82,
        "gini_coefficient": 0.84,
        "transaction_entropy": 1.1,
        "structural_depth": 12.0,
    },
}


def _ensure_model_artifacts() -> None:
    if Path(MODEL_PATH).exists() and Path(FEATURE_NAMES_PATH).exists():
        return
    print("Artifacts not found. Running train_model.py...")
    train_main()


def run_data_smoke() -> None:
    df = build_dataset(total_n=3000)
    validate_dataset(df)
    assert list(df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    print("data_smoke: OK")


def run_risk_smoke() -> None:
    _ensure_model_artifacts()

    legit = predict_risk(PRESETS["legit"])
    suspicious = predict_risk(PRESETS["suspicious"])
    pyramid = predict_risk(PRESETS["pyramid"])

    assert 0.0 <= legit["probability"] <= 1.0
    assert 0.0 <= suspicious["probability"] <= 1.0
    assert 0.0 <= pyramid["probability"] <= 1.0
    assert pyramid["probability"] > legit["probability"], "Expected pyramid risk > legit risk"

    top_features = explain(PRESETS["pyramid"], top_k=5)
    assert len(top_features) == 5
    print(
        "risk_smoke: OK | "
        f"legit={legit['probability']:.4f}, "
        f"suspicious={suspicious['probability']:.4f}, "
        f"pyramid={pyramid['probability']:.4f}"
    )


def run_graph_smoke() -> None:
    low_case = PRESETS["suspicious"].copy()
    low_case["centralization_index"] = 0.15
    low_case["structural_depth"] = 8.0

    high_case = PRESETS["suspicious"].copy()
    high_case["centralization_index"] = 0.90
    high_case["structural_depth"] = 8.0

    low_graph = build_transaction_graph(low_case)
    high_graph = build_transaction_graph(high_case)

    low_metrics = compute_hub_metrics(low_graph)
    high_metrics = compute_hub_metrics(high_graph)

    assert high_metrics["hub_in_degree_share"] > low_metrics["hub_in_degree_share"], (
        "Expected higher centralization to increase hub_in_degree_share"
    )
    print(
        "graph_smoke: OK | "
        f"low_hub_share={low_metrics['hub_in_degree_share']:.4f}, "
        f"high_hub_share={high_metrics['hub_in_degree_share']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["data", "risk", "graph", "all"],
        default="all",
        help="Run only one smoke group or all.",
    )
    args = parser.parse_args()

    if args.only in {"data", "all"}:
        run_data_smoke()
    if args.only in {"risk", "all"}:
        run_risk_smoke()
    if args.only in {"graph", "all"}:
        run_graph_smoke()
    print("tests_smoke: SUCCESS")


if __name__ == "__main__":
    main()
