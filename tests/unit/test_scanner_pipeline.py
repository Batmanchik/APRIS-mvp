from __future__ import annotations

import pandas as pd
import pytest

from apris.data_generator import FEATURE_COLUMNS
from apris.frontend.scanner_pipeline import (
    deterministic_case_id,
    prepare_cases_from_feature_df,
    prepare_cases_from_tx_df,
)


def test_deterministic_case_id_is_stable() -> None:
    payload = {"a": 1.0, "b": 2.0}
    first = deterministic_case_id("case", 0, 42, payload)
    second = deterministic_case_id("case", 0, 42, payload)
    changed = deterministic_case_id("case", 0, 43, payload)

    assert first == second
    assert first != changed


def test_prepare_cases_from_feature_df_returns_v2_case_payloads() -> None:
    row = {name: 0.5 for name in FEATURE_COLUMNS}
    row["avg_holding_time"] = 30.0
    row["structural_depth"] = 6.0
    df = pd.DataFrame([row, row])

    cases = prepare_cases_from_feature_df(df, seed=7, feature_columns=FEATURE_COLUMNS)

    assert len(cases) == 2
    assert cases[0]["case_id"] != cases[1]["case_id"]
    assert len(cases[0]["events"]) == 1
    assert set(cases[0]["tabular_features"]) == set(FEATURE_COLUMNS)


def test_prepare_cases_from_tx_df_builds_entity_cases_and_caps_events() -> None:
    tx_df = pd.DataFrame(
        [
            {"sender_id": "A", "receiver_id": "B", "amount": 100.0, "timestamp": "2026-03-21T10:00:00Z"},
            {"sender_id": "B", "receiver_id": "C", "amount": 50.0, "timestamp": "2026-03-21T11:00:00Z"},
            {"sender_id": "A", "receiver_id": "C", "amount": 70.0, "timestamp": "2026-03-21T12:00:00Z"},
            {"sender_id": "C", "receiver_id": "A", "amount": 90.0, "timestamp": "2026-03-21T13:00:00Z"},
        ]
    )

    cases = prepare_cases_from_tx_df(tx_df, seed=21, max_events_per_case=2)
    case_ids = {case["case_id"] for case in cases}

    assert len(cases) == 3
    assert len(case_ids) == 3
    assert all(len(case["events"]) <= 2 for case in cases)
    assert all(event["ts"].endswith("+00:00") for case in cases for event in case["events"])
    assert all(set(case["tabular_features"]) == set(FEATURE_COLUMNS) for case in cases)


def test_prepare_cases_from_tx_df_validates_required_columns() -> None:
    bad_df = pd.DataFrame([{"sender_id": "A", "receiver_id": "B", "amount": 100.0}])
    with pytest.raises(ValueError, match="Missing required transaction columns"):
        prepare_cases_from_tx_df(bad_df, seed=1)


def test_prepare_cases_from_tx_df_handles_low_volume_entities() -> None:
    tx_df = pd.DataFrame(
        [
            {"sender_id": "S1", "receiver_id": "R1", "amount": 5.0, "timestamp": "2026-03-21T10:00:00Z"},
        ]
    )
    cases = prepare_cases_from_tx_df(tx_df, seed=10, max_events_per_case=3)

    assert len(cases) == 2
    assert all("tabular_features" in case for case in cases)


def test_prepare_cases_from_tx_df_normalizes_entity_ids_for_dedup() -> None:
    tx_df = pd.DataFrame(
        [
            {
                "sender_id": " too-001 ",
                "receiver_id": "wallet-01",
                "amount": 50.0,
                "timestamp": "2026-03-21T10:00:00Z",
            },
            {
                "sender_id": "TOO001",
                "receiver_id": "WALLET01",
                "amount": 75.0,
                "timestamp": "2026-03-21T10:05:00Z",
            },
        ]
    )

    cases = prepare_cases_from_tx_df(tx_df, seed=22, max_events_per_case=3)

    assert len(cases) == 2
