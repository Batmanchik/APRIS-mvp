from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from apris.data_generator import FEATURE_COLUMNS
from apris.etl import aggregate_to_operational, load_transactions, process_external_dataset


def _build_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sender_id": "A",
                "receiver_id": "B",
                "amount": 100.0,
                "timestamp": "2026-03-01T10:00:00",
            },
            {
                "sender_id": "C",
                "receiver_id": "B",
                "amount": 250.0,
                "timestamp": "2026-03-01T11:00:00",
            },
            {
                "sender_id": "B",
                "receiver_id": "D",
                "amount": 80.0,
                "timestamp": "2026-03-01T12:00:00",
            },
        ]
    )


def test_load_transactions_csv(tmp_path: Path) -> None:
    path = tmp_path / "transactions.csv"
    _build_transactions().to_csv(path, index=False)

    df = load_transactions(path)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_transactions_json(tmp_path: Path) -> None:
    path = tmp_path / "transactions.json"
    _build_transactions().to_json(path, orient="records")

    df = load_transactions(path)
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_transactions_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    pd.DataFrame([{"sender_id": "A", "amount": 5.0}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_transactions(path)


def test_load_transactions_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "transactions.txt"
    path.write_text("invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_transactions(path)


def test_aggregate_to_operational_for_target_entity() -> None:
    tx_df = _build_transactions()
    operational = aggregate_to_operational(tx_df, target_entity_id="B")

    expected_keys = {
        "tx_count_total",
        "unique_counterparties",
        "new_clients_current",
        "new_clients_previous",
        "referred_clients_current",
        "incoming_funds",
        "payouts_total",
        "top1_wallet_share",
        "top10_wallet_share",
        "avg_holding_days",
        "repeat_investor_share",
        "max_referral_depth",
    }
    assert set(operational.keys()) == expected_keys
    assert operational["tx_count_total"] == 3.0


def test_aggregate_to_operational_raises_for_missing_entity() -> None:
    tx_df = _build_transactions()
    with pytest.raises(ValueError, match="No transactions found"):
        aggregate_to_operational(tx_df, target_entity_id="Z")


def test_process_external_dataset_accepts_feature_table(tmp_path: Path) -> None:
    row = {name: 0.5 for name in FEATURE_COLUMNS}
    row["label"] = 1
    path = tmp_path / "features.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    df = process_external_dataset(path)
    assert all(col in df.columns for col in FEATURE_COLUMNS)
    assert "label" in df.columns
