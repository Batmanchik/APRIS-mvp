"""
Scanner data preparation pipeline for Streamlit page.

This module isolates case-building logic from UI code so it can be tested and
optimized independently.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from apris.cheops.domain.entity_resolution import (
    ENTITY_TYPE_UNKNOWN,
    normalize_external_entity_id,
    resolve_entity_key,
)
from apris.etl import aggregate_to_operational
from apris.risk_engine import OPERATIONAL_INPUT_BOUNDS, operational_to_features

_REQUIRED_TX_COLUMNS = {"sender_id", "receiver_id", "amount", "timestamp"}


def deterministic_case_id(prefix: str, index: int, seed: int, payload: dict[str, Any]) -> str:
    """Builds a deterministic case id from payload content."""
    fingerprint = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(f"{prefix}|{index}|{seed}|{fingerprint}".encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def event_stub(case_id: str, base_amount: float) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "event_id": f"evt-{case_id}",
        "ts": now,
        "amount": float(max(base_amount, 1.0)),
        "currency": "USD",
        "sender_id": f"src-{case_id}",
        "receiver_id": f"dst-{case_id}",
        "sender_type": "company",
        "receiver_type": "wallet",
        "channel": "legal",
        "jurisdiction": "KZ",
        "asset_type": "fiat",
    }


def _build_case_from_features(
    feature_payload: dict[str, float],
    *,
    index: int,
    seed: int,
) -> dict[str, Any]:
    case_id = deterministic_case_id("case", index, seed, feature_payload)
    return {
        "case_id": case_id,
        "window_hours": 24,
        "events": [event_stub(case_id, base_amount=100.0)],
        "tabular_features": feature_payload,
    }


def prepare_cases_from_feature_df(
    df: pd.DataFrame,
    *,
    seed: int,
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Converts feature rows into v2 case payloads."""
    feature_rows = df[list(feature_columns)].astype(float).to_dict(orient="records")
    return [_build_case_from_features(row, index=i, seed=seed) for i, row in enumerate(feature_rows)]


def _normalize_transactions(tx_df: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED_TX_COLUMNS - set(tx_df.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")

    work_df = tx_df.copy()
    work_df["sender_id"] = work_df["sender_id"].astype("string").fillna("")
    work_df["receiver_id"] = work_df["receiver_id"].astype("string").fillna("")
    work_df["amount"] = pd.to_numeric(work_df["amount"], errors="coerce")
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], errors="coerce", utc=True)

    work_df["sender_id"] = work_df["sender_id"].map(normalize_external_entity_id)
    work_df["receiver_id"] = work_df["receiver_id"].map(normalize_external_entity_id)

    work_df = work_df[
        (work_df["sender_id"] != "")
        & (work_df["receiver_id"] != "")
        & work_df["amount"].notna()
        & work_df["timestamp"].notna()
    ].copy()
    if work_df.empty:
        raise ValueError("No valid transactions after normalization.")

    work_df["sender_id"] = work_df["sender_id"].astype(str)
    work_df["receiver_id"] = work_df["receiver_id"].astype(str)
    return work_df.reset_index(drop=True)


def _build_entity_tx_index_map(work_df: pd.DataFrame) -> dict[str, list[int]]:
    tx_idx = work_df.index.to_numpy()
    sender_map = pd.DataFrame({"entity_id": work_df["sender_id"].to_numpy(), "tx_idx": tx_idx})
    receiver_map = pd.DataFrame({"entity_id": work_df["receiver_id"].to_numpy(), "tx_idx": tx_idx})
    association = pd.concat([sender_map, receiver_map], ignore_index=True).drop_duplicates()
    grouped = association.groupby("entity_id", sort=True)["tx_idx"].agg(list)
    return {str(entity): [int(idx) for idx in indices] for entity, indices in grouped.items()}


def _event_from_row(case_id: str, event_idx: int, row: Any) -> dict[str, Any]:
    timestamp: datetime = getattr(row, "timestamp")
    return {
        "event_id": f"evt-{case_id}-{event_idx}",
        "ts": timestamp.isoformat(),
        "amount": float(max(getattr(row, "amount"), 1.0)),
        "currency": "USD",
        "sender_id": str(getattr(row, "sender_id")),
        "receiver_id": str(getattr(row, "receiver_id")),
        "sender_type": str(getattr(row, "sender_type", "company")),
        "receiver_type": str(getattr(row, "receiver_type", "wallet")),
        "channel": str(getattr(row, "channel", "legal")),
        "jurisdiction": str(getattr(row, "jurisdiction", "KZ")),
        "asset_type": str(getattr(row, "asset_type", "fiat")),
    }


def _normalize_operational_for_model(ops: dict[str, float]) -> dict[str, float]:
    normalized = dict(ops)
    for key, (low, high) in OPERATIONAL_INPUT_BOUNDS.items():
        value = float(normalized.get(key, low))
        normalized[key] = float(max(low, min(high, value)))

    if normalized["referred_clients_current"] > normalized["new_clients_current"]:
        normalized["referred_clients_current"] = normalized["new_clients_current"]
    if normalized["top1_wallet_share"] > normalized["top10_wallet_share"]:
        normalized["top1_wallet_share"] = normalized["top10_wallet_share"]
    return normalized


def prepare_cases_from_tx_df(
    tx_df: pd.DataFrame,
    *,
    seed: int,
    max_events_per_case: int = 6,
) -> list[dict[str, Any]]:
    """
    Converts raw transaction log into v2 case payloads.

    Performance notes:
    - builds entity->transaction index map once (vectorized),
    - avoids full-table filtering for each entity.
    """
    work_df = _normalize_transactions(tx_df)
    entity_index_map = _build_entity_tx_index_map(work_df)

    cases: list[dict[str, Any]] = []
    for idx, (entity_id, tx_indices) in enumerate(entity_index_map.items()):
        entity_tx = work_df.loc[tx_indices].sort_values("timestamp", ascending=False, kind="mergesort")
        ops = _normalize_operational_for_model(
            aggregate_to_operational(entity_tx, target_entity_id=entity_id)
        )
        features = operational_to_features(ops)
        entity_key = resolve_entity_key(
            {
                "entity_type": ENTITY_TYPE_UNKNOWN,
                "source_entity_id": entity_id,
                "name": entity_id,
                "jurisdiction": "KZ",
            }
        )
        case_id = deterministic_case_id("entity", idx, seed, {"entity_key": entity_key, **features})

        events = [
            _event_from_row(case_id, event_idx=event_idx, row=row)
            for event_idx, row in enumerate(
                entity_tx.head(max_events_per_case).itertuples(index=False),
                start=1,
            )
        ]
        if not events:
            events = [event_stub(case_id, base_amount=float(ops["incoming_funds"]))]

        cases.append(
            {
                "case_id": case_id,
                "window_hours": 24,
                "events": events,
                "tabular_features": features,
            }
        )
    return cases


def read_uploaded_frame(file_obj: Any) -> pd.DataFrame:
    name = str(file_obj.name).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    if name.endswith(".json"):
        return pd.read_json(file_obj)
    raise ValueError("Unsupported format. Use CSV or JSON.")
