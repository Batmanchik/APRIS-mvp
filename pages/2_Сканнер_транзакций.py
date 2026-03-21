"""Transaction scanner page (batch ETL -> API v2 scoring)."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from apris.data_generator import FEATURE_COLUMNS, generate_live_batch
from apris.etl import aggregate_to_operational
from apris.frontend import api_client
from apris.risk_engine import operational_to_features


st.set_page_config(page_title="Scanner | Cheops AI", page_icon="📡", layout="wide")

st.title("📡 Mass Transaction Scanner")
st.caption("Batch scan through API v2 (/api/v2/score/batch). No local model inference in UI.")


def _risk_level(prob: float) -> str:
    if prob >= 0.85:
        return "Critical"
    if prob >= 0.70:
        return "High"
    if prob >= 0.45:
        return "Medium"
    return "Low"


def _deterministic_case_id(prefix: str, index: int, seed: int, payload: dict[str, Any]) -> str:
    fingerprint = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(f"{prefix}|{index}|{seed}|{fingerprint}".encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _event_stub(case_id: str, base_amount: float) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "event_id": f"evt-{case_id}",
        "ts": now.isoformat(),
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


def _build_case_from_features(feature_payload: dict[str, float], index: int, seed: int) -> dict[str, Any]:
    case_id = _deterministic_case_id("case", index, seed, feature_payload)
    return {
        "case_id": case_id,
        "window_hours": 24,
        "events": [_event_stub(case_id, base_amount=100.0)],
        "tabular_features": feature_payload,
    }


def _prepare_cases_from_feature_df(df: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    feature_rows = df[FEATURE_COLUMNS].astype(float).to_dict(orient="records")
    return [_build_case_from_features(row, index=i, seed=seed) for i, row in enumerate(feature_rows)]


def _prepare_cases_from_tx_df(tx_df: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    work_df = tx_df.copy()
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], errors="coerce")
    work_df = work_df.dropna(subset=["timestamp"])

    entities = sorted(set(work_df["sender_id"].astype(str)) | set(work_df["receiver_id"].astype(str)))
    cases: list[dict[str, Any]] = []
    for idx, entity_id in enumerate(entities):
        ops = aggregate_to_operational(work_df, target_entity_id=entity_id)
        features = operational_to_features(ops)
        case_id = _deterministic_case_id("entity", idx, seed, {"entity_id": entity_id, **features})

        recent = work_df[
            (work_df["sender_id"].astype(str) == entity_id)
            | (work_df["receiver_id"].astype(str) == entity_id)
        ].sort_values("timestamp", ascending=False)

        events: list[dict[str, Any]] = []
        for event_idx, row in enumerate(recent.head(6).itertuples(index=False), start=1):
            ts = row.timestamp.to_pydatetime()
            events.append(
                {
                    "event_id": f"evt-{case_id}-{event_idx}",
                    "ts": ts.astimezone(timezone.utc).isoformat() if ts.tzinfo else ts.replace(tzinfo=timezone.utc).isoformat(),
                    "amount": float(max(getattr(row, "amount"), 1.0)),
                    "currency": "USD",
                    "sender_id": str(getattr(row, "sender_id")),
                    "receiver_id": str(getattr(row, "receiver_id")),
                    "sender_type": "company",
                    "receiver_type": "wallet",
                    "channel": "legal",
                    "jurisdiction": "KZ",
                    "asset_type": "fiat",
                }
            )

        if not events:
            events = [_event_stub(case_id, base_amount=float(ops["incoming_funds"]))]

        cases.append(
            {
                "case_id": case_id,
                "window_hours": 24,
                "events": events,
                "tabular_features": features,
            }
        )
    return cases


def _read_uploaded(file_obj: Any) -> pd.DataFrame:
    name = str(file_obj.name).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file_obj)
    if name.endswith(".json"):
        return pd.read_json(file_obj)
    raise ValueError("Unsupported format. Use CSV or JSON.")


def _build_cases(uploaded: Any, simulate: bool, batch_size: int) -> tuple[list[dict[str, Any]], int, str]:
    if simulate:
        synthetic_df, seed = generate_live_batch(total_n=batch_size, seed=42)
        return _prepare_cases_from_feature_df(synthetic_df, seed=seed), seed, "synthetic"

    if uploaded is None:
        raise ValueError("Upload CSV/JSON or enable simulation.")

    df = _read_uploaded(uploaded)
    seed = 42
    cols = set(df.columns)
    if set(FEATURE_COLUMNS).issubset(cols):
        return _prepare_cases_from_feature_df(df, seed=seed), seed, "features"

    required_tx = {"sender_id", "receiver_id", "amount", "timestamp"}
    if required_tx.issubset(cols):
        return _prepare_cases_from_tx_df(df, seed=seed), seed, "transactions"

    raise ValueError(
        "Input file must contain either model feature columns or raw tx columns "
        "(sender_id, receiver_id, amount, timestamp)."
    )


def _render_summary(scored: pd.DataFrame, failed: int, mode: str) -> None:
    total = int(len(scored))
    critical = int((scored["risk_prob"] >= 0.85).sum())
    high = int(((scored["risk_prob"] >= 0.70) & (scored["risk_prob"] < 0.85)).sum())
    medium = int(((scored["risk_prob"] >= 0.45) & (scored["risk_prob"] < 0.70)).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scanned", total)
    c2.metric("Critical", critical)
    c3.metric("High", high)
    c4.metric("Medium", medium)
    c5.metric("Failures", failed)
    st.caption(f"Source mode: {mode}")


uploaded_file = st.file_uploader("Upload transactions or features (CSV/JSON)", type=["csv", "json"])
simulate_scan = st.checkbox("Use synthetic simulation", value=True)
batch_size = st.slider("Simulation entity count", min_value=500, max_value=5000, value=1500, step=500)

if st.button("Run batch scan", type="primary", use_container_width=True):
    progress = st.progress(0, text="Checking API availability")
    try:
        api_client.health_check_v2_model()
    except Exception as exc:
        progress.empty()
        st.error(f"API unavailable. Start backend first. Details: {exc}")
        st.stop()

    try:
        progress.progress(10, text="Preparing cases")
        cases, seed_used, source_mode = _build_cases(uploaded_file, simulate_scan, batch_size)
        if not cases:
            raise ValueError("No cases prepared from selected source.")

        progress.progress(45, text=f"Scoring {len(cases)} cases via API v2 batch")
        response = api_client.score_batch_v2(cases)

        progress.progress(75, text="Building dashboard dataset")
        results = response.get("results", [])
        failures = response.get("failures", [])

        score_map = {row["case_id"]: row for row in results}
        rows: list[dict[str, Any]] = []
        for case in cases:
            item = score_map.get(case["case_id"])
            if item is None:
                continue
            prob = float(item["global_risk"])
            row: dict[str, Any] = {
                "company_id": case["case_id"],
                "risk_prob": prob,
                "risk_level": _risk_level(prob),
                "risk_band": item["risk_band"],
                "model_version": item["model_version"],
                "calibration_version": item["calibration_version"],
            }
            for feature_name, feature_value in case.get("tabular_features", {}).items():
                row[feature_name] = float(feature_value)
            for typology, value in item.get("typology_probs", {}).items():
                row[f"typology_{typology.lower()}"] = float(value)
            rows.append(row)

        scored_df = pd.DataFrame(rows).sort_values("risk_prob", ascending=False).reset_index(drop=True)
        st.session_state["scan_results"] = scored_df
        st.session_state["last_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["last_scan_seed"] = seed_used

        progress.progress(100, text="Completed")
        _render_summary(scored_df, failed=len(failures), mode=source_mode)

        if failures:
            with st.expander("Batch failures", expanded=False):
                st.dataframe(pd.DataFrame(failures), use_container_width=True)

        st.success("Scan finished. Open the anomaly dashboard to inspect dossiers.")
        if st.button("Open anomaly dashboard"):
            st.switch_page("pages/1_Дашборд_аномалий.py")

    except Exception as exc:
        progress.empty()
        st.error(f"Scan failed: {exc}")
