"""Transaction scanner page (batch ETL -> API v2 scoring)."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from apris.data_generator import FEATURE_COLUMNS, generate_live_batch
from apris.frontend import api_client
from apris.frontend.scanner_pipeline import (
    prepare_cases_from_feature_df,
    prepare_cases_from_tx_df,
    read_uploaded_frame,
)


st.set_page_config(page_title="Scanner | Cheops AI", page_icon="рџ“Ў", layout="wide")

st.title("рџ“Ў Mass Transaction Scanner")
st.caption("Batch scan through API v2 (/api/v2/score/batch). No local model inference in UI.")


def _render_api_unavailable(error: Exception) -> None:
    st.error(f"API unavailable: {error}")
    st.caption("Check backend (`scripts/app.ps1 status`) and `CHEOPS_API_BASE_URL` configuration.")
    if st.button("Retry API check", use_container_width=True):
        st.rerun()


def _risk_level(prob: float) -> str:
    if prob >= 0.85:
        return "Critical"
    if prob >= 0.70:
        return "High"
    if prob >= 0.45:
        return "Medium"
    return "Low"


def _build_cases(uploaded: Any, simulate: bool, batch_size: int) -> tuple[list[dict[str, Any]], int, str]:
    if simulate:
        synthetic_df, seed = generate_live_batch(total_n=batch_size, seed=42)
        return (
            prepare_cases_from_feature_df(
                synthetic_df,
                seed=seed,
                feature_columns=FEATURE_COLUMNS,
            ),
            seed,
            "synthetic",
        )

    if uploaded is None:
        raise ValueError("Upload CSV/JSON or enable simulation.")

    df = read_uploaded_frame(uploaded)
    seed = 42
    cols = set(df.columns)
    if set(FEATURE_COLUMNS).issubset(cols):
        return (
            prepare_cases_from_feature_df(
                df,
                seed=seed,
                feature_columns=FEATURE_COLUMNS,
            ),
            seed,
            "features",
        )

    required_tx = {"sender_id", "receiver_id", "amount", "timestamp"}
    if required_tx.issubset(cols):
        return prepare_cases_from_tx_df(df, seed=seed), seed, "transactions"

    raise ValueError(
        "Input file must contain either model feature columns or raw tx columns "
        "(sender_id, receiver_id, amount, timestamp)."
    )


def _render_summary(scored: pd.DataFrame, failed: int, mode: str, elapsed_seconds: float) -> None:
    total = int(len(scored))
    if scored.empty:
        critical = 0
        high = 0
        medium = 0
    else:
        critical = int((scored["risk_prob"] >= 0.85).sum())
        high = int(((scored["risk_prob"] >= 0.70) & (scored["risk_prob"] < 0.85)).sum())
        medium = int(((scored["risk_prob"] >= 0.45) & (scored["risk_prob"] < 0.70)).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scanned", total)
    c2.metric("Critical", critical)
    c3.metric("High", high)
    c4.metric("Medium", medium)
    c5.metric("Failures", failed)
    st.caption(f"Source mode: {mode} | Duration: {elapsed_seconds:.2f}s")


def _render_model_health_snapshot(details: dict[str, Any]) -> None:
    health = details.get("health", {})
    metrics = details.get("metrics", {})
    drift = details.get("drift", {})

    st.caption(
        "Model health: "
        f"status={details.get('status', 'unknown')}, "
        f"ready={health.get('ready', False)}, "
        f"model_version={health.get('model_version', 'n/a')}"
    )

    with st.expander("Model details (v2)", expanded=False):
        branch_modes = health.get("branch_modes", {})
        if branch_modes:
            st.write("Branch modes:")
            st.json(branch_modes)
        if metrics:
            st.write("Branch metrics:")
            st.json(metrics)
        if drift:
            st.write("Drift snapshot:")
            st.json(drift)


def _render_runtime_health_snapshot(runtime_payload: dict[str, Any]) -> None:
    runtime = runtime_payload.get("runtime", {})
    total_requests = int(runtime.get("requests_total", runtime.get("total_requests", 0)))
    total_errors = int(runtime.get("errors_total", runtime.get("total_errors", 0)))
    error_rate = float(runtime.get("error_rate_total", runtime.get("error_rate", 0.0)))

    st.caption(
        "Runtime health: "
        f"requests={total_requests}, "
        f"errors={total_errors}, "
        f"error_rate={error_rate:.2%}"
    )
    if error_rate >= 0.05 and total_requests > 20:
        st.warning("Runtime is degraded: elevated API error-rate detected.")

    endpoints = runtime.get("endpoints", {})
    if not endpoints:
        return

    focus_paths = ["/api/v2/score/batch", "/api/v2/score", "/api/v2/explain"]
    focus_rows: list[dict[str, Any]] = []
    for path in focus_paths:
        matched = [
            (endpoint_name, metrics)
            for endpoint_name, metrics in endpoints.items()
            if isinstance(metrics, dict) and str(endpoint_name).endswith(path)
        ]
        for endpoint_name, metrics in matched:
            focus_rows.append(
                {
                    "endpoint": endpoint_name,
                    "requests": int(metrics.get("requests", 0)),
                    "errors": int(metrics.get("errors", 0)),
                    "error_rate": float(metrics.get("error_rate", 0.0)),
                    "p95_ms": float(metrics.get("latency_p95_ms", 0.0)),
                }
            )

    if focus_rows:
        with st.expander("Runtime details (v2)", expanded=False):
            st.dataframe(pd.DataFrame(focus_rows), use_container_width=True, hide_index=True)


uploaded_file = st.file_uploader("Upload transactions or features (CSV/JSON)", type=["csv", "json"])
simulate_scan = st.checkbox("Use synthetic simulation", value=True)
batch_size = st.slider("Simulation entity count", min_value=500, max_value=5000, value=1500, step=500)

if st.button("Run batch scan", type="primary", use_container_width=True):
    started_at = time.perf_counter()
    progress = st.progress(0, text="Checking API availability")
    try:
        api_client.health_check_v2_model()
        model_details: dict[str, Any] = {}
        runtime_details: dict[str, Any] = {}
        try:
            model_details = api_client.health_check_v2_model_details()
        except Exception:
            model_details = {}
        try:
            runtime_details = api_client.health_check_v2_runtime()
        except Exception:
            runtime_details = {}
    except api_client.ApiClientError as exc:
        progress.empty()
        _render_api_unavailable(exc)
        st.stop()
    except Exception as exc:
        progress.empty()
        st.error(f"Unexpected health-check failure: {exc}")
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

        case_map = {case["case_id"]: case for case in cases}
        rows: list[dict[str, Any]] = []
        for item in results:
            case_id = item.get("case_id")
            case = case_map.get(str(case_id))
            if case is None:
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

        if rows:
            scored_df = pd.DataFrame(rows).sort_values("risk_prob", ascending=False).reset_index(drop=True)
        else:
            scored_df = pd.DataFrame(columns=["company_id", "risk_prob", "risk_level"])
        st.session_state["scan_results"] = scored_df
        st.session_state["last_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["last_scan_seed"] = seed_used
        st.session_state["last_scan_duration_s"] = time.perf_counter() - started_at

        progress.progress(100, text="Completed")
        _render_summary(
            scored=scored_df,
            failed=len(failures),
            mode=source_mode,
            elapsed_seconds=float(st.session_state["last_scan_duration_s"]),
        )

        if model_details:
            _render_model_health_snapshot(model_details)
        if runtime_details:
            _render_runtime_health_snapshot(runtime_details)

        if failures:
            failure_ratio = float(len(failures) / max(1, len(cases)))
            if failure_ratio >= 0.5:
                st.warning(
                    "Batch finished in degraded mode: "
                    f"{len(failures)}/{len(cases)} cases failed."
                )
            with st.expander("Batch failures", expanded=False):
                st.dataframe(pd.DataFrame(failures), use_container_width=True)

        st.success("Scan finished. Open the anomaly dashboard to inspect dossiers.")
        if st.button("Open anomaly dashboard"):
            st.switch_page("pages/1_Anomaly_Dashboard.py")

    except api_client.ApiClientError as exc:
        progress.empty()
        _render_api_unavailable(exc)
    except Exception as exc:
        progress.empty()
        st.error(f"Scan failed: {exc}")
