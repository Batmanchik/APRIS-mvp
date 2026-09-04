"""
Anomaly Dashboard (Alert Inbox).

Primary analyst workspace: alert filtering, case selection, and dossier review.
"""
from __future__ import annotations

import base64
import hashlib
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from apris.crypto_ponzi.aggregator import aggregate_transactions
from apris.crypto_ponzi.tx_generator import generate_company_transactions
from apris.crypto_ponzi.visualizations import (
    plot_counterparty_network,
    plot_inflow_structure_pie,
)
from apris.data_generator import FEATURE_COLUMNS
from apris.frontend import api_client


st.set_page_config(page_title="Anomaly Dashboard | Cheops AI", page_icon="🚨", layout="wide")


st.markdown(
    """
    <style>
    .metric-card { background: var(--apris-surface); border: 1px solid var(--apris-border); border-radius: 12px; padding: 1.25rem; }
    .metric-val { font-size: 2rem; font-weight: 700; color: var(--apris-text); line-height: 1; margin-bottom: 0.25rem; }
    .metric-label { font-size: 0.85rem; color: var(--apris-text-secondary); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    .red-val { color: #ef4444 !important; }
    .dossier-header { background: #1a1a1a; color: white; padding: 1.5rem 2rem; border-radius: 12px 12px 0 0; margin-top: 2rem; }
    .dossier-body { border: 1px solid var(--apris-border); border-top: none; border-radius: 0 0 12px 12px; padding: 2rem; background: var(--apris-bg); }
    .risk-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-critical { background: #fee2e2; color: #991b1b; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-ok { background: #d1fae5; color: #065f46; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_figure_inline(fig: plt.Figure, width: str = "100%") -> None:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    st.markdown(
        f"<img src='data:image/png;base64,{encoded}' style='width:{width};height:auto;display:block;border-radius:12px;' />",
        unsafe_allow_html=True,
    )
    plt.close(fig)


def _risk_threshold(selection: str) -> float:
    if selection == "Medium+":
        return 0.45
    if selection == "High+":
        return 0.70
    if selection == "Critical":
        return 0.85
    return 0.0


def _risk_emoji(prob: float) -> str:
    if prob >= 0.85:
        return "🚨"
    if prob >= 0.70:
        return "⚠️"
    return "✅"


def _extract_model_features(entity_row: pd.Series) -> dict[str, float]:
    features: dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        value = entity_row.get(name, None)
        if value is None:
            continue
        try:
            features[name] = float(value)
        except (TypeError, ValueError):
            continue
    return features


def _build_explain_payload(case_id: str, features: dict[str, float]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "window_hours": 24,
        "events": [
            {
                "event_id": f"evt-{case_id}",
                "ts": datetime.now(timezone.utc).isoformat(),
                "amount": 100.0,
                "currency": "USD",
                "sender_id": f"src-{case_id}",
                "receiver_id": f"dst-{case_id}",
                "sender_type": "company",
                "receiver_type": "wallet",
                "channel": "legal",
                "jurisdiction": "KZ",
                "asset_type": "fiat",
            }
        ],
        "tabular_features": features,
    }


@st.cache_data(show_spinner=False, max_entries=200)
def _cached_crypto_case(selected_id: str, preset: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    fake_company_name = f"Smart Contract {selected_id[:6]}"
    seed = int(hashlib.sha256(selected_id.encode("utf-8")).hexdigest()[:8], 16)
    case = generate_company_transactions(preset, seed=seed)
    transactions = case["transactions"]
    metrics = aggregate_transactions(transactions)
    return fake_company_name, transactions, metrics


def _format_option(row: pd.Series) -> str:
    return (
        f"{_risk_emoji(float(row['risk_prob']))} {row['company_id']} - "
        f"{float(row['risk_prob']):.2%} [{row['risk_level']}]"
    )


st.title("🚨 Detected Anomalies Dashboard")

if "scan_results" not in st.session_state:
    st.info(
        "No scan data available. Open **📡 Transaction Scanner** and run a batch scan first."
    )
    if st.button("Open Scanner"):
        st.switch_page("pages/2_Transaction_Scanner.py")
    st.stop()

df = st.session_state["scan_results"].copy()
if df.empty:
    st.warning("The last scan produced no successful cases. Run a new scan.")
    if st.button("Open Scanner"):
        st.switch_page("pages/2_Transaction_Scanner.py")
    st.stop()

scan_time = st.session_state.get("last_scan_time", "Unknown")
scan_duration = st.session_state.get("last_scan_duration_s")
total_scanned = len(df)
critical = int((df["risk_prob"] >= 0.85).sum())
high = int(((df["risk_prob"] >= 0.70) & (df["risk_prob"] < 0.85)).sum())
medium = int(((df["risk_prob"] >= 0.45) & (df["risk_prob"] < 0.70)).sum())

if scan_duration is not None:
    st.markdown(f"**Last update:** {scan_time}  |  **Scan duration:** {scan_duration:.2f}s")
else:
    st.markdown(f"**Last update:** {scan_time}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        (
            f"<div class='metric-card'><div class='metric-val'>{total_scanned}</div>"
            "<div class='metric-label'>Entities Scanned</div></div>"
        ),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='metric-card'><div class='metric-val red-val'>{critical}</div><div class='metric-label'>Critical</div></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"<div class='metric-card'><div class='metric-val' style='color:#f97316;'>{high}</div><div class='metric-label'>High</div></div>",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"<div class='metric-card'><div class='metric-val' style='color:#eab308;'>{medium}</div><div class='metric-label'>Medium</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.subheader("📋 Alert Inbox")

f1, f2, f3 = st.columns([1.1, 1, 1.9])
with f1:
    risk_filter = st.selectbox("Risk Filter", ["All", "Medium+", "High+", "Critical"], index=1)
with f2:
    top_n = st.slider("Top N", min_value=20, max_value=500, value=100, step=20)
with f3:
    search_query = st.text_input("Search by ID", value="", placeholder="e.g. case-")

display_df = df.sort_values(by="risk_prob", ascending=False).copy()
display_df = display_df[display_df["risk_prob"] >= _risk_threshold(risk_filter)]
if search_query.strip():
    search_value = search_query.strip().lower()
    display_df = display_df[display_df["company_id"].astype(str).str.lower().str.contains(search_value)]

if display_df.empty:
    st.warning("No records match the current filters.")
    st.stop()

inbox_df = (
    display_df[["company_id", "risk_prob", "risk_level"]]
    .head(top_n)
    .rename(columns={"company_id": "entity_id", "risk_prob": "global_risk"})
)
st.dataframe(inbox_df, use_container_width=True, hide_index=True)

top_slice = display_df.head(top_n)
options = top_slice["company_id"].astype(str).tolist()
entity_lookup = {str(row["company_id"]): row for _, row in top_slice.iterrows()}

selected_id = st.selectbox(
    "Select Entity for Dossier",
    options=options,
    format_func=lambda case_id: _format_option(entity_lookup[case_id]),
)
entity_data = entity_lookup[selected_id]

prob = float(entity_data["risk_prob"])
badge_class = "badge-critical" if prob >= 0.70 else ("badge-warning" if prob >= 0.45 else "badge-ok")

st.markdown(
    f"""
    <div class='dossier-header'>
        <h2 style='margin:0; font-size:1.5rem; letter-spacing:1px;'>ENTITY DOSSIER: <code>{selected_id}</code></h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='dossier-body'>", unsafe_allow_html=True)
col_summary, col_ml = st.columns([1, 1.2])

with col_summary:
    st.markdown(
        f"#### System Verdict: <span class='risk-badge {badge_class}'>{entity_data['risk_level']} ({prob:.1%})</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Potential multi-channel fraud pattern detected based on route topology and "
        "counterparty concentration."
    )

    model_features = _extract_model_features(entity_data)
    explain_rendered = False
    if model_features:
        try:
            v2_explain = api_client.explain_case_v2(_build_explain_payload(selected_id, model_features))
            st.markdown("#### 🔎 Explain (v2)")
            st.info(v2_explain.get("summary", "Summary unavailable."))

            branch_scores = v2_explain.get("branch_scores", {})
            if branch_scores:
                st.caption("Branch scores")
                st.json(branch_scores)

            tabular_factors = v2_explain.get("tabular_factors", [])
            if tabular_factors:
                st.markdown("#### 🚩 Key Factors")
                for factor in tabular_factors[:3]:
                    name = factor.get("feature", "unknown")
                    value = float(factor.get("value", 0.0))
                    impact = float(factor.get("impact", 0.0))
                    st.markdown(f"**{name}**: value `{value:.3f}`, impact `{impact:.3f}`")
            explain_rendered = True
        except Exception:
            explain_rendered = False

    if not explain_rendered:
        fallback_features = entity_data.drop(
            ["company_id", "risk_prob", "risk_level", "pred_label", "label", "is_borderline"],
            errors="ignore",
        ).to_dict()
        try:
            explanations = api_client.explain_features(fallback_features, top_k=3)
            st.markdown("#### 🚩 Top-3 Indicators")
            for exp in explanations:
                feature = exp["feature"]
                st.markdown(
                    f"**{feature}**: importance `{exp['importance']:.3f}`"
                    f"  \n<small style='color: #6e6e80;'>Value: {float(fallback_features[feature]):.3f}</small>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.warning("Explainability API is temporarily unavailable.")

with col_ml:
    st.markdown("#### 🕸️ Route Visualization")
    preset = "PYRAMID" if prob >= 0.70 else ("DANGEROUS" if prob >= 0.45 else "SAFE")
    with st.spinner("Building transaction graph..."):
        fake_company_name, transactions, metrics = _cached_crypto_case(selected_id, preset)
        g1, g2 = st.columns(2)
        with g1:
            fig_network = plot_counterparty_network(transactions, company_name=fake_company_name)
            _render_figure_inline(fig_network)
            st.caption("Transfer structure")
        with g2:
            fig_pie = plot_inflow_structure_pie(metrics)
            _render_figure_inline(fig_pie)
            st.caption("Incoming flow concentration")

st.markdown("</div>", unsafe_allow_html=True)
