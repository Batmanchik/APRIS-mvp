from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, RISK_THRESHOLDS, generate_live_batch
from graph_module import build_transaction_graph, compute_hub_metrics, plot_graph
from population_map import (
    build_population_map_figure,
    check_feature_order,
    check_no_nan,
    check_pca_dimensions,
    fit_population_pca,
    load_feature_names as load_population_feature_names,
    load_population_dataset,
    project_current_case,
)
from risk_engine import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    OPERATIONAL_INPUT_BOUNDS,
    explain,
    load_artifacts,
    operational_to_features,
    predict_risk,
)


PRESETS: dict[str, dict[str, float]] = {
    "Legit": {
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
    "Suspicious": {
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
    "Pyramid": {
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

OPERATIONAL_PRESETS: dict[str, dict[str, float]] = {
    "Legit": {
        "tx_count_total": 8000.0,
        "unique_counterparties": 2200.0,
        "new_clients_current": 260.0,
        "new_clients_previous": 245.0,
        "referred_clients_current": 70.0,
        "incoming_funds": 1_800_000.0,
        "payouts_total": 1_050_000.0,
        "top1_wallet_share": 0.18,
        "top10_wallet_share": 0.46,
        "avg_holding_days": 62.0,
        "repeat_investor_share": 0.44,
        "max_referral_depth": 4.0,
    },
    "Suspicious": {
        "tx_count_total": 12000.0,
        "unique_counterparties": 1800.0,
        "new_clients_current": 420.0,
        "new_clients_previous": 300.0,
        "referred_clients_current": 190.0,
        "incoming_funds": 2_000_000.0,
        "payouts_total": 1_200_000.0,
        "top1_wallet_share": 0.32,
        "top10_wallet_share": 0.62,
        "avg_holding_days": 34.0,
        "repeat_investor_share": 0.60,
        "max_referral_depth": 8.0,
    },
    "Pyramid": {
        "tx_count_total": 32000.0,
        "unique_counterparties": 1300.0,
        "new_clients_current": 1500.0,
        "new_clients_previous": 650.0,
        "referred_clients_current": 1220.0,
        "incoming_funds": 3_800_000.0,
        "payouts_total": 4_500_000.0,
        "top1_wallet_share": 0.72,
        "top10_wallet_share": 0.92,
        "avg_holding_days": 15.0,
        "repeat_investor_share": 0.86,
        "max_referral_depth": 13.0,
    },
}

PRIMARY_COLOR = "#0B6E99"
SECONDARY_COLOR = "#1F9E89"
ALERT_COLOR = "#C2410C"
LEGIT_COLOR = "#1D4ED8"
PYRAMID_COLOR = "#DC2626"

FEATURE_LABELS = {
    "growth_rate": "growth_rate",
    "referral_ratio": "referral_ratio",
    "payout_dependency": "payout_dependency",
    "centralization_index": "centralization_index",
    "avg_holding_time": "avg_holding_time (days, observed window)",
    "reinvestment_rate": "reinvestment_rate",
    "gini_coefficient": "gini_coefficient",
    "transaction_entropy": "transaction_entropy",
    "structural_depth": "structural_depth",
}

OPERATIONAL_LABELS = {
    "tx_count_total": "transactions_count_total",
    "unique_counterparties": "unique_counterparties",
    "new_clients_current": "new_clients_current_period",
    "new_clients_previous": "new_clients_previous_period",
    "referred_clients_current": "referred_clients_current_period",
    "incoming_funds": "incoming_funds_total",
    "payouts_total": "payouts_total",
    "top1_wallet_share": "top1_wallet_share",
    "top10_wallet_share": "top10_wallet_share",
    "avg_holding_days": "avg_holding_days",
    "repeat_investor_share": "repeat_investor_share",
    "max_referral_depth": "max_referral_depth",
}


def _guard_streamlit_entrypoint() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is None:
            print("This is a Streamlit app.")
            print("Run it with: streamlit run app.py")
            raise SystemExit(0)
    except Exception:
        return


def _set_style() -> None:
    st.markdown(
        """
        <style>
            html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
                color: #0f172a !important;
            }
            .stApp {
                background:
                    radial-gradient(1200px 500px at -20% -10%, #d1ecff 0%, transparent 60%),
                    radial-gradient(900px 480px at 120% 0%, #ffe9d4 0%, transparent 50%),
                    linear-gradient(180deg, #f8fbff 0%, #eef5fb 100%);
            }
            header[data-testid="stHeader"] {
                background: rgba(248, 251, 255, 0.88) !important;
                border-bottom: 1px solid rgba(15, 23, 42, 0.08);
            }
            [data-testid="collapsedControl"] {
                display: none;
            }
            [data-testid="stSidebar"] {
                display: none;
            }
            div.stButton > button, div.stFormSubmitButton > button {
                background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%) !important;
                color: #0f172a !important;
                border: 1px solid #94a3b8 !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                min-height: 40px !important;
            }
            div.stButton > button:hover, div.stFormSubmitButton > button:hover {
                border-color: #0b6e99 !important;
                color: #0b6e99 !important;
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"],
            div.stButton > button[kind="primary"] {
                background: linear-gradient(180deg, #ef4444 0%, #dc2626 100%) !important;
                color: #ffffff !important;
                border: 1px solid #b91c1c !important;
            }
            [data-testid="stMetric"] {
                background: rgba(255,255,255,0.80);
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 12px;
                padding: 8px 10px;
            }
            [data-testid="stMetricLabel"] {
                color: #475569 !important;
            }
            [data-testid="stMetricValue"] {
                color: #0f172a !important;
                font-weight: 800 !important;
            }
            [data-baseweb="tab-list"] button {
                color: #0f172a !important;
                font-weight: 600;
            }
            [data-baseweb="tab-panel"] {
                background: rgba(255,255,255,0.60);
                border-radius: 12px;
                padding: 10px 12px 16px 12px;
            }
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                background: #ffffff !important;
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 10px;
            }
            .apris-card {
                background: rgba(255,255,255,0.86);
                border: 1px solid rgba(15, 23, 42, 0.10);
                border-radius: 14px;
                padding: 14px 16px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            }
            .apris-chip {
                display: inline-block;
                background: #0b6e9920;
                color: #0b6e99;
                border: 1px solid #0b6e9944;
                border-radius: 999px;
                padding: 4px 10px;
                margin-right: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            .risk-banner {
                border-radius: 12px;
                padding: 12px 14px;
                margin-bottom: 8px;
                border: 1px solid rgba(15, 23, 42, 0.12);
                font-weight: 600;
            }
            .risk-banner-low {
                background: #dcfce7;
                color: #166534 !important;
                border-color: #86efac;
            }
            .risk-banner-medium {
                background: #ffedd5;
                color: #9a3412 !important;
                border-color: #fdba74;
            }
            .risk-banner-high {
                background: #fee2e2;
                color: #991b1b !important;
                border-color: #fca5a5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_matplotlib_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#E2E8F0",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#334155",
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def _ensure_model_available() -> tuple[Any, list[str]]:
    if not Path(MODEL_PATH).exists() or not Path(FEATURE_NAMES_PATH).exists():
        st.error("Model not found. Run training first.")
        st.code("python train_model.py")
        st.stop()
    try:
        return load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model artifacts: {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def _load_population_df_cached() -> pd.DataFrame:
    return load_population_dataset()


@st.cache_resource(show_spinner=False)
def _load_population_projection_cached() -> tuple[pd.DataFrame, Any, Any, list[str]]:
    feature_names = load_population_feature_names()
    dataset = load_population_dataset()
    check_no_nan(dataset, feature_names)
    if not check_feature_order(feature_names):
        raise ValueError("feature_names.json order does not match expected model feature schema.")
    projected, scaler, pca = fit_population_pca(dataset, feature_names)
    check_pca_dimensions(projected, pca)
    return projected, scaler, pca, feature_names


def _ensure_defaults() -> None:
    for feature in FEATURE_COLUMNS:
        if feature not in st.session_state:
            st.session_state[feature] = PRESETS["Suspicious"][feature]
    for name, val in OPERATIONAL_PRESETS["Suspicious"].items():
        if name not in st.session_state:
            st.session_state[name] = val
    if "input_source_mode" not in st.session_state:
        st.session_state["input_source_mode"] = "Operational facts (recommended)"


def _set_preset(name: str) -> None:
    for feature in FEATURE_COLUMNS:
        st.session_state[feature] = float(PRESETS[name][feature])
    for key, value in OPERATIONAL_PRESETS[name].items():
        st.session_state[key] = float(value)


def _risk_band(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "High"
    if prob >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def _risk_css_class(level: str) -> str:
    level_norm = level.lower()
    if level_norm == "low":
        return "risk-banner risk-banner-low"
    if level_norm == "medium":
        return "risk-banner risk-banner-medium"
    return "risk-banner risk-banner-high"


def _sample_population_case(dataset: pd.DataFrame) -> dict[str, float]:
    row = dataset.sample(n=1).iloc[0]
    return {feature: float(row[feature]) for feature in FEATURE_COLUMNS}


def _score_live_batch(df: pd.DataFrame, model: Any) -> dict[str, Any]:
    x = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).copy()
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)

    scored = df.copy()
    scored["risk_prob"] = proba
    scored["pred_label"] = pred
    scored["risk_level"] = scored["risk_prob"].apply(_risk_band)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "recall_pyramid": float(recall),
        "precision_pyramid": float(precision),
        "confusion_matrix": cm.tolist(),
    }
    return {"scored": scored, "metrics": metrics}


def _render_input_guide() -> None:
    with st.expander("How to input data", expanded=False):
        st.markdown(
            """
            1. Use **Operational facts (recommended)** if you have transaction facts from monitoring systems.
            2. Fill values for current period.
            3. Click **Score Case**.
            4. Check:
            - Risk probability and risk level
            - Dynamic case-signal chart
            - Transaction graph concentration metrics
            """
        )
        st.caption(
            "Validation rules: referred_clients_current <= new_clients_current, "
            "top1_wallet_share <= top10_wallet_share."
        )


def _render_operational_input_main() -> tuple[dict[str, float], dict[str, float]]:
    st.markdown("#### Operational facts")
    cols = st.columns(3)
    raw: dict[str, float] = {}
    int_fields = {
        "tx_count_total",
        "unique_counterparties",
        "new_clients_current",
        "new_clients_previous",
        "referred_clients_current",
        "max_referral_depth",
    }
    money_fields = {"incoming_funds", "payouts_total"}

    for idx, (key, (low, high)) in enumerate(OPERATIONAL_INPUT_BOUNDS.items()):
        col = cols[idx % 3]
        with col:
            label = OPERATIONAL_LABELS.get(key, key)
            current = float(st.session_state[key])
            if key in int_fields:
                val = st.number_input(
                    label,
                    min_value=int(low),
                    max_value=int(high),
                    value=int(round(current)),
                    step=1,
                    format="%d",
                    key=f"op_{key}",
                )
            elif key in money_fields:
                val = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(current),
                    step=1000.0,
                    format="%.2f",
                    key=f"op_{key}",
                )
            else:
                val = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(current),
                    step=0.0001,
                    format="%.4f",
                    key=f"op_{key}",
                )
            st.session_state[key] = float(val)
            raw[key] = float(st.session_state[key])

    derived = operational_to_features(raw)
    return raw, derived


def _render_feature_input_main() -> dict[str, float]:
    st.markdown("#### Model features (advanced)")
    mode = st.radio(
        "Input precision",
        ["Precise inputs", "Quick sliders"],
        horizontal=True,
        index=0,
        key="feature_input_mode",
    )
    cols = st.columns(3)
    features: dict[str, float] = {}
    for idx, feature in enumerate(FEATURE_COLUMNS):
        low, high = FEATURE_BOUNDS[feature]
        current = float(st.session_state[feature])
        label = FEATURE_LABELS[feature]
        with cols[idx % 3]:
            if feature == "structural_depth":
                if mode == "Precise inputs":
                    val = st.number_input(
                        label,
                        min_value=int(low),
                        max_value=int(high),
                        value=int(round(current)),
                        step=1,
                        format="%d",
                        key=f"adv_{feature}",
                    )
                else:
                    val = st.slider(
                        label,
                        min_value=int(low),
                        max_value=int(high),
                        value=int(round(current)),
                        step=1,
                        key=f"adv_{feature}",
                    )
            else:
                precise_step = 0.1 if feature == "avg_holding_time" else 0.0001
                precise_format = "%.1f" if feature == "avg_holding_time" else "%.4f"
                slider_step = 0.1 if feature == "avg_holding_time" else 0.01
                if mode == "Precise inputs":
                    val = st.number_input(
                        label,
                        min_value=float(low),
                        max_value=float(high),
                        value=float(current),
                        step=float(precise_step),
                        format=precise_format,
                        key=f"adv_{feature}",
                    )
                else:
                    val = st.slider(
                        label,
                        min_value=float(low),
                        max_value=float(high),
                        value=float(current),
                        step=float(slider_step),
                        key=f"adv_{feature}",
                    )
            st.session_state[feature] = float(val)
            features[feature] = float(val)
    return features


def _plot_case_signal_breakdown(
    features: dict[str, float],
    feature_names: list[str],
    importances: np.ndarray,
) -> None:
    importance_map = {f: float(w) for f, w in zip(feature_names, importances)}
    baseline = PRESETS["Legit"]
    rows: list[tuple[str, float]] = []
    for feature in FEATURE_COLUMNS:
        low, high = FEATURE_BOUNDS[feature]
        span = max(1e-9, float(high - low))
        deviation = abs(float(features[feature]) - float(baseline[feature])) / span
        weighted = deviation * importance_map.get(feature, 0.0)
        rows.append((feature, weighted))
    rows.sort(key=lambda x: x[1], reverse=True)

    names = [x[0] for x in rows]
    values = [x[1] for x in rows]
    colors = []
    for idx, _ in enumerate(names):
        if idx == 0:
            colors.append("#DC2626")
        elif idx == 1:
            colors.append("#EA580C")
        elif idx == 2:
            colors.append("#F59E0B")
        else:
            colors.append(PRIMARY_COLOR)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.barh(names, values, color=colors)
    ax.invert_yaxis()
    ax.set_title("Case Signal Breakdown (dynamic)")
    ax.set_xlabel("Weighted deviation from legit baseline")

    for idx in range(min(3, len(values))):
        ax.text(
            values[idx] + max(values) * 0.01 if max(values) > 0 else 0.001,
            idx,
            "TOP",
            va="center",
            fontsize=9,
            color="#991B1B",
            fontweight="bold",
        )

    st.pyplot(fig, clear_figure=True)
    st.caption("Weighted deviation from legit baseline. Top-3 signals are highlighted.")


def _plot_global_feature_importance(feature_names: list[str], importances: np.ndarray) -> None:
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in ranked]
    values = [float(x[1]) for x in ranked]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.barh(names, values, color="#2563eb")
    ax.invert_yaxis()
    ax.set_title("Global Feature Importance (model-wide)")
    ax.set_xlabel("importance")
    st.pyplot(fig, clear_figure=True)


def _plot_feature_gallery(scored: pd.DataFrame, marker_features: dict[str, float] | None = None) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, feature in enumerate(FEATURE_COLUMNS):
        ax = axes[idx]
        legit = scored[scored["label"] == 0][feature].astype(float)
        pyramid = scored[scored["label"] == 1][feature].astype(float)
        ax.hist(legit, bins=24, alpha=0.55, color=LEGIT_COLOR, density=True, label="legit")
        ax.hist(pyramid, bins=24, alpha=0.48, color=PYRAMID_COLOR, density=True, label="pyramid")
        if marker_features is not None and feature in marker_features:
            ax.axvline(float(marker_features[feature]), color=ALERT_COLOR, linestyle="--", linewidth=2.0)
        ax.set_title(feature, fontsize=10)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Live Feature Gallery: one chart per variable", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    st.pyplot(fig, clear_figure=True)


def _plot_live_confusion_and_scores(scored: pd.DataFrame, cm: list[list[int]]) -> None:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        matrix = np.array(cm, dtype=float)
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title("Live Batch Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], labels=["legit", "pyramid"])
        ax.set_yticks([0, 1], labels=["legit", "pyramid"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#0f172a")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        legit_probs = scored[scored["label"] == 0]["risk_prob"]
        pyramid_probs = scored[scored["label"] == 1]["risk_prob"]
        ax.hist(legit_probs, bins=24, alpha=0.55, color=LEGIT_COLOR, label="legit", density=True)
        ax.hist(pyramid_probs, bins=24, alpha=0.52, color=PYRAMID_COLOR, label="pyramid", density=True)
        ax.axvline(RISK_THRESHOLDS["medium"], color=ALERT_COLOR, linestyle="--", linewidth=1.8)
        ax.axvline(RISK_THRESHOLDS["high"], color=ALERT_COLOR, linestyle="-.", linewidth=1.8)
        ax.set_title("Risk Probability Distribution (live batch)")
        ax.set_xlabel("P(pyramid)")
        ax.legend(frameon=False)
        st.pyplot(fig, clear_figure=True)


def _plot_stage_summary(scored: pd.DataFrame) -> None:
    counts = scored["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    ax.bar(counts.index, counts.values, color=[SECONDARY_COLOR, "#D97706", ALERT_COLOR], width=0.58)
    ax.set_title("Live Batch Risk Levels")
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def main() -> None:
    st.set_page_config(page_title="APRIS Stage Demo", layout="wide", initial_sidebar_state="collapsed")
    _set_style()
    _apply_matplotlib_theme()

    model, feature_names = _ensure_model_available()
    _ensure_defaults()
    population_error: str | None = None
    population_df = pd.DataFrame()
    projected_population = pd.DataFrame()
    pop_scaler: Any | None = None
    pop_pca: Any | None = None
    try:
        population_df = _load_population_df_cached()
        projected_population, pop_scaler, pop_pca, pop_feature_names = _load_population_projection_cached()
        if pop_feature_names != feature_names:
            raise ValueError("Feature order mismatch between model and population map artifacts.")
    except Exception as exc:
        population_error = str(exc)

    st.markdown(
        "<div class='apris-card'><h1 style='margin:0;'>APRIS Live Demonstrator</h1>"
        "<p style='margin:8px 0 0 0;'>Operational-first case scoring for financial security analysts.</p>"
        "<span class='apris-chip'>Local model</span>"
        "<span class='apris-chip'>No external API</span>"
        "<span class='apris-chip'>Live proof mode</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "All data shown in this app is synthetic (generated locally). "
        "No real personal or financial records are used."
    )
    st.caption(
        "This is a decision-support prototype: it helps prioritize cases, "
        "but does not replace formal investigation."
    )

    tab_single, tab_live = st.tabs(["Single Object Scoring", "Live Stage Proof"])

    with tab_single:
        st.subheader("Quick presets")
        p1, p2, p3, p4 = st.columns(4)
        if p1.button("Preset: Legit", use_container_width=True):
            _set_preset("Legit")
        if p2.button("Preset: Suspicious", use_container_width=True):
            _set_preset("Suspicious")
        if p3.button("Preset: Pyramid", use_container_width=True):
            _set_preset("Pyramid")
        if p4.button("Generate Random Case", use_container_width=True):
            if population_df.empty:
                st.warning("Population dataset unavailable for random case generation.")
            else:
                sampled_features = _sample_population_case(population_df)
                for feature in FEATURE_COLUMNS:
                    st.session_state[feature] = float(sampled_features[feature])
                st.session_state["input_source_mode"] = "Model features (advanced)"
                st.session_state["single_result"] = predict_risk(
                    sampled_features, model=model, feature_names=feature_names
                )
                st.session_state["single_explain"] = explain(
                    sampled_features, model=model, feature_names=feature_names
                )
                st.session_state["single_features"] = sampled_features
                st.session_state["single_raw_operational"] = None
                st.success("Random synthetic case generated and scored.")

        _render_input_guide()
        input_source = st.radio(
            "Input source",
            ["Operational facts (recommended)", "Model features (advanced)"],
            horizontal=True,
            key="input_source_mode",
        )

        if input_source == "Operational facts (recommended)":
            raw_operational, features = _render_operational_input_main()
        else:
            raw_operational, features = None, _render_feature_input_main()

        if st.button("Score Case", type="primary", use_container_width=True):
            try:
                st.session_state["single_result"] = predict_risk(
                    features, model=model, feature_names=feature_names
                )
                st.session_state["single_explain"] = explain(
                    features, model=model, feature_names=feature_names
                )
                st.session_state["single_features"] = features
                st.session_state["single_raw_operational"] = raw_operational
            except Exception as exc:
                st.error(f"Input validation error: {exc}")

        if "single_result" in st.session_state:
            result = st.session_state["single_result"]
            object_features = st.session_state["single_features"]
            top_features = st.session_state["single_explain"]
            raw_operational_view = st.session_state.get("single_raw_operational")

            level = str(result["label_text"])
            risk_prob = float(result["prob"])
            st.markdown(
                (
                    f"<div class='{_risk_css_class(level)}'>"
                    f"Risk probability: {risk_prob:.3f} | Risk level: {level}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk probability", f"{risk_prob:.3f}")
            m2.metric("Risk level", level)
            m3.metric("Thresholds", f"{RISK_THRESHOLDS['medium']:.1f} / {RISK_THRESHOLDS['high']:.1f}")

            st.write("Top features (global importance):")
            st.table(pd.DataFrame(top_features))

            if raw_operational_view is not None:
                with st.expander("Derived model features from operational facts", expanded=False):
                    st.table(
                        pd.DataFrame(
                            [{"feature": k, "value": float(v)} for k, v in object_features.items()]
                        )
                    )

            c1, c2 = st.columns(2)
            with c1:
                _plot_case_signal_breakdown(
                    object_features,
                    feature_names,
                    np.asarray(model.feature_importances_, dtype=float),
                )
                with st.expander("Model-wide global importance (reference)", expanded=False):
                    _plot_global_feature_importance(
                        feature_names,
                        np.asarray(model.feature_importances_, dtype=float),
                    )
            with c2:
                graph = build_transaction_graph(object_features)
                graph_metrics = compute_hub_metrics(graph)
                st.pyplot(plot_graph(graph, hub_node=int(graph_metrics["hub_node"])), clear_figure=True)
                g1, g2 = st.columns(2)
                g1.metric("hub_in_degree_share", f"{graph_metrics['hub_in_degree_share']:.3f}")
                g2.metric("hub_betweenness", f"{graph_metrics['hub_betweenness']:.3f}")
                st.caption("High hub concentration may indicate centralized payout structure.")

            st.markdown("#### Population Risk Map")
            if population_error is not None or pop_scaler is None or pop_pca is None:
                st.warning(f"Population map unavailable: {population_error}")
            else:
                current_point = project_current_case(
                    object_features,
                    feature_names,
                    pop_scaler,
                    pop_pca,
                )
                pop_fig = build_population_map_figure(projected_population, current_point=current_point)
                st.pyplot(pop_fig, clear_figure=True)
                st.caption("PCA projection of synthetic population with current case overlay.")
        else:
            st.info("Fill values and click 'Score Case'.")

    with tab_live:
        st.subheader("On-stage synthetic generation")
        controls1, controls2, controls3 = st.columns([1, 1, 2])
        with controls1:
            batch_size = st.number_input(
                "Batch size",
                min_value=200,
                max_value=5000,
                value=800,
                step=100,
            )
        with controls2:
            use_random_seed = st.checkbox("Random seed each run", value=True)
            manual_seed = st.number_input("Manual seed", min_value=1, max_value=2**31 - 1, value=42)
        with controls3:
            st.markdown(
                "<div class='apris-card'><b>Live-proof script:</b> "
                "Generate fresh synthetic organizations, score with trained model, render updated charts immediately."
                "</div>",
                unsafe_allow_html=True,
            )

        if st.button("Generate New Synthetic Batch Now", type="primary"):
            selected_seed = None if use_random_seed else int(manual_seed)
            live_df, used_seed = generate_live_batch(total_n=int(batch_size), seed=selected_seed)
            payload = _score_live_batch(live_df, model=model)
            st.session_state["live_scored"] = payload["scored"]
            st.session_state["live_metrics"] = payload["metrics"]
            st.session_state["live_seed"] = used_seed

        if "live_scored" not in st.session_state:
            selected_seed = None if use_random_seed else int(manual_seed)
            live_df, used_seed = generate_live_batch(total_n=int(batch_size), seed=selected_seed)
            payload = _score_live_batch(live_df, model=model)
            st.session_state["live_scored"] = payload["scored"]
            st.session_state["live_metrics"] = payload["metrics"]
            st.session_state["live_seed"] = used_seed

        live_scored: pd.DataFrame = st.session_state["live_scored"]
        live_metrics: dict[str, float | list[list[int]]] = st.session_state["live_metrics"]
        live_seed = st.session_state["live_seed"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Seed used", str(live_seed))
        k2.metric("Recall (pyramid)", f"{float(live_metrics['recall_pyramid']):.3f}")
        k3.metric("ROC-AUC", f"{float(live_metrics['roc_auc']):.3f}")
        k4.metric("Accuracy", f"{float(live_metrics['accuracy']):.3f}")

        _plot_live_confusion_and_scores(live_scored, live_metrics["confusion_matrix"])  # type: ignore[arg-type]
        _plot_stage_summary(live_scored)

        marker = st.session_state.get("single_features")
        _plot_feature_gallery(live_scored, marker_features=marker)

        preview_cols = FEATURE_COLUMNS + ["label", "pred_label", "risk_prob", "risk_level", "is_borderline"]
        st.dataframe(live_scored[preview_cols].head(25), use_container_width=True)


if __name__ == "__main__":
    _guard_streamlit_entrypoint()
    main()
