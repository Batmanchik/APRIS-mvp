from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from io import BytesIO

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
from crypto_ponzi.tx_generator import generate_company_transactions, PRESET_ORDER, PRESET_CONFIGS
from crypto_ponzi.aggregator import aggregate_transactions
from crypto_ponzi.risk_scorer import compute_crypto_ponzi_score, FACTOR_LABELS_RU
from crypto_ponzi.explainer import generate_explanation
from crypto_ponzi.visualizations import (
    plot_inflow_outflow_timeline,
    plot_inflow_structure_pie,
    plot_counterparty_network,
    plot_factor_contributions,
    build_metrics_table,
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

PRIMARY_COLOR = "#10a37f"
SECONDARY_COLOR = "#10a37f"
ALERT_COLOR = "#ef4444"
LEGIT_COLOR = "#10a37f"
PYRAMID_COLOR = "#ef4444"

FEATURE_LABELS = {
    "growth_rate": "Темп роста вкладчиков",
    "referral_ratio": "Доля реферальных участников",
    "payout_dependency": "Зависимость выплат от притока",
    "centralization_index": "Индекс централизации",
    "avg_holding_time": "Среднее удержание средств (дни)",
    "reinvestment_rate": "Доля повторных вложений",
    "gini_coefficient": "Коэффициент Джини",
    "transaction_entropy": "Транзакционная энтропия",
    "structural_depth": "Глубина структуры",
}

OPERATIONAL_LABELS = {
    "tx_count_total": "Общее число транзакций",
    "unique_counterparties": "Уникальные контрагенты",
    "new_clients_current": "Новые клиенты (текущий период)",
    "new_clients_previous": "Новые клиенты (предыдущий период)",
    "referred_clients_current": "Реферальные клиенты (текущий период)",
    "incoming_funds": "Общий входящий поток средств",
    "payouts_total": "Общий объем выплат",
    "top1_wallet_share": "Доля топ-1 кошелька",
    "top10_wallet_share": "Доля топ-10 кошельков",
    "avg_holding_days": "Среднее удержание (дни)",
    "repeat_investor_share": "Доля повторных инвесторов",
    "max_referral_depth": "Макс. глубина реферальной структуры",
}


def _guard_streamlit_entrypoint() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is None:
            print("Это приложение Streamlit.")
            print("Запустите его так: streamlit run app.py")
            raise SystemExit(0)
    except Exception:
        return


def _set_style() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --apris-bg: #ffffff;
                --apris-surface: #f7f7f8;
                --apris-border: #e5e5e5;
                --apris-border-hover: #cdcdcd;
                --apris-text: #1a1a1a;
                --apris-text-secondary: #6e6e80;
                --apris-accent: #10a37f;
                --apris-accent-hover: #0d8a6a;
                --apris-accent-light: rgba(16, 163, 127, 0.08);
                --apris-shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
                --apris-shadow-md: 0 2px 8px rgba(0,0,0,0.06);
                --apris-shadow-lg: 0 4px 16px rgba(0,0,0,0.08);
                --apris-radius: 12px;
                --apris-radius-lg: 16px;
            }

            /* ── Global ── */
            html, body, .stApp {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                color: var(--apris-text) !important;
                -webkit-font-smoothing: antialiased;
            }
            .stApp {
                background: var(--apris-bg) !important;
            }
            .stApp p, .stApp li, .stApp label, .stApp span,
            [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
            .stMarkdown, .stCaption {
                color: var(--apris-text) !important;
                font-family: 'Inter', -apple-system, sans-serif !important;
            }
            .stApp p, .stApp li, .stApp label {
                font-weight: 400 !important;
                line-height: 1.6 !important;
                font-size: 0.925rem !important;
            }
            .stApp h1 {
                font-weight: 700 !important;
                letter-spacing: -0.02em !important;
                color: var(--apris-text) !important;
                font-size: 1.75rem !important;
            }
            .stApp h2 {
                font-weight: 600 !important;
                letter-spacing: -0.01em !important;
                color: var(--apris-text) !important;
                font-size: 1.25rem !important;
            }
            .stApp h3, .stApp h4 {
                font-weight: 600 !important;
                color: var(--apris-text) !important;
            }
            .stCaption, [data-testid="stCaptionContainer"] {
                color: var(--apris-text-secondary) !important;
            }

            /* ── Header ── */
            header[data-testid="stHeader"] {
                background: var(--apris-bg) !important;
                border-bottom: 1px solid var(--apris-border) !important;
            }
            [data-testid="stToolbar"] { display: none; }
            [data-testid="collapsedControl"] { display: none; }
            [data-testid="stSidebar"] { display: none; }

            /* ── Buttons ── */
            div.stButton > button, div.stFormSubmitButton > button {
                background: var(--apris-bg) !important;
                color: var(--apris-text) !important;
                border: 1px solid var(--apris-border) !important;
                border-radius: 20px !important;
                font-family: 'Inter', sans-serif !important;
                font-weight: 500 !important;
                font-size: 0.875rem !important;
                min-height: 40px !important;
                padding: 6px 20px !important;
                transition: all 0.15s ease !important;
                box-shadow: var(--apris-shadow-sm) !important;
            }
            div.stButton > button:hover, div.stFormSubmitButton > button:hover {
                border-color: var(--apris-border-hover) !important;
                background: var(--apris-surface) !important;
                box-shadow: var(--apris-shadow-md) !important;
                color: var(--apris-text) !important;
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"],
            div.stButton > button[kind="primary"] {
                background: var(--apris-accent) !important;
                color: #ffffff !important;
                border: 1px solid var(--apris-accent) !important;
                font-weight: 600 !important;
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
            div.stButton > button[kind="primary"]:hover {
                background: var(--apris-accent-hover) !important;
                border-color: var(--apris-accent-hover) !important;
                box-shadow: 0 2px 12px rgba(16, 163, 127, 0.25) !important;
            }

            /* ── Metric cards ── */
            [data-testid="stMetric"] {
                background: var(--apris-bg) !important;
                border: 1px solid var(--apris-border) !important;
                border-radius: var(--apris-radius) !important;
                padding: 12px 14px !important;
                box-shadow: var(--apris-shadow-sm) !important;
                transition: box-shadow 0.15s ease;
            }
            [data-testid="stMetric"]:hover {
                box-shadow: var(--apris-shadow-md) !important;
            }
            [data-testid="stMetricLabel"] {
                color: var(--apris-text-secondary) !important;
                opacity: 1 !important;
                font-size: 0.8rem !important;
                text-transform: uppercase !important;
                letter-spacing: 0.04em !important;
            }
            [data-testid="stMetricValue"] {
                color: var(--apris-text) !important;
                font-weight: 700 !important;
                opacity: 1 !important;
            }
            [data-testid="stMetricValue"] > div {
                color: var(--apris-text) !important;
                opacity: 1 !important;
            }
            [data-testid="stMetric"] * { text-shadow: none !important; }

            /* ── Tabs ── */
            [data-baseweb="tab-list"] {
                gap: 0 !important;
                border-bottom: 1px solid var(--apris-border) !important;
                background: transparent !important;
            }
            [data-baseweb="tab-list"] button {
                font-family: 'Inter', sans-serif !important;
                font-weight: 500 !important;
                font-size: 0.9rem !important;
                color: var(--apris-text-secondary) !important;
                border-bottom: 2px solid transparent !important;
                padding: 10px 20px !important;
                transition: all 0.15s ease !important;
                background: transparent !important;
            }
            [data-baseweb="tab-list"] button:hover {
                color: var(--apris-text) !important;
            }
            [data-baseweb="tab-list"] button[aria-selected="true"] {
                color: var(--apris-accent) !important;
                border-bottom-color: var(--apris-accent) !important;
                font-weight: 600 !important;
            }
            [data-baseweb="tab-panel"] {
                background: transparent !important;
                border-radius: 0 !important;
                padding: 16px 4px !important;
            }
            [data-baseweb="tab-highlight"] {
                background-color: var(--apris-accent) !important;
            }

            /* ── Tables & DataFrames ── */
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                border: 1px solid var(--apris-border) !important;
                border-radius: var(--apris-radius) !important;
                overflow: hidden !important;
            }

            /* ── Input fields — clean light style ── */
            [data-baseweb="input"], [data-baseweb="base-input"] {
                background: var(--apris-bg) !important;
                border: 1px solid var(--apris-border) !important;
                border-radius: 10px !important;
                transition: border-color 0.15s ease !important;
            }
            [data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within {
                border-color: var(--apris-accent) !important;
                box-shadow: 0 0 0 3px var(--apris-accent-light) !important;
            }
            [data-baseweb="input"] > div, [data-baseweb="base-input"] > div {
                background: transparent !important;
            }
            [data-baseweb="input"] input, [data-baseweb="base-input"] input {
                background: transparent !important;
                color: var(--apris-text) !important;
                -webkit-text-fill-color: var(--apris-text) !important;
                font-weight: 500 !important;
                font-family: 'Inter', sans-serif !important;
            }
            [data-baseweb="input"] [role="button"], [data-baseweb="base-input"] [role="button"] {
                color: var(--apris-text-secondary) !important;
            }

            /* ── Cards ── */
            .apris-card {
                background: var(--apris-surface);
                border: 1px solid var(--apris-border);
                border-radius: var(--apris-radius-lg);
                padding: 20px 24px;
                box-shadow: var(--apris-shadow-sm);
            }

            /* ── Hero section ── */
            .apris-hero {
                text-align: center;
                padding: 40px 24px 28px;
                margin-bottom: 8px;
            }
            .apris-hero h1 {
                font-size: 2rem !important;
                margin-bottom: 6px !important;
                letter-spacing: -0.03em !important;
                color: var(--apris-text) !important;
            }
            .apris-hero .apris-subtitle {
                color: var(--apris-text-secondary);
                font-size: 1.05rem;
                margin: 0;
                font-weight: 400;
                line-height: 1.5;
            }
            .apris-hero .apris-badge {
                display: inline-block;
                background: var(--apris-accent-light);
                color: var(--apris-accent);
                font-weight: 600;
                font-size: 0.75rem;
                padding: 4px 14px;
                border-radius: 20px;
                margin-bottom: 14px;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            /* ── Risk banners ── */
            .risk-banner {
                border-radius: var(--apris-radius);
                padding: 16px 20px;
                margin-bottom: 12px;
                font-weight: 600;
                font-size: 0.95rem;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .risk-banner-low {
                background: #ecfdf5;
                color: #065f46 !important;
                border: 1px solid #a7f3d0;
            }
            .risk-banner-medium {
                background: #fffbeb;
                color: #92400e !important;
                border: 1px solid #fde68a;
            }
            .risk-banner-high {
                background: #fef2f2;
                color: #991b1b !important;
                border: 1px solid #fecaca;
            }
            .risk-banner-critical {
                background: #450a0a;
                color: #fecaca !important;
                border: 1px solid #991b1b;
                animation: pulse-critical 2s ease-in-out infinite;
            }
            @keyframes pulse-critical {
                0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.3); }
                50% { box-shadow: 0 0 12px 4px rgba(239, 68, 68, 0.15); }
            }

            /* ── Info / warning alerts ── */
            [data-testid="stAlert"] {
                border-radius: var(--apris-radius) !important;
                border: 1px solid var(--apris-border) !important;
                font-size: 0.875rem !important;
            }

            /* ── Expanders ── */
            [data-testid="stExpander"] {
                border: 1px solid var(--apris-border) !important;
                border-radius: var(--apris-radius) !important;
                box-shadow: none !important;
            }

            /* ── Code blocks ── */
            .stMarkdown code {
                background: var(--apris-surface) !important;
                color: var(--apris-text) !important;
                border: 1px solid var(--apris-border) !important;
                border-radius: 6px !important;
                padding: 2px 6px !important;
                font-size: 0.85rem !important;
            }

            /* ── Divider ── */
            hr {
                border-color: var(--apris-border) !important;
                opacity: 0.5 !important;
            }

            /* ── Scrollbar ── */
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_matplotlib_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#e5e5e5",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#e5e5e5",
            "axes.titleweight": "bold",
            "font.size": 10,
            "font.family": "sans-serif",
            "text.color": "#1a1a1a",
            "axes.labelcolor": "#1a1a1a",
            "xtick.color": "#6e6e80",
            "ytick.color": "#6e6e80",
        }
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


def _ensure_model_available() -> tuple[Any, list[str]]:
    if not Path(MODEL_PATH).exists() or not Path(FEATURE_NAMES_PATH).exists():
        st.error("Модель не найдена. Сначала запустите обучение.")
        st.code("python train_model.py")
        st.stop()
    try:
        return load_artifacts()
    except Exception as exc:
        st.error(f"Ошибка загрузки артефактов модели: {exc}")
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
        raise ValueError("Порядок признаков в feature_names.json не совпадает с ожидаемой схемой.")
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
        st.session_state["input_source_mode"] = "Операционные факты (рекомендуется)"


def _set_preset(name: str) -> None:
    for feature in FEATURE_COLUMNS:
        st.session_state[feature] = float(PRESETS[name][feature])
    for key, value in OPERATIONAL_PRESETS[name].items():
        st.session_state[key] = float(value)


def _risk_band(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "Высокий"
    if prob >= RISK_THRESHOLDS["medium"]:
        return "Средний"
    return "Низкий"


def _risk_css_class(level: str) -> str:
    level_norm = level.lower()
    if level_norm in {"low", "низкий"}:
        return "risk-banner risk-banner-low"
    if level_norm in {"medium", "средний"}:
        return "risk-banner risk-banner-medium"
    return "risk-banner risk-banner-high"


def _translate_risk_level(level: str) -> str:
    mapping = {
        "low": "Низкий",
        "medium": "Средний",
        "high": "Высокий",
        "низкий": "Низкий",
        "средний": "Средний",
        "высокий": "Высокий",
    }
    return mapping.get(level.lower(), level)


def _risk_emoji(level: str) -> str:
    level_norm = level.lower()
    if level_norm in {"low", "низкий"}:
        return "✅"
    if level_norm in {"medium", "средний"}:
        return "⚠️"
    return "🚨"


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
    with st.expander("📖 Как вводить данные", expanded=False):
        st.markdown(
            """
            1. Выберите **Операционные факты (рекомендуется)**, если есть факты из мониторинга.
            2. Заполните значения за текущий период.
            3. Нажмите **Оценить кейс**.
            4. Проверьте:
            - Вероятность риска и уровень риска
            - Динамический разбор сигналов кейса
            - Метрики концентрации транзакционного графа
            """
        )
        st.caption(
            "Правила валидации: referred_clients_current <= new_clients_current, "
            "top1_wallet_share <= top10_wallet_share."
        )


def _render_operational_input_main() -> tuple[dict[str, float], dict[str, float]]:
    st.markdown("#### 📊 Операционные факты")
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
    st.markdown("#### 🔬 Признаки модели (расширенный режим)")
    mode = st.radio(
        "Режим ввода",
        ["Точный ввод", "Быстрые слайдеры"],
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
                if mode == "Точный ввод":
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
                if mode == "Точный ввод":
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
            colors.append("#ef4444")
        elif idx == 1:
            colors.append("#f97316")
        elif idx == 2:
            colors.append("#eab308")
        else:
            colors.append("#10a37f")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.barh(names, values, color=colors, height=0.6, edgecolor="none")
    ax.invert_yaxis()
    ax.set_title("Разбор сигналов кейса (динамика)", fontsize=13, pad=12)
    ax.set_xlabel("Взвешенное отклонение от легитимного baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for idx in range(min(3, len(values))):
        ax.text(
            values[idx] + max(values) * 0.01 if max(values) > 0 else 0.001,
            idx,
            "ТОП",
            va="center",
            fontsize=9,
            color="#991B1B",
            fontweight="bold",
        )

    _render_figure_inline(fig)
    st.caption("Взвешенное отклонение от легитимного baseline. Выделены топ-3 сигнала.")


def _plot_global_feature_importance(feature_names: list[str], importances: np.ndarray) -> None:
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in ranked]
    values = [float(x[1]) for x in ranked]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.barh(names, values, color="#10a37f", height=0.6, edgecolor="none")
    ax.invert_yaxis()
    ax.set_title("Глобальная важность признаков (по модели)", fontsize=13, pad=12)
    ax.set_xlabel("важность")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _render_figure_inline(fig)


def _plot_feature_gallery(scored: pd.DataFrame, marker_features: dict[str, float] | None = None) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, feature in enumerate(FEATURE_COLUMNS):
        ax = axes[idx]
        legit = scored[scored["label"] == 0][feature].astype(float)
        pyramid = scored[scored["label"] == 1][feature].astype(float)
        ax.hist(legit, bins=24, alpha=0.55, color=LEGIT_COLOR, density=True, label="легит")
        ax.hist(pyramid, bins=24, alpha=0.48, color=PYRAMID_COLOR, density=True, label="пирамида")
        if marker_features is not None and feature in marker_features:
            ax.axvline(float(marker_features[feature]), color="#f97316", linestyle="--", linewidth=2.0)
        ax.set_title(feature, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Галерея признаков: отдельный график по каждой переменной", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    _render_figure_inline(fig)


def _plot_live_confusion_and_scores(scored: pd.DataFrame, cm: list[list[int]]) -> None:
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        matrix = np.array(cm, dtype=float)
        im = ax.imshow(matrix, cmap="Greens")
        ax.set_title("Матрица ошибок (live batch)", fontsize=13, pad=12)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Факт")
        ax.set_xticks([0, 1], labels=["легит", "пирамида"])
        ax.set_yticks([0, 1], labels=["легит", "пирамида"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#1a1a1a", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _render_figure_inline(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        legit_probs = scored[scored["label"] == 0]["risk_prob"]
        pyramid_probs = scored[scored["label"] == 1]["risk_prob"]
        ax.hist(legit_probs, bins=24, alpha=0.55, color=LEGIT_COLOR, label="легит", density=True)
        ax.hist(pyramid_probs, bins=24, alpha=0.52, color=PYRAMID_COLOR, label="пирамида", density=True)
        ax.axvline(RISK_THRESHOLDS["medium"], color="#f97316", linestyle="--", linewidth=1.8)
        ax.axvline(RISK_THRESHOLDS["high"], color="#ef4444", linestyle="-.", linewidth=1.8)
        ax.set_title("Распределение вероятности риска (live batch)", fontsize=13, pad=12)
        ax.set_xlabel("P(пирамида)")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _render_figure_inline(fig)


def _plot_stage_summary(scored: pd.DataFrame) -> None:
    counts = scored["risk_level"].value_counts().reindex(["Низкий", "Средний", "Высокий"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    bar_colors = ["#10a37f", "#eab308", "#ef4444"]
    ax.bar(counts.index, counts.values, color=bar_colors, width=0.58, edgecolor="none")
    ax.set_title("Уровни риска в live batch", fontsize=13, pad=12)
    ax.set_ylabel("Количество")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _render_figure_inline(fig)


def _render_system_guide() -> None:
    st.subheader("📚 Подробный гид по панели APRIS")
    st.markdown(
        """
        Этот раздел объясняет, что показывает каждый график и зачем он нужен в аналитической работе.
        """
    )

    with st.expander("🎯 Цель системы", expanded=True):
        st.markdown(
            """
            - APRIS оценивает вероятность того, что финансовая структура имеет пирамидальные признаки.
            - Система не выносит юридический вердикт, а приоритизирует кейсы для проверки.
            - Все данные в MVP синтетические, расчеты выполняются локально.
            """
        )

    with st.expander("📋 Вкладка: Оценка одного кейса", expanded=True):
        st.markdown(
            """
            1. **Вероятность риска и уровень риска**: итоговый скор конкретного объекта.
            2. **Топ признаков (глобальная важность)**: какие признаки в целом наиболее значимы для модели.
            3. **Разбор сигналов кейса (динамика)**:
               сравнение текущего кейса с легитимным baseline с учетом весов модели.
               Топ-3 сигнала показывают, что именно «тянет» риск вверх.
            4. **Синтетический граф транзакций**:
               модельно-восстановленная структура потоков; визуально показывает концентрацию вокруг hub.
            5. **hub_in_degree_share / hub_betweenness**:
               количественная оценка централизации выплат и роли центрального узла.
            6. **Карта рисков популяции (PCA)**:
               позиция текущего кейса на фоне синтетической популяции (легит/пирамида/borderline).
            """
        )

    with st.expander("📡 Вкладка: Live-демонстрация", expanded=True):
        st.markdown(
            """
            Live-раздел нужен для демонстрации, что система работает не на заранее заготовленных картинках.

            1. Генерируется новый синтетический батч прямо во время демонстрации.
            2. Модель пересчитывает метрики качества на новом батче:
               **Recall (pyramid), ROC-AUC, Accuracy, Confusion Matrix**.
            3. **Матрица ошибок** показывает, где модель ошибается (FP/FN).
            4. **Распределение вероятностей риска** показывает отделимость классов и влияние порогов 0.4/0.7.
            5. **Уровни риска в live batch** показывают, как объекты распределяются по Low/Medium/High.
            6. **Галерея признаков** (по каждой переменной) показывает, как распределения отличаются
               между легитимными и пирамидальными объектами.
            """
        )

    with st.expander("💡 Как интерпретировать результат", expanded=False):
        st.markdown(
            """
            - **Низкий риск (< 0.4)**: сигналов мало, кейс низкого приоритета.
            - **Средний риск (0.4-0.7)**: есть аномалии, нужен углубленный анализ.
            - **Высокий риск (>= 0.7)**: выраженный набор признаков пирамидальной схемы, кейс в приоритет.
            """
        )


def _render_crypto_ponzi_tab() -> None:
    """Render the Company-Level Crypto-Ponzi Detection tab."""
    st.markdown(
        """
        <div style='text-align:center; margin-bottom: 24px;'>
            <div class='apris-badge'>Company-Level Analysis</div>
            <h2 style='margin-top:8px; font-weight:700;'>Crypto-Ponzi Detection Engine</h2>
            <p style='color: var(--apris-text-secondary); font-size:0.95rem;'>
            Анализ поведения компании за период — агрегация транзакций, метрики, риск-скор
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Preset buttons ─────────────────────────────────────────────
    st.markdown("#### Выберите компанию для анализа")
    preset_emojis = {"SAFE": "🏢", "MODERATE": "🏢", "DANGEROUS": "🏢", "PYRAMID": "🏢"}

    cols = st.columns(4)
    for i, preset_name in enumerate(PRESET_ORDER):
        company = PRESET_CONFIGS[preset_name]["company_name"]
        with cols[i]:
            if st.button(
                f"{preset_emojis[preset_name]} {company}",
                key=f"cp_preset_{preset_name}",
                use_container_width=True,
            ):
                st.session_state["cp_active_preset"] = preset_name

    active_preset = st.session_state.get("cp_active_preset")
    if active_preset is None:
        st.markdown(
            """
            <div style='text-align:center; padding: 60px 20px; color: var(--apris-text-secondary);'>
                <p style='font-size:2.5rem; margin-bottom: 8px;'>🏢</p>
                <p style='font-size:1.1rem;'>Выберите пресет компании для начала анализа</p>
                <p style='font-size:0.85rem;'>Система сгенерирует синтетические транзакции за 90 дней</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Generate data ──────────────────────────────────────────────
    with st.spinner("Генерация транзакций и расчёт метрик…"):
        case = generate_company_transactions(active_preset, seed=42)
        transactions = case["transactions"]
        metrics = aggregate_transactions(transactions)
        score_result = compute_crypto_ponzi_score(metrics)
        explanation = generate_explanation(metrics, score_result, company_name=case["company_name"])

    # ── Company info card ──────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background: var(--apris-surface); border-radius: var(--apris-radius);
                    padding: 20px 24px; border: 1px solid var(--apris-border); margin-bottom: 20px;'>
            <div style='display:flex; gap: 40px; flex-wrap: wrap;'>
                <div><span style='color: var(--apris-text-secondary); font-size:0.8rem;'>КОМПАНИЯ</span><br>
                     <strong>{case['company_name']}</strong></div>
                <div><span style='color: var(--apris-text-secondary); font-size:0.8rem;'>ID</span><br>
                     <code>{case['company_id']}</code></div>
                <div><span style='color: var(--apris-text-secondary); font-size:0.8rem;'>ВИД ДЕЯТЕЛЬНОСТИ</span><br>
                     {case['declared_business_type']}</div>
                <div><span style='color: var(--apris-text-secondary); font-size:0.8rem;'>ПЕРИОД</span><br>
                     {case['reporting_period']}</div>
                <div><span style='color: var(--apris-text-secondary); font-size:0.8rem;'>ТРАНЗАКЦИЙ</span><br>
                     <strong>{len(transactions)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Risk banner ────────────────────────────────────────────────
    prob = score_result["probability"]
    emoji = score_result["risk_emoji"]
    level_ru = score_result["risk_level_ru"]
    css_class = score_result["risk_css"]

    st.markdown(
        f"""
        <div class='risk-banner {css_class}'>
            {emoji} Вероятность крипто-пирамиды: &nbsp;<strong>{prob:.3f}</strong>
            &nbsp;│&nbsp; Уровень риска: &nbsp;<strong>{level_ru}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Key metrics as st.metric columns ───────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Крипто-экспозиция", f"{metrics['crypto_exposure_ratio']:.1%}")
    m2.metric("Зависимость выплат", f"{metrics['dependency_ratio']:.2f}")
    m3.metric("Физ. лица (приток)", f"{metrics['physical_inflow_ratio']:.1%}")
    m4.metric("Ср. удержание", f"{metrics['avg_holding_time']:.0f} дн.")
    m5.metric("Концентрация top-5", f"{metrics['concentration_index']:.2f}")

    st.markdown("---")

    # ── Visualizations ─────────────────────────────────────────────
    # 1 & 2: Timeline + Pie side by side
    vc1, vc2 = st.columns([1.6, 1])
    with vc1:
        fig_timeline = plot_inflow_outflow_timeline(transactions)
        _render_figure_inline(fig_timeline)
        plt.close(fig_timeline)
    with vc2:
        fig_pie = plot_inflow_structure_pie(metrics)
        _render_figure_inline(fig_pie)
        plt.close(fig_pie)

    # 3 & 4: Network + Factor contributions
    vc3, vc4 = st.columns([1.2, 1])
    with vc3:
        fig_network = plot_counterparty_network(transactions, company_name=case["company_name"])
        _render_figure_inline(fig_network)
        plt.close(fig_network)
    with vc4:
        fig_factors = plot_factor_contributions(score_result)
        _render_figure_inline(fig_factors)
        plt.close(fig_factors)

    # 5: Metrics table
    st.markdown("#### 📊 Таблица ключевых метрик")
    metrics_df = build_metrics_table(metrics)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # ── Explanation ────────────────────────────────────────────────
    st.markdown("#### 📝 Объяснение оценки")
    st.markdown(explanation)

    # ── Raw transactions preview ───────────────────────────────────
    with st.expander("📋 Первые 30 транзакций (синтетические)", expanded=False):
        st.dataframe(transactions.head(30), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="APRIS — Система анализа риска", layout="wide", initial_sidebar_state="collapsed")
    _ensure_defaults()
    _set_style()
    _apply_matplotlib_theme()

    model, feature_names = _ensure_model_available()
    population_error: str | None = None
    population_df = pd.DataFrame()
    projected_population = pd.DataFrame()
    pop_scaler: Any | None = None
    pop_pca: Any | None = None
    try:
        population_df = _load_population_df_cached()
        projected_population, pop_scaler, pop_pca, pop_feature_names = _load_population_projection_cached()
        if pop_feature_names != feature_names:
            raise ValueError("Несовпадение порядка признаков модели и карты рисков популяции.")
    except Exception as exc:
        population_error = str(exc)

    # ── Hero section ──
    st.markdown(
        "<div class='apris-hero'>"
        "<div class='apris-badge'>AI-Powered Risk Detection</div>"
        "<h1>APRIS</h1>"
        "<p class='apris-subtitle'>AI Pyramid Risk Intelligence System — панель поддержки решений для раннего скрининга пирамидных рисков</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Quick info section ──
    with st.expander("📋 Краткая инструкция", expanded=False):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown(
                """
                1. Выберите источник ввода: операционные факты (рекомендуется) или признаки модели.
                2. Используйте пресет или введите значения кейса вручную.
                3. Нажмите кнопку «Оценить кейс» и проверьте результаты.
                """
            )
        with c2:
            st.markdown("**Формулы и пороги**")
            st.markdown(
                """
                `growth_rate = (Новые_тек - Новые_пред) / max(Новые_пред, 1)`
                `referral_ratio = Реферальные_тек / max(Новые_тек, 1)`
                `payout_dependency = Выплаты / max(Вход_поток, 1)`
                """
            )
            st.caption("Пороги: < 0.4 — низкий, 0.4-0.7 — средний, >= 0.7 — высокий.")
    st.info("🔒 Все данные синтетические и обрабатываются локально. Реальные персональные записи не используются.")

    tab_single, tab_live, tab_crypto, tab_guide = st.tabs(
        ["🔍 Оценка кейса", "📡 Live-демонстрация", "🏢 Crypto-Ponzi (Компания)", "📚 Гид"]
    )

    with tab_single:
        st.markdown("##### Быстрые пресеты")
        p1, p2, p3, p4 = st.columns(4)
        if p1.button("✅ Легит", use_container_width=True):
            _set_preset("Legit")
        if p2.button("⚠️ Подозрительно", use_container_width=True):
            _set_preset("Suspicious")
        if p3.button("🚨 Пирамида", use_container_width=True):
            _set_preset("Pyramid")
        if p4.button("🎲 Случайный кейс", use_container_width=True):
            if population_df.empty:
                st.warning("Синтетический датасет недоступен для генерации случайного кейса.")
            else:
                sampled_features = _sample_population_case(population_df)
                for feature in FEATURE_COLUMNS:
                    st.session_state[feature] = float(sampled_features[feature])
                st.session_state["input_source_mode"] = "Признаки модели (расширенный режим)"
                st.session_state["single_result"] = predict_risk(
                    sampled_features, model=model, feature_names=feature_names
                )
                st.session_state["single_explain"] = explain(
                    sampled_features, model=model, feature_names=feature_names
                )
                st.session_state["single_features"] = sampled_features
                st.session_state["single_raw_operational"] = None
                st.success("Случайный синтетический кейс сгенерирован и оценен.")

        st.markdown("---")
        _render_input_guide()
        input_source = st.radio(
            "Источник ввода",
            ["Операционные факты (рекомендуется)", "Признаки модели (расширенный режим)"],
            horizontal=True,
            key="input_source_mode",
        )

        if input_source == "Операционные факты (рекомендуется)":
            raw_operational, features = _render_operational_input_main()
        else:
            raw_operational, features = None, _render_feature_input_main()

        st.markdown("")  # spacer
        if st.button("🔍 Оценить кейс", type="primary", use_container_width=True):
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
                st.error(f"Ошибка валидации входных данных: {exc}")

        if "single_result" in st.session_state:
            result = st.session_state["single_result"]
            object_features = st.session_state["single_features"]
            top_features = st.session_state["single_explain"]
            raw_operational_view = st.session_state.get("single_raw_operational")

            level = _translate_risk_level(str(result["label_text"]))
            risk_prob = float(result["prob"])
            emoji = _risk_emoji(level)
            st.markdown(
                (
                    f"<div class='{_risk_css_class(level)}'>"
                    f"{emoji} Вероятность риска: <strong>{risk_prob:.3f}</strong> &nbsp;|&nbsp; Уровень риска: <strong>{level}</strong>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            _, _, threshold_col = st.columns(3)
            with threshold_col:
                st.metric("Пороги", f"{RISK_THRESHOLDS['medium']:.1f} / {RISK_THRESHOLDS['high']:.1f}")

            st.markdown("**Топ признаков (глобальная важность):**")
            st.table(pd.DataFrame(top_features).rename(columns={"feature": "Признак", "importance": "Важность"}))

            if raw_operational_view is not None:
                with st.expander("📐 Рассчитанные признаки модели из операционных фактов", expanded=False):
                    st.table(
                        pd.DataFrame(
                            [{"Признак": k, "Значение": float(v)} for k, v in object_features.items()]
                        )
                    )

            c1, c2 = st.columns(2)
            with c1:
                _plot_case_signal_breakdown(
                    object_features,
                    feature_names,
                    np.asarray(model.feature_importances_, dtype=float),
                )
                with st.expander("📊 Глобальная важность по модели (справочно)", expanded=False):
                    _plot_global_feature_importance(
                        feature_names,
                        np.asarray(model.feature_importances_, dtype=float),
                    )
            with c2:
                graph = build_transaction_graph(object_features)
                graph_metrics = compute_hub_metrics(graph)
                graph_fig = plot_graph(graph, hub_node=int(graph_metrics["hub_node"]))
                _render_figure_inline(graph_fig)
                g1, g2 = st.columns(2)
                g1.metric("Доля входящих у hub", f"{graph_metrics['hub_in_degree_share']:.3f}")
                g2.metric("Betweenness hub", f"{graph_metrics['hub_betweenness']:.3f}")
                st.caption("Высокая концентрация на hub может указывать на централизованную структуру выплат.")

            st.markdown("#### 🗺️ Карта рисков популяции")
            if population_error is not None or pop_scaler is None or pop_pca is None:
                st.warning(f"Карта популяции недоступна: {population_error}")
            else:
                current_point = project_current_case(
                    object_features,
                    feature_names,
                    pop_scaler,
                    pop_pca,
                )
                pop_fig = build_population_map_figure(projected_population, current_point=current_point)
                _render_figure_inline(pop_fig)
                st.caption("PCA-проекция синтетической популяции с наложением текущего кейса.")
        else:
            st.info("Заполните значения и нажмите «Оценить кейс».")

    with tab_live:
        st.subheader("📡 Live-генерация синтетических данных")
        controls1, controls2, controls3 = st.columns([1, 1, 2])
        with controls1:
            batch_size = st.number_input(
                "Размер батча",
                min_value=200,
                max_value=5000,
                value=800,
                step=100,
            )
        with controls2:
            use_random_seed = st.checkbox("Случайный seed при каждом запуске", value=True)
            manual_seed = st.number_input("Ручной seed", min_value=1, max_value=2**31 - 1, value=42)
        with controls3:
            st.markdown(
                "<div class='apris-card'><b>Live-сценарий:</b> "
                "Сгенерировать новый синтетический набор организаций, оценить моделью и сразу отрисовать обновленные графики."
                "</div>",
                unsafe_allow_html=True,
            )

        if st.button("🚀 Сгенерировать новый синтетический батч", type="primary"):
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
        k1.metric("Использованный seed", str(live_seed))
        k2.metric("Recall (пирамида)", f"{float(live_metrics['recall_pyramid']):.3f}")
        k3.metric("ROC-AUC", f"{float(live_metrics['roc_auc']):.3f}")
        k4.metric("Точность (Accuracy)", f"{float(live_metrics['accuracy']):.3f}")

        _plot_live_confusion_and_scores(live_scored, live_metrics["confusion_matrix"])  # type: ignore[arg-type]
        _plot_stage_summary(live_scored)

        marker = st.session_state.get("single_features")
        _plot_feature_gallery(live_scored, marker_features=marker)

        preview_cols = FEATURE_COLUMNS + ["label", "pred_label", "risk_prob", "risk_level", "is_borderline"]
        st.dataframe(live_scored[preview_cols].head(25), use_container_width=True)

    with tab_crypto:
        _render_crypto_ponzi_tab()

    with tab_guide:
        _render_system_guide()


if __name__ == "__main__":
    _guard_streamlit_entrypoint()
    main()
