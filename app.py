from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from io import BytesIO

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score  # type: ignore

from data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, RISK_THRESHOLDS, generate_live_batch  # type: ignore
from graph_module import build_transaction_graph, compute_hub_metrics, plot_graph
from population_map import (  # type: ignore
    build_population_map_figure,
    check_feature_order,
    check_no_nan,
    check_pca_dimensions,
    fit_population_pca,
    load_feature_names as load_population_feature_names,
    load_population_dataset,
    project_current_case,
)
from risk_engine import (  # type: ignore
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
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

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
        <style>
            :root {
                --apris-border-soft: rgba(226, 232, 240, 0.8);
                --background-color: #f8fafc;
                --secondary-background-color: #ffffff;
                --text-color: #0f172a;
                --primary-color: #6366f1; /* Indigo */
                --primary-hover: #4f46e5;
                --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
                --card-shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
            }
            /* Global Font Settings */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            html, body, .stApp {
                font-family: 'Inter', sans-serif !important;
                color: var(--text-color) !important;
                background-color: var(--background-color) !important;
                -webkit-font-smoothing: antialiased;
            }
            /* Clean up background */
            .stApp {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
            }
            .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span,
            [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], .stMarkdown, .stCaption {
                color: var(--text-color) !important;
            }
            /* Headings */
            .stApp h1 {
                font-weight: 800 !important;
                letter-spacing: -0.025em;
                background: linear-gradient(90deg, #312e81, #6366f1);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            .stApp h2, .stApp h3, .stApp h4 {
                font-weight: 700 !important;
                letter-spacing: -0.015em;
                color: #1e293b !important;
            }
            /* Header and navigation clean up */
            header[data-testid="stHeader"] {
                background: transparent !important;
                box-shadow: none !important;
            }
            [data-testid="stToolbar"], [data-testid="collapsedControl"], [data-testid="stSidebar"] {
                display: none !important;
            }
            /* Buttons */
            div.stButton > button, div.stFormSubmitButton > button {
                background: var(--secondary-background-color);
                color: #334155;
                border: 1px solid var(--apris-border-soft);
                border-radius: 12px !important;
                font-weight: 600 !important;
                min-height: 48px !important;
                padding: 0 1.5rem !important;
                transition: all 0.2s ease-in-out !important;
                box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            }
            div.stButton > button:hover, div.stFormSubmitButton > button:hover {
                border-color: #cbd5e1 !important;
                background-color: #f8fafc !important;
                color: #0f172a !important;
                transform: translateY(-1px);
                box-shadow: var(--card-shadow);
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"],
            div.stButton > button[kind="primary"] {
                background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
                color: #ffffff !important;
                border: none !important;
                box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.4), 0 2px 4px -2px rgba(99, 102, 241, 0.4) !important;
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
            div.stButton > button[kind="primary"]:hover {
                box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.5), 0 4px 6px -4px rgba(99, 102, 241, 0.5) !important;
                transform: translateY(-2px);
            }
            div[data-testid="stFormSubmitButton"] button[kind="primary"]:active,
            div.stButton > button[kind="primary"]:active {
                transform: translateY(0);
            }
            /* Metrics Cards */
            [data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.5);
                border-radius: 16px;
                padding: 1rem 1.25rem;
                box-shadow: var(--card-shadow);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            [data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                box-shadow: var(--card-shadow-hover);
            }
            [data-testid="stMetricLabel"] {
                color: #64748b !important;
                font-size: 0.875rem !important;
                font-weight: 500 !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.25rem;
            }
            [data-testid="stMetricValue"] {
                color: #0f172a !important;
                font-weight: 800 !important;
                font-size: 2rem !important;
                line-height: 1 !important;
            }
            /* Tabs styling */
            [data-baseweb="tab-list"] {
                gap: 1rem;
                border-bottom: 2px solid #e2e8f0;
            }
            [data-baseweb="tab"] {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
            }
            [data-baseweb="tab"] p {
                font-weight: 600;
                font-size: 1.05rem;
                color: #64748b !important;
            }
            [aria-selected="true"] p {
                color: var(--primary-color) !important;
            }
            [data-baseweb="tab-highlight"] {
                background-color: var(--primary-color) !important;
                height: 3px !important;
                border-radius: 3px 3px 0 0 !important;
            }
            [data-baseweb="tab-panel"] {
                background: transparent;
                padding: 1.5rem 0;
            }
            /* Data tables */
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                border: none !important;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: var(--card-shadow);
                background: white;
            }
            /* Main Content Cards */
            .apris-card {
                background: rgba(255, 255, 255, 0.9);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.6);
                border-radius: 20px;
                padding: 1.5rem 2rem;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
                margin-bottom: 1.5rem;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .apris-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
            }
            /* Input fields styling */
            [data-baseweb="input"], [data-baseweb="base-input"] {
                background: #f8fafc !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 12px !important;
                transition: all 0.2s ease;
            }
            [data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within {
                border-color: var(--primary-color) !important;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
                background: #ffffff !important;
            }
            [data-baseweb="input"] input, [data-baseweb="base-input"] input {
                color: #0f172a !important;
                -webkit-text-fill-color: #0f172a !important;
                font-weight: 500 !important;
                font-size: 1rem !important;
            }
            /* Risk banners */
            .risk-banner {
                border-radius: 16px;
                padding: 1rem 1.5rem;
                margin-bottom: 1.5rem;
                font-weight: 700;
                font-size: 1.125rem;
                display: flex;
                align-items: center;
                box-shadow: var(--card-shadow);
            }
            .risk-banner::before {
                content: '';
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 12px;
            }
            .risk-banner-low {
                background: linear-gradient(to right, #ecfdf5, #d1fae5);
                color: #065f46 !important;
                border: 1px solid #10b981;
            }
            .risk-banner-low::before { background-color: #10b981; box-shadow: 0 0 8px #10b981; }
            .risk-banner-medium {
                background: linear-gradient(to right, #fffbeb, #fef3c7);
                color: #b45309 !important;
                border: 1px solid #f59e0b;
            }
            .risk-banner-medium::before { background-color: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
            .risk-banner-high {
                background: linear-gradient(to right, #fef2f2, #fee2e2);
                color: #b91c1c !important;
                border: 1px solid #ef4444;
            }
            .risk-banner-high::before { background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }
            
            /* Expanders */
            [data-testid="stExpander"] {
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                background: white;
                box-shadow: var(--card-shadow);
                overflow: hidden;
            }
            [data-testid="stExpander"] > details > summary {
                padding: 1rem 1.5rem;
                font-weight: 600;
                background: #f8fafc;
            }
            [data-testid="stExpander"] > details > summary:hover {
                background: #f1f5f9;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _apply_matplotlib_theme() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "none",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#E2E8F0",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.color": "#334155",
            "axes.titleweight": "bold",
            "font.size": 10,
            "text.color": "#0F172A",
            "axes.labelcolor": "#0F172A",
            "xtick.color": "#0F172A",
            "ytick.color": "#0F172A",
        }
    )


def _render_figure_inline(fig: plt.Figure, width: str = "100%") -> None:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    st.markdown(
        f"<img src='data:image/png;base64,{encoded}' style='width:{width};height:auto;display:block;' />",
        unsafe_allow_html=True,
    )
    plt.close(fig)


def _ensure_model_available() -> tuple[Any, list[str]]:
    if not Path(MODEL_PATH).exists() or not Path(FEATURE_NAMES_PATH).exists():
        st.error("Модель не найдена. Сначала запустите обучение.")
        st.code("python train_model.py")
        st.stop()
        raise RuntimeError("Stopped by Streamlit")
    try:
        return load_artifacts()
    except Exception as exc:
        st.error(f"Ошибка загрузки артефактов модели: {exc}")
        st.stop()
        raise RuntimeError("Stopped by Streamlit")


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
    with st.expander("Как вводить данные", expanded=False):
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
    st.markdown("#### Операционные факты")
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
                    min_value=low,
                    max_value=high,
                    value=float(current),
                    step=1000.0,
                    format="%.2f",
                    key=f"op_{key}",
                )
            else:
                val = st.number_input(
                    label,
                    min_value=low,
                    max_value=high,
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
    st.markdown("#### Признаки модели (расширенный режим)")
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
                        min_value=low,
                        max_value=high,
                        value=float(current),
                        step=float(precise_step),
                        format=precise_format,
                        key=f"adv_{feature}",
                    )
                else:
                    val = st.slider(
                        label,
                        min_value=low,
                        max_value=high,
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
    ax.set_title("Разбор сигналов кейса (динамика)")
    ax.set_xlabel("Взвешенное отклонение от легитимного baseline")

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
    ax.barh(names, values, color="#2563eb")
    ax.invert_yaxis()
    ax.set_title("Глобальная важность признаков (по модели)")
    ax.set_xlabel("важность")
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
        if isinstance(marker_features, dict):
            val = marker_features.get(str(feature))
            if val is not None:
                ax.axvline(float(val), color=ALERT_COLOR, linestyle="--", linewidth=2.0)
        ax.set_title(str(feature), fontsize=10)
        ax.tick_params(labelsize=8)
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
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title("Матрица ошибок (live batch)")
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Факт")
        ax.set_xticks([0, 1], labels=["легит", "пирамида"])
        ax.set_yticks([0, 1], labels=["легит", "пирамида"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#0f172a")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _render_figure_inline(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        legit_probs = scored[scored["label"] == 0]["risk_prob"]
        pyramid_probs = scored[scored["label"] == 1]["risk_prob"]
        ax.hist(legit_probs, bins=24, alpha=0.55, color=LEGIT_COLOR, label="легит", density=True)
        ax.hist(pyramid_probs, bins=24, alpha=0.52, color=PYRAMID_COLOR, label="пирамида", density=True)
        ax.axvline(RISK_THRESHOLDS["medium"], color=ALERT_COLOR, linestyle="--", linewidth=1.8)
        ax.axvline(RISK_THRESHOLDS["high"], color=ALERT_COLOR, linestyle="-.", linewidth=1.8)
        ax.set_title("Распределение вероятности риска (live batch)")
        ax.set_xlabel("P(пирамида)")
        ax.legend(frameon=False)
        _render_figure_inline(fig)


def _plot_stage_summary(scored: pd.DataFrame) -> None:
    counts = scored["risk_level"].value_counts().reindex(["Низкий", "Средний", "Высокий"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    ax.bar(counts.index, counts.values, color=[SECONDARY_COLOR, "#D97706", ALERT_COLOR], width=0.58)
    ax.set_title("Уровни риска в live batch")
    ax.set_ylabel("Количество")
    _render_figure_inline(fig)


def _render_system_guide() -> None:
    st.subheader("Подробный гид по панели APRIS")
    st.markdown(
        """
        Этот раздел объясняет, что показывает каждый график и зачем он нужен в аналитической работе.
        """
    )

    with st.expander("Цель системы", expanded=True):
        st.markdown(
            """
            - APRIS оценивает вероятность того, что финансовая структура имеет пирамидальные признаки.
            - Система не выносит юридический вердикт, а приоритизирует кейсы для проверки.
            - Все данные в MVP синтетические, расчеты выполняются локально.
            """
        )

    with st.expander("Вкладка: Оценка одного кейса", expanded=True):
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

    with st.expander("Вкладка: Live-демонстрация", expanded=True):
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

    with st.expander("Как интерпретировать результат", expanded=False):
        st.markdown(
            """
            - **Низкий риск (< 0.4)**: сигналов мало, кейс низкого приоритета.
            - **Средний риск (0.4-0.7)**: есть аномалии, нужен углубленный анализ.
            - **Высокий риск (>= 0.7)**: выраженный набор признаков пирамидальной схемы, кейс в приоритет.
            """
        )


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

    st.markdown(
        "<div class='apris-card'><h1 style='margin:0;'>APRIS — AI Pyramid Risk Intelligence System</h1>"
        "<p style='margin:8px 0 0 0;'>Панель поддержки решений для раннего скрининга пирамидных рисков.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    intro_left, intro_right = st.columns([1.2, 1])
    with intro_left:
        st.markdown(
            """
            **Краткая инструкция**
            1. Выберите источник ввода: операционные факты (рекомендуется) или признаки модели.
            2. Используйте пресет или введите значения кейса вручную.
            3. Нажмите кнопку «Оценить кейс» и проверьте:
            - Вероятность риска и уровень риска
            - Разбор сигналов кейса (ключевые драйверы)
            - Синтетический транзакционный граф и hub-метрики
            - Карта рисков популяции (кейс на фоне синтетической популяции)
            """
        )
        st.info("Все данные в интерфейсе синтетические и генерируются локально. Реальные персональные записи не используются.")
    with intro_right:
        st.markdown("#### Формулы и пороги")
        st.markdown(
            """
            `growth_rate = (Новые_клиенты_текущего - Новые_клиенты_предыдущего) / max(Новые_клиенты_предыдущего, 1)`  
            `referral_ratio = Реферальные_клиенты_текущего / max(Новые_клиенты_текущего, 1)`  
            `payout_dependency = Общий_объем_выплат / max(Общий_входящий_поток, 1)`  
            `Risk = P(пирамида | признаки)`
            """
        )
        st.caption("Пороги интерпретации: < 0.4 — низкий риск, 0.4-0.7 — средний, >= 0.7 — высокий.")

    tab_single, tab_live, tab_guide = st.tabs(
        ["Оценка одного кейса", "Live-демонстрация", "Как читать систему"]
    )

    with tab_single:
        st.subheader("Быстрые пресеты")
        p1, p2, p3, p4 = st.columns(4)
        if p1.button("Пресет: Легит", use_container_width=True):
            _set_preset("Legit")
        if p2.button("Пресет: Подозрительно", use_container_width=True):
            _set_preset("Suspicious")
        if p3.button("Пресет: Пирамида", use_container_width=True):
            _set_preset("Pyramid")
        if p4.button("Случайный кейс", use_container_width=True):
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

        if st.button("Оценить кейс", type="primary", use_container_width=True):
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
            st.markdown(
                (
                    f"<div class='{_risk_css_class(level)}'>"
                    f"Вероятность риска: {risk_prob:.3f} | Уровень риска: {level}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            _, _, threshold_col = st.columns(3)
            with threshold_col:
                st.metric("Пороги", f"{RISK_THRESHOLDS['medium']:.1f} / {RISK_THRESHOLDS['high']:.1f}")

            st.write("Топ признаков (глобальная важность):")
            st.table(pd.DataFrame(top_features).rename(columns={"feature": "Признак", "importance": "Важность"}))

            if raw_operational_view is not None:
                with st.expander("Рассчитанные признаки модели из операционных фактов", expanded=False):
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
                with st.expander("Глобальная важность по модели (справочно)", expanded=False):
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

            st.markdown("#### Карта рисков популяции")
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
        st.subheader("Сценарий live-генерации синтетических данных")
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

        if st.button("Сгенерировать новый синтетический батч", type="primary"):
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

    with tab_guide:
        _render_system_guide()


if __name__ == "__main__":
    _guard_streamlit_entrypoint()
    main()
