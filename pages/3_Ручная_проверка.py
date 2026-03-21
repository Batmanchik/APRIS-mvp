"""
Page 1: Оценка одного кейса (Single Case Assessment).

Этот файл содержит всю логику вкладки «Оценка кейса»,
включая ввод признаков, вызов модели и визуализацию результатов.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so shared helpers can be imported.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import base64
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from apris.data_generator import FEATURE_BOUNDS, FEATURE_COLUMNS, RISK_THRESHOLDS
from apris.graph_module import build_transaction_graph, compute_hub_metrics, plot_graph
from apris.population_map import (
    build_population_map_figure,
    check_feature_order,
    check_no_nan,
    check_pca_dimensions,
    fit_population_pca,
    load_feature_names as load_population_feature_names,
    load_population_dataset,
    project_current_case,
)
from apris.risk_engine import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    OPERATIONAL_INPUT_BOUNDS,
    explain,
    load_artifacts,
    operational_to_features,
    predict_risk,
)

# ── Constants ─────────────────────────────────────────────────────
PRESETS: dict[str, dict[str, float]] = {
    "Legit": {
        "growth_rate": 0.08, "referral_ratio": 0.20, "payout_dependency": 0.55,
        "centralization_index": 0.25, "avg_holding_time": 60.0, "reinvestment_rate": 0.42,
        "gini_coefficient": 0.34, "transaction_entropy": 3.8, "structural_depth": 4.0,
    },
    "Suspicious": {
        "growth_rate": 0.22, "referral_ratio": 0.56, "payout_dependency": 0.92,
        "centralization_index": 0.54, "avg_holding_time": 34.0, "reinvestment_rate": 0.60,
        "gini_coefficient": 0.56, "transaction_entropy": 2.4, "structural_depth": 8.0,
    },
    "Pyramid": {
        "growth_rate": 0.62, "referral_ratio": 0.86, "payout_dependency": 1.35,
        "centralization_index": 0.86, "avg_holding_time": 14.0, "reinvestment_rate": 0.82,
        "gini_coefficient": 0.84, "transaction_entropy": 1.1, "structural_depth": 12.0,
    },
}

OPERATIONAL_PRESETS: dict[str, dict[str, float]] = {
    "Legit": {
        "tx_count_total": 8000.0, "unique_counterparties": 2200.0,
        "new_clients_current": 260.0, "new_clients_previous": 245.0,
        "referred_clients_current": 70.0, "incoming_funds": 1_800_000.0,
        "payouts_total": 1_050_000.0, "top1_wallet_share": 0.18,
        "top10_wallet_share": 0.46, "avg_holding_days": 62.0,
        "repeat_investor_share": 0.44, "max_referral_depth": 4.0,
    },
    "Suspicious": {
        "tx_count_total": 12000.0, "unique_counterparties": 1800.0,
        "new_clients_current": 420.0, "new_clients_previous": 300.0,
        "referred_clients_current": 190.0, "incoming_funds": 2_000_000.0,
        "payouts_total": 1_200_000.0, "top1_wallet_share": 0.32,
        "top10_wallet_share": 0.62, "avg_holding_days": 34.0,
        "repeat_investor_share": 0.60, "max_referral_depth": 8.0,
    },
    "Pyramid": {
        "tx_count_total": 32000.0, "unique_counterparties": 1300.0,
        "new_clients_current": 1500.0, "new_clients_previous": 650.0,
        "referred_clients_current": 1220.0, "incoming_funds": 3_800_000.0,
        "payouts_total": 4_500_000.0, "top1_wallet_share": 0.72,
        "top10_wallet_share": 0.92, "avg_holding_days": 15.0,
        "repeat_investor_share": 0.86, "max_referral_depth": 13.0,
    },
}

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


# ── Helpers ───────────────────────────────────────────────────────

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
        st.code("python -m apris.train_model")
        st.stop()
    try:
        return load_artifacts()
    except Exception as exc:
        st.error(f"Ошибка загрузки артефактов модели: {exc}")
        st.stop()


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
        "low": "Низкий", "medium": "Средний", "high": "Высокий",
        "низкий": "Низкий", "средний": "Средний", "высокий": "Высокий",
    }
    return mapping.get(level.lower(), level)


def _risk_emoji(level: str) -> str:
    level_norm = level.lower()
    if level_norm in {"low", "низкий"}:
        return "✅"
    if level_norm in {"medium", "средний"}:
        return "⚠️"
    return "🚨"


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


def _sample_population_case(dataset: pd.DataFrame) -> dict[str, float]:
    row = dataset.sample(n=1).iloc[0]
    return {feature: float(row[feature]) for feature in FEATURE_COLUMNS}


# ── Input renderers ───────────────────────────────────────────────

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
        "tx_count_total", "unique_counterparties", "new_clients_current",
        "new_clients_previous", "referred_clients_current", "max_referral_depth",
    }
    money_fields = {"incoming_funds", "payouts_total"}

    for idx, (key, (low, high)) in enumerate(OPERATIONAL_INPUT_BOUNDS.items()):
        col = cols[idx % 3]
        with col:
            label = OPERATIONAL_LABELS.get(key, key)
            current = float(st.session_state[key])
            if key in int_fields:
                val = st.number_input(label, min_value=int(low), max_value=int(high),
                    value=int(round(current)), step=1, format="%d", key=f"op_{key}")
            elif key in money_fields:
                val = st.number_input(label, min_value=float(low), max_value=float(high),
                    value=float(current), step=1000.0, format="%.2f", key=f"op_{key}")
            else:
                val = st.number_input(label, min_value=float(low), max_value=float(high),
                    value=float(current), step=0.0001, format="%.4f", key=f"op_{key}")
            st.session_state[key] = float(val)
            raw[key] = float(st.session_state[key])
    derived = operational_to_features(raw)
    return raw, derived


def _render_feature_input_main() -> dict[str, float]:
    st.markdown("#### 🔬 Признаки модели (расширенный режим)")
    mode = st.radio("Режим ввода", ["Точный ввод", "Быстрые слайдеры"],
                    horizontal=True, index=0, key="feature_input_mode")
    cols = st.columns(3)
    features: dict[str, float] = {}
    for idx, feature in enumerate(FEATURE_COLUMNS):
        low, high = FEATURE_BOUNDS[feature]
        current = float(st.session_state[feature])
        label = FEATURE_LABELS[feature]
        with cols[idx % 3]:
            if feature == "structural_depth":
                if mode == "Точный ввод":
                    val = st.number_input(label, min_value=int(low), max_value=int(high),
                        value=int(round(current)), step=1, format="%d", key=f"adv_{feature}")
                else:
                    val = st.slider(label, min_value=int(low), max_value=int(high),
                        value=int(round(current)), step=1, key=f"adv_{feature}")
            else:
                precise_step = 0.1 if feature == "avg_holding_time" else 0.0001
                precise_format = "%.1f" if feature == "avg_holding_time" else "%.4f"
                slider_step = 0.1 if feature == "avg_holding_time" else 0.01
                if mode == "Точный ввод":
                    val = st.number_input(label, min_value=float(low), max_value=float(high),
                        value=float(current), step=float(precise_step), format=precise_format,
                        key=f"adv_{feature}")
                else:
                    val = st.slider(label, min_value=float(low), max_value=float(high),
                        value=float(current), step=float(slider_step), key=f"adv_{feature}")
            st.session_state[feature] = float(val)
            features[feature] = float(val)
    return features


# ── Plot helpers ──────────────────────────────────────────────────

def _plot_case_signal_breakdown(
    features: dict[str, float], feature_names: list[str], importances: np.ndarray,
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
        if idx == 0:   colors.append("#ef4444")
        elif idx == 1: colors.append("#f97316")
        elif idx == 2: colors.append("#eab308")
        else:          colors.append("#10a37f")
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
            idx, "ТОП", va="center", fontsize=9, color="#991B1B", fontweight="bold",
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


# ── Main page logic ──────────────────────────────────────────────

_ensure_defaults()
model, feature_names = _ensure_model_available()

# Load population data
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

st.header("🔍 Оценка одного кейса")

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
    horizontal=True, key="input_source_mode",
)

if input_source == "Операционные факты (рекомендуется)":
    raw_operational, features = _render_operational_input_main()
else:
    raw_operational, features = None, _render_feature_input_main()

st.markdown("")
if st.button("🔍 Оценить кейс", type="primary", use_container_width=True):
    try:
        st.session_state["single_result"] = predict_risk(features, model=model, feature_names=feature_names)
        st.session_state["single_explain"] = explain(features, model=model, feature_names=feature_names)
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
        f"<div class='{_risk_css_class(level)}'>"
        f"{emoji} Вероятность риска: <strong>{risk_prob:.3f}</strong> &nbsp;|&nbsp; Уровень риска: <strong>{level}</strong>"
        "</div>",
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
                pd.DataFrame([{"Признак": k, "Значение": float(v)} for k, v in object_features.items()])
            )

    c1, c2 = st.columns(2)
    with c1:
        _plot_case_signal_breakdown(
            object_features, feature_names, np.asarray(model.feature_importances_, dtype=float),
        )
        with st.expander("📊 Глобальная важность по модели (справочно)", expanded=False):
            _plot_global_feature_importance(feature_names, np.asarray(model.feature_importances_, dtype=float))
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
        current_point = project_current_case(object_features, feature_names, pop_scaler, pop_pca)
        pop_fig = build_population_map_figure(projected_population, current_point=current_point)
        _render_figure_inline(pop_fig)
        st.caption("PCA-проекция синтетической популяции с наложением текущего кейса.")
else:
    st.info("Заполните значения и нажмите «Оценить кейс».")
