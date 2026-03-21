"""
Дашборд аномалий (Alert Inbox).
Основной экран банковского аналитика. Показывает результаты последнего сканирования,
отсортированные по уровню риска. При клике на компанию открывается её "Досье".
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import base64
import hashlib
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from apris.frontend import api_client
from apris.crypto_ponzi.tx_generator import generate_company_transactions, PRESET_CONFIGS
from apris.crypto_ponzi.aggregator import aggregate_transactions
from apris.crypto_ponzi.visualizations import (
    plot_counterparty_network,
    plot_inflow_structure_pie,
)


st.set_page_config(page_title="Дашборд аномалий | Cheops AI", page_icon="🚨", layout="wide")


# ── Styles ────────────────────────────────────────────────────────
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

# ── Main Page ─────────────────────────────────────────────────────

st.title("🚨 Дашборд выявленных аномалий")

if "scan_results" not in st.session_state:
    st.info("Нет данных сканирования. Перейдите в модуль **📡 Сканнер транзакций** и запустите обработку реестра.")
    if st.button("Перейти в Сканнер"):
        st.switch_page("pages/2_Сканнер_транзакций.py")
    st.stop()

df = st.session_state["scan_results"]
scan_time = st.session_state.get("last_scan_time", "Неизвестно")

total_scanned = len(df)
critical = len(df[df["risk_prob"] >= 0.7])
suspicious = len(df[(df["risk_prob"] >= 0.4) & (df["risk_prob"] < 0.7)])

st.markdown(f"**Последнее обновление:** {scan_time}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{total_scanned}</div><div class='metric-label'>Субъектов проверено</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='metric-card'><div class='metric-val red-val'>{critical}</div><div class='metric-label'>Критических (Пирамиды)</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#eab308;'>{suspicious}</div><div class='metric-label'>Подозрительных (Наблюдение)</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#10a37f;'>{total_scanned - critical - suspicious}</div><div class='metric-label'>Легитимных (Чистая база)</div></div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("📋 Лидерборд рисков (Alert Inbox)")

# Prepare display table
display_df = df.copy()
display_df = display_df.sort_values(by="risk_prob", ascending=False)
display_cols = ["company_id", "risk_prob", "risk_level"]

# Let the user select an entity to view its dossier
def format_row(row):
    emoji = "🚨" if row["risk_prob"] >= 0.7 else ("⚠️" if row["risk_prob"] >= 0.4 else "✅")
    return f"{emoji} {row['company_id']} — Вероятность: {row['risk_prob']:.2%} [{row['risk_level']}]"

selected_label = st.selectbox(
    "Выберите подозрительный объект для детального расследования (Досье):", 
    display_df.head(50).apply(format_row, axis=1).tolist()
)

if selected_label:
    # Extract ID from selected label
    selected_id = selected_label.split(" — ")[0].split(" ")[1]
    entity_data = display_df[display_df["company_id"] == selected_id].iloc[0]
    
    # ── DOSSIER RENDERING ──────────────────────────────────────────
    prob = float(entity_data['risk_prob'])
    badge_class = "badge-critical" if prob >= 0.7 else ("badge-warning" if prob >= 0.4 else "badge-ok")
    
    st.markdown(
        f"""
        <div class='dossier-header'>
            <h2 style='margin:0; font-size:1.5rem; letter-spacing:1px;'>ДОСЬЕ ОБЪЕКТА: <code>{selected_id}</code></h2>
        </div>
        """, unsafe_allow_html=True
    )
    
    with st.container():
        st.markdown("<div class='dossier-body'>", unsafe_allow_html=True)
        
        col_summary, col_ml = st.columns([1, 1.2])
        
        with col_summary:
            st.markdown(f"#### Вердикт системы: <span class='risk-badge {badge_class}'>{entity_data['risk_level']} ({prob:.1%})</span>", unsafe_allow_html=True)
            st.markdown("Подозрение на мультиканальные махинации по паттернам распределения потоков и контрагентов.")
            
            # Request explainability directly from FastAPI via api_client
            # Prepare features for the API
            model_features = entity_data.drop(["company_id", "risk_prob", "risk_level", "pred_label", "label", "is_borderline"], errors="ignore").to_dict()
            try:
                explanations = api_client.explain_features(model_features, top_k=3)
                
                st.markdown("#### 🚩 Топ-3 индикатора (почему поднят алерт):")
                for exp in explanations:
                    st.markdown(f"**{exp['feature']}**: Вес: `{exp['importance']:.3f}`  \n"
                                f"<small style='color: #6e6e80;'>Значение у объекта: {model_features[exp['feature']]:.2f}</small>", 
                                unsafe_allow_html=True)
            except Exception as e:
                st.warning("Не удалось получить Explainability с сервера API.")
        
        with col_ml:
            st.markdown("#### 🕸️ Крипто-сеть транзакций (Подозрительный паттерн)")
            
            # Since this is a demo based on synthetic data, if the risk is high, we map it to PYRAMID pattern
            preset = "PYRAMID" if prob >= 0.7 else ("DANGEROUS" if prob >= 0.4 else "SAFE")
            
            with st.spinner("Рендеринг транзакционного графа смарт-контракта..."):
                # Simulate parsing the specific subgraph for this entity using our crypto_ponzi engine
                fake_company_name = f"Smart Contract {selected_id[:6]}"
                seed = int(hashlib.sha256(selected_id.encode("utf-8")).hexdigest()[:8], 16)
                case = generate_company_transactions(preset, seed=seed)
                case["company_name"] = fake_company_name
                transactions = case["transactions"]
                
                # Render graph & pie side by side
                g1, g2 = st.columns(2)
                with g1:
                    fig_network = plot_counterparty_network(transactions, company_name=fake_company_name)
                    _render_figure_inline(fig_network)
                    st.caption("Структура переводов (Центральный узел — смарт-контракт)")
                with g2:
                    metrics = aggregate_transactions(transactions)
                    fig_pie = plot_inflow_structure_pie(metrics)
                    _render_figure_inline(fig_pie)
                    st.caption("Концентрация входящих кошельков")
                    
        st.markdown("</div>", unsafe_allow_html=True)
