"""Досье кандидата: всё на экране посчитано по его собственным событиям.

Прошлая версия этой страницы рисовала «крипто-сеть транзакций» так: брала
вероятность риска, выбирала по ней пресет (PYRAMID / DANGEROUS / SAFE),
генерировала транзакции из хеша идентификатора объекта и показывала
получившуюся картинку как структуру этого объекта. Картинка всегда
подтверждала вердикт, потому что была из него выведена.

Здесь граф строится из событий кандидата, признаки считаются теми же
функциями, что и в ветках модели, а оценку возвращает сервис: интерфейс не
загружает модель и не считает inference сам. Режимы веток показаны как есть —
если ветка работает эвристикой, а не обученной моделью, это написано на
экране.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from apris.cheops.infrastructure.ml.case_pipeline import case_features
from apris.frontend import api_client
from apris.frontend.candidate_view import (
    candidate_payload,
    events_table,
    features_table,
    plot_candidate_graph,
    plot_feature_bars,
    window_hours_for,
)
from apris.frontend.session import current_state

st.set_page_config(page_title="Досье кандидата | Cheops AI", page_icon="🗂️", layout="wide")

FEATURE_LABELS = {
    "graph_density": "плотность графа переводов",
    "graph_hub_share": "доля входящей суммы у крупнейшего получателя (схождение)",
    "graph_fanout_share": "доля исходящей суммы у крупнейшего отправителя (расхождение)",
    "graph_relay_share": "доля средств, прошедших транзитом от источника к другому выходу",
    "graph_weight_cv_norm": "разброс сумм по рёбрам",
    "event_rate_hour": "интенсивность событий в час",
    "burst_ratio_90s": "наибольшая доля событий в одном окне 90 секунд",
    "median_delta_inverse": "обратная медиана интервала между событиями",
    "amount_cv_norm": "разброс сумм по событиям",
    "unique_sender_ratio": "доля уникальных отправителей",
}

BRANCH_MODE_LABELS = {
    "trained_tabular_v2": "обученная модель",
    "trained_sequence_v2": "обученная модель",
    "trained_graph_v2": "обученная модель",
    "trained_logistic_meta": "обученная мета-модель",
    "heuristic_proxy": "эвристика, не обученная модель",
    "legacy_or_proxy": "легаси-модель или прокси",
    "weighted_fallback": "фиксированные веса, не обучение",
}

st.title("🗂️ Досье кандидата")

with st.spinner("Подготовка кандидатов…"):
    _world, dataset = current_state()

if dataset.size == 0:
    st.info(
        "На текущем мире поиск не предложил ни одного кандидата. "
        "Смените seed или масштаб на странице **Поиск сетей**."
    )
    st.stop()

by_id = {candidate.candidate_id: candidate for candidate in dataset.candidates}
labels = dict(zip(by_id, dataset.labels.tolist()))

order = sorted(by_id, key=lambda cid: by_id[cid].size, reverse=True)
selected_id = st.selectbox(
    "Кандидат",
    order,
    format_func=lambda cid: f"{cid} — {by_id[cid].size} счетов, {len(by_id[cid].events)} событий",
)
candidate = by_id[selected_id]

st.caption(
    "Кандидат предложен по общим ресурсам: "
    + ", ".join(candidate.link_reasons)
    + ". Метка известна только для оценки и модели не передаётся."
)

# ── Оценка сервиса ────────────────────────────────────────────────
payload = candidate_payload(
    selected_id, candidate.events, window_hours=window_hours_for(candidate.events)
)
sent = len(payload["events"])

score: dict | None = None
explanation: dict | None = None
api_error: str | None = None
try:
    score = api_client.score_case_v2(payload)
    explanation = api_client.explain_case_v2(payload)
except api_client.ApiClientError as exc:
    api_error = str(exc)
except Exception as exc:  # соединение, таймаут, недоступный сервис
    api_error = str(exc)

if api_error is not None:
    st.error(
        "Сервис оценки недоступен, поэтому оценка не показана. "
        "Интерфейс намеренно не считает inference сам: то, что видит аналитик, "
        "должно приходить из того же сервиса, что работает в проде.\n\n"
        f"Причина: {api_error}"
    )
    st.code("python -m uvicorn apris.api.main:app --port 8000", language="bash")
else:
    assert score is not None and explanation is not None
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Общий риск", f"{float(score['global_risk']):.3f}")
    c2.metric("Полоса", str(score["risk_band"]))
    c3.metric(
        "Версия модели / калибровки",
        f"{score['model_version']} / {score['calibration_version']}",
    )
    if sent < len(candidate.events):
        st.caption(
            f"В запрос ушли последние {sent} событий из {len(candidate.events)} — "
            "хвост окна, а не случайная выборка."
        )

    st.markdown("#### Ветки модели")
    branch_scores = dict(explanation["branch_scores"])
    branch_modes = dict(explanation["branch_modes"])
    branch_rows = [
        {
            "Ветка": name,
            "Оценка": float(value),
            "Режим": BRANCH_MODE_LABELS.get(branch_modes.get(name, ""), branch_modes.get(name, "")),
        }
        for name, value in branch_scores.items()
    ]
    st.dataframe(pd.DataFrame(branch_rows), use_container_width=True, hide_index=True)

    proxies = [
        row["Ветка"]
        for row in branch_rows
        if branch_modes.get(row["Ветка"], "").startswith(("heuristic", "weighted", "legacy"))
    ]
    if proxies:
        st.warning(
            "Не всё здесь — обученные модели: "
            + ", ".join(proxies)
            + ". Пока артефакты этих веток не обучены, их вклад в общий риск нельзя "
            "предъявлять как результат обучения."
        )

    st.markdown("#### Типологии")
    typology_frame = pd.DataFrame(
        [{"Типология": k, "Вероятность": float(v)} for k, v in score["typology_probs"].items()]
    ).sort_values("Вероятность", ascending=False)
    st.dataframe(typology_frame, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Структура и признаки ──────────────────────────────────────────
left, right = st.columns([1.15, 1])

with left:
    st.markdown("#### Структура переводов")
    st.pyplot(plot_candidate_graph(candidate.events), use_container_width=True)

with right:
    st.markdown("#### Признаки, посчитанные по событиям")
    features = case_features(candidate)
    st.pyplot(plot_feature_bars(features), use_container_width=True)

with st.expander("Что означает каждый признак", expanded=False):
    st.dataframe(
        features_table(features, FEATURE_LABELS), use_container_width=True, hide_index=True
    )
    st.caption(
        "graph_relay_share — доля средств, прошедшая от одного доминирующего источника "
        "к другому доминирующему выходу через посредников. На симуляции он разделяет "
        "сети от всего остального; на реальном датасете Elliptic он дал AUC 0.515, и "
        "этот отрицательный результат стоит в аудите как есть."
    )

with st.expander(f"События кандидата ({len(candidate.events)})", expanded=False):
    st.dataframe(events_table(candidate.events), use_container_width=True, hide_index=True)

with st.expander("Счета кандидата", expanded=False):
    st.write(", ".join(candidate.member_ids))

with st.expander("Метка (только для оценки)", expanded=False):
    st.write("Сеть" if labels[selected_id] else "Не сеть")
    st.caption(
        "Метка приложена после того, как кандидат был предложен. Ни поиск, ни признаки, "
        "ни модель её не видят."
    )
