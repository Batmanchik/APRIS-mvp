"""Поиск сетей: кандидаты предлагаются из потока событий, а не из ответов.

Эта страница заменила пакетный сканнер, и подмена здесь принципиальная.
Сканнер получал готовый список объектов и оценивал каждый по отдельности —
то есть самую трудную половину задачи, «какие счета вообще образуют одну
структуру», кто-то решал за него. В симуляции это решал файл с ответами,
и любая метрика после такого измеряла не детектор (Находка 4).

Здесь кластеры предлагает `discover_candidates`, который читает только
события: общий банкомат в узком окне, общий предок по деньгам, общий
получатель. Метки прикладываются после и нужны ровно для одной величины —
coverage, доли реальных сетей, которую предложение вообще накрыло. Это
потолок полноты: сеть, не попавшая ни в одного кандидата, не будет найдена
никакой моделью дальше.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from apris.frontend.session import (
    DEFAULT_SCALE,
    DEFAULT_SEED,
    SCALES,
    ensure_state,
    world_summary_rows,
)

st.set_page_config(page_title="Поиск сетей | Cheops AI", page_icon="🕸️", layout="wide")

st.title("🕸️ Поиск сетей в потоке событий")
st.caption(
    "Кандидаты строятся из событий. Файл с ответами при построении не читается — "
    "метки прикладываются после, чтобы измерить coverage."
)


with st.sidebar:
    st.markdown("### Параметры мира")
    scale_key = st.radio(
        "Масштаб",
        list(SCALES),
        format_func=lambda key: SCALES[key].label,
        index=list(SCALES).index(DEFAULT_SCALE),
    )
    st.caption(SCALES[scale_key].note)
    seed = st.number_input(
        "Seed", min_value=1, max_value=10_000, value=DEFAULT_SEED, step=1
    )
    st.button("Построить мир и найти кандидатов", type="primary", use_container_width=True)

with st.spinner("Генерация мира и поиск кандидатов…"):
    world, dataset = ensure_state(scale_key, int(seed))

st.markdown("#### Мир")
cols = st.columns(4)
for col, (label, value) in zip(cols, world_summary_rows(world)):
    col.metric(label, value)

st.markdown("#### Что предложил поиск")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Кандидатов", dataset.size)
c2.metric("Из них сети мулов", dataset.positives)
c3.metric("Базовая доля", f"{dataset.base_rate:.3f}")
c4.metric("Coverage по сетям мулов", f"{dataset.coverage:.3f}")

missed = dataset.discovery.missed_network_ids
if missed:
    st.warning(
        f"Не предложено ни разу: {len(missed)} сетей мулов из "
        f"{dataset.discovery.networks_total}. Это потолок полноты — их не найдёт "
        "никакая модель дальше по конвейеру."
    )
else:
    st.success(
        f"Все {dataset.discovery.networks_total} сетей мулов попали хотя бы в одного "
        "кандидата. Потолок полноты не ограничивает модель на этом мире."
    )
st.caption(
    "Coverage считается только по сетям мулов: пирамида — это не кластер счетов, "
    "а один организатор с потоком вкладчиков, и ищется она иначе. Поэтому число "
    "сетей в мире выше числа сетей, которые здесь оцениваются."
)

st.markdown("---")
st.markdown("#### Кандидаты")

if dataset.size == 0:
    st.info("Поиск не предложил ни одного кандидата. Попробуйте другой seed или масштаб.")
    st.stop()

rows = []
for candidate, label in zip(dataset.candidates, dataset.labels):
    stamps = [event.ts for event in candidate.events]
    rows.append(
        {
            "ID": candidate.candidate_id,
            "Счетов": candidate.size,
            "Событий": len(candidate.events),
            "Начало": min(stamps),
            "Длительность, ч": round((max(stamps) - min(stamps)).total_seconds() / 3600.0, 1),
            "Основания связи": ", ".join(candidate.link_reasons),
            "Метка": int(label),
        }
    )
frame = pd.DataFrame(rows)

show_labels = st.checkbox(
    "Показать метки (только для оценки, модели они недоступны)", value=False
)
view = frame if show_labels else frame.drop(columns=["Метка"])
st.dataframe(view.sort_values("Счетов", ascending=False), use_container_width=True, height=420)

st.caption(
    "Столбец «Основания связи» показывает, по каким общим ресурсам счета были "
    "соединены. Одно основание — совпадение; несколько независимых — сигнал: "
    "именно корроборация, а не поиск сообществ, отделяет структуры друг от друга."
)

st.markdown("#### Размеры кандидатов")
size_frame = pd.DataFrame({"Счетов в кандидате": frame["Счетов"]})
st.bar_chart(size_frame["Счетов в кандидате"].value_counts().sort_index())

st.info(
    "Дальше: откройте **Досье кандидата**, чтобы посмотреть конкретную структуру и "
    "оценку сервиса, или **Валидацию**, чтобы увидеть, сколько эта оценка стоит."
)
