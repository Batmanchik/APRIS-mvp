"""Очередь аналитика: то, что система кладёт человеку на стол.

Эта страница — единственное место в интерфейсе, где виден конечный продукт,
а не его части. Остальные показывают, как устроен конвейер; здесь стоит
результат его работы: список дел, порог, по которому он отрезан, и цена
этого порога в проверках на одну находку.

Три вещи здесь принципиальны.

**Очередь режется порогом, а не бюджетом.** Прогон по редкости (Р6) показал,
что «проверяем верхние 10 %» при редких мошенниках тратит почти весь бюджет
на честных людей. Порог под заданную полноту — правильная политика, и длина
очереди при ней **результат, а не настройка**: тихая неделя даёт короткий
список, и это верное поведение.

**Порог выбран на прошлом.** Он читается по предыдущим отрезкам walk-forward,
а очередь режется на последнем, которого модель не видела. Порог, подобранный
на тех же строках, по которым потом отчитываются, — самый старый способ
получить число, не воспроизводимое в понедельник.

**Метка показана последней колонкой и не участвовала ни в чём до отреза.**
В настоящем банке этой колонки нет; здесь она есть только затем, чтобы
страница могла сказать, сколько дел из списка оказались настоящими.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from apris.cheops.infrastructure.pipeline import QUEUE_PATH, read_queue

st.set_page_config(page_title="Очередь аналитика | Cheops AI", page_icon="📋", layout="wide")

st.title("📋 Очередь аналитика")
st.caption(
    "Результат полного конвейера: мир → поиск кандидатов → признаки → детектор → "
    "порог. Список отрезан порогом под заданную полноту, а не фиксированным бюджетом."
)

report = read_queue()

if report is None:
    st.warning(
        "Конвейер ещё не запускался. Соберите очередь одной командой:\n\n"
        "```\npython scripts/run_pipeline.py --preset full\n```\n\n"
        f"Она запишет `{QUEUE_PATH}`, и страница подхватит файл сама."
    )
    st.stop()

world = report.get("world", {})
head = st.columns(4)
head[0].metric("Мир", str(report.get("preset", "—")))
head[1].metric("Событий", f"{int(world.get('events', 0)):,}".replace(",", " "))
head[2].metric("Счетов", f"{int(world.get('accounts', 0)):,}".replace(",", " "))
head[3].metric("Прогон", str(report.get("run_id", "—")))
st.caption(
    f"Детектор {report.get('detector', '—')}, собрано {report.get('generated_at', '—')}, "
    f"{float(report.get('seconds', 0)):.0f} с на весь конвейер."
)

COLUMN_LABELS = {
    "rank": "№",
    "key": "объект",
    "score": "оценка",
    "members": "участников",
    "events": "событий",
    "amount_total": "сумма",
    "first_seen": "первое событие",
    "last_seen": "последнее",
    "truth": "настоящий",
}

for outcome in report.get("outcomes", []):
    st.markdown(f"### Очередь: {outcome['unit']}")

    cols = st.columns(5)
    cols[0].metric("Дел в очереди", outcome["queued"])
    cols[1].metric(
        "Из них настоящих",
        "—" if outcome["queued"] == 0 else f"{outcome['precision']:.0%}",
    )
    cols[2].metric("Поймано в блоке", f"{outcome['recall']:.0%}")
    cols[3].metric(
        "Проверок на находку",
        "—" if not outcome.get("reviews_per_catch") else f"{outcome['reviews_per_catch']:.1f}",
    )
    cols[4].metric("Потолок уровня", f"{outcome['unit_ceiling']:.3f}")

    st.caption(
        f"Порог {outcome['threshold']:.3f} выбран по прошлым отрезкам под полноту "
        f"{outcome['target_recall']:.0%}. Отрезан последний блок: "
        f"{outcome['block_rows']} строк, настоящих в нём {outcome['block_positives']} "
        f"(доля {outcome['block_prevalence']:.3f}). Потолок уровня — сколько объектов "
        "этот уровень анализа вообще способен увидеть до всякой модели."
    )

    items = outcome.get("items", [])
    if not items:
        st.info(
            f"Очередь пустая: порог не пропустил никого из {outcome['block_rows']} "
            f"строк блока. Это ответ «здесь тихо», а не ошибка — но на блоке такого "
            "размера он и не мог быть уверенным."
        )
        continue

    frame = pd.DataFrame(items)
    frame = frame[list(COLUMN_LABELS)]
    frame["truth"] = frame["truth"].map({1: "да", 0: "нет"})
    frame["amount_total"] = frame["amount_total"].map(
        lambda value: f"{value:,.0f}".replace(",", " ")
    )
    frame["score"] = frame["score"].map(lambda value: f"{value:.3f}")
    st.dataframe(
        frame.rename(columns=COLUMN_LABELS),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.markdown(
    "**Чего эта страница не показывает.** Мир смоделирован нами, поэтому колонка "
    "«настоящий» здесь есть, а в банке её нет — там цена очереди меряется тем, "
    "что вернул аналитик. Доля мошенников в этом мире выше жизненной; как меняются "
    "числа при настоящей редкости — в `docs/RESULTS.md`, результат Р6."
)
