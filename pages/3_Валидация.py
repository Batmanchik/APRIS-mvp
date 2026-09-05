"""Валидация: сколько стоит оценка, которую показывает досье.

Все цифры на этой странице считаются здесь и сейчас на том мире, который
построен на странице поиска. Ни одна не зашита в код — в этом смысл:
результат, который нельзя перезапустить, это утверждение, а не результат.

Три вещи показаны рядом намеренно.

**Purged walk-forward** — обучение только на прошлом, тест на следующем
блоке, и всё, что попадает в зазор перед границей, выбрасывается из
обучения, а не тихо остаётся в нём.

**Лестница квинтилей** — один AUC говорит, что оценка где-то разделяет.
Лестница показывает, упорядочивает ли она популяцию. Оценка может иметь
хороший AUC и ломаться на верхнем квинтиле, и тогда порог двигать нельзя.

**Наивное правило** — то, что написал бы любой человек первым делом.
Оно ловит честных людей чаще, чем мулов, и это тоже измерение.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from apris.cheops.infrastructure.ml.case_pipeline import run_case_validation
from apris.cheops.infrastructure.simulation.acceptance import evaluate as evaluate_acceptance
from apris.frontend.session import current_state

st.set_page_config(page_title="Валидация | Cheops AI", page_icon="📐", layout="wide")

FIGURE_DIR = _PROJECT_ROOT / "artifacts" / "figures"

st.title("📐 Валидация")
st.caption("Цифры считаются на текущем мире при открытии страницы, а не берутся из документа.")

with st.spinner("Подготовка мира…"):
    world, dataset = current_state()


@st.cache_data(show_spinner=False)
def _validation(cache_key: tuple[str, int, int]):
    return run_case_validation(dataset)


@st.cache_data(show_spinner=False)
def _acceptance(cache_key: tuple[str, int, int]):
    return evaluate_acceptance(world)


cache_key = (str(st.session_state.get("cheops_scale")), dataset.size, dataset.positives)

with st.spinner("Purged walk-forward…"):
    report = _validation(cache_key)

# ── Итог ──────────────────────────────────────────────────────────
st.markdown("#### Классификация кандидатов")

if report.roc_auc is None:
    st.warning(
        f"Метрика не посчитана: {report.note}.\n\n"
        "Это не поломка, а отказ выдавать цифру, которую не на чем измерить. "
        "Возьмите полный мир на странице поиска."
    )
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC (out-of-fold)", f"{report.roc_auc:.4f}")
    c2.metric("PR-AUC", f"{report.pr_auc:.4f}")
    c3.metric("Базовая доля", f"{report.base_rate:.3f}")
    c4.metric("Оценённых фолдов", f"{report.scored_folds} из {len(report.folds)}")

    if report.roc_auc >= 0.99:
        st.error(
            "**Почти идеальное разделение — это симптом, а не достижение.** "
            "В этом проекте такой результат уже дважды означал дефект: сначала "
            "равномерные интервалы в генераторе дали AUC 1.0000 на возрасте "
            "счёта, потом кейсы собирались из файла с ответами и разделялись "
            "по любому столбцу. Здесь наиболее вероятная причина проще: на "
            "быстром мире доля мошеннических кандидатов "
            f"({report.base_rate:.3f}) непропорционально высока, а честных "
            "популяций мало. Возьмите полный мир."
        )

    if report.ladder is not None:
        if report.ladder.monotonic:
            st.success(f"Лестница квинтилей: {report.ladder.describe()}")
        else:
            st.warning(
                f"Лестница квинтилей: {report.ladder.describe()}. "
                "Оценка разделяет, но не упорядочивает — один AUC это скрыл бы полностью."
            )

st.metric("Coverage поиска", f"{report.coverage:.3f}")
st.caption(
    "Потолок полноты: доля реальных сетей, попавших хотя бы в одного кандидата. "
    "При старой схеме, где кейсы собирались из файла с ответами, эта величина "
    "равнялась единице по построению и ничего не значила."
)

# ── Фолды ─────────────────────────────────────────────────────────
if report.folds:
    st.markdown("#### Фолды")
    fold_rows = [
        {
            "Фолд": fold.index,
            "Обучение": fold.train_size,
            "Тест": fold.test_size,
            "Выброшено зазором": fold.purged,
            "Сетей в тесте": fold.positives_in_test,
            "ROC-AUC": "—" if fold.roc_auc is None else round(fold.roc_auc, 4),
        }
        for fold in report.folds
    ]
    st.dataframe(pd.DataFrame(fold_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Фолды без оценки не выброшены, а показаны: в их обучающем блоке остался "
        "один класс. Спрятать их значило бы показать метрику и скрыть, на скольких "
        "фолдах она получена."
    )

# ── Лестница ──────────────────────────────────────────────────────
if report.ladder is not None:
    st.markdown("#### Лестница квинтилей")
    ladder_frame = pd.DataFrame(
        {
            "Квинтиль": [f"Q{i + 1}" for i in range(len(report.ladder.bucket_rates))],
            "Доля сетей": list(report.ladder.bucket_rates),
        }
    ).set_index("Квинтиль")
    st.bar_chart(ladder_frame)
    st.caption(
        "Слева — кандидаты с самой низкой оценкой, справа — с самой высокой. "
        "У честного сигнала доля растёт слева направо."
    )

# ── Признаки по одному ────────────────────────────────────────────
if report.single_feature_auc:
    st.markdown("#### Каждый признак по отдельности")
    feature_frame = (
        pd.DataFrame(
            [{"Признак": k, "AUC in-sample": v} for k, v in report.single_feature_auc.items()]
        )
        .sort_values("AUC in-sample", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(feature_frame, use_container_width=True, hide_index=True)
    if report.roc_auc is not None:
        best = feature_frame.iloc[0]
        st.caption(
            f"Сильнейший одиночный признак — {best['Признак']} с AUC "
            f"{best['AUC in-sample']:.4f} на всей выборке, тогда как модель целиком "
            f"даёт {report.roc_auc:.4f} вне обучения. Эти числа несравнимы напрямую, "
            "но разрыв показывает, сколько отбирает временной сплит: признак, "
            "разделяющий весь датасет, не обязан переноситься вперёд по времени."
        )

# ── Наивное правило ───────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Наивное правило: во что обходится очевидное решение")

with st.spinner("Проверка наивных правил…"):
    acceptance = _acceptance(cache_key)

check_rows = [
    {"Проверка": check.name, "Пройдена": "да" if check.passed else "нет", "Детали": check.detail}
    for check in acceptance.checks
]
st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)
st.caption(
    "Правило «быстрый вывод после поступления» срабатывает на студентах чаще, "
    "чем на мулах. Система, построенная на первом же очевидном признаке, указывала "
    "бы не на мошенничество, а мимо него."
)

# ── Фигуры ────────────────────────────────────────────────────────
figures = sorted(FIGURE_DIR.glob("*.png")) if FIGURE_DIR.exists() else []
if figures:
    st.markdown("---")
    st.markdown("#### Фигуры отчёта")
    st.caption("Собраны отдельным прогоном и лежат в artifacts/figures.")
    for path in figures:
        st.image(str(path), use_container_width=True)
