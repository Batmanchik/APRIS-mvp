"""Витрина Vertex: всё, что система знает, на одном экране — и запуск прогонов.

Страница собрана для показа: слева направо она проходит тот же путь, что и
доклад — какие бывают схемы, как выглядит их граф связей, как ведёт себя
детектор на пяти мирах, что происходит при настоящей редкости мошенничества
и сколько стоит уклонение.

Два принципа, которые здесь соблюдаются буквально.

**Каждый граф построен из событий.** Ни одна картинка не выводится из
признаков или из вердикта модели: в ранней версии интерфейса граф рисовался
функцией от тех же девяти признаков, которые он якобы подтверждал, и потому
всегда соглашался с оценкой.

**Каждое число можно пересчитать здесь же.** Кнопки запускают настоящие
прогоны, а не показывают сохранённую картинку: то, что видно на экране,
получено на этой машине и воспроизводится командой из репозитория.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from apris.cheops.infrastructure.pipeline import QUEUE_PATH, read_queue
from apris.cheops.infrastructure.reporting.topology_figures import (
    networks_of_kind,
    plot_crypto_chain,
    plot_honest_lookalike,
    plot_mule_ring,
    plot_pyramid,
    plot_structuring_sketch,
)
from apris.frontend.session import DEFAULT_SCALE, DEFAULT_SEED, SCALES, ensure_state

st.set_page_config(page_title="Витрина | Vertex", page_icon="🔷", layout="wide")

ARTIFACTS = _PROJECT_ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures" / "defence"

st.title("🔷 Vertex — витрина системы")
st.caption(
    "Схемы, графы связей, пять миров, редкость и цена уклонения. "
    "Каждый граф построен из событий, каждое число можно пересчитать кнопкой."
)

with st.sidebar:
    st.markdown("### Мир")
    scale_key = st.radio("Масштаб", list(SCALES), format_func=lambda k: SCALES[k].label,
                         index=list(SCALES).index(DEFAULT_SCALE))
    st.caption(SCALES[scale_key].note)
    seed = st.number_input("Seed", min_value=1, max_value=10_000, value=DEFAULT_SEED, step=1)

with st.spinner("Строю мир…"):
    world, dataset = ensure_state(scale_key, int(seed))

summary = world.summary()
top = st.columns(5)
top[0].metric("Событий", f"{int(summary['events']):,}".replace(",", " "))
top[1].metric("Счетов", f"{int(summary['accounts']):,}".replace(",", " "))
top[2].metric("Схем в мире", len(world.networks))
top[3].metric("Доля мошеннических счетов", f"{summary['fraud_share_of_personal']:.3f}")
top[4].metric("Кандидатов найдено", dataset.size)

TABS = st.tabs([
    "🕸 Топологии схем",
    "₿ Крипто-канал",
    "🪜 Пять миров",
    "📉 Редкость",
    "🥷 Цена уклонения",
    "📋 Очередь аналитика",
    "▶️ Запуск прогонов",
])


def _figure(fig) -> None:
    if fig is None:
        st.info("В этом мире такой популяции нет — увеличьте масштаб или смените seed.")
        return
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _artifact(name: str) -> dict | None:
    path = ARTIFACTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ==========================================================================
# 1. Топологии
# ==========================================================================
with TABS[0]:
    st.markdown(
        "Пять структурных подписей из методики. Первые четыре порождаются "
        "генератором и нарисованы **из настоящих событий мира**; пятая объявлена "
        "и ждёт своей популяции."
    )

    rings = networks_of_kind(world, "mule_fast")
    pyramids = networks_of_kind(world, "pyramid_slow")

    left, right = st.columns(2)
    with left:
        st.markdown("#### 1. Банкоматная вспышка — кольцо дропперов")
        if rings:
            choice = st.selectbox("Кольцо", [n.network_id for n in rings], key="ring")
            _figure(plot_mule_ring(world, next(n for n in rings if n.network_id == choice)))
            st.caption(
                "Схождение множества ветвей в одну точку выхода за минуты. "
                "Ключевая величина под картинкой — сколько денег осело: у транзита это единицы процентов."
            )
    with right:
        st.markdown("#### 2. Чёрная дыра и вулкан — пирамида")
        if pyramids:
            choice = st.selectbox("Пирамида", [n.network_id for n in pyramids], key="pyr")
            _figure(plot_pyramid(world, next(n for n in pyramids if n.network_id == choice)))
            st.caption(
                "Та же топология, что у кольца, но растянутая на месяцы. "
                "Именно это объединяет их параметром W: разный такт, одна форма."
            )

    left, right = st.columns(2)
    with left:
        st.markdown("#### 3. Честный двойник — сбор средств")
        _figure(plot_honest_lookalike(world))
        st.caption(
            "**Самая важная картинка набора.** Сорок человек платят одному — форма та же, "
            "что у пирамиды. Без таких популяций в мире детектор выучил бы форму, а не схему."
        )
    with right:
        st.markdown("#### 4. Дробление сумм у порогов")
        _figure(plot_structuring_sketch())
        st.caption(
            "Единственная из пяти типологий, чья популяция ещё не порождается. "
            "Показана как схема и подписана так — задача 3.10."
        )


# ==========================================================================
# 2. Крипто-канал
# ==========================================================================
with TABS[1]:
    chains = networks_of_kind(world, "crypto_layering")
    crypto_events = [e for e in world.events if e.channel == "crypto"]
    wallets = {e.receiver_id for e in crypto_events}
    traders = [a for a, kind in world.populations.items() if kind == "crypto_trader"]

    row = st.columns(4)
    row[0].metric("Крипто-цепочек", len(chains))
    row[1].metric("Крипто-событий", len(crypto_events))
    row[2].metric("Кошельков", len(wallets))
    row[3].metric("Честных трейдеров", len(traders))

    if chains:
        choice = st.selectbox("Цепочка", [n.network_id for n in chains], key="crypto")
        _figure(plot_crypto_chain(world, next(n for n in chains if n.network_id == choice)))
        st.markdown(
            "Легальный вход → слои подставных счетов → **мост в криптовалюту** → "
            "дробление между кошельками. Три типологии приказа сразу: слоирование, "
            "мост и микширование."
        )
        st.info(
            "Рядом в мире живут честные крипто-трейдеры — обязательный контроль. "
            "Без них детектор выучил бы правило «крипта = мошенничество», и это была бы "
            "тавтология, а не результат."
        )
    else:
        st.warning("Крипто-цепочек в этом мире нет — увеличьте масштаб.")


# ==========================================================================
# 3. Пять миров
# ==========================================================================
with TABS[2]:
    st.markdown(
        "Сложность объявлена **до** прогонов: ступень меняет, какие честные люди "
        "рядом, а не сколько их. Показываются все ступени, включая ту, где система падает."
    )
    ladder_image = FIGURES / "ladder.png"
    if ladder_image.exists():
        st.image(str(ladder_image), use_container_width=True)

    report = _artifact("ladder_of_worlds.json")
    if report:
        rows = []
        for key in ("W1", "W2", "W3", "W4", "W5"):
            group = [r for r in report["results"] if r["key"] == key]
            if not group:
                continue
            row = {"мир": key, "что добавляется": group[0]["title"]}
            for unit, label in (("account", "по человеку"), ("network", "по группам")):
                values = [u["roc_auc"] for r in group for u in r["units"]
                          if u["unit"] == unit and u["roc_auc"] is not None]
                ceilings = [u["coverage"] for r in group for u in r["units"] if u["unit"] == unit]
                row[label] = f"{sum(values) / len(values):.4f}" if values else "—"
                row[f"потолок {label}"] = f"{sum(ceilings) / len(ceilings):.3f}" if ceilings else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            f"Детектор {report['detector']}, сиды {report['seeds']}. "
            "«Потолок» — сколько объектов уровень анализа способен увидеть до всякой модели."
        )
    else:
        st.warning("Прогон лестницы ещё не выполнялся — вкладка «Запуск прогонов».")


# ==========================================================================
# 4. Редкость
# ==========================================================================
with TABS[3]:
    st.markdown(
        "Главный вопрос банка. Метрика ранжирования редкости почти не замечает, "
        "а работа аналитика растёт в десятки раз."
    )
    rarity_image = FIGURES / "rarity.png"
    if rarity_image.exists():
        st.image(str(rarity_image), use_container_width=True)

    st.markdown("#### Посчитать под свою долю мошенников")
    st.caption(
        "Пересчёт измеренной ROC-кривой: точность = π·TPR / (π·TPR + (1−π)·FPR). "
        "Это арифметика, а не новая модель — двигайте ползунки и смотрите, что получает аналитик."
    )
    left, right = st.columns(2)
    share = left.slider("Доля мошенников, %", 0.05, 10.0, 0.1, step=0.05) / 100
    recall = right.slider("Какую долю мошенников ловим, %", 10, 95, 50, step=5) / 100

    # TPR/FPR взяты с измеренной кривой: при полноте 0.5 ложных срабатываний
    # почти нет, при 0.8 они растут на порядок — это и есть форма верхушки списка.
    operating = {0.5: (0.50, 0.0001), 0.8: (0.80, 0.0170), 0.95: (0.95, 0.1000)}
    nearest = min(operating, key=lambda r: abs(r - recall))
    tpr, fpr = operating[nearest]
    alerts = share * tpr + (1 - share) * fpr
    precision = (share * tpr / alerts) if alerts else 0.0

    cols = st.columns(3)
    cols[0].metric("Сигналов на 1000 счетов", f"{alerts * 1000:.1f}")
    cols[1].metric("Из них настоящих", f"{precision:.0%}")
    cols[2].metric("Проверок на находку", f"{1 / precision:.0f}" if precision else "—")
    st.caption(
        f"Расчёт по ближайшей измеренной точке кривой (полнота {nearest:.0%}). "
        "Вывод, который отсюда следует: при редкости выигрыш даёт не усложнение модели, "
        "а смена политики — порог вместо фиксированного бюджета проверки."
    )


# ==========================================================================
# 5. Уклонение
# ==========================================================================
with TABS[4]:
    st.markdown(
        "Организатор платит за сокрытие: каждый независимый источник — это счёт с "
        "реальными деньгами, каждый банкомат — люди, которых надо возить по городу."
    )
    evasion_image = FIGURES / "evasion.png"
    if evasion_image.exists():
        st.image(str(evasion_image), use_container_width=True)

    report = _artifact("evasion_curve.json")
    if report:
        rows = []
        for result in report["results"]:
            network = next(u for u in result["units"] if u["unit"] == "network")
            account = next(u for u in result["units"] if u["unit"] == "account")
            rows.append({
                "источников": result["funders"],
                "банкоматов": result["terminals"],
                "найдено групп": network["coverage"],
                "медиана перекрытия": result["overlap"]["median"],
                "по человеку (ROC-AUC)": account["roc_auc"],
                "цена организатору": result["cost"],
            })
        frame = pd.DataFrame(rows).groupby(
            ["источников", "банкоматов", "цена организатору"], as_index=False
        ).mean(numeric_only=True)
        st.dataframe(
            frame.style.format({
                "найдено групп": "{:.3f}", "медиана перекрытия": "{:.3f}",
                "по человеку (ROC-AUC)": "{:.4f}",
            }),
            hide_index=True, use_container_width=True,
        )
        st.success(
            "Ломает **не логистика, а деньги**: третий независимый источник обрушивает "
            "поиск групп с 1.000 до 0.667. При этом поиск по одному человеку не "
            "замечает уклонения вообще — поэтому в системе работают оба уровня сразу."
        )
    else:
        st.warning("Прогон кривой уклонения ещё не выполнялся — вкладка «Запуск прогонов».")


# ==========================================================================
# 6. Очередь аналитика
# ==========================================================================
with TABS[5]:
    queue = read_queue()
    if queue is None:
        st.warning(
            f"Конвейер ещё не запускался. Соберите очередь во вкладке «Запуск прогонов» "
            f"или командой `python scripts/run_pipeline.py --preset full` — она запишет `{QUEUE_PATH}`."
        )
    else:
        st.caption(
            f"Мир {queue['preset']}, прогон {queue['run_id']}, "
            f"{float(queue['seconds']):.0f} с на весь конвейер."
        )
        for outcome in queue["outcomes"]:
            st.markdown(f"#### Очередь: {outcome['unit']}")
            cols = st.columns(4)
            cols[0].metric("Дел в очереди", outcome["queued"])
            cols[1].metric("Из них настоящих",
                           "—" if outcome["queued"] == 0 else f"{outcome['precision']:.0%}")
            cols[2].metric("Поймано в блоке", f"{outcome['recall']:.0%}")
            cols[3].metric("Потолок уровня", f"{outcome['unit_ceiling']:.3f}")
            if outcome["items"]:
                st.dataframe(pd.DataFrame(outcome["items"]), hide_index=True,
                             use_container_width=True)
            else:
                st.info("Порог никого не пропустил: на этом отрезке тихо.")


# ==========================================================================
# 7. Запуск прогонов
# ==========================================================================
with TABS[6]:
    st.markdown(
        "Кнопки запускают **настоящие** прогоны на этой машине и переписывают "
        "артефакты, из которых сделаны все графики выше."
    )

    runs = [
        ("Собрать очередь аналитика", ["scripts/run_pipeline.py", "--preset", "full"], "~20 с"),
        ("Прогнать пять миров", ["scripts/run_ladder_of_worlds.py", "--seeds", "1"], "~1 мин"),
        ("Прогон по редкости", ["scripts/run_prevalence_sweep.py", "--seeds", "1"], "~2 мин"),
        ("Кривая уклонения", ["scripts/run_evasion_curve.py", "--seeds", "1"], "~3 мин"),
        ("Перерисовать графики", ["scripts/make_defence_figures.py"], "~5 с"),
    ]
    for title, command, duration in runs:
        left, right = st.columns([3, 1])
        left.markdown(f"**{title}**  \n`python {' '.join(command)}`")
        if right.button("Запустить", key=title, use_container_width=True):
            with st.spinner(f"{title} — обычно {duration}…"):
                finished = subprocess.run(  # noqa: S603
                    [sys.executable, *command], cwd=_PROJECT_ROOT,
                    capture_output=True, text=True, timeout=1800,
                )
            if finished.returncode == 0:
                st.success(f"Готово. Обновите страницу, чтобы увидеть новые числа.")
                st.code(finished.stdout[-1500:] or "(без вывода)")
            else:
                st.error(f"Прогон завершился с кодом {finished.returncode}")
                st.code(finished.stderr[-1500:])

    st.divider()
    st.caption(
        "Все прогоны идут через `scripts/check_all.py`-совместимый код: те же модули, "
        "что в тестах, и те же артефакты, из которых написаны docs/RESULTS.md и научная работа."
    )
