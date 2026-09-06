"""
Cheops AI - Система интеллектуального анализа мультиканальных рисков.

Главная точка входа Streamlit. Основная логика находится в pages/.
Этот модуль применяет общие стили и отображает стартовый экран.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_ASSETS_DIR = _SRC_DIR / "apris" / "frontend" / "assets"


def _read_asset(name: str) -> str:
    path = _ASSETS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Ресурс не найден: {path}")
    return path.read_text(encoding="utf-8")


def _render_html(body: str) -> None:
    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        html_renderer(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def _guard_streamlit_entrypoint() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is None:
            print("Это Streamlit-приложение.")
            print("Запуск: streamlit run app.py")
            raise SystemExit(0)
    except Exception:
        return


def _set_style() -> None:
    css = _read_asset("streamlit_theme.css")
    _render_html(
        f"""
        <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"> 
        <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
        <link href=\"https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,200..700,0..1,-50..200\" rel=\"stylesheet\">
        <link href=\"https://fonts.googleapis.com/icon?family=Material+Icons\" rel=\"stylesheet\">
        <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap\" rel=\"stylesheet\"> 
        <style>{css}</style>
        """
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


def main() -> None:
    st.set_page_config(
        page_title="Cheops AI - Интеллектуальный скрининг рисков",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _set_style()
    _apply_matplotlib_theme()

    _render_html(_read_asset("hero.html"))

    with st.expander("Как устроен интерфейс", expanded=True):
        c1, c2 = st.columns([1.25, 1])
        with c1:
            st.markdown(
                """
                1. **Поиск сетей** — кандидаты строятся из потока событий.
                   Файл с ответами при этом не читается, поэтому coverage
                   (доля реальных сетей, попавших хотя бы в одного кандидата)
                   является настоящим потолком полноты.
                2. **Досье кандидата** — структура, признаки и оценка сервиса
                   для выбранного кластера. Всё посчитано по его событиям.
                3. **Валидация** — purged walk-forward, лестница квинтилей и
                   проверка наивного правила. Цифры считаются при открытии
                   страницы, а не берутся из документа.
                4. **Ручная проверка** — девять признаков исходной модели для
                   ручного ввода, с явной оговоркой о некалиброванных порогах.
                """
            )
        with c2:
            st.markdown("**Что здесь не показывается**")
            st.markdown(
                """
                - Граф, построенный из признаков, которые он подтверждает.
                - Картинка структуры, выведенная из вердикта.
                - Метрика, посчитанная на разметке, которую поиск получил
                  заранее.

                Каждый пункт — исправленный дефект, а не осторожность:
                подробности в `docs/reviews/AUDIT_FINDINGS_2026-09-04.md`.
                """
            )

    st.info(
        "Данные синтетические и обрабатываются локально. Оценку возвращает API — "
        "интерфейс намеренно не считает inference сам."
    )
    st.markdown("---")
    st.markdown("Выберите страницу в боковом меню для продолжения.")


if __name__ == "__main__":
    _guard_streamlit_entrypoint()
    main()
