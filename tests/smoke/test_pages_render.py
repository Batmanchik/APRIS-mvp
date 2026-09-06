"""Каждая страница интерфейса открывается без исключения.

Зачем отдельный тест, когда есть тесты ядра: ядро переписывалось несколько
раз — вынесены пресеты, изменились сигнатуры признаков, появился конвейер, —
а страницы при этом никто не открывал. Интерфейс, который падает на первом
клике, обнуляет любые метрики позади него, и узнать об этом на защите хуже,
чем в CI.

Проверяется ровно одно: страница отрисовывается целиком и не бросает
исключение. Это не проверка вёрстки и не проверка чисел — на них есть тесты
ядра. Это проверка того, что интерфейс собран из того же кода, что и всё
остальное.

API здесь намеренно не поднимается. Страницы обращаются к нему только по
кнопке, поэтому отрисовка обязана работать и без него — иначе демо на
ноутбуке без запущенного бэкенда встречает человека трассировкой.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAGES = [
    "app.py",
    "pages/1_Поиск_сетей.py",
    "pages/2_Досье_кандидата.py",
    "pages/3_Валидация.py",
    "pages/4_Ручная_проверка.py",
    "pages/5_Очередь_аналитика.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str) -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=300).run()

    assert not app.exception, (
        f"{page} упала при отрисовке: "
        + "; ".join(str(item.value)[:300] for item in app.exception)
    )


def test_every_page_file_is_covered_by_this_test() -> None:
    """Новая страница без строки в PAGES не проверяется никем, и узнать об
    этом можно только на защите. Список сверяется с каталогом."""
    on_disk = sorted(
        f"pages/{path.name}" for path in (PROJECT_ROOT / "pages").glob("*.py")
    )
    assert on_disk == sorted(page for page in PAGES if page.startswith("pages/"))
