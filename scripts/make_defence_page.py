"""Веб-версия защиты: тот же материал, что в слайдах, одной страницей.

    python scripts/make_defence_page.py

Берёт `artifacts/site/defence_template.html`, вшивает в него три графика как
data-URI и пишет `artifacts/site/defence.html` — самодостаточный файл, который
открывается двойным щелчком и не тянет ничего из сети, кроме шрифтов.

Картинки вшиваются, а не подключаются файлами, потому что страницу будут
пересылать и открывать с флешки: HTML с отвалившимися картинками на защите
хуже, чем отсутствие страницы.
"""

from __future__ import annotations

import base64
from pathlib import Path

SITE = Path("artifacts") / "site"
FIG = Path("artifacts") / "figures" / "defence"
SLOTS = {"{{LADDER}}": "ladder.png", "{{RARITY}}": "rarity.png", "{{EVASION}}": "evasion.png"}


def main() -> int:
    page = (SITE / "defence_template.html").read_text(encoding="utf-8")
    for slot, name in SLOTS.items():
        encoded = base64.b64encode((FIG / name).read_bytes()).decode("ascii")
        page = page.replace(slot, f"data:image/png;base64,{encoded}")
    if "{{" in page:
        raise SystemExit("в шаблоне остался незаполненный слот")
    out = SITE / "defence.html"
    out.write_text(page, encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
