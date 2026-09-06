"""Картинки под защиту — из уже посчитанных артефактов, без пересчёта.

    python scripts/make_defence_figures.py

Читает `artifacts/*.json`, которые пишут прогоны, и рисует три PNG в
`artifacts/figures/defence/`. Ничего не измеряет сам: если числа на слайде
отличаются от `docs/RESULTS.md`, значит устарел артефакт, а не картинка, и
чинится это перезапуском прогона.

Три картинки — три утверждения доклада, по одному на каждую:

1. **Редкость.** Одна метрика почти не двигается, вторая взлетает в
   восемьдесят раз. Два подграфика с общей осью X, а не две оси на одной
   картинке: ось Y у них измеряет разное, и совмещать их — способ
   нарисовать любую взаимосвязь, какая понравится.
2. **Уклонение.** Где именно ломается поиск групп и какая из двух ручек это
   делает.
3. **Лестница миров.** Пять миров нарастающей сложности, два уровня анализа,
   и обрыв сетевого уровня на пятой ступени.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = Path("artifacts") / "figures" / "defence"

# Проверено `scripts/validate_palette.js`: разделение по всем видам
# дальтонизма ΔE 27+, контраст к фону выше 3:1.
BLUE = "#2563EB"
AMBER = "#B45309"
INK = "#1f2328"
MUTED = "#6b7280"
GRID = "#e5e7eb"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _style(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _load(name: str) -> dict:
    return json.loads((Path("artifacts") / name).read_text(encoding="utf-8"))


# ==========================================================================
# 1. Редкость
# ==========================================================================


def figure_prevalence() -> Path:
    report = _load("prevalence_sweep.json")
    cells: dict[float, list[dict]] = {}
    for cell in report["natural"]:
        cells.setdefault(round(st.mean(c["achieved_prevalence"] for c in report["natural"]), 4), []).append(cell)
    for cell in report["measured"]:
        cells.setdefault(cell["target_prevalence"], []).append(cell)

    shares = sorted(cells, reverse=True)
    roc = [st.mean(c["roc_auc"] for c in cells[s]) for s in shares]
    per_catch = [st.mean(1.0 / c["precision_at_budget"] for c in cells[s]) for s in shares]
    x = [s * 100 for s in shares]

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.4), sharex=True, gridspec_kw={"hspace": 0.28}
    )

    top.plot(x, roc, color=BLUE, linewidth=2, marker="o", markersize=8)
    top.set_ylim(0.5, 1.02)
    top.set_ylabel("ROC-AUC")
    top.set_title("Метрика, которой все хвастаются, редкости почти не замечает", loc="left")
    for xi, yi in zip(x, roc):
        top.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=10, color=INK)
    _style(top)

    bottom.plot(x, per_catch, color=AMBER, linewidth=2, marker="o", markersize=8)
    bottom.set_ylabel("проверок на одну находку")
    bottom.set_title("А работа аналитика вырастает в восемьдесят раз", loc="left")
    for xi, yi in zip(x, per_catch):
        bottom.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=10, color=INK)
    bottom.set_xscale("log")
    bottom.set_xlabel("доля мошенников в мире, %   →   ближе к жизни")
    bottom.set_xticks(x)
    bottom.set_xticklabels([f"{v:.1f}" for v in x])
    # Ось развёрнута: слева наш мир, справа настоящая редкость. Иначе движение
    # вправо читается как улучшение, а показать надо обратное.
    bottom.invert_xaxis()
    bottom.set_ylim(0, max(per_catch) * 1.25)
    _style(bottom)

    figure.suptitle("Один и тот же детектор при разной редкости мошенников",
                    fontsize=15, fontweight="bold", x=0.045, ha="left", y=0.985)
    figure.text(0.045, 0.005,
                "Три сида, purged walk-forward. Правило: ROC-AUC без указанной доли "
                "мошенников рядом — цифра ни о чём.",
                fontsize=9, color=MUTED, ha="left")

    path = OUT_DIR / "rarity.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


# ==========================================================================
# 2. Уклонение
# ==========================================================================


def figure_evasion() -> Path:
    report = _load("evasion_curve.json")
    by_key: dict[str, list[dict]] = {}
    for result in report["results"]:
        by_key.setdefault(result["key"], []).append(result)

    def coverage(key: str) -> float:
        group = by_key[key]
        return st.mean(u["coverage"] for r in group for u in r["units"] if u["unit"] == "network")

    funders = [1, 2, 3, 4, 6]
    funder_cover = [coverage(k) for k in ("naive", "f2", "f3", "f4", "f6")]
    terminals = [1, 2, 3, 4]
    terminal_cover = [coverage(k) for k in ("naive", "t2", "t3", "t4")]
    both = coverage("f6t4")

    figure, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(funders, funder_cover, color=AMBER, linewidth=2, marker="o", markersize=9,
            label="деньги приходят с N разных счетов")
    ax.plot(terminals, terminal_cover, color=BLUE, linewidth=2, marker="s", markersize=8,
            label="снимают в N разных банкоматах")

    ax.annotate("деньги с 3 счетов —\nтеряем треть банд",
                xy=(3, funder_cover[2]), xytext=(1.6, 0.42),
                fontsize=10, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER, linewidth=1.3))
    ax.annotate("4 банкомата —\nвсё ещё видим 3 из 4",
                xy=(4, terminal_cover[3]), xytext=(4.15, 0.94),
                fontsize=10, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.3))
    ax.scatter([6], [both], s=130, color=INK, zorder=5, marker="X")
    ax.annotate(f"и то, и другое сразу:\nне видим НИ ОДНОЙ ({both:.3f})",
                xy=(6, both), xytext=(3.9, 0.17), fontsize=10,
                color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=INK, linewidth=1.2))

    ax.set_xlabel("во сколько раз организатор раздробил канал")
    ax.set_ylabel("доля банд, которые система находит")
    ax.set_ylim(-0.04, 1.12)
    ax.set_xlim(0.7, 6.6)
    ax.set_xticks([1, 2, 3, 4, 6])
    ax.set_title("Цена уклонения: ломают деньги, а не логистика", loc="left", fontsize=15)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    _style(ax)
    figure.text(0.045, -0.02,
                "Мир W4, три сида. Поиск по одному человеку на этой кривой не двигается "
                "вообще: 0.950 → 0.970.",
                fontsize=9, color=MUTED, ha="left")

    path = OUT_DIR / "evasion.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


# ==========================================================================
# 3. Лестница миров
# ==========================================================================


def figure_ladder() -> Path:
    report = _load("ladder_of_worlds.json")
    order = ["W1", "W2", "W3", "W4", "W5"]
    titles = {
        "W1": "непохожие\nнегативы",
        "W2": "+ тоже\nопустошают счёт",
        "W3": "+ та же\nформа",
        "W4": "+ вторая схема,\nдругой такт",
        "W5": "+ организатор\nпрячется",
    }

    def unit_score(key: str, unit: str) -> float | None:
        values = [
            u["roc_auc"] for r in report["results"] if r["key"] == key
            for u in r["units"] if u["unit"] == unit and u["roc_auc"] is not None
        ]
        return st.mean(values) if values else None

    account = [unit_score(k, "account") for k in order]
    network = [unit_score(k, "network") for k in order]
    x = range(len(order))

    figure, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(x, account, color=BLUE, linewidth=2, marker="o", markersize=9,
            label="поиск по одному человеку")
    visible = [(i, v) for i, v in zip(x, network) if v is not None]
    ax.plot([i for i, _ in visible], [v for _, v in visible], color=AMBER,
            linewidth=2, marker="s", markersize=8, label="поиск по группам")

    for i, value in zip(x, account):
        ax.annotate(f"{value:.3f}", (i, value), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=10, color=BLUE)
    for i, value in visible:
        ax.annotate(f"{value:.3f}", (i, value), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, color=AMBER)

    # Стрелка указывает туда, где точки НЕТ: на W5 у группового уровня нет
    # оценки вообще, и это главное содержание картинки.
    ax.annotate("на W5 у группового уровня\nвообще нет точки:\nон не ошибается — он слепнет",
                xy=(4, 0.999), xytext=(2.25, 0.845),
                fontsize=10, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER, linewidth=1.4))

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{k}\n{titles[k]}" for k in order], fontsize=10)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.75, 1.03)
    ax.set_title("Лестница миров: сложность объявлена ДО прогонов", loc="left", fontsize=15)
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    _style(ax)
    figure.text(0.045, -0.04,
                "Показаны все пять ступеней, включая ту, где мы падаем. Три сида, один "
                "и тот же детектор — меняется только мир.",
                fontsize=9, color=MUTED, ha="left")

    path = OUT_DIR / "ladder.png"
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for builder in (figure_prevalence, figure_evasion, figure_ladder):
        path = builder()
        print(f"  {path}")
    print("\nЧисла взяты из artifacts/*.json — тех же, из которых написан docs/RESULTS.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
