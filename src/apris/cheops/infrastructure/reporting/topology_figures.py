"""Рисунки топологий: структура схемы из настоящих событий мира.

Каждая картинка здесь строится **из событий**, а не из признаков и не из
вердикта модели. Это принципиально: в ранней версии интерфейса граф рисовался
функцией от тех же девяти признаков, которые он якобы подтверждал, поэтому
картинка всегда соглашалась с оценкой. Здесь узлы и рёбра берутся из
`world.events`, ширина линии равна сумме перевода, а раскладка задаётся
ролью узла в схеме, а не силовым алгоритмом — силовая раскладка красива, но
прячет то единственное, что нужно увидеть: куда сходится поток.

Пять структурных подписей из методики (раздел 2.1 научной работы):

* схождение множества ветвей в одну точку выхода — «банкоматная вспышка»;
* лавинообразный приток от несвязанных вершин — «чёрная дыра» пирамиды;
* взрывной исходящий поток — «вулкан» выплат;
* цепочка слоёв с мостом в криптовалюту;
* дробление сумм у порогов — структура объявлена, популяция в разработке.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from apris.cheops.domain.models import TransactionEvent  # noqa: E402
from apris.cheops.infrastructure.simulation.generator import (  # noqa: E402
    SimulatedNetwork,
    SimulatedWorld,
)

# Та же палитра, что у остальных рисунков защиты.
SOURCE = "#A11D21"
MULE = "#E4694E"
EXIT = "#6B3FA0"
CRYPTO = "#0E7C86"
HONEST = "#8FA3C8"
INK = "#1f2328"
MUTED = "#6b7280"
EDGE = "#e5e7eb"

_ROLE_COLOR = {
    "source": SOURCE,
    "mule": MULE,
    "exit": EXIT,
    "crypto": CRYPTO,
    "honest": HONEST,
}


@dataclass(frozen=True)
class Layer:
    """Один ярус раскладки: подпись слева и узлы в ряд."""

    title: str
    role: str
    nodes: tuple[str, ...]


def _style(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, fontweight="bold", color=INK, loc="left", pad=30)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=10,
                color=MUTED, va="bottom")


def _positions(layers: list[Layer]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    height = max(len(layers) - 1, 1)
    for index, layer in enumerate(layers):
        y = 1.0 - index / height
        count = max(len(layer.nodes), 1)
        for order, node in enumerate(layer.nodes):
            x = 0.5 if count == 1 else order / (count - 1)
            positions[node] = (x, y)
    return positions


def _draw(
    ax,
    layers: list[Layer],
    edges: list[tuple[str, str, float]],
    *,
    max_width: float = 4.0,
) -> None:
    positions = _positions(layers)
    amounts = [amount for _, _, amount in edges] or [1.0]
    biggest = max(amounts)

    for sender, receiver, amount in edges:
        if sender not in positions or receiver not in positions:
            continue
        x1, y1 = positions[sender]
        x2, y2 = positions[receiver]
        width = 0.4 + max_width * (amount / biggest) ** 0.6
        ax.plot([x1, x2], [y1, y2], color="#B9C2D4", linewidth=width,
                alpha=0.75, solid_capstyle="round", zorder=1)

    for layer in layers:
        color = _ROLE_COLOR[layer.role]
        for node in layer.nodes:
            x, y = positions[node]
            marker = "s" if layer.role in {"exit", "crypto"} else "o"
            size = 260 if len(layer.nodes) == 1 else max(70, 220 - 6 * len(layer.nodes))
            ax.scatter([x], [y], s=size, color=color, marker=marker, zorder=3,
                       edgecolors="white", linewidths=1.2)
        if layer.nodes:
            _, y = positions[layer.nodes[0]]
            ax.text(-0.09, y, layer.title, fontsize=10, fontweight="bold",
                    color=color, ha="right", va="center")

    ax.set_xlim(-0.42, 1.08)
    ax.set_ylim(-0.12, 1.12)


def _events_between(events, members: set[str]):
    return [e for e in events if e.sender_id in members or e.receiver_id in members]


# ==========================================================================
# 1. Банкоматная вспышка: кольцо дропперов
# ==========================================================================


def plot_mule_ring(world: SimulatedWorld, network: SimulatedNetwork):
    """Схождение ветвей в одну точку выхода — подпись «банкоматная вспышка»."""
    members = set(network.account_ids)
    organizers = set(network.organizer_ids)
    mules = tuple(sorted(members - organizers))
    events = _events_between(world.events, members)

    terminals = tuple(sorted({e.receiver_id for e in events if e.receiver_type == "atm"}))
    layers = [
        Layer("ИСТОЧНИК", "source", tuple(sorted(organizers))),
        Layer("ДРОПЫ", "mule", mules),
        Layer("ВЫХОД", "exit", terminals or ("ATM",)),
    ]
    edges = [(e.sender_id, e.receiver_id, e.amount) for e in events]

    figure, ax = plt.subplots(figsize=(8, 5))
    _draw(ax, layers, edges)

    incoming = sum(e.amount for e in events if e.receiver_id in members)
    outgoing = sum(e.amount for e in events if e.sender_id in members)
    retained = max(incoming - outgoing, 0.0)
    span = (max(e.ts for e in events) - min(e.ts for e in events)).total_seconds() / 60

    _style(ax, f"Кольцо дропперов {network.network_id}",
           "Транзит без удержания: деньги входят и в тот же час выходят наличными")
    ax.text(0.0, -0.09,
            f"вошло {incoming:,.0f} ₸   вышло {outgoing:,.0f} ₸   "
            f"осело {retained:,.0f} ₸ ({retained / incoming:.1%})   "
            f"вся операция за {span:.0f} мин".replace(",", " "),
            transform=ax.transAxes, fontsize=10, color=INK, family="monospace")
    figure.tight_layout()
    return figure


# ==========================================================================
# 2. Пирамида: чёрная дыра и вулкан на одной картинке
# ==========================================================================


def plot_pyramid(world: SimulatedWorld, network: SimulatedNetwork):
    """Приток от несвязанных вершин и выплата обратно — та же форма, другой такт."""
    members = set(network.account_ids)
    organizers = set(network.organizer_ids)
    events = _events_between(world.events, members)

    investors = tuple(sorted({e.sender_id for e in events if e.receiver_id in organizers})[:24])
    payouts = tuple(sorted({e.receiver_id for e in events if e.sender_id in organizers})[:24])
    layers = [
        Layer("ВКЛАДЧИКИ", "honest", investors),
        Layer("ОРГАНИЗАТОР", "source", tuple(sorted(organizers))),
        Layer("ВЫПЛАТЫ", "mule", payouts),
    ]
    edges = [
        (e.sender_id, e.receiver_id, e.amount)
        for e in events
        if e.sender_id in set(investors) | organizers or e.receiver_id in organizers | set(payouts)
    ]

    figure, ax = plt.subplots(figsize=(8, 5))
    _draw(ax, layers, edges, max_width=2.6)

    days = (max(e.ts for e in events) - min(e.ts for e in events)).days
    _style(ax, f"Финансовая пирамида {network.network_id}",
           "Та же топология, что у кольца, но растянутая на месяцы — это и объединяет параметр W")
    ax.text(0.0, -0.09,
            f"вкладчиков {len(investors)}   выплат {len(payouts)}   "
            f"схема живёт {days} дн.",
            transform=ax.transAxes, fontsize=10, color=INK, family="monospace")
    figure.tight_layout()
    return figure


# ==========================================================================
# 3. Крипто-цепочка: легальный вход, слои, мост, дробление по кошелькам
# ==========================================================================


def plot_crypto_chain(world: SimulatedWorld, network: SimulatedNetwork):
    """Легальный вход → слои подставных счетов → мост в крипту → кошельки."""
    members = set(network.account_ids)
    events = _events_between(world.events, members)

    crypto_events = [e for e in events if e.channel == "crypto"]
    wallets = tuple(sorted({e.receiver_id for e in crypto_events}))
    bridges = tuple(sorted({e.sender_id for e in crypto_events}))
    entry = tuple(sorted({e.sender_id for e in events if e.sender_id not in members})[:8])
    layers_middle = tuple(sorted(members - set(bridges) - set(wallets)))

    layers = [
        Layer("ЛЕГАЛЬНЫЙ ВХОД", "honest", entry or ("вход",)),
        Layer("СЛОИ", "mule", layers_middle),
        Layer("МОСТ", "source", bridges),
        Layer("КОШЕЛЬКИ", "crypto", wallets),
    ]
    edges = [(e.sender_id, e.receiver_id, e.amount) for e in events]

    figure, ax = plt.subplots(figsize=(8, 5.4))
    _draw(ax, layers, edges, max_width=3.2)

    _style(ax, f"Крипто-слоирование {network.network_id}",
           "Три типологии приказа сразу: слоирование, мост в криптовалюту, микширование")
    ax.text(0.0, -0.09,
            f"слоёв {len(layers_middle)}   крипто-переводов {len(crypto_events)}   "
            f"кошельков {len(wallets)}",
            transform=ax.transAxes, fontsize=10, color=INK, family="monospace")
    legend = [
        mpatches.Patch(color=CRYPTO, label="криптовалютный канал"),
        mpatches.Patch(color=MULE, label="подставные счета"),
        mpatches.Patch(color=HONEST, label="легальные плательщики"),
    ]
    ax.legend(handles=legend, loc="center right", frameon=False, fontsize=9)
    figure.tight_layout()
    return figure


# ==========================================================================
# 4. Честный двойник: то, что нельзя ловить
# ==========================================================================


def plot_honest_lookalike(world: SimulatedWorld, population: str = "crowd_collector"):
    """Сбор средств: та же звезда, что у пирамиды, и это честные люди.

    Самая важная картинка набора. Без неё «много платят одному» выглядит
    исчерпывающим признаком мошенничества, а это ровно тот случай, когда
    детектор учится не тому.
    """
    accounts = [a for a, kind in world.populations.items() if kind == population]
    grouped: dict[str, list[TransactionEvent]] = defaultdict(list)
    for event in world.events:
        if event.receiver_id in set(accounts):
            grouped[event.receiver_id].append(event)
    if not grouped:
        return None

    collector = max(grouped, key=lambda key: len(grouped[key]))
    events = grouped[collector][:40]
    senders = tuple(sorted({e.sender_id for e in events}))

    layers = [
        Layer("ЛЮДИ", "honest", senders),
        Layer("СБОР", "source", (collector,)),
    ]
    edges = [(e.sender_id, e.receiver_id, e.amount) for e in events]

    figure, ax = plt.subplots(figsize=(8, 4.2))
    _draw(ax, layers, edges, max_width=2.2)
    _style(ax, "Честный сбор средств — та же форма",
           "Сорок человек платят одному. Отличие от пирамиды не в форме, а во времени удержания")
    ax.text(0.0, -0.12,
            f"плательщиков {len(senders)}   переводов {len(events)}   "
            f"это НЕ мошенничество",
            transform=ax.transAxes, fontsize=10, color=INK, family="monospace")
    figure.tight_layout()
    return figure


# ==========================================================================
# 5. Дробление сумм: структура объявлена, популяция в разработке
# ==========================================================================


def plot_structuring_sketch():
    """Схема дробления у порогов — единственная из пяти, чья популяция ещё
    не порождается генератором. Показывается как схема и подписана так."""
    layers = [
        Layer("ИСТОЧНИК", "source", ("S",)),
        Layer("ДРОБЛЕНИЕ", "mule", tuple(f"m{i}" for i in range(8))),
        Layer("СБОРКА", "exit", ("E",)),
    ]
    edges = [("S", f"m{i}", 1.0) for i in range(8)] + [(f"m{i}", "E", 1.0) for i in range(8)]

    figure, ax = plt.subplots(figsize=(8, 4.2))
    _draw(ax, layers, edges, max_width=2.0)
    _style(ax, "Дробление сумм у порогов (structuring)",
           "Крупная сумма дробится ниже порога контроля и собирается обратно")
    ax.text(0.0, -0.12,
            "структура объявлена в методике; популяция в генераторе — ближайшая задача",
            transform=ax.transAxes, fontsize=10, color=MUTED, style="italic")
    figure.tight_layout()
    return figure


def networks_of_kind(world: SimulatedWorld, kind: str) -> list[SimulatedNetwork]:
    return [n for n in world.networks if n.kind == kind]


__all__ = [
    "networks_of_kind",
    "plot_crypto_chain",
    "plot_honest_lookalike",
    "plot_mule_ring",
    "plot_pyramid",
    "plot_structuring_sketch",
]
