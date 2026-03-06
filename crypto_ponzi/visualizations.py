"""
Visualization module for Company-Level Crypto-Ponzi Detection.

Provides 5 chart types + metrics table for the Streamlit dashboard.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd


# ── Color palette (consistent with app theme) ─────────────────────
_GREEN = "#10a37f"
_RED = "#ef4444"
_ORANGE = "#f97316"
_YELLOW = "#eab308"
_BLUE = "#3b82f6"
_PURPLE = "#8b5cf6"
_GRAY = "#6b7280"
_LIGHT_BG = "#ffffff"
_BORDER = "#e5e5e5"


def _apply_chart_style(ax: plt.Axes) -> None:
    """Apply consistent clean styling to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_BORDER)
    ax.spines["bottom"].set_color(_BORDER)
    ax.tick_params(colors="#6e6e80", labelsize=9)
    ax.set_facecolor(_LIGHT_BG)


# ── 1. Inflow / Outflow timeline ──────────────────────────────────

def plot_inflow_outflow_timeline(transactions: pd.DataFrame) -> plt.Figure:
    """
    Daily inflow vs outflow line chart with area fill.
    """
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    daily = df.groupby(["date", "direction"])["amount"].sum().unstack(fill_value=0)
    if "in" not in daily.columns:
        daily["in"] = 0.0
    if "out" not in daily.columns:
        daily["out"] = 0.0
    daily = daily.sort_index()
    daily.index = pd.to_datetime(daily.index)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    _apply_chart_style(ax)

    ax.fill_between(daily.index, daily["in"], alpha=0.15, color=_GREEN)
    ax.fill_between(daily.index, daily["out"], alpha=0.12, color=_RED)
    ax.plot(daily.index, daily["in"], color=_GREEN, linewidth=2, label="Приток (in)")
    ax.plot(daily.index, daily["out"], color=_RED, linewidth=2, label="Отток (out)")

    ax.set_title("Динамика притока и оттока по дням", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Сумма (KZT)", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


# ── 2. Inflow structure pie chart ─────────────────────────────────

def plot_inflow_structure_pie(metrics: dict[str, float]) -> plt.Figure:
    """
    Pie chart showing physical / legal / exchange share of inflows.
    """
    phys = metrics.get("physical_inflow_ratio", 0.0)
    legal = metrics.get("legal_inflow_ratio", 0.0)
    exch = metrics.get("exchange_inflow_ratio", 0.0)

    # Normalize to ensure they sum to 1
    total = phys + legal + exch
    if total > 0:
        phys, legal, exch = phys / total, legal / total, exch / total

    labels = ["Физ. лица", "Юр. лица", "Биржи (крипто)"]
    values = [phys, legal, exch]
    colors = [_ORANGE, _GREEN, _PURPLE]
    explode = (0.04, 0.02, 0.04)

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        explode=explode,
        startangle=90,
        textprops={"fontsize": 10, "color": "#1a1a1a"},
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)

    ax.set_title("Структура входящих поступлений", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig


# ── 3. Counterparty network graph ─────────────────────────────────

def plot_counterparty_network(
    transactions: pd.DataFrame,
    company_name: str = "Компания",
    top_k: int = 12,
) -> plt.Figure:
    """
    Network graph showing company ↔ top counterparties with edge widths
    proportional to transaction volume.
    """
    df = transactions.copy()
    by_cp = df.groupby(["counterparty_name", "counterparty_type", "direction"]).agg(
        volume=("amount", "sum"),
        count=("amount", "count"),
    ).reset_index()

    # Top counterparties by volume
    cp_totals = by_cp.groupby("counterparty_name")["volume"].sum().nlargest(top_k)
    top_names = set(cp_totals.index)

    G = nx.DiGraph()
    G.add_node(company_name, node_type="company")

    for _, row in by_cp.iterrows():
        if row["counterparty_name"] not in top_names:
            continue
        cp = row["counterparty_name"]
        G.add_node(cp, node_type=row["counterparty_type"])
        if row["direction"] == "in":
            G.add_edge(cp, company_name, weight=row["volume"])
        else:
            G.add_edge(company_name, cp, weight=row["volume"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor(_LIGHT_BG)

    if len(G.nodes) <= 1:
        ax.text(0.5, 0.5, "Недостаточно данных", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node colors by type
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        ntype = G.nodes[node].get("node_type", "legal")
        if ntype == "company":
            node_colors.append(_GREEN)
            node_sizes.append(700)
        elif ntype == "exchange":
            node_colors.append(_PURPLE)
            node_sizes.append(350)
        elif ntype == "physical":
            node_colors.append(_ORANGE)
            node_sizes.append(300)
        else:
            node_colors.append(_BLUE)
            node_sizes.append(350)

    # Edge widths proportional to volume
    edges = G.edges(data=True)
    max_w = max((d.get("weight", 1) for _, _, d in edges), default=1)
    edge_widths = [0.5 + 3.0 * d.get("weight", 1) / max_w for _, _, d in edges]
    edge_colors = [
        _GREEN if v == company_name else _RED
        for u, v, _ in edges
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color=edge_colors,
        alpha=0.5, arrows=True, arrowsize=12, ax=ax,
        connectionstyle="arc3,rad=0.08",
    )

    # Labels
    labels = {}
    for node in G.nodes:
        name = str(node)
        if len(name) > 18:
            name = name[:16] + "…"
        labels[node] = name
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color="#1a1a1a", ax=ax)

    ax.set_title("Сетевой граф компании и топ-контрагентов", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── 4. Factor contribution bar chart ──────────────────────────────

def plot_factor_contributions(score_result: dict[str, Any]) -> plt.Figure:
    """
    Horizontal bar chart showing each factor's contribution to the risk score.
    """
    contributions = score_result.get("factor_contributions", {})

    factor_labels = {
        "dependency_ratio": "Зависимость выплат",
        "crypto_exposure_ratio": "Крипто-экспозиция",
        "concentration_index": "Концентрация",
        "inverse_holding_time": "Короткое удержание",
        "inverse_entropy": "Неравномерность",
    }

    names = [factor_labels.get(k, k) for k in contributions.keys()]
    values = list(contributions.values())

    # Color by magnitude
    colors = []
    for v in values:
        if v >= 0.15:
            colors.append(_RED)
        elif v >= 0.08:
            colors.append(_ORANGE)
        elif v >= 0.04:
            colors.append(_YELLOW)
        else:
            colors.append(_GREEN)

    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_chart_style(ax)

    bars = ax.barh(names, values, color=colors, height=0.55, edgecolor="none")
    ax.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, color="#1a1a1a", fontweight="500",
        )

    total = sum(values)
    ax.set_title(
        f"Вклад факторов в итоговый риск-скор ({total:.3f})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Вклад в вероятность", fontsize=10)
    fig.tight_layout()
    return fig


# ── 5. Metrics summary table ──────────────────────────────────────

def build_metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    """
    Build a display-ready metrics DataFrame.
    """
    display_rows = [
        ("Общий приток", f"{metrics.get('total_inflow', 0):,.0f} KZT"),
        ("Общий отток", f"{metrics.get('total_outflow', 0):,.0f} KZT"),
        ("Крипто-приток", f"{metrics.get('crypto_inflow', 0):,.0f} KZT"),
        ("Крипто-отток", f"{metrics.get('crypto_outflow', 0):,.0f} KZT"),
        ("Доля физ. лиц (приток)", f"{metrics.get('physical_inflow_ratio', 0):.1%}"),
        ("Крипто-экспозиция", f"{metrics.get('crypto_exposure_ratio', 0):.1%}"),
        ("Коэффициент зависимости", f"{metrics.get('dependency_ratio', 0):.3f}"),
        ("Ср. время удержания", f"{metrics.get('avg_holding_time', 0):.1f} дней"),
        ("Индекс концентрации (top-5)", f"{metrics.get('concentration_index', 0):.3f}"),
        ("Энтропия потоков", f"{metrics.get('entropy_of_flows', 0):.4f}"),
        ("", ""),
        ("Всего транзакций", f"{metrics.get('total_transactions', 0):.0f}"),
        ("Приток (кол-во)", f"{metrics.get('inflow_count', 0):.0f}"),
        ("Отток (кол-во)", f"{metrics.get('outflow_count', 0):.0f}"),
        ("Уникальные контрагенты", f"{metrics.get('unique_counterparties', 0):.0f}"),
    ]
    return pd.DataFrame(display_rows, columns=["Метрика", "Значение"])
