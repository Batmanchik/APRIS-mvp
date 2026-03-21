"""
Synthetic transaction generator for Company-Level Crypto-Ponzi Detection.

Generates realistic transaction datasets for 4 preset profiles:
SAFE, MODERATE, DANGEROUS, PYRAMID.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ── Preset configurations ──────────────────────────────────────────
PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "SAFE": {
        "company_name": "ТОО «GreenField Logistics»",
        "business_type": "Логистика и грузоперевозки",
        "tx_count_range": (500, 800),
        "physical_inflow_share": 0.15,
        "legal_inflow_share": 0.80,
        "exchange_inflow_share": 0.02,
        "exchange_outflow_share": 0.01,
        "crypto_amount_multiplier": 0.5,
        "outflow_ratio": 0.45,
        "avg_amount_in": 450_000.0,
        "avg_amount_out": 380_000.0,
        "amount_std_factor": 0.35,
        "holding_days_mean": 135.0,
        "concentration_top5": 0.12,
        "growth_pattern": "flat",
    },
    "MODERATE": {
        "company_name": "ТОО «CryptoTrade Advisors»",
        "business_type": "Финансовый консалтинг",
        "tx_count_range": (800, 1200),
        "physical_inflow_share": 0.40,
        "legal_inflow_share": 0.30,
        "exchange_inflow_share": 0.20,
        "exchange_outflow_share": 0.12,
        "crypto_amount_multiplier": 1.5,
        "outflow_ratio": 0.65,
        "avg_amount_in": 320_000.0,
        "avg_amount_out": 280_000.0,
        "amount_std_factor": 0.40,
        "holding_days_mean": 75.0,
        "concentration_top5": 0.30,
        "growth_pattern": "linear",
    },
    "DANGEROUS": {
        "company_name": "ТОО «DigitalVault Investments»",
        "business_type": "Инвестиционная платформа",
        "tx_count_range": (1000, 1500),
        "physical_inflow_share": 0.55,
        "legal_inflow_share": 0.10,
        "exchange_inflow_share": 0.35,
        "exchange_outflow_share": 0.25,
        "crypto_amount_multiplier": 2.5,
        "outflow_ratio": 0.78,
        "avg_amount_in": 180_000.0,
        "avg_amount_out": 250_000.0,
        "amount_std_factor": 0.50,
        "holding_days_mean": 40.0,
        "concentration_top5": 0.45,
        "growth_pattern": "exponential",
    },
    "PYRAMID": {
        "company_name": "ТОО «UltraYield Global»",
        "business_type": "Высокодоходная инвестиционная программа",
        "tx_count_range": (1500, 2000),
        "physical_inflow_share": 0.45,
        "legal_inflow_share": 0.02,
        "exchange_inflow_share": 0.53,
        "exchange_outflow_share": 0.50,
        "crypto_amount_multiplier": 3.0,
        "outflow_ratio": 0.92,
        "avg_amount_in": 95_000.0,
        "avg_amount_out": 210_000.0,
        "amount_std_factor": 0.65,
        "holding_days_mean": 18.0,
        "concentration_top5": 0.65,
        "growth_pattern": "exponential_steep",
    },
}

PRESET_ORDER = ["SAFE", "MODERATE", "DANGEROUS", "PYRAMID"]

# ── Counterparty name pools ────────────────────────────────────────
_LEGAL_NAMES = [
    "ТОО «АльфаТрейд»", "АО «БетаСервис»", "ТОО «ГаммаЛогистик»",
    "АО «ДельтаФинанс»", "ТОО «ЭпсилонГрупп»", "АО «ЗетаКапитал»",
    "ТОО «ТетаИнвест»", "АО «ЙотаКонсалт»", "ТОО «КаппаТех»",
    "АО «ЛямбдаСтрой»", "ТОО «МюИндустри»", "АО «НюЭнерго»",
    "ТОО «СигмаТранс»", "АО «ОмегаРесурс»", "ТОО «ФиДевелоп»",
]
_PHYSICAL_NAMES = [
    "Иванов А.С.", "Петрова М.К.", "Сидоров В.Н.", "Козлова Е.А.",
    "Михайлов Д.И.", "Новикова О.П.", "Федоров С.Л.", "Морозова Т.В.",
    "Волков Р.Г.", "Алексеева Н.Ю.", "Лебедев И.О.", "Семенова Д.Р.",
    "Егоров А.К.", "Павлова Л.М.", "Дмитриев Б.Т.", "Кузьмина В.С.",
    "Степанов Г.Н.", "Маркова Ж.Э.", "Андреев П.Ф.", "Тарасова У.Б.",
]
_EXCHANGE_NAMES = [
    "Binance", "Bybit", "OKX", "KuCoin", "Gate.io",
    "MEXC", "Bitget", "HTX (Huobi)", "Crypto.com", "Kraken",
]
_BANK_NAMES = [
    "Halyk Bank", "Kaspi Bank", "Forte Bank", "Jusan Bank",
    "Bank CenterCredit", "Altyn Bank", "Eurasian Bank",
    "First Heartland Jusan", "Freedom Bank", "Bereke Bank",
]


def _generate_timestamps(
    n: int,
    days: int,
    growth_pattern: str,
    rng: np.random.Generator,
) -> list[datetime]:
    """Generate transaction timestamps with the specified growth pattern."""
    base = datetime(2025, 10, 1)

    if growth_pattern == "flat":
        offsets = rng.uniform(0, days, size=n)
    elif growth_pattern == "linear":
        offsets = rng.triangular(0, days * 0.7, days, size=n)
    elif growth_pattern == "exponential":
        raw = rng.exponential(scale=days / 3.0, size=n)
        offsets = days - np.clip(raw, 0, days)
    elif growth_pattern == "exponential_steep":
        raw = rng.exponential(scale=days / 5.0, size=n)
        offsets = days - np.clip(raw, 0, days)
    else:
        offsets = rng.uniform(0, days, size=n)

    offsets = np.sort(offsets)
    timestamps = []
    for off in offsets:
        day_offset = timedelta(days=float(off))
        hour_offset = timedelta(hours=float(rng.uniform(8, 20)))
        minute_offset = timedelta(minutes=float(rng.integers(0, 60)))
        timestamps.append(base + day_offset + hour_offset + minute_offset)
    return timestamps


def _pick_counterparty(
    direction: str,
    cp_type: str,
    rng: np.random.Generator,
    concentration_top5: float,
    _top_pool: dict[str, list[str]],
) -> tuple[str, str]:
    """Pick a counterparty name and bank, biased by concentration."""
    if cp_type == "exchange":
        pool = _EXCHANGE_NAMES
    elif cp_type == "physical":
        pool = _PHYSICAL_NAMES
    else:
        pool = _LEGAL_NAMES

    key = f"{cp_type}_{direction}"
    if key not in _top_pool:
        top_k = min(5, len(pool))
        _top_pool[key] = list(rng.choice(pool, size=top_k, replace=False))

    if rng.random() < concentration_top5:
        name = str(rng.choice(_top_pool[key]))
    else:
        name = str(rng.choice(pool))

    bank = str(rng.choice(_BANK_NAMES)) if cp_type != "exchange" else "—"
    return name, bank


def generate_company_transactions(
    preset: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Generate a full synthetic company case with transactions.

    Returns dict with keys:
        company_id, company_name, declared_business_type,
        reporting_period, transactions (pd.DataFrame)
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {PRESET_ORDER}")

    cfg = PRESET_CONFIGS[preset]
    rng = np.random.default_rng(seed)

    lo, hi = cfg["tx_count_range"]
    n_total = int(rng.integers(lo, hi + 1))

    # Determine in/out split
    outflow_ratio = cfg["outflow_ratio"]
    n_in = int(n_total * (1 / (1 + outflow_ratio)))
    n_out = n_total - n_in

    # Build direction list
    directions = ["in"] * n_in + ["out"] * n_out
    rng.shuffle(directions)

    # Generate timestamps
    timestamps = _generate_timestamps(n_total, 90, cfg["growth_pattern"], rng)

    # Counterparty type distribution for inflows
    phys_share = cfg["physical_inflow_share"]
    legal_share = cfg["legal_inflow_share"]
    exch_in_share = cfg["exchange_inflow_share"]
    exch_out_share = cfg["exchange_outflow_share"]

    rows: list[dict[str, Any]] = []
    _top_pool: dict[str, list[str]] = {}

    for i in range(n_total):
        direction = directions[i]

        # Choose counterparty type
        if direction == "in":
            r = rng.random()
            if r < phys_share:
                cp_type = "physical"
            elif r < phys_share + legal_share:
                cp_type = "legal"
            else:
                cp_type = "exchange"
        else:
            r = rng.random()
            if r < exch_out_share:
                cp_type = "exchange"
            elif r < exch_out_share + 0.3:
                cp_type = "legal"
            else:
                cp_type = "physical"

        # Generate amount
        avg = cfg["avg_amount_in"] if direction == "in" else cfg["avg_amount_out"]
        std = avg * cfg["amount_std_factor"]
        amount = max(1000.0, float(rng.normal(avg, std)))

        # Exchange transactions carry higher amounts (crypto multiplier)
        if cp_type == "exchange":
            amount *= cfg.get("crypto_amount_multiplier", 1.0)

        # Currency
        if cp_type == "exchange":
            currency = str(rng.choice(["USDT", "BTC", "ETH", "USDC"]))
        else:
            currency = "KZT"

        cp_name, cp_bank = _pick_counterparty(
            direction, cp_type, rng, cfg["concentration_top5"], _top_pool
        )

        rows.append({
            "transaction_id": str(uuid.uuid4())[:12],
            "timestamp": timestamps[i],
            "amount": round(amount, 2),
            "currency": currency,
            "direction": direction,
            "counterparty_type": cp_type,
            "counterparty_name": cp_name,
            "counterparty_bank": cp_bank,
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return {
        "company_id": f"KZ-{rng.integers(100000, 999999)}",
        "company_name": cfg["company_name"],
        "declared_business_type": cfg["business_type"],
        "reporting_period": "2025-10-01 / 2025-12-30 (90 дней)",
        "transactions": df,
        "preset": preset,
    }
