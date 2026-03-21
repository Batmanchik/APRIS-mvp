"""
Transaction aggregator for Company-Level Crypto-Ponzi Detection.

Aggregates raw transaction-level data into company-level metrics.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def aggregate_transactions(transactions: pd.DataFrame) -> dict[str, float]:
    """
    Aggregate raw transactions into company-level metrics.

    Expected columns:
        timestamp, amount, direction, counterparty_type,
        counterparty_name, counterparty_bank

    Returns dict with 10 metrics + auxiliary breakdowns.
    """
    df = transactions.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)

    inflows = df[df["direction"] == "in"]
    outflows = df[df["direction"] == "out"]

    # ── Core totals ────────────────────────────────────────────────
    total_inflow = float(inflows["amount"].sum())
    total_outflow = float(outflows["amount"].sum())
    total_volume = total_inflow + total_outflow

    # ── Crypto flows ───────────────────────────────────────────────
    crypto_inflow = float(
        inflows[inflows["counterparty_type"] == "exchange"]["amount"].sum()
    )
    crypto_outflow = float(
        outflows[outflows["counterparty_type"] == "exchange"]["amount"].sum()
    )

    # ── Physical inflow ratio ──────────────────────────────────────
    physical_inflow = float(
        inflows[inflows["counterparty_type"] == "physical"]["amount"].sum()
    )
    physical_inflow_ratio = physical_inflow / max(total_inflow, 1.0)

    # ── Legal inflow ratio (auxiliary) ─────────────────────────────
    legal_inflow = float(
        inflows[inflows["counterparty_type"] == "legal"]["amount"].sum()
    )
    legal_inflow_ratio = legal_inflow / max(total_inflow, 1.0)

    # ── Exchange inflow ratio (auxiliary) ──────────────────────────
    exchange_inflow_ratio = crypto_inflow / max(total_inflow, 1.0)

    # ── Crypto exposure ratio ──────────────────────────────────────
    crypto_exposure_ratio = (crypto_inflow + crypto_outflow) / max(total_volume, 1.0)

    # ── Dependency ratio ───────────────────────────────────────────
    dependency_ratio = total_outflow / max(total_inflow, 1.0)

    # ── Avg holding time ───────────────────────────────────────────
    avg_holding_time = _estimate_holding_time(df)

    # ── Concentration index (top 5) ────────────────────────────────
    concentration_index = _compute_concentration_index(df, top_k=5)

    # ── Entropy of flows ───────────────────────────────────────────
    entropy_of_flows = _compute_flow_entropy(df)

    return {
        # 10 required metrics
        "total_inflow": total_inflow,
        "total_outflow": total_outflow,
        "crypto_inflow": crypto_inflow,
        "crypto_outflow": crypto_outflow,
        "physical_inflow_ratio": physical_inflow_ratio,
        "crypto_exposure_ratio": crypto_exposure_ratio,
        "dependency_ratio": dependency_ratio,
        "avg_holding_time": avg_holding_time,
        "concentration_index": concentration_index,
        "entropy_of_flows": entropy_of_flows,
        # Auxiliary breakdowns
        "legal_inflow_ratio": legal_inflow_ratio,
        "exchange_inflow_ratio": exchange_inflow_ratio,
        "total_transactions": len(df),
        "inflow_count": len(inflows),
        "outflow_count": len(outflows),
        "unique_counterparties": df["counterparty_name"].nunique(),
        "total_volume": total_volume,
    }


def _estimate_holding_time(df: pd.DataFrame) -> float:
    """
    Estimate average holding time using cumulative balance analysis.

    Measures how long funds remain in the company by tracking the
    weighted-average "age" of the cumulative balance over time.
    """
    df_sorted = df.sort_values("timestamp").copy()
    df_sorted["date"] = pd.to_datetime(df_sorted["timestamp"]).dt.date

    daily = df_sorted.groupby("date").apply(
        lambda g: pd.Series({
            "daily_in": g[g["direction"] == "in"]["amount"].sum(),
            "daily_out": g[g["direction"] == "out"]["amount"].sum(),
        }),
        include_groups=False,
    ).reset_index()

    if daily.empty or daily["daily_in"].sum() == 0:
        return 90.0

    daily = daily.sort_values("date")
    total_in = daily["daily_in"].sum()
    total_out = daily["daily_out"].sum()

    if total_out == 0:
        return 90.0

    # Match each outflow chunk to the earliest available inflow, FIFO.
    inflow_queue: list[tuple[int, float]] = []  # (day_index, remaining_amount)
    holding_weighted_sum = 0.0
    total_matched = 0.0

    for idx, row in daily.iterrows():
        day_i = int(daily.index.get_loc(idx)) if not isinstance(idx, int) else idx
        day_idx = len(inflow_queue)  # just use sequential day index

    # Re-do with proper day indexing
    daily = daily.reset_index(drop=True)
    for day_idx in range(len(daily)):
        row = daily.iloc[day_idx]
        if row["daily_in"] > 0:
            inflow_queue.append((day_idx, float(row["daily_in"])))

        remaining_out = float(row["daily_out"])
        while remaining_out > 0 and inflow_queue:
            in_day, in_amt = inflow_queue[0]
            matched = min(remaining_out, in_amt)
            holding_days = max(1, day_idx - in_day)
            holding_weighted_sum += matched * holding_days
            total_matched += matched
            remaining_out -= matched
            in_amt -= matched
            if in_amt <= 0.01:
                inflow_queue.pop(0)
            else:
                inflow_queue[0] = (in_day, in_amt)

    if total_matched == 0:
        return 90.0

    avg_holding = holding_weighted_sum / total_matched
    return round(max(1.0, avg_holding), 1)


def _compute_concentration_index(df: pd.DataFrame, top_k: int = 5) -> float:
    """
    Compute concentration index as share of total volume
    attributed to the top-K counterparties.
    """
    total_volume = df["amount"].sum()
    if total_volume == 0:
        return 0.0

    by_cp = df.groupby("counterparty_name")["amount"].sum()
    top = by_cp.nlargest(top_k).sum()
    return float(top / total_volume)


def _compute_flow_entropy(df: pd.DataFrame) -> float:
    """
    Shannon entropy of daily flow distribution.
    Higher entropy = more uniform distribution = healthier.
    """
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
    daily_volumes = df_copy.groupby("date")["amount"].sum().values.astype(float)

    if len(daily_volumes) == 0 or daily_volumes.sum() == 0:
        return 0.0

    probs = daily_volumes / daily_volumes.sum()
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log2(probs)))

    # Normalize by max possible entropy
    max_entropy = math.log2(max(len(probs), 2))
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return round(normalized, 4)
