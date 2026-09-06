import math
import random
from datetime import datetime, timedelta

import pytest
from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.flow_weight import flow_weight, LAMBDA_FAST, LAMBDA_SLOW
from apris.cheops.infrastructure.simulation.config import SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world

def _make_event(sender: str, receiver: str, amount: float, ts: datetime, event_id: str) -> TransactionEvent:
    return TransactionEvent(
        event_id=event_id,
        ts=ts,
        amount=amount,
        currency="KZT",
        sender_id=sender,
        receiver_id=receiver,
        sender_type="person",
        receiver_type="person",
        channel="legal",
        jurisdiction="KZ",
        asset_type="fiat"
    )

def test_pure_transit():
    start = datetime(2026, 1, 1, 12, 0)
    events = [
        _make_event("S1", "ACC", 100.0, start, "ev1"),
        _make_event("ACC", "R1", 100.0, start + timedelta(minutes=10), "ev2"),
    ]
    w = flow_weight("ACC", events, lam_per_hour=LAMBDA_FAST)
    assert w is not None
    assert math.isclose(w, 1.0, abs_tol=0.1) # 10 mins is 1/6 hour, exp(-0.5 * 1/6) = exp(-0.0833) = 0.92

def test_transit_30_days_fast_lambda():
    start = datetime(2026, 1, 1, 12, 0)
    events = [
        _make_event("S1", "ACC", 100.0, start, "ev1"),
        _make_event("ACC", "R1", 100.0, start + timedelta(days=30), "ev2"),
    ]
    w = flow_weight("ACC", events, lam_per_hour=LAMBDA_FAST)
    assert w is not None
    assert math.isclose(w, 0.0, abs_tol=1e-5)

def test_only_spends_returns_none():
    start = datetime(2026, 1, 1, 12, 0)
    events = [
        _make_event("ACC", "R1", 100.0, start, "ev1"),
        _make_event("ACC", "R2", 50.0, start + timedelta(minutes=10), "ev2"),
    ]
    w = flow_weight("ACC", events, lam_per_hour=LAMBDA_FAST)
    assert w is None

def test_generated_world_accounts_w_bounds():
    config = SimulationConfig(days=10)
    world = generate_world(config)
    accounts = list(world.accounts.keys())
    random.shuffle(accounts)
    sample = accounts[:200]
    from collections import defaultdict
    grouped = defaultdict(list)
    for e in world.events:
        grouped[e.sender_id].append(e)
        if e.receiver_id != e.sender_id:
            grouped[e.receiver_id].append(e)
            
    for acc in sample:
        acc_events = grouped.get(acc, [])
        w = flow_weight(acc, acc_events, lam_per_hour=LAMBDA_FAST)
        if w is not None:
            assert 0.0 <= w <= 1.0

def test_reordering_events_invariant():
    start = datetime(2026, 1, 1, 12, 0)
    events = [
        _make_event("S1", "ACC", 100.0, start, "ev1"),
        _make_event("ACC", "R1", 40.0, start + timedelta(minutes=10), "ev2"),
        _make_event("S2", "ACC", 20.0, start + timedelta(minutes=20), "ev3"),
        _make_event("ACC", "R2", 80.0, start + timedelta(minutes=30), "ev4"),
    ]
    w1 = flow_weight("ACC", events, lam_per_hour=LAMBDA_FAST)
    
    events_reversed = list(reversed(events))
    w2 = flow_weight("ACC", events_reversed, lam_per_hour=LAMBDA_FAST)
    
    assert w1 == w2

def test_shop_and_mule_differ():
    start = datetime(2026, 1, 1, 12, 0)
    
    # Mule: receives 1M, cash out 1M in 5 mins
    mule_events = [
        _make_event("S1", "MULE", 1000000.0, start, "ev1"),
        _make_event("MULE", "ATM", 1000000.0, start + timedelta(minutes=5), "ev2"),
    ]
    w_mule = flow_weight("MULE", mule_events, lam_per_hour=LAMBDA_FAST)
    
    # Shop: receives 1M (in small chunks), pays out 1M after 3 days
    shop_events = []
    for i in range(10):
        shop_events.append(_make_event(f"CUST{i}", "SHOP", 100000.0, start + timedelta(minutes=i*30), f"in{i}"))
    shop_events.append(_make_event("SHOP", "SUPPLIER", 1000000.0, start + timedelta(days=3), "out"))
    
    w_shop = flow_weight("SHOP", shop_events, lam_per_hour=LAMBDA_FAST)
    
    assert w_mule is not None
    assert w_shop is not None
    assert w_mule > 0.9
    assert w_shop < 0.1
