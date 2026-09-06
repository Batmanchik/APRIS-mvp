import time
import os
import psutil
import pytest
from datetime import datetime
from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.oes_sampler import one_side_edge_sampling

def test_oes_sampler_stress():
    print("Generating massive synthetic graph...")
    num_edges = 1_000_000
    
    events = []
    hub_id = "PAY_AGGREGATOR_MAIN"
    
    # Half of the edges go from regular users to the hub
    # Half of the edges go from the hub to regular users
    ts = datetime(2026, 1, 1)
    
    # Optimized generation using list comprehension
    half = num_edges // 2
    events_in = [
        TransactionEvent(
            event_id=f"evt_in_{i}",
            sender_id=f"user_{i}",
            receiver_id=hub_id,
            amount=100.0,
            ts=ts,
            currency="USD",
            sender_type="person",
            receiver_type="person",
            channel="legal",
            jurisdiction="KZ",
            asset_type="fiat"
        ) for i in range(half)
    ]
    events_out = [
        TransactionEvent(
            event_id=f"evt_out_{i}",
            sender_id=hub_id,
            receiver_id=f"user_{i}",
            amount=50.0,
            ts=ts,
            currency="USD",
            sender_type="person",
            receiver_type="person",
            channel="legal",
            jurisdiction="KZ",
            asset_type="fiat"
        ) for i in range(half)
    ]
    
    events = events_in + events_out
    
    print(f"Total events generated: {len(events)}")
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)
    
    start_time = time.time()
    
    sampled_events = one_side_edge_sampling(events, target_degree=50.0)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Memory before: {mem_before:.2f} MB, Memory after: {mem_after:.2f} MB, Spike: {mem_after - mem_before:.2f} MB")
    
    assert elapsed < 10.0, f"OES sampler timed out: {elapsed:.2f}s > 10.0s"
    assert len(sampled_events) > 0, "Sampler filtered out all events!"
