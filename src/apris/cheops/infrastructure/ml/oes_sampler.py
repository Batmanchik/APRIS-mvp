"""
One-Side Edge Sampling (OES) for transaction graphs.

This module provides a highly optimized, vectorized implementation of One-Side Edge Sampling.
OES preserves edges that are structurally significant to at least one of their endpoints.
By computing a retention probability p(u) for each node based on its local density (e.g., degree),
an edge (u, v) is retained if it is sampled by u OR v. 

This filters out massive financial noise (like large exchange hot wallets or payroll hubs) 
while preserving the local topology of low-degree nodes (e.g., individual fraudsters or mules),
ensuring the fraud topology remains intact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections.abc import Sequence

from apris.cheops.domain.models import TransactionEvent

def one_side_edge_sampling(
    events: Sequence[TransactionEvent],
    target_degree: float = 50.0,
    random_state: int = 42,
) -> list[TransactionEvent]:
    """
    Applies One-Side Edge Sampling to a sequence of TransactionEvents.
    
    Complexity: O(E) where E is the number of events.
    Uses vectorized NumPy and Pandas operations to process millions of edges efficiently.
    """
    if len(events) < 2:
        return list(events)
        
    # Map string IDs to integer indices for fast array operations
    # pd.factorize is highly optimized in C for this purpose
    senders = [e.sender_id for e in events]
    receivers = [e.receiver_id for e in events]
    
    all_nodes = np.array(senders + receivers)
    codes, uniques = pd.factorize(all_nodes)
    
    n_events = len(events)
    u_idx = codes[:n_events]
    v_idx = codes[n_events:]
    
    # Compute total degrees (in-degree + out-degree)
    out_degrees = np.bincount(u_idx, minlength=len(uniques))
    in_degrees = np.bincount(v_idx, minlength=len(uniques))
    total_degrees = out_degrees + in_degrees
    
    # Compute retention probabilities: p = min(1.0, target_degree / total_degree)
    p_node = np.minimum(1.0, target_degree / np.maximum(total_degrees, 1.0))
    
    # Edge retention probability uses independent OR logic: P_keep = P_u + P_v - (P_u * P_v)
    p_u = p_node[u_idx]
    p_v = p_node[v_idx]
    edge_keep_prob = p_u + p_v - (p_u * p_v)
    
    # Vectorized random draw
    rng = np.random.RandomState(random_state)
    random_draws = rng.rand(n_events)
    
    # Boolean mask of edges to keep
    mask = random_draws < edge_keep_prob
    
    # Filter events using the fast boolean mask
    sampled_events = [events[i] for i, keep in enumerate(mask) if keep]
    
    return sampled_events
