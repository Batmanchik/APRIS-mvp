from __future__ import annotations

import math
from typing import Sequence

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.legacy_features_v2 import fifo_match

# Mule bursts occur within minutes to hours. A lambda of 0.5 per hour gives
# a half-life of roughly 1.4 hours, strongly discounting money held for days
# while keeping intra-hour flows near 1.0.
LAMBDA_FAST = 0.5

# Pyramids stretch over months. A lambda of 0.001 per hour gives a half-life
# of roughly 693 hours (almost a month), allowing the scheme to hold funds
# much longer before they are completely discounted.
LAMBDA_SLOW = 0.001


def flow_weight(account: str, events: Sequence[TransactionEvent], *, lam_per_hour: float) -> float | None:
    """Calculate the volume-weighted share of flow with individual decay.
    
    Returns the sum of matched parcels scaled by their retention time,
    divided by total inflow.
    If total inflow is zero, returns None.
    """
    matches, total_in, total_out, unmatched_out = fifo_match(account, events)
    if total_in <= 0:
        return None
        
    numerator = sum(m.amount * math.exp(-lam_per_hour * (m.held_seconds / 3600.0)) for m in matches)
    return numerator / total_in
