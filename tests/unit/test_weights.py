import pytest
from datetime import datetime
from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.ml.event_features_v2 import graph_features_from_events
from apris.cheops.infrastructure.ml.graph_v2 import extract_graph_features_from_case_window, CaseWindow

def test_weights_are_summed_correctly():
    ts = datetime(2026, 1, 1)
    events = [
        TransactionEvent(
            event_id="e1", ts=ts, amount=100.0, currency="USD",
            sender_id="A", receiver_id="B", sender_type="P", receiver_type="P",
            channel="C", jurisdiction="J", asset_type="A"
        ),
        TransactionEvent(
            event_id="e2", ts=ts, amount=150.0, currency="USD",
            sender_id="A", receiver_id="B", sender_type="P", receiver_type="P",
            channel="C", jurisdiction="J", asset_type="A"
        ),
        TransactionEvent(
            event_id="e3", ts=ts, amount=50.0, currency="USD",
            sender_id="A", receiver_id="B", sender_type="P", receiver_type="P",
            channel="C", jurisdiction="J", asset_type="A"
        ),
        TransactionEvent(
            event_id="e4", ts=ts, amount=300.0, currency="USD",
            sender_id="B", receiver_id="C", sender_type="P", receiver_type="P",
            channel="C", jurisdiction="J", asset_type="A"
        )
    ]
    cw = CaseWindow(case_id="cw1", events=tuple(events), start_ts=ts, end_ts=ts, window_hours=24)
    features = extract_graph_features_from_case_window(cw)
    assert features["graph_weight_cv_norm"] == 0.0, "Weights were not summed properly; CV should be 0 for identical edge sums."
