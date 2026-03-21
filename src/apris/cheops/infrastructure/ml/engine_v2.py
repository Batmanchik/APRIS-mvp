from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx

from apris.cheops.application.dto import ExplainOutput, ScoreOutput
from apris.cheops.domain.contracts import map_events_to_typology_labels
from apris.cheops.domain.models import CaseWindow
from apris.cheops.domain.typologies import FraudTypology, TYPOLOGY_NAMES
from apris.risk_engine import (
    OPERATIONAL_INPUT_BOUNDS,
    load_artifacts,
    operational_to_features,
    predict_risk,
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class MultiBranchRiskEngine:
    model: Any | None = None
    feature_names: list[str] | None = None
    model_version: str = "cheops-v2-fusion"
    calibration_version: str = "iso-v1"
    auto_load_artifacts: bool = True

    def __post_init__(self) -> None:
        if (self.model is None or self.feature_names is None) and self.auto_load_artifacts:
            try:
                self.model, self.feature_names = load_artifacts()
            except FileNotFoundError:
                self.model = None
                self.feature_names = None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "tabular_model_loaded": self.model is not None,
            "sequence_branch": "heuristic_tcn_proxy",
            "graph_branch": "heuristic_graphsage_proxy",
            "fusion": "weighted_meta_head",
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
        }

    def _aggregate_operational(self, case_window: CaseWindow) -> dict[str, float]:
        events = case_window.events
        tx_count_total = float(len(events))
        counterparties = set()
        incoming_funds = 0.0
        payouts_total = 0.0

        sender_amounts: dict[str, float] = {}
        for event in events:
            counterparties.add(event.sender_id)
            counterparties.add(event.receiver_id)
            incoming_funds += event.amount
            sender_amounts[event.sender_id] = sender_amounts.get(event.sender_id, 0.0) + event.amount
            if event.sender_type in {"company", "legal_entity", "merchant"}:
                payouts_total += event.amount

        sender_volumes = sorted(sender_amounts.values(), reverse=True)
        top1 = sender_volumes[0] if sender_volumes else 0.0
        top10 = sum(sender_volumes[:10]) if sender_volumes else 0.0
        top1_wallet_share = top1 / max(incoming_funds, 1.0)
        top10_wallet_share = top10 / max(incoming_funds, 1.0)

        unique_senders = len({event.sender_id for event in events})
        new_clients_current = float(max(1, unique_senders))
        new_clients_previous = float(max(1, int(new_clients_current * 0.82)))
        referred_clients_current = float(max(0, int(new_clients_current * 0.31)))

        hours = max((case_window.end_ts - case_window.start_ts).total_seconds() / 3600.0, 1.0)
        avg_holding_days = float(max(1.0, min(365.0, (hours / 24.0) * 2.5)))

        operational = {
            "tx_count_total": tx_count_total,
            "unique_counterparties": float(max(2, len(counterparties))),
            "new_clients_current": new_clients_current,
            "new_clients_previous": new_clients_previous,
            "referred_clients_current": min(referred_clients_current, new_clients_current),
            "incoming_funds": float(max(1.0, incoming_funds)),
            "payouts_total": float(max(0.0, payouts_total)),
            "top1_wallet_share": float(_clip01(top1_wallet_share)),
            "top10_wallet_share": float(_clip01(max(top10_wallet_share, top1_wallet_share))),
            "avg_holding_days": avg_holding_days,
            "repeat_investor_share": float(_clip01(unique_senders / max(len(events), 1))),
            "max_referral_depth": float(max(1.0, min(30.0, math.log2(max(len(counterparties), 2)) + 1.0))),
        }
        for key, (low, high) in OPERATIONAL_INPUT_BOUNDS.items():
            operational[key] = float(max(low, min(high, operational[key])))
        return operational

    def _derive_tabular_features(
        self,
        case_window: CaseWindow,
        tabular_features: dict[str, float] | None,
    ) -> dict[str, float]:
        if tabular_features:
            return {str(k): float(v) for k, v in tabular_features.items()}
        operational = self._aggregate_operational(case_window)
        return operational_to_features(operational)

    def _score_tabular(self, tabular_features: dict[str, float]) -> float:
        if self.model is not None and self.feature_names is not None:
            result = predict_risk(tabular_features, model=self.model, feature_names=self.feature_names)
            return _clip01(float(result["probability"]))

        proxy = (
            tabular_features.get("growth_rate", 0.0) * 0.35
            + tabular_features.get("payout_dependency", 0.0) * 0.35
            + tabular_features.get("centralization_index", 0.0) * 0.30
        )
        return _clip01(proxy)

    def _score_sequence(self, case_window: CaseWindow) -> float:
        span_hours = max((case_window.end_ts - case_window.start_ts).total_seconds() / 3600.0, 1.0)
        burst_rate = len(case_window.events) / span_hours
        temporal_jumps = 0
        events = case_window.events
        for idx in range(1, len(events)):
            delta_sec = (events[idx].ts - events[idx - 1].ts).total_seconds()
            if delta_sec <= 90:
                temporal_jumps += 1
        density = temporal_jumps / max(len(events) - 1, 1)
        raw = 0.55 * burst_rate + 2.1 * density
        return _clip01(_sigmoid(raw - 1.8))

    def _score_graph(self, case_window: CaseWindow) -> float:
        graph = nx.DiGraph()
        for event in case_window.events:
            graph.add_edge(event.sender_id, event.receiver_id, weight=event.amount)

        if graph.number_of_nodes() == 0:
            return 0.0

        in_degrees = dict(graph.in_degree())
        hub_share = max(in_degrees.values()) / max(sum(in_degrees.values()), 1)
        density = nx.density(graph)
        components = nx.number_weakly_connected_components(graph)
        raw = 1.8 * hub_share + 1.1 * density + (0.3 if components <= 2 else -0.1)
        return _clip01(_sigmoid(raw))

    def _fuse(self, tabular_prob: float, sequence_prob: float, graph_prob: float) -> float:
        return _clip01(0.58 * tabular_prob + 0.22 * sequence_prob + 0.20 * graph_prob)

    def _typology_probs(
        self,
        labels: dict[str, int],
        tabular_prob: float,
        sequence_prob: float,
        graph_prob: float,
    ) -> dict[str, float]:
        probs: dict[str, float] = {}
        for name in TYPOLOGY_NAMES:
            label_boost = 0.35 if labels.get(name, 0) == 1 else 0.0
            if name == FraudTypology.LEGAL_LAYERING.value:
                score = 0.20 + 0.55 * graph_prob + 0.25 * tabular_prob + label_boost
            elif name == FraudTypology.LEGAL_TO_CRYPTO_BRIDGE.value:
                score = 0.20 + 0.45 * sequence_prob + 0.35 * tabular_prob + label_boost
            elif name == FraudTypology.CRYPTO_MIXING.value:
                score = 0.20 + 0.55 * graph_prob + 0.25 * sequence_prob + label_boost
            elif name == FraudTypology.STRUCTURED_SPLITTING.value:
                score = 0.20 + 0.40 * sequence_prob + 0.35 * tabular_prob + label_boost
            else:
                score = 0.20 + 0.45 * tabular_prob + 0.25 * sequence_prob + label_boost
            probs[name] = _clip01(score)
        return probs

    def _risk_band(self, global_risk: float) -> str:
        if global_risk >= 0.85:
            return "CRITICAL"
        if global_risk >= 0.70:
            return "HIGH"
        if global_risk >= 0.45:
            return "MEDIUM"
        return "LOW"

    def score_case(
        self,
        case_window: CaseWindow,
        *,
        tabular_features: dict[str, float] | None = None,
    ) -> ScoreOutput:
        features = self._derive_tabular_features(case_window, tabular_features)
        tabular_prob = self._score_tabular(features)
        sequence_prob = self._score_sequence(case_window)
        graph_prob = self._score_graph(case_window)
        global_risk = self._fuse(tabular_prob, sequence_prob, graph_prob)

        labels = map_events_to_typology_labels(list(case_window.events))
        typology_probs = self._typology_probs(labels, tabular_prob, sequence_prob, graph_prob)
        return ScoreOutput(
            case_id=case_window.case_id,
            global_risk=global_risk,
            typology_probs=typology_probs,
            risk_band=self._risk_band(global_risk),
            model_version=self.model_version,
            calibration_version=self.calibration_version,
            explanation_ready=True,
        )

    def explain_case(
        self,
        case_window: CaseWindow,
        *,
        tabular_features: dict[str, float] | None = None,
    ) -> ExplainOutput:
        features = self._derive_tabular_features(case_window, tabular_features)
        score = self.score_case(case_window, tabular_features=features)

        ranked_tabular = sorted(
            ((name, float(value)) for name, value in features.items()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        tabular_factors: list[dict[str, float | str]] = [
            {"feature": name, "value": value} for name, value in ranked_tabular
        ]

        deltas = []
        for idx in range(1, len(case_window.events)):
            delta_min = (case_window.events[idx].ts - case_window.events[idx - 1].ts).total_seconds() / 60.0
            deltas.append(delta_min)
        avg_delta = float(sum(deltas) / max(len(deltas), 1))
        sequence_factors: list[dict[str, float | str]] = [
            {"factor": "event_count", "value": float(len(case_window.events))},
            {"factor": "avg_delta_minutes", "value": avg_delta},
            {"factor": "window_hours", "value": float(case_window.window_hours)},
        ]

        graph = nx.DiGraph()
        for event in case_window.events:
            graph.add_edge(event.sender_id, event.receiver_id)
        graph_factors: list[dict[str, float | str]] = [
            {"factor": "nodes", "value": float(graph.number_of_nodes())},
            {"factor": "edges", "value": float(graph.number_of_edges())},
            {"factor": "density", "value": float(nx.density(graph) if graph.number_of_nodes() > 1 else 0.0)},
        ]

        strongest_typology = max(score.typology_probs.items(), key=lambda item: item[1])[0]

        summary = (
            f"Case {case_window.case_id} has global risk {score.global_risk:.3f} "
            f"({score.risk_band}). Strongest typology: "
            f"{strongest_typology}."
        )
        return ExplainOutput(
            summary=summary,
            tabular_factors=tabular_factors,
            sequence_factors=sequence_factors,
            graph_factors=graph_factors,
            confidence=_clip01(0.55 + 0.35 * score.global_risk),
        )
