from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apris.cheops.application.dto import ExplainOutput, ScoreOutput
from apris.cheops.domain.contracts import map_events_to_typology_labels
from apris.cheops.domain.models import CaseWindow
from apris.cheops.domain.typologies import FraudTypology, TYPOLOGY_NAMES
from apris.cheops.infrastructure.ml.drift_v2 import (
    DRIFT_REPORT_V2_PATH,
    FEATURE_PROFILE_V2_PATH,
)
from apris.cheops.infrastructure.ml.fusion_v2 import (
    FUSION_V2_ARTIFACT_PATH,
    FUSION_V2_METRICS_PATH,
    load_fusion_artifact,
    predict_fusion_meta,
)
from apris.cheops.infrastructure.ml.graph_v2 import (
    GRAPH_V2_ARTIFACT_PATH,
    GRAPH_V2_METRICS_PATH,
    extract_graph_features_from_case_window,
    heuristic_graph_from_case_window,
    load_graph_artifact,
    predict_graph_from_case_window,
)
from apris.cheops.infrastructure.ml.sequence_v2 import (
    SEQUENCE_V2_ARTIFACT_PATH,
    SEQUENCE_V2_METRICS_PATH,
    extract_sequence_features_from_case_window,
    heuristic_sequence_from_case_window,
    load_sequence_artifact,
    predict_sequence_from_case_window,
)
from apris.cheops.infrastructure.ml.tabular_v2 import (
    TABULAR_V2_BUNDLE_PATH,
    TABULAR_V2_METRICS_PATH,
    load_tabular_bundle,
    predict_tabular_bundle,
)
from apris.cheops.infrastructure.ml.model_registry_v2 import MODEL_REGISTRY_V2_PATH
from apris.risk_engine import (
    OPERATIONAL_INPUT_BOUNDS,
    load_artifacts,
    operational_to_features,
    predict_risk,
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass
class MultiBranchRiskEngine:
    model: Any | None = None
    feature_names: list[str] | None = None
    tabular_bundle: dict[str, Any] | None = None
    tabular_bundle_path: str | Path = TABULAR_V2_BUNDLE_PATH
    sequence_model: dict[str, Any] | None = None
    sequence_model_path: str | Path = SEQUENCE_V2_ARTIFACT_PATH
    graph_model: dict[str, Any] | None = None
    graph_model_path: str | Path = GRAPH_V2_ARTIFACT_PATH
    fusion_meta: dict[str, Any] | None = None
    fusion_meta_path: str | Path = FUSION_V2_ARTIFACT_PATH
    model_version: str = "cheops-v2-fusion"
    calibration_version: str = "iso-v1"
    auto_load_artifacts: bool = True

    def __post_init__(self) -> None:
        if self.tabular_bundle is None and self.auto_load_artifacts:
            try:
                self.tabular_bundle = load_tabular_bundle(self.tabular_bundle_path)
                bundle_version = str(self.tabular_bundle.get("bundle_version", "cheops-tabular-v2"))
                self.model_version = f"cheops-v2-fusion+{bundle_version}"
                self.calibration_version = "isotonic-tabular-v2"
            except FileNotFoundError:
                self.tabular_bundle = None

        if self.fusion_meta is None and self.auto_load_artifacts:
            try:
                self.fusion_meta = load_fusion_artifact(self.fusion_meta_path)
            except FileNotFoundError:
                self.fusion_meta = None

        if self.sequence_model is None and self.auto_load_artifacts:
            try:
                self.sequence_model = load_sequence_artifact(self.sequence_model_path)
            except FileNotFoundError:
                self.sequence_model = None

        if self.graph_model is None and self.auto_load_artifacts:
            try:
                self.graph_model = load_graph_artifact(self.graph_model_path)
            except FileNotFoundError:
                self.graph_model = None

        if (
            self.tabular_bundle is None
            and (self.model is None or self.feature_names is None)
            and self.auto_load_artifacts
        ):
            try:
                self.model, self.feature_names = load_artifacts()
            except FileNotFoundError:
                self.model = None
                self.feature_names = None

        if self.fusion_meta is not None:
            self.model_version = f"{self.model_version}+fusion-meta-v2"
            self.calibration_version = f"{self.calibration_version}+fusion-meta-v2"
        if self.sequence_model is not None:
            self.model_version = f"{self.model_version}+sequence-v2"
            self.calibration_version = f"{self.calibration_version}+sequence-v2"
        if self.graph_model is not None:
            self.model_version = f"{self.model_version}+graph-v2"
            self.calibration_version = f"{self.calibration_version}+graph-v2"

    def health(self) -> dict[str, Any]:
        tabular_loaded = self.tabular_bundle is not None or self.model is not None
        return {
            "status": "ok",
            "tabular_model_loaded": tabular_loaded,
            "sequence_branch": "trained_tcn_surrogate" if self.sequence_model is not None else "heuristic_tcn_proxy",
            "graph_branch": "trained_graphsage_surrogate" if self.graph_model is not None else "heuristic_graphsage_proxy",
            "fusion": "logistic_meta_head" if self.fusion_meta is not None else "weighted_meta_head",
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
        }

    def _safe_load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def health_details(self) -> dict[str, Any]:
        health = self.health()
        artifacts = {
            "tabular_bundle": Path(self.tabular_bundle_path).exists(),
            "tabular_metrics": TABULAR_V2_METRICS_PATH.exists(),
            "sequence_model": Path(self.sequence_model_path).exists(),
            "sequence_metrics": SEQUENCE_V2_METRICS_PATH.exists(),
            "graph_model": Path(self.graph_model_path).exists(),
            "graph_metrics": GRAPH_V2_METRICS_PATH.exists(),
            "fusion_model": Path(self.fusion_meta_path).exists(),
            "fusion_metrics": FUSION_V2_METRICS_PATH.exists(),
            "feature_profile": FEATURE_PROFILE_V2_PATH.exists(),
            "drift_report": DRIFT_REPORT_V2_PATH.exists(),
            "model_registry": MODEL_REGISTRY_V2_PATH.exists(),
        }

        tabular_metrics = self._safe_load_json(TABULAR_V2_METRICS_PATH) or {}
        sequence_metrics = self._safe_load_json(SEQUENCE_V2_METRICS_PATH) or {}
        graph_metrics = self._safe_load_json(GRAPH_V2_METRICS_PATH) or {}
        fusion_metrics = self._safe_load_json(FUSION_V2_METRICS_PATH) or {}
        drift_report = self._safe_load_json(DRIFT_REPORT_V2_PATH) or {}
        model_registry = self._safe_load_json(MODEL_REGISTRY_V2_PATH) or {}

        return {
            "status": "ok",
            "health": health,
            "artifacts": artifacts,
            "metrics": {
                "tabular_global_roc_auc": tabular_metrics.get("global", {}).get("roc_auc"),
                "tabular_global_ece": tabular_metrics.get("global", {}).get("ece"),
                "sequence_roc_auc": sequence_metrics.get("sequence_head", {}).get("roc_auc"),
                "sequence_ece": sequence_metrics.get("sequence_head", {}).get("ece"),
                "graph_roc_auc": graph_metrics.get("graph_head", {}).get("roc_auc"),
                "graph_ece": graph_metrics.get("graph_head", {}).get("ece"),
                "fusion_roc_auc": fusion_metrics.get("meta_head", {}).get("roc_auc"),
                "fusion_ece": fusion_metrics.get("meta_head", {}).get("ece"),
            },
            "drift": {
                "overall_level": drift_report.get("overall_level"),
                "overall_psi": drift_report.get("overall_psi"),
                "features_high_drift": drift_report.get("features_high_drift"),
                "features_moderate_drift": drift_report.get("features_moderate_drift"),
            },
            "registry": {
                "selected_tabular_candidate": model_registry.get("selected_tabular_candidate"),
                "selection_reason": model_registry.get("selection_reason"),
                "generated_at": model_registry.get("generated_at"),
                "updated_at": model_registry.get("updated_at"),
            },
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
        if tabular_features is not None:
            normalized: dict[str, float] = {}
            for key, value in tabular_features.items():
                name = str(key)
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"tabular_features['{name}'] must be numeric.") from exc
                if not math.isfinite(number):
                    raise ValueError(f"tabular_features['{name}'] must be finite.")
                normalized[name] = number
            return normalized
        operational = self._aggregate_operational(case_window)
        return operational_to_features(operational)

    def _score_tabular(
        self,
        tabular_features: dict[str, float],
    ) -> tuple[float, dict[str, float] | None]:
        if self.tabular_bundle is not None:
            global_score, typology_scores = predict_tabular_bundle(tabular_features, self.tabular_bundle)
            return _clip01(global_score), {k: _clip01(v) for k, v in typology_scores.items()}

        if self.model is not None and self.feature_names is not None:
            result = predict_risk(tabular_features, model=self.model, feature_names=self.feature_names)
            return _clip01(float(result["probability"])), None

        proxy = (
            tabular_features.get("growth_rate", 0.0) * 0.35
            + tabular_features.get("payout_dependency", 0.0) * 0.35
            + tabular_features.get("centralization_index", 0.0) * 0.30
        )
        return _clip01(proxy), None

    def _score_sequence(self, case_window: CaseWindow) -> float:
        if self.sequence_model is not None:
            return _clip01(predict_sequence_from_case_window(case_window, self.sequence_model))
        return _clip01(heuristic_sequence_from_case_window(case_window))

    def _score_graph(self, case_window: CaseWindow) -> float:
        if self.graph_model is not None:
            return _clip01(predict_graph_from_case_window(case_window, self.graph_model))
        return _clip01(heuristic_graph_from_case_window(case_window))

    def _fuse(self, tabular_prob: float, sequence_prob: float, graph_prob: float) -> float:
        if self.fusion_meta is not None:
            return predict_fusion_meta(tabular_prob, sequence_prob, graph_prob, self.fusion_meta)
        return _clip01(0.58 * tabular_prob + 0.22 * sequence_prob + 0.20 * graph_prob)

    def _branch_modes(self) -> dict[str, str]:
        return {
            "tabular": "trained_tabular_v2" if self.tabular_bundle is not None else "legacy_or_proxy",
            "sequence": "trained_sequence_v2" if self.sequence_model is not None else "heuristic_proxy",
            "graph": "trained_graph_v2" if self.graph_model is not None else "heuristic_proxy",
            "fusion": "trained_logistic_meta" if self.fusion_meta is not None else "weighted_fallback",
        }

    def _score_components(
        self,
        case_window: CaseWindow,
        tabular_features: dict[str, float],
    ) -> dict[str, Any]:
        tabular_prob, tabular_typology_probs = self._score_tabular(tabular_features)
        sequence_prob = self._score_sequence(case_window)
        graph_prob = self._score_graph(case_window)
        global_risk = self._fuse(tabular_prob, sequence_prob, graph_prob)
        return {
            "tabular_prob": _clip01(tabular_prob),
            "sequence_prob": _clip01(sequence_prob),
            "graph_prob": _clip01(graph_prob),
            "global_risk": _clip01(global_risk),
            "tabular_typology_probs": tabular_typology_probs,
        }

    def _typology_probs(
        self,
        labels: dict[str, int],
        tabular_prob: float,
        sequence_prob: float,
        graph_prob: float,
        tabular_typology_probs: dict[str, float] | None = None,
    ) -> dict[str, float]:
        probs: dict[str, float] = {}
        for name in TYPOLOGY_NAMES:
            base_tabular_prob = (
                _clip01(tabular_typology_probs.get(name, tabular_prob))
                if tabular_typology_probs is not None
                else tabular_prob
            )
            label_boost = 0.35 if labels.get(name, 0) == 1 else 0.0
            if name == FraudTypology.LEGAL_LAYERING.value:
                score = 0.15 + 0.50 * graph_prob + 0.25 * sequence_prob + 0.25 * base_tabular_prob + label_boost
            elif name == FraudTypology.LEGAL_TO_CRYPTO_BRIDGE.value:
                score = 0.15 + 0.45 * sequence_prob + 0.20 * graph_prob + 0.35 * base_tabular_prob + label_boost
            elif name == FraudTypology.CRYPTO_MIXING.value:
                score = 0.15 + 0.50 * graph_prob + 0.20 * sequence_prob + 0.35 * base_tabular_prob + label_boost
            elif name == FraudTypology.STRUCTURED_SPLITTING.value:
                score = 0.15 + 0.35 * sequence_prob + 0.20 * graph_prob + 0.40 * base_tabular_prob + label_boost
            else:
                score = 0.15 + 0.25 * sequence_prob + 0.20 * graph_prob + 0.40 * base_tabular_prob + label_boost
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
        components = self._score_components(case_window, features)
        tabular_prob = float(components["tabular_prob"])
        sequence_prob = float(components["sequence_prob"])
        graph_prob = float(components["graph_prob"])
        global_risk = float(components["global_risk"])
        tabular_typology_probs = components["tabular_typology_probs"]

        labels = map_events_to_typology_labels(list(case_window.events))
        typology_probs = self._typology_probs(
            labels,
            tabular_prob,
            sequence_prob,
            graph_prob,
            tabular_typology_probs=tabular_typology_probs,
        )
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
        components = self._score_components(case_window, features)
        global_risk = float(components["global_risk"])
        tabular_prob = float(components["tabular_prob"])
        sequence_prob = float(components["sequence_prob"])
        graph_prob = float(components["graph_prob"])
        labels = map_events_to_typology_labels(list(case_window.events))
        tabular_typology_probs = components["tabular_typology_probs"]
        typology_probs = self._typology_probs(
            labels,
            tabular_prob,
            sequence_prob,
            graph_prob,
            tabular_typology_probs=tabular_typology_probs,
        )
        risk_band = self._risk_band(global_risk)

        ranked_tabular = sorted(
            ((name, float(value)) for name, value in features.items()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]
        tabular_factors: list[dict[str, float | str]] = [
            {"feature": name, "value": value} for name, value in ranked_tabular
        ]

        sequence_features = extract_sequence_features_from_case_window(case_window)
        sequence_factors: list[dict[str, float | str]] = [
            {"factor": name, "value": float(value)} for name, value in sequence_features.items()
        ]
        sequence_factors.insert(0, {"factor": "sequence_prob", "value": float(sequence_prob)})

        graph_features = extract_graph_features_from_case_window(case_window)
        graph_factors: list[dict[str, float | str]] = [
            {"factor": name, "value": float(value)} for name, value in graph_features.items()
        ]
        graph_factors.insert(0, {"factor": "graph_prob", "value": float(graph_prob)})

        strongest_typology = max(typology_probs.items(), key=lambda item: item[1])[0]

        summary = (
            f"Case {case_window.case_id} has global risk {global_risk:.3f} "
            f"({risk_band}). Strongest typology: "
            f"{strongest_typology}."
        )
        return ExplainOutput(
            summary=summary,
            tabular_factors=tabular_factors,
            sequence_factors=sequence_factors,
            graph_factors=graph_factors,
            branch_scores={
                "tabular": float(tabular_prob),
                "sequence": float(sequence_prob),
                "graph": float(graph_prob),
                "fusion": float(global_risk),
            },
            branch_modes=self._branch_modes(),
            confidence=_clip01(0.55 + 0.35 * global_risk),
        )
