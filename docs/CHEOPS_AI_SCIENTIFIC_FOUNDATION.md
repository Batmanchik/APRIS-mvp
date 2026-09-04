# Cheops AI: Scientific Foundation (v1)

Date: 2026-03-21  
Status: Working scientific section for judges, reviewers, and technical audit.

## 1. Purpose
This document formalizes the current Cheops AI methodology:
- mathematical definitions and notation;
- what is treated as "pyramid-like" behavior in the model;
- formulas for feature engineering and scoring;
- branch fusion, calibration, and drift control.

Important: this is an operational/ML definition for risk screening, not a legal qualification.

## 2. Notation
- Event: `e_i`
- Case window: `W(c, H)` where `c` is `case_id`, `H` is window length in hours
- Number of events: `n = |W|`
- Event amount: `a_i > 0`
- Event timestamp: `t_i`
- Sender/receiver ids: `s_i`, `r_i`
- Global binary target in training: `y in {0, 1}`
- Typology target for typology `k`: `y_k in {0, 1}`
- Global risk score: `p_global in [0, 1]`

## 3. Data Contract Level
Canonical event schema (`TransactionEvent`) includes:
- `event_id, ts, amount, currency, sender_id, receiver_id, sender_type, receiver_type, channel, jurisdiction, asset_type`
- optional: `tx_hash, case_id, metadata`

Channel domain: `channel in {"legal", "crypto"}`.

Case window construction:
- sort events by `ts`,
- keep only events in last `H` hours from `end_ts`,
- resulting object: `CaseWindow(case_id, events, start_ts, end_ts, window_hours)`.

## 4. Operational Facts -> Model Features
For operational inputs:
- `new_clients_current`,
- `new_clients_previous`,
- `referred_clients_current`,
- `incoming_funds`,
- `payouts_total`,
- `top1_wallet_share`,
- `top10_wallet_share`,
- `avg_holding_days`,
- `repeat_investor_share`,
- `unique_counterparties`,
- `tx_count_total`,
- `max_referral_depth`.

Core formulas:
- `growth_rate = (new_clients_current - new_clients_previous) / max(new_clients_previous, 1)`
- `referral_ratio = referred_clients_current / max(new_clients_current, 1)`
- `payout_dependency = payouts_total / max(incoming_funds, 1)`
- `centralization_index = top1_wallet_share`
- `avg_holding_time = avg_holding_days`
- `reinvestment_rate = repeat_investor_share`
- `gini_coefficient = 0.12 + 0.72 * top10_wallet_share + 0.22 * top1_wallet_share`
- `entropy_ratio = log(1 + unique_counterparties) / log(1 + max(tx_count_total, unique_counterparties + 1))`
- `transaction_entropy = 0.3 + 4.7 * entropy_ratio * (1 - 0.55 * top1_wallet_share)`
- `structural_depth = max_referral_depth`

Each feature is clipped to predefined bounds (`FEATURE_BOUNDS`).

## 5. What Is "Pyramid-Like" in Current ML System
Cheops AI currently uses three layers of evidence:

1. Statistical class (`y=1`) in training data:
- synthetic "pyramid-like" distribution is generated with higher:
  - growth,
  - referral share,
  - payout dependency,
  - centralization,
  - inequality (`gini`),
  - depth;
- and lower entropy / lower holding time than legitimate profiles.

2. Rule-typology layer from event graph/time behavior:
- deterministic event-level labels (Section 6).

3. Continuous risk layer:
- final `p_global` from multi-branch fusion (Section 9).

Operationally, a case is treated as pyramid-like/high-risk if:
- `p_global` enters `HIGH`/`CRITICAL` band, and
- typology probabilities show strong mass on route-related suspicious patterns.

## 6. Event-Level Typology Rules (Deterministic Labels)
The current event mapper produces labels:

- `LEGAL_TO_CRYPTO_BRIDGE = 1` iff both channels exist in case (`legal` and `crypto`).
- `STRUCTURED_SPLITTING = 1` iff:
  - unique receivers `>= 5`, and
  - count of small parts `>= max(4, floor(n/3))`,
  - where "small part" means `a_i < 0.35 * mean(a)`.
- `CRYPTO_MIXING = 1` iff:
  - number of crypto events `>= 6`,
  - unique crypto receivers `>= 4`.
- `LEGAL_LAYERING = 1` iff:
  - directed graph depth `>= 4`,
  - legal events count `>= 3`.
- `CASH_OUT = 1` iff:
  - outgoing share from company/legal sender types `>= 0.62`,
  - `n >= 6`.

## 7. Typology Targets for Tabular Training
Besides event rules, tabular branch creates typology targets from features:

- `LEGAL_LAYERING_raw = ((depth >= 8.0) and (central >= 0.45)) or ((gini >= 0.62) and (payout >= 0.85))`
- `LEGAL_TO_CRYPTO_BRIDGE_raw = ((payout >= 0.92) and (entropy <= 2.35)) or ((growth >= 0.23) and (holding <= 40.0))`
- `CRYPTO_MIXING_raw = ((entropy <= 2.2) and (reinvest >= 0.56)) or ((central >= 0.58) and (depth >= 7.0))`
- `STRUCTURED_SPLITTING_raw = ((referral >= 0.52) and (depth >= 7.0)) or ((growth >= 0.28) and (entropy <= 2.6))`
- `CASH_OUT_raw = ((payout >= 1.00) and (holding <= 32.0)) or ((central >= 0.56) and (gini >= 0.68))`

Fallback signal for low-positive-rate balancing:
- `fallback = 0.55 * y_global + 0.25 * payout + 0.20 * central`

If a typology has too few positives, top fallback-ranked rows are promoted to keep minimum positive rate.

## 8. Branch Models
### 8.1 Tabular Branch
- Model family: LightGBM binary classifiers.
- Heads:
  - global head (`p_tabular_global`),
  - one head per typology (`p_tabular_k`).
- Calibration: isotonic regression per head.

### 8.2 Sequence Branch
Two modes:
- trained surrogate (`cheops_v2_sequence.joblib`) if available,
- heuristic fallback otherwise.

Training matrix from normalized tabular features (`x~`):
- `event_rate_hour = 0.08 + 0.92 * (0.42*growth~ + 0.19*referral~ + 0.19*depth~ + 0.20*reinvest~)`
- `burst_ratio_90s = 0.05 + 0.95 * (0.46*growth~ + 0.31*holding_short~ + 0.23*entropy_low~)`
- `median_delta_inverse = 0.06 + 0.94 * (0.44*holding_short~ + 0.30*entropy_low~ + 0.26*growth~)`
- `amount_cv_norm = 0.05 + 0.95 * (0.37*central~ + 0.34*payout~ + 0.29*gini~)`
- `unique_sender_ratio = 0.04 + 0.96 * (0.43*referral~ + 0.30*depth~ + 0.27*(1-central~))`

Runtime extraction from events:
- `span_hours = max((end_ts - start_ts), 60 sec) / 3600`
- `rate = n / span_hours`
- `event_rate_hour = 1 - exp(-rate / 4)`
- `burst_ratio_90s = count(delta_t <= 90 sec) / max(n-1, 1)`
- `median_delta_inverse = 1 / (1 + median(delta_t_minutes)/60)`
- `amount_cv_norm = min((std(amount)/mean(amount))/2, 1)`
- `unique_sender_ratio = |unique_senders| / max(n, 1)`

Sequence heuristic score:
- `p_seq_heur = 0.39*event_rate_hour + 0.29*burst_ratio_90s + 0.20*median_delta_inverse + 0.07*amount_cv_norm + 0.05*unique_sender_ratio`

### 8.3 Graph Branch
Two modes:
- trained surrogate (`cheops_v2_graph.joblib`) if available,
- heuristic fallback otherwise.

Training matrix from normalized tabular features:
- `graph_density = 0.05 + 0.95*(0.38*central~ + 0.34*depth~ + 0.28*entropy_low~)`
- `graph_hub_share = 0.04 + 0.96*(0.55*central~ + 0.30*gini~ + 0.15*payout~)`
- `graph_component_compactness = 0.06 + 0.94*(0.42*depth~ + 0.34*referral~ + 0.24*(1-entropy_low~))`
- `graph_transitivity = 0.03 + 0.97*(0.40*central~ + 0.32*depth~ + 0.28*referral~)`
- `graph_weight_cv_norm = 0.05 + 0.95*(0.47*gini~ + 0.31*payout~ + 0.22*central~)`

Runtime extraction from event graph:
- `graph_density = density(G)`
- `graph_hub_share = max_in_degree / sum_in_degree`
- `graph_component_compactness = 1 / number_of_weakly_connected_components`
- `graph_transitivity = transitivity(undirected(G))` if nodes `>= 3`, else `0`
- `graph_weight_cv_norm = min((std(edge_weight)/mean(edge_weight))/2, 1)`

Graph heuristic score:
- `p_graph_heur = 0.34*hub_share + 0.30*density + 0.18*compactness + 0.10*transitivity + 0.08*weight_cv_norm`

## 9. Fusion and Final Risk
If fusion artifact exists:
- logistic meta-model on `[p_tabular, p_sequence, p_graph]`,
- then isotonic calibration.

Fallback fusion (deterministic):
- `p_global = 0.58*p_tabular + 0.22*p_sequence + 0.20*p_graph`

Risk bands in v2 runtime:
- `CRITICAL`: `p_global >= 0.85`
- `HIGH`: `0.70 <= p_global < 0.85`
- `MEDIUM`: `0.45 <= p_global < 0.70`
- `LOW`: `p_global < 0.45`

## 10. Calibration and Reliability Metrics
Per branch/head:
- ROC-AUC
- Brier score
- ECE (Expected Calibration Error)

ECE formula (histogram bins):
- let bins be `B_j`,
- `conf_j = mean(p_i | i in B_j)`,
- `acc_j = mean(y_i | i in B_j)`,
- `w_j = |B_j| / N`,
- then `ECE = sum_j w_j * |acc_j - conf_j|`.

## 11. Drift Control (PSI)
For each feature:
- baseline histogram rates `q_j`,
- current histogram rates `p_j`,
- `PSI = sum_j (q_j - p_j) * ln(q_j / p_j)`.

Thresholds:
- `PSI < 0.10`: stable
- `0.10 <= PSI < 0.25`: moderate drift
- `PSI >= 0.25`: high drift

Overall drift is the mean PSI across model features.

## 12. Explainability Outputs
`/api/v2/explain` returns:
- `summary`,
- `tabular_factors`,
- `sequence_factors`,
- `graph_factors`,
- `branch_scores` (`tabular`, `sequence`, `graph`, `fusion`),
- `branch_modes` (trained vs fallback),
- `confidence`.

This allows analyst and judge to trace not only final score but branch-level contribution.

## 13. Current Limitations (Important for Judges)
- Current high metrics are on synthetic benchmark data; external domain validation is a separate phase.
- "Pyramid-like" is an ML operational label, not a final legal verdict.
- Typology thresholds are currently rule-based/engineered and should be refined on real labeled cases.

## 14. Reproducibility References
- Feature/threshold definitions: `src/apris/risk_engine.py`, `src/apris/data_generator.py`
- Typology rules: `src/apris/cheops/domain/contracts.py`
- Tabular branch: `src/apris/cheops/infrastructure/ml/tabular_v2.py`
- Sequence branch: `src/apris/cheops/infrastructure/ml/sequence_v2.py`
- Graph branch: `src/apris/cheops/infrastructure/ml/graph_v2.py`
- Fusion: `src/apris/cheops/infrastructure/ml/fusion_v2.py`
- Drift: `src/apris/cheops/infrastructure/ml/drift_v2.py`
- Runtime scorer: `src/apris/cheops/infrastructure/ml/engine_v2.py`
