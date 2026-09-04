# Audit Findings, 2026-09-04

Status: measured, not estimated. Every number below was produced by running
the project's own code on its own artifacts.

These are recorded because they are **results**, not just defects. Each one
is defensible material for the scientific write-up, and each one would be
found by a technical reviewer who reads the source.

---

## Finding 1 — the three ML branches are one vector renamed three times

**Severity: critical. Status: addressed by `ml/event_features_v2.py`.**

`graph_v2.build_graph_matrix_from_tabular` and
`sequence_v2.build_sequence_matrix_from_tabular` build their matrices as
hand-written linear combinations of the same nine aggregate features:

```python
graph_density   = 0.05 + 0.95 * (0.38*central + 0.34*depth + 0.28*entropy_low)
burst_ratio_90s = 0.05 + 0.95 * (0.46*growth + 0.31*holding_short + 0.23*entropy_low)
_proxy_graph_prob = 0.36*central + 0.28*depth + 0.22*gini + 0.14*payout
```

Consequences:

1. **The graph branch never reads a graph; the sequence branch never reads a
   sequence.** Both are deterministic transformations of one vector, so
   fusing the three branches adds no information — the fusion input is a
   rank-deficient map of the tabular input.
2. **`burst_ratio_90s` promises a 90-second window** while being derived from
   `avg_holding_time`, which is measured in days. The name asserts a time
   resolution the input does not carry.
3. `fusion_v2` names its own functions `_proxy_sequence_prob` and
   `_proxy_graph_prob`, which is honest in the code and invisible in the
   architecture diagram.

This is the same defect class as the original `graph_module.py`, which drew a
transaction graph from the very features the graph was presented as
supporting — one architectural layer up and considerably more convincing.

**Fix.** `event_features_v2.py` computes both matrices from an actual event
stream. `burst_ratio_90s` now measures a real 90-second window by a
two-pointer scan.

### The feature that carries the structure

Degree alone cannot express a mule network. Measured means over four case
kinds, full-scale world:

| Feature | Crowd collection | **Mule network** | Payroll | Pyramid |
|---|---|---|---|---|
| `graph_hub_share` (convergence) | 0.580 | 0.494 | 0.039 | 0.523 |
| `graph_fanout_share` (divergence) | 0.420 | 0.506 | 1.000 | 0.477 |
| **`graph_relay_share`** | **0.000** | **0.494** | **0.000** | **0.000** |
| `event_rate_hour` | 0.004 | 0.874 | 0.010 | 0.033 |
| `median_delta_inverse` | 0.019 | 0.954 | 0.137 | 0.084 |
| `burst_ratio_90s` | 0.023 | 0.230 | 0.012 | 0.002 |

`graph_relay_share` is the share of value relayed from a dominant source to a
*different* dominant sink through intermediaries — the textbook definition of
layering, encoded rather than fitted.

**A first attempt used the geometric mean of convergence and divergence and
failed**: a whip-round scored 0.492 against a mule network's 0.500, because
the collector is simultaneously the largest sender and the largest receiver.
Requiring `source != sink` and an actual two-hop path is what separates them.
Both the failure and the fix are pinned by tests.

---

## Finding 2 — the legacy model ranks perfectly and detects nothing

**Severity: critical. Status: open, requires recalibration and retraining.**

The nine legacy features were computed from an independent event stream for
the first time — external validation that was structurally impossible while
the generator wrote the features itself.

```
ROC-AUC = 1.0000        looks excellent
Pyramids found: 0 / 30  at the system's own high-risk threshold of 0.70
```

All thirty pyramids scored near **0.50**. The high-risk threshold is **0.70**.
On independent data the model would not have flagged a single pyramid while
ranking them flawlessly.

**AUC 1.0 is compatible with zero detections.** Ranking and decision are
different things; reporting the first while operating on the second is
self-deception. This is the concrete cost of the uncalibrated `predict_proba`
of a random forest.

### Why: the features do not mean what their names say

| Feature | Pyramid | Honest | Reading |
|---|---|---|---|
| `gini_coefficient` | 0.336 | 0.345 | **no signal** — and this carried 34.7 % of the model's importance |
| `centralization_index` | 0.011 | 0.019 | a pyramid with 400 investors has a top-depositor share near 1/400; the model trained on 0.2–0.9 |
| `transaction_entropy` | 5.000 | 4.704 | saturated at the upper bound |
| `structural_depth` | 16.0 | 2.0 | saturated at the upper bound |
| `payout_dependency` | 0.921 | 0.224 | **works** — and carried only 0.04 of importance |
| `avg_holding_time` | 6.0 d | 47.0 d | **works** |

The feature carrying a third of the model's decisions has no signal once
computed from actual money flows: Gini was an artefact of the old generator,
where it was deliberately strengthened by a correlation adjustment. Meanwhile
`payout_dependency` — the literal definition of a Ponzi scheme — was almost
switched off and turns out to be one of the strongest.

**Caveat, stated before anyone asks.** `referral_ratio` (0.991 vs 0.000) and
`structural_depth` (16 vs 2) separate perfectly because no honest population
pays referral bonuses. Until an honest referral-based business exists in the
simulation, the ROC-AUC figure for pyramids must not be quoted as a result.

---

## Finding 3 — the first simulator leaked, and how

**Severity: critical at the time. Status: fixed, documented so it is not
repeated.**

The first version of the event generator produced **ROC-AUC 1.0000** on
account-level features: the previous project's mistake, one storey up.

| Feature | AUC then | Cause |
|---|---|---|
| `account_age_days` | 0.978 | mule accounts opened 3–90 days ago against 60–1500 for everyone else |
| `total_in` | 0.964 | mule amounts 150k–900k against 30k–260k, almost disjoint |
| `transit_median_min` | 0.908 | mule delay 0.5–6 min against 1–40 min |

All three came from defining populations with **uniform intervals**. A uniform
interval gives a hard edge that a model splits on without any meaning behind
it. Replaced by log-normal distributions with heavy overlap, which is also
more realistic: real schemes push whatever the fraud yielded, mules sometimes
withdraw half an hour later, and accounts are often old ones whose access was
bought.

After the fix, against hard negatives: **ROC-AUC 0.9019**, best single feature
0.778. Recorded honestly: the acceptance ceiling is 0.90, so this sits two
thousandths above it and remains a judgement call rather than a passed gate.

Also fixed in the same pass:

- **Pyramid investors were counted as fraudulent accounts.** They are victims.
  Counting them inflated the fraud share threefold.
- **Employers paid on random days per employee.** Real payroll pays everyone
  at once, which is what creates honest fan-out.
- **The analysis window was taken as min/max over all events**, and ordinary
  spending drags the tail far past the horizon, so the midpoint drifted and
  `growth_rate` came out as exactly 0.000 for every pyramid.

---

## What is still open

1. An honest referral-based business, so `referral_ratio` stops separating for
   free (blocks any pyramid figure being quoted).
2. Retraining and calibrating the legacy model on flow-derived features.
3. Wiring the real event matrices into `engine_v2` in place of the
   `*_from_tabular` builders.
4. The evasion sweep and the detectability curve.
