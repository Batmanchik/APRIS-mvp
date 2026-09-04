# Research Plan — RKNP 2026

Date: 2026-09-04
Target: third round, end of October 2026.
Author works solo.

## 1. What the work claims

Not "we detect money mules" — they are already detected, and the contents of
the systems doing it are closed, so no honest comparison is possible.

> **We build an environment in which flow-based detection can be measured,
> and we measure where it breaks and what evasion costs.**

That claim rests on a published baseline (the four AFRD criteria), matches a
roadmap item the National Bank announced itself, and cannot be refuted by
pointing at proprietary industrial systems.

## 2. Context and the gap

| Fact | Value |
|---|---|
| Anti-Fraud Centre operating since | July 2024 |
| Connected participants | 200+ |
| Incidents registered by 2026-01-01 | 80 871 |
| Funds blocked | 2.8 bn KZT |
| Mule share of incidents | 6.87 % |
| Criminal cases since September 2025 | 700+ |
| Dropper identification rules approved | April 2026 |

The April rules define a mule by four criteria: transfers to listed persons,
a shared phone, a shared IP or device, and deviation from the customer's usual
profile. All four are about **identity and device**; none is about the shape
of the money flow. A fresh network of clean accounts, each with its own phone,
none on any list, passes the first three, and the fourth requires a "usual
profile" a newly opened account does not have.

The National Bank's own development plan names the next stage: transition
"from reaction to prevention", **transactional antifraud at the level of
national payment systems**, AI for scheme forecasting. That stage does not
exist yet, so no data from it exists for anyone.

## 3. Two hypotheses

**H1 — unit of analysis.** Raising the unit from the account to the network
yields a larger gain than increasing model complexity at the same unit.

A mule network is not a property of an account: at that level there is an
ordinary student withdrawing money. Measured evidence already supports the
premise — account-level features give ROC-AUC 0.902 against hard negatives,
while `graph_relay_share` alone separates network cases at 0.494 vs 0.000.

**H2 — speed.** A fast detector with moderate recall saves more money than a
precise slow one.

Money leaves in minutes. The metric is therefore not "how many networks were
found" but **what share of value had already left at the moment of the alert**
— the same quantity the Anti-Fraud Centre reports.

Both are falsifiable by the same grid of measurements.

## 4. Architecture

```
Layer 0  SIMULATION          events only: who paid whom, how much, when
         8 honest populations (4 confusable) + 2 fraudulent structures
                  |
Layer 1  FEATURES            derived by the detector, never by the generator
         account · neighbourhood · network, over a sliding window
                  |
Layer 2  DETECTOR LADDER     rules -> logistic -> forest -> boosting -> graph
         plus an unsupervised branch
                  |
Layer 3  ADVERSARIAL EVAL    evasion curves · time to alert · value lost
```

Window width `W` is the scale parameter: minutes for mule networks, months
for pyramids. The same code covers both, which is what makes the old nine
features a special case of the new layer rather than discarded work.

## 5. Status

| Component | State |
|---|---|
| Layer 0 simulator | **done**, accepted, 21 tests |
| Acceptance criterion | **done**, with a second gate after the first proved too weak |
| Account-level features | done (in the previous working tree, to be ported) |
| Real graph and sequence features | **done** — `ml/event_features_v2.py` |
| Case builder | **done** — four paired case kinds |
| Legacy model external validation | **done** — see AUDIT_FINDINGS |
| Detector ladder on real matrices | open |
| Evasion sweep | open |
| Online evaluation | open |

## 6. Experiments

Core — the work does not exist without these.

| # | Experiment | Output |
|---|---|---|
| E1 | Baseline (AFRD-4) vs the ladder | first measured answer to "why ML" |
| E2 | Effect of the unit of analysis, full grid | key table, tests H1 |
| E3 | Detectability curve over `funders` | main figure; look for the knee |

Extension — added on top of a standing bench.

| # | Experiment | Output |
|---|---|---|
| E4 | Curves over the other three knobs | which dial is cheapest for the fraudster |
| E5 | Two-dimensional evasion surface | where to place friction |
| E6 | Time to alert, per ladder cell | distribution; the tail matters |
| E7 | Share of value already gone | tests H2; comparable with AFC reporting |
| E8 | Single-bank vs three-bank view | quantifies the value of interbank exchange |
| E9 | One detector across two scales via `W` | tests the unifying idea |
| E10 | Labelled vs unlabelled | applicability where labels do not exist |
| E11 | Base rate at 6.87 %, 1 %, 0.5 %, 0.1 % | analyst workload |
| E12 | Distribution shift | robustness |
| E13 | Proxy-discrimination check on age | ethics section becomes a measurement |
| E14 | Curve translated into a countermeasure | the practical conclusion |

## 7. Ethics

Mules are usually recruited, often minors, frequently victims rather than
organisers. Three rules, all implemented rather than merely stated:

- **Age is not a model feature.** It exists in the data for exactly one
  purpose: E13, which checks whether the model learned it indirectly through
  behaviour. Using it directly would be proxy discrimination.
- **The target is the organiser.** The correct output is "a network of forty
  accounts with a common source", not "account 17 is suspicious". This is both
  more accurate and defensible.
- **Prioritisation, not verdict.** The system ranks cases for human review.

## 8. Limits, stated before they are asked about

- The data is synthetic. Absolute figures such as "we find 96 %" are not
  claimed and must not appear on a slide. Only **comparative** statements are
  made, and those are fully provable inside a controlled environment.
- **The generator is still written by us.** Separating events from features
  removes the crudest circularity, but what counts as fraudulent behaviour is
  still our model of it. The circle widens; it does not open. Only real data
  closes it.
- FIFO matching of inflows to outflows is a design decision, not a fact:
  money is fungible and the assignment is not identifiable.
- Prior art exists — PaySim, AMLSim. The difference here is Kazakh mechanics
  (ATM cash-out as the exit point) and measurement against the four published
  AFRD criteria.

## 9. Schedule

| Weeks | Work |
|---|---|
| 1–2 | port account-level features; honest referral business; wire real matrices into `engine_v2` |
| 3 | E1 and E2. **Checkpoint: does the network level beat the account level?** |
| 4–5 | E3, then E6 and E7 |
| 6–7 | write-up, ~30 pages |
| 8 | polish, hostile review, rehearsal |

The science must be finished by the end of week 5. That is the one deadline
that cannot move: a paper cannot be written without results, and results
cannot be defended without a paper.

**Checkpoint rule.** If at the end of week 3 the network level does not beat
the account level, the hypothesis is wrong and that is reported as the
finding — not hidden, not re-fitted until it passes.
