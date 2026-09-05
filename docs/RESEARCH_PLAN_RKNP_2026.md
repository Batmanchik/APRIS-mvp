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
| Criminal cases since September 2025 | **needs verification** — see note |
| Dropper identification rules approved | April 2026 |

**Number of criminal cases: do not quote either figure yet.** Secondary
sources give both "700+ since September 2025" and "49 cases in four months,
13 sent to court", an order of magnitude apart — probably different offences
or different periods. Take the figure from the Prosecutor General's Office
or the Committee on Legal Statistics before it goes near a slide. Every
other number in the table above traces to a National Bank publication.

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

## 5. Status — 2026-09-06

| Component | State |
|---|---|
| Layer 0 simulator | **done**, accepted, tests |
| Acceptance criterion | **done**, second gate added after the first proved weak |
| Account-level features | **done** — `experiments/ladder.py` |
| Real graph and sequence features | **done** — `ml/event_features_v2.py` |
| Legacy nine from raw events | **done** — `ml/legacy_features_v2.py` |
| Case builder | replaced by blind discovery with a reported ceiling |
| Detector ladder on real matrices | **done** — E1/E2, `experiments/ladder.py` |
| Published-rule baseline | **done** — `ml/baseline_afrd.py`, three of four criteria |
| Ladder of worlds W1–W5 | **done** — `experiments/ladder_of_worlds.py` |
| Five standard diagnostics | **done** — `reporting/diagnostics.py` |
| Evasion sweep | two points measured, curve open |
| Realistic base rate (E11) | **open, and it is the next thing** |
| Crypto typologies in the generator | open — delegated |
| Online evaluation (E6, E7) | open |

Measured results and what each one does not prove: **`docs/RESULTS.md`**.

### The week-3 checkpoint fired, and here is the answer

§9 says that if the network level does not beat the account level, that is
reported as the finding rather than hidden. It fired, and the answer turned
out to be more useful than either half of H1.

The network unit **does** beat the account unit — 0.965 to 0.999 against
0.965 to 0.982 — on every rung it can see at all. It does not then degrade
under evasion. It **disappears**: at six independent funders and four
terminals, discovery no longer places even two members of most rings into one
candidate, median overlap 1.000 → 0.000. The account unit does not notice,
0.9648 → 0.9646.

So neither unit dominates. One is stronger; the other is the one that
survives an adversary who pays. H1 as worded — that the unit matters more
than the model — is not what was found, and the finding that replaced it
prices both sides of the exchange, which is a better sentence to defend.

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
