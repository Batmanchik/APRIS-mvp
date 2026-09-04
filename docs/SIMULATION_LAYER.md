# Simulation Layer (Layer 0)

Date: 2026-09-04
Status: implemented, accepted, covered by 21 tests.

## 1. What it is

`src/apris/cheops/infrastructure/simulation/` generates a synthetic world of
accounts, ATMs and transactions with known ground truth, so that detectors
can be built and compared on data whose answers are known exactly.

It exists because the next stage of anti-fraud in Kazakhstan — transactional
antifraud at the level of national payment systems, announced by the National
Bank — does not exist yet. No data from that stage exists for anyone,
including the regulator. Simulation is therefore not a substitute for real
data here; it is the only instrument available for the problem.

## 2. The rule the whole layer obeys

**The generator writes only events. Every feature is derived by the detector.**

Not one metric is written into the stream. No `transit`, no `gini`, no
`time_to_cashout`. A generator that writes `gini = 0.67` into a pyramid's row
hands the answer to the model, which then reads it instead of finding it —
and the resulting accuracy measures nothing at all. That is precisely how the
previous generation of this project produced 0.96 that could not be defended.

Ground truth (network membership, population labels, the referral tree) is
returned in a separate structure and never reaches the event stream. A test
pins this.

## 3. What the world contains

Eight honest populations, four of which are deliberately confusable with
fraud, plus two fraudulent structures.

| Population | Behaviour | Why it exists |
|---|---|---|
| Salary earner | monthly income, gradual spending | ordinary negative |
| Freelancer | income from many distinct payers | breaks "many new counterparties" |
| Trader | many small sales, periodic large cash-out | breaks "large cash withdrawal" |
| **Fast spender** | money arrives, fully withdrawn in minutes | **hard negative 1** — indistinguishable from a mule at account level |
| **Marketplace seller** | a new unknown buyer every time, then cash | **hard negative 2** — breaks "unknown counterparty" |
| Family circle | transfers inside a small closed group | breaks dense-community signals |
| **Crowd collection** | ~40 people send to one, spent over weeks | **hard negative 3** — honest fan-in |
| **Employer** | pays all staff on one payday | **hard negative 4** — honest fan-out |
| Mule network | source → mules → ATM inside minutes | positive, fast scale |
| Pyramid | payouts funded by inflow, referral tree | positive, slow scale |

The two scales are one invariant at different observation windows: transit
without own income, compressed into minutes or stretched over months.

## 4. Evasion knobs

`EvasionKnobs` are the dials that make a network less visible. Each carries a
real cost to the organiser, and the cost — not the dial — is what the study
reports.

| Knob | Range | Breaks | Cost to the organiser |
|---|---|---|---|
| `funders` | 1 → 40 | shared source | each funder is a real account with real money |
| `terminals` | 1 → 20 | convergence on one exit | driving people across the city |
| `time_spread_minutes` | 2 → 480 | temporal tightness | the operation stops being fast |
| `split_factor` | 1 → 10 | amount thresholds | more operations, more traces |

## 5. Acceptance criterion — the check that must not be skipped

`simulation/acceptance.py`. Three naive rules must each **misfire on honest
people**, and two of them must still catch the fraud:

```
[OK] A misfires on the fast-spending student   601/650
[OK] B misfires on the payday employer          25/25
[OK] C misfires on the whip-round               25/40
[OK] A still catches mules                     372/531
[OK] B still catches the network source          11/24
```

If the rules stop misfiring, the generator has become too kind and every
result built above it describes a fiction.

**The first version of this criterion was too weak and is documented here so
the mistake is not repeated.** It checked that naive *rules* misfire — and
they did — while a gradient boosting model on the same account-level features
still separated the classes at ROC-AUC 1.0000. Rules failing is not evidence
that a task is hard. A second gate exists now:
`ACCOUNT_LEVEL_AUC_CEILING = 0.90` in `config.py`.

## 6. Side finding, pinned by a test

The most intuitive graph rule — "many senders into one account" — catches
**0 of 531 mules** while flagging 25 of 40 honest whip-rounds.

In a mule network the fan spreads *out* from the source and converges on the
*ATM*; the mule account itself sees no convergence at all. The first idea that
comes to mind when looking at a transaction graph gives zero recall and
maximum false positives simultaneously.

## 7. How to use it

```python
from apris.cheops.infrastructure.simulation import generate_world, SimulationConfig
from apris.cheops.infrastructure.simulation.cases import build_cases
from apris.cheops.infrastructure.simulation.acceptance import evaluate

world = generate_world(SimulationConfig())     # ~2 minutes at full scale
report = evaluate(world)                        # acceptance criterion
cases = build_cases(world)                      # labelled cases for detectors
```

Full-scale generation produces roughly 320 000 events over 23 000 accounts,
54 networks, with a mule share of personal accounts near the 6.87 % reported
by the Anti-Fraud Centre.

For tests use a reduced config — see `tests/unit/test_simulation_layer.py`.
Note that the fan-out rule needs at least ~20 employees per employer, so a
shrunken fixture must keep that ratio or acceptance check B cannot fire.
