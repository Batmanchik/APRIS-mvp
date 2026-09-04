"""Parameters of the simulated world.

Every constant here describes BEHAVIOUR (who pays whom, when, how much),
never a derived metric. Writing something like ``transit = 1.0`` into this
module would hand the answer to the detector and invalidate every result
computed downstream.

Source tags used below:
    OFFICIAL   published regulator data
    REPORTED   descriptions of real schemes in press and AFRD rules
    ASSUMED    our own assumption; documented as such in the paper
    REVIEW     needs an explicit decision from the project owner
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Channels and types compatible with apris.cheops.domain.models
# --------------------------------------------------------------------------

CHANNEL_LEGAL = "legal"

ASSET_FIAT = "fiat"
ASSET_CASH = "cash"

TYPE_PERSON = "person"
TYPE_COMPANY = "company"
TYPE_MERCHANT = "merchant"
TYPE_ATM = "atm"

JURISDICTION = "KZ"
CURRENCY = "KZT"

REGIONS: tuple[str, ...] = (
    "Shymkent", "Almaty", "Astana", "Karaganda",
    "Aktobe", "Taraz", "Pavlodar", "Kyzylorda",
)

AGE_BANDS: tuple[str, ...] = ("16-19", "20-29", "30-44", "45-59", "60+")


# --------------------------------------------------------------------------
# Evasion knobs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EvasionKnobs:
    """Dials that make a mule network less visible.

    Each has a real cost to the organiser, and that cost — not the dial
    itself — is what the study reports. The naive network sets every dial
    to its minimum; the detectability curve sweeps one dial at a time.
    """

    # REVIEW. Independent accounts funding the network.
    # 1 = naive (single source), 40 = upper bound of the sweep.
    # Cost: each funder is a real account holding real money.
    funders: int = 1

    # Distinct ATMs used by one network.
    # Cost: driving people across the city.
    terminals: int = 1

    # REPORTED. Window the whole operation fits into. The observed
    # mechanic ("they take the kid to the ATM") gives minutes, not hours.
    # Cost: the operation stops being fast, exposure grows.
    time_spread_minutes: float = 12.0

    # Pieces each transfer is split into.
    # Cost: more operations, more traces.
    split_factor: int = 1


# --------------------------------------------------------------------------
# World configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42

    # ASSUMED. 120 days let both scales live in one dataset: mule networks
    # (minutes) and pyramids (months).
    days: int = 120

    # Honest populations
    salary_earners: int = 900
    freelancers: int = 260
    traders: int = 180
    fast_spenders: int = 650        # hard negative 1: withdraws everything at once
    family_circles: int = 70        # groups, not accounts
    crowd_collections: int = 40     # hard negative 2: honest fan-in
    marketplace_sellers: int = 300  # hard negative 3: unknown counterparty each time
    employers: int = 25             # hard negative 4: honest fan-out on payday

    # Fraudulent structures
    mule_networks: int = 24
    pyramids: int = 30

    # Infrastructure
    terminals: int = 60
    banks: int = 3                  # for the cross-bank visibility experiment
    merchants: int = 400

    evasion: EvasionKnobs = field(default_factory=EvasionKnobs)

    # OFFICIAL. Mule incidents were 6.87 % of all incidents registered by
    # the National Bank Anti-Fraud Centre as of 2026-01-01 (80 871 total).
    # Used as a sanity reference for the resulting share, not for generation.
    reference_fraud_share: float = 0.0687


# --------------------------------------------------------------------------
# Monetary magnitudes, KZT
# --------------------------------------------------------------------------
# ASSUMED throughout. Orders of magnitude are everyday ones: salary, gig
# work, small trade revenue. Recorded as assumptions in the paper.

SALARY_RANGE = (180_000.0, 650_000.0)
FREELANCE_JOB_RANGE = (25_000.0, 320_000.0)
TRADE_SALE_RANGE = (1_500.0, 45_000.0)
EVERYDAY_SPEND_RANGE = (800.0, 35_000.0)
FAMILY_TRANSFER_RANGE = (5_000.0, 120_000.0)
CROWD_CONTRIBUTION_RANGE = (2_000.0, 50_000.0)
PYRAMID_DEPOSIT_RANGE = (50_000.0, 1_200_000.0)

# Log-normal, and the overlap is deliberate. An earlier version drew mule
# amounts from (150k, 900k) and fast-spender amounts from (30k, 260k) —
# almost disjoint intervals — and a model separated the two by amount alone
# at AUC 0.964. That was an artefact of the generator, not a signal: real
# schemes push whatever the fraud yielded, including 50 thousand.
MULE_AMOUNT_MEDIAN = 250_000.0
MULE_AMOUNT_SIGMA = 0.85
FAST_SPENDER_AMOUNT_MEDIAN = 160_000.0
FAST_SPENDER_AMOUNT_SIGMA = 0.95

ATM_DAILY_LIMIT = 2_000_000.0

# REPORTED. Reporting threshold that structured amounts cluster below.
REPORTING_THRESHOLD = 500_000.0


# --------------------------------------------------------------------------
# Time magnitudes, minutes
# --------------------------------------------------------------------------
# Log-normal for the same reason as amounts. An earlier version used
# (0.5, 6) for mules against (1, 40) for students — separable intervals,
# AUC 0.908 on timing alone. A student who just received money also often
# withdraws it within a minute.

MULE_DELAY_MEDIAN_MIN = 3.0
MULE_DELAY_SIGMA = 0.95
FAST_SPENDER_DELAY_MEDIAN_MIN = 7.0
FAST_SPENDER_DELAY_SIGMA = 1.10

# ASSUMED. Ordinary people spend money over days.
NORMAL_SPEND_DELAY_HOURS = (2.0, 240.0)


# --------------------------------------------------------------------------
# Naive rules — used ONLY by the layer-0 acceptance check.
# Detection models never see these thresholds and must derive their own.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NaiveRuleThresholds:
    # REVIEW. "Cashed out shortly after money arrived."
    fast_cashout_minutes: float = 5.0

    # "Received from many distinct senders within the window."
    fan_in_count: int = 20
    fan_in_window_hours: float = 24.0

    # "Sent to many distinct receivers within the window."
    fan_out_count: int = 20
    fan_out_window_hours: float = 1.0


DEFAULT_NAIVE_RULES = NaiveRuleThresholds()

# Ceiling for the acceptance criterion: a model built on SINGLE-ACCOUNT
# features must not separate a mule from a hard negative better than this.
# Above it, the generator leaks and the network-level hypothesis would hold
# trivially rather than because networks carry information accounts do not.
ACCOUNT_LEVEL_AUC_CEILING = 0.90
