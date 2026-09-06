"""Event-level generator of a synthetic financial world.

The generator emits ``TransactionEvent`` objects and nothing else: who paid
whom, how much, when, over which channel. It never writes a derived metric.
Detection layers must compute transit, retention, holding time and graph
structure from the events themselves.

That separation is what makes evaluation honest. The previous generation of
this project wrote ``gini = 0.67`` straight into a pyramid's feature row, so
the model read the answer instead of finding it, and the resulting accuracy
measured nothing.

Ground truth (which accounts belong to which network, which population an
account was drawn from) is returned alongside the events in a separate
structure and is never mixed into the stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from apris.cheops.domain.models import TransactionEvent
from apris.cheops.infrastructure.simulation.config import (
    ASSET_CASH,
    ASSET_FIAT,
    ATM_DAILY_LIMIT,
    CHANNEL_LEGAL,
    CROWD_CONTRIBUTION_RANGE,
    CURRENCY,
    EVERYDAY_SPEND_RANGE,
    FAMILY_TRANSFER_RANGE,
    FAST_SPENDER_AMOUNT_MEDIAN,
    FAST_SPENDER_AMOUNT_SIGMA,
    FAST_SPENDER_DELAY_MEDIAN_MIN,
    FAST_SPENDER_DELAY_SIGMA,
    FREELANCE_JOB_RANGE,
    JURISDICTION,
    MULE_AMOUNT_MEDIAN,
    MULE_COUNT_MAX,
    MULE_COUNT_MIN,
    MULE_AMOUNT_SIGMA,
    MULE_DELAY_MEDIAN_MIN,
    MULE_DELAY_SIGMA,
    NORMAL_SPEND_DELAY_HOURS,
    PYRAMID_DEPOSIT_RANGE,
    REGIONS,
    REPORTING_THRESHOLD,
    SALARY_RANGE,
    TRADE_SALE_RANGE,
    TYPE_ATM,
    TYPE_COMPANY,
    TYPE_MERCHANT,
    TYPE_PERSON,
    SimulationConfig,
)

SIMULATION_START = datetime(2026, 1, 1, 0, 0, 0)


# ==========================================================================
# Ground truth records — kept apart from the event stream
# ==========================================================================


@dataclass(frozen=True)
class SimulatedAccount:
    account_id: str
    opened_at: datetime
    account_type: str
    region: str
    age_band: str
    bank_id: int


@dataclass(frozen=True)
class SimulatedNetwork:
    network_id: str
    kind: str            # "mule_fast" | "pyramid_slow"
    scale: str           # "fast" | "slow"
    account_ids: tuple[str, ...]
    organizer_ids: tuple[str, ...]
    referrals: dict[str, str] = field(default_factory=dict)


@dataclass
class SimulatedWorld:
    config: SimulationConfig
    events: list[TransactionEvent] = field(default_factory=list)
    accounts: dict[str, SimulatedAccount] = field(default_factory=dict)
    terminals: list[str] = field(default_factory=list)
    networks: list[SimulatedNetwork] = field(default_factory=list)
    populations: dict[str, str] = field(default_factory=dict)

    def fraud_account_ids(self) -> set[str]:
        return {a for n in self.networks for a in n.account_ids}

    def summary(self) -> dict[str, float]:
        personal = [a for a in self.accounts.values() if a.account_type == TYPE_PERSON]
        fraud = self.fraud_account_ids()
        return {
            "accounts": float(len(self.accounts)),
            "personal_accounts": float(len(personal)),
            "terminals": float(len(self.terminals)),
            "events": float(len(self.events)),
            "networks": float(len(self.networks)),
            "fraud_accounts": float(len(fraud)),
            "fraud_share_of_personal": (len(fraud) / len(personal)) if personal else 0.0,
        }


# ==========================================================================
# World builder
# ==========================================================================


class _Builder:
    """Mutable helper used while the world is being constructed."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.world = SimulatedWorld(config=config)
        self._event_no = 0
        self._counters: dict[str, int] = {}

    # ---------------- entities ----------------

    def _uid(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        return f"{prefix}{n:05d}"

    def new_account(
        self,
        prefix: str,
        *,
        account_type: str = TYPE_PERSON,
        age_band: str | None = None,
        opened_days_ago: tuple[int, int] = (60, 1500),
    ) -> str:
        if age_band is None:
            age_band = str(
                self.rng.choice(
                    ["16-19", "20-29", "30-44", "45-59", "60+"],
                    p=[0.10, 0.30, 0.32, 0.20, 0.08],
                )
            )
        lo, hi = opened_days_ago
        opened = SIMULATION_START - timedelta(days=int(self.rng.integers(lo, hi + 1)))
        account = SimulatedAccount(
            account_id=self._uid(prefix),
            opened_at=opened,
            account_type=account_type,
            region=str(self.rng.choice(REGIONS)),
            age_band=age_band,
            bank_id=int(self.rng.integers(1, self.config.banks + 1)),
        )
        self.world.accounts[account.account_id] = account
        return account.account_id

    def new_terminal(self) -> str:
        terminal_id = self._uid("ATM")
        self.world.terminals.append(terminal_id)
        self.world.accounts[terminal_id] = SimulatedAccount(
            account_id=terminal_id,
            opened_at=SIMULATION_START - timedelta(days=900),
            account_type=TYPE_ATM,
            region=str(self.rng.choice(REGIONS)),
            age_band="",
            bank_id=int(self.rng.integers(1, self.config.banks + 1)),
        )
        return terminal_id

    # ---------------- events ----------------

    def emit(
        self,
        sender: str,
        receiver: str,
        amount: float,
        when: datetime,
        *,
        asset_type: str = ASSET_FIAT,
    ) -> None:
        if amount <= 0 or sender == receiver:
            return
        when = max(when, SIMULATION_START)
        self._event_no += 1
        sender_acc = self.world.accounts.get(sender)
        receiver_acc = self.world.accounts.get(receiver)
        self.world.events.append(
            TransactionEvent(
                event_id=f"EV{self._event_no:08d}",
                ts=when,
                amount=round(float(amount), 2),
                currency=CURRENCY,
                sender_id=sender,
                receiver_id=receiver,
                sender_type=sender_acc.account_type if sender_acc else TYPE_PERSON,
                receiver_type=receiver_acc.account_type if receiver_acc else TYPE_PERSON,
                channel=CHANNEL_LEGAL,
                jurisdiction=JURISDICTION,
                asset_type=asset_type,
            )
        )

    def cash_out(self, sender: str, terminal: str, amount: float, when: datetime) -> None:
        self.emit(sender, terminal, min(amount, ATM_DAILY_LIMIT), when, asset_type=ASSET_CASH)

    # ---------------- helpers ----------------

    def uniform(self, bounds: tuple[float, float]) -> float:
        return float(self.rng.uniform(bounds[0], bounds[1]))

    def lognormal(
        self, median: float, sigma: float, lo: float | None = None, hi: float | None = None
    ) -> float:
        """Log-normal draw with a given median.

        Used wherever two populations must OVERLAP. Uniform intervals give
        hard edges that a model splits on without any meaning behind it —
        that is how the first version of this generator leaked.
        """
        value = float(median) * float(np.exp(self.rng.normal(0.0, sigma)))
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        return value

    def moment(self, day: int, hours: tuple[int, int] = (7, 23)) -> datetime:
        return SIMULATION_START + timedelta(
            days=day,
            hours=int(self.rng.integers(hours[0], hours[1])),
            minutes=int(self.rng.integers(0, 60)),
            seconds=int(self.rng.integers(0, 60)),
        )

    def mark(self, account_ids: list[str], population: str) -> None:
        for account_id in account_ids:
            self.world.populations.setdefault(account_id, population)


# ==========================================================================
# Honest populations
# ==========================================================================


def _spend_down(
    b: _Builder,
    account: str,
    amount: float,
    after: datetime,
    merchants: list[str],
    terminals: list[str],
    delay_hours: tuple[float, float] = NORMAL_SPEND_DELAY_HOURS,
) -> None:
    """Ordinary spending: several payments and occasional cash.

    Leaves part of the money on the account. That residue is retention, and
    it is what separates a person from a pass-through account — but we do
    not write retention anywhere, it emerges from the events.
    """
    remaining = amount * float(b.rng.uniform(0.55, 0.92))
    moment = after
    while remaining > 1000:
        step = min(remaining, b.uniform(EVERYDAY_SPEND_RANGE))
        moment = moment + timedelta(hours=float(b.rng.uniform(*delay_hours)) / 6.0)
        if b.rng.random() < 0.22 and len(terminals) > 0:
            b.cash_out(account, str(b.rng.choice(terminals)), step, moment)
        else:
            b.emit(account, str(b.rng.choice(merchants)), step, moment)
        remaining -= step


def _gen_salary_earners(
    b: _Builder, merchants: list[str], terminals: list[str], employers: list[str]
) -> list[str]:
    """Monthly income, gradual spending.

    Every employer pays ALL of its people on one day, as payroll actually
    works. That creates honest fan-out: one company sends money to hundreds
    of different people within an hour — the very shape a naive graph rule
    reads as handing money to mules.
    """
    ids: list[str] = []
    pay_days = {e: int(b.rng.integers(1, 28)) for e in employers}
    for _ in range(b.config.salary_earners):
        account = b.new_account("ACC")
        ids.append(account)
        salary = b.uniform(SALARY_RANGE)
        employer = str(b.rng.choice(employers))
        for month_start in range(0, b.config.days, 30):
            day = month_start + pay_days[employer]
            if day >= b.config.days:
                break
            when = SIMULATION_START + timedelta(
                days=day, hours=10, minutes=int(b.rng.integers(0, 90))
            )
            b.emit(employer, account, salary, when)
            _spend_down(b, account, salary, when, merchants, terminals)
    return ids


def _gen_freelancers(b: _Builder, merchants: list[str], terminals: list[str]) -> list[str]:
    """Irregular income from MANY distinct payers.

    Breaks the "many new counterparties" signal: an honest freelancer has
    as many as a mule does.
    """
    ids: list[str] = []
    for _ in range(b.config.freelancers):
        account = b.new_account("ACC")
        ids.append(account)
        for _ in range(int(b.rng.integers(6, 26))):
            payer = b.new_account("CLI", account_type=TYPE_COMPANY)
            when = b.moment(int(b.rng.integers(0, b.config.days)))
            amount = b.uniform(FREELANCE_JOB_RANGE)
            b.emit(payer, account, amount, when)
            _spend_down(b, account, amount, when, merchants, terminals)
    return ids


def _gen_traders(b: _Builder, merchants: list[str], terminals: list[str]) -> list[str]:
    """Many small sales, periodic LARGE cash withdrawal.

    Breaks the "large cash-out" signal: honest revenue is withdrawn too.
    """
    ids: list[str] = []
    for _ in range(b.config.traders):
        account = b.new_account("ACC", account_type=TYPE_COMPANY)
        ids.append(account)
        atm = str(b.rng.choice(terminals))
        for day in range(b.config.days):
            taken = 0.0
            for _ in range(int(b.rng.poisson(7))):
                amount = b.uniform(TRADE_SALE_RANGE)
                b.emit(str(b.rng.choice(merchants)), account, amount, b.moment(day, (9, 21)))
                taken += amount
            if day % int(b.rng.integers(3, 8)) == 0 and taken > 0:
                b.cash_out(
                    account, atm, taken * float(b.rng.uniform(0.4, 0.8)), b.moment(day, (18, 22))
                )
    return ids


def _gen_fast_spenders(b: _Builder, terminals: list[str], employers: list[str]) -> list[str]:
    """HARD NEGATIVE 1 — a student who withdraws everything at once.

    Money arrives and is fully withdrawn within minutes. At the level of a
    SINGLE account this is indistinguishable from a mule: transit near one,
    retention near zero, cash-out within minutes.

    This population is what makes account-level detection impossible and
    forces the analysis up to the network. Without it the task would be
    falsely easy.
    """
    ids: list[str] = []
    for _ in range(b.config.fast_spenders):
        account = b.new_account(
            "ACC", age_band=str(b.rng.choice(["16-19", "20-29"], p=[0.55, 0.45]))
        )
        ids.append(account)
        atm = str(b.rng.choice(terminals))
        sender = str(b.rng.choice(employers))
        for _ in range(int(b.rng.integers(3, 13))):
            when = b.moment(int(b.rng.integers(0, b.config.days)), (9, 22))
            amount = b.lognormal(
                FAST_SPENDER_AMOUNT_MEDIAN, FAST_SPENDER_AMOUNT_SIGMA, 15_000.0, 2_000_000.0
            )
            b.emit(sender, account, amount, when)
            delay = b.lognormal(
                FAST_SPENDER_DELAY_MEDIAN_MIN, FAST_SPENDER_DELAY_SIGMA, 0.4, 240.0
            )
            b.cash_out(
                account, atm, amount * float(b.rng.uniform(0.9, 1.0)),
                when + timedelta(minutes=delay),
            )
    return ids


def _gen_marketplace_sellers(b: _Builder, terminals: list[str]) -> list[str]:
    """HARD NEGATIVE 2 — sold something to a stranger and took the cash.

    Sold a phone, received a transfer from someone never seen before,
    withdrew the money. A new sender every time, cash-out every time.

    Exists to kill the "unknown counterparty" signal: a mule's sender is
    unknown too, and that turns out to prove nothing.
    """
    ids: list[str] = []
    for _ in range(b.config.marketplace_sellers):
        account = b.new_account("ACC")
        ids.append(account)
        atm = str(b.rng.choice(terminals))
        for _ in range(int(b.rng.integers(2, 9))):
            buyer = b.new_account("BUY")
            when = b.moment(int(b.rng.integers(0, b.config.days)), (10, 21))
            amount = b.lognormal(
                FAST_SPENDER_AMOUNT_MEDIAN * 0.8, FAST_SPENDER_AMOUNT_SIGMA,
                10_000.0, 1_500_000.0,
            )
            b.emit(buyer, account, amount, when)
            delay = b.lognormal(
                FAST_SPENDER_DELAY_MEDIAN_MIN * 1.5, FAST_SPENDER_DELAY_SIGMA, 0.5, 600.0
            )
            b.cash_out(
                account, atm, amount * float(b.rng.uniform(0.85, 1.0)),
                when + timedelta(minutes=delay),
            )
    return ids


def _gen_family_circles(b: _Builder, merchants: list[str], terminals: list[str]) -> list[str]:
    """Transfers inside a small closed group. Breaks dense-community signals."""
    ids: list[str] = []
    for _ in range(b.config.family_circles):
        group = np.array([b.new_account("ACC") for _ in range(int(b.rng.integers(3, 6)))])
        ids.extend(group.tolist())
        for day in range(0, b.config.days, 3):
            src, dst = b.rng.choice(group, size=2, replace=False)
            amount = b.uniform(FAMILY_TRANSFER_RANGE)
            when = b.moment(day)
            b.emit(str(src), str(dst), amount, when)
            _spend_down(b, str(dst), amount * 0.6, when, merchants, terminals)
    return ids


def _gen_crowd_collections(b: _Builder, merchants: list[str], terminals: list[str]) -> None:
    """HARD NEGATIVE 3 — a whip-round for a common cause.

    Forty people send money to one person who spends it over weeks. This is
    HONEST fan-in — exactly the structure a naive graph rule reads as a mule
    network. Contributions arrive in a burst of a day or two, the way they
    actually do after a post is shared.

    The difference from a network is only in the dynamics: the money settles
    and is spent outwards rather than leaving through one point in minutes.
    Finding that difference is the detector's job, not ours.
    """
    for _ in range(b.config.crowd_collections):
        collector = b.new_account("ACC")
        b.world.populations[collector] = "crowd_collector"
        day0 = int(b.rng.integers(0, max(1, b.config.days - 30)))
        total = 0.0
        for _ in range(int(b.rng.integers(18, 55))):
            donor = b.new_account("ACC")
            b.world.populations[donor] = "crowd_donor"
            amount = b.uniform(CROWD_CONTRIBUTION_RANGE)
            b.emit(donor, collector, amount, b.moment(day0 + int(b.rng.integers(0, 2))))
            total += amount
        _spend_down(
            b, collector, total, b.moment(day0 + 5), merchants, terminals,
            delay_hours=(12.0, 400.0),
        )


# ==========================================================================
# Fraudulent structures
# ==========================================================================


def _gen_mule_network(
    b: _Builder,
    merchants: list[str],
    terminals: list[str],
    employers: list[str],
    index: int,
) -> SimulatedNetwork:
    """A fast mule network: source(s) -> mules -> ATM inside a tight window.

    Evasion knobs from the config decide how visible it is. We describe
    BEHAVIOUR here; not one metric is computed in this function.
    """
    knobs = b.config.evasion
    # REPORTED. Network size is heavy-tailed: most rings are small, a few
    # are very large. A scheme uncovered in Aktobe region involved 150
    # droppers and 3.5 bn KZT, so the earlier cap of 40 was four times below
    # what actually occurs. A Pareto draw keeps the median near a dozen while
    # letting the tail reach the observed scale.
    mule_count = int(min(MULE_COUNT_MAX, MULE_COUNT_MIN + b.rng.pareto(1.6) * 9.0))

    # Account age is NOT a marker. An earlier version opened mule accounts
    # 3-90 days back against 60-1500 for everyone else, and a model split
    # on that single column at AUC 0.978. Real networks use both fresh
    # accounts and long-standing ones whose access was simply bought.
    mules: list[str] = []
    for _ in range(mule_count):
        fresh = bool(b.rng.random() < 0.45)
        mules.append(
            b.new_account(
                "MUL",
                age_band=str(b.rng.choice(["16-19", "20-29", "30-44"], p=[0.45, 0.38, 0.17])),
                opened_days_ago=(3, 120) if fresh else (60, 1500),
            )
        )

    # A mule is a person, not a single-purpose account. Before and after the
    # operation they draw an income and buy things like anybody else, so the
    # ring's events sit inside an ordinary history rather than being the whole
    # of it. Without this the account itself is a giveaway and the network
    # never has to be found at all.
    for mule in mules:
        if b.rng.random() < 0.75:
            employer = str(b.rng.choice(employers))
            wage = b.lognormal(SALARY_RANGE[0], 0.5, 40_000.0, 900_000.0)
            for month_start in range(0, b.config.days, 30):
                day = month_start + int(b.rng.integers(1, 28))
                if day >= b.config.days:
                    break
                when = b.moment(day, (9, 18))
                b.emit(employer, mule, wage, when)
                _spend_down(b, mule, wage, when, merchants, terminals)

    funder_count = max(1, min(knobs.funders, mule_count))
    funders = [b.new_account("FND", account_type=TYPE_COMPANY) for _ in range(funder_count)]
    for funder in funders:
        b.world.populations[funder] = "mule_funder"

    pool = list(
        b.rng.choice(terminals, size=min(knobs.terminals, len(terminals)), replace=False)
    )

    start = b.moment(int(b.rng.integers(0, b.config.days)), (10, 21))
    spread = timedelta(minutes=float(knobs.time_spread_minutes))

    for position, mule in enumerate(mules):
        funder = funders[position % len(funders)]
        atm = str(pool[position % len(pool)])
        arrival = start + spread * float(b.rng.random())

        total = b.lognormal(MULE_AMOUNT_MEDIAN, MULE_AMOUNT_SIGMA, 15_000.0, 2_000_000.0)
        parts = max(1, int(knobs.split_factor))
        if parts > 1:
            # Structured amounts stick just below the reporting threshold.
            # That is a consequence of the concealment mechanic, not a
            # separately injected feature.
            per_part = min(
                total / parts, REPORTING_THRESHOLD * float(b.rng.uniform(0.90, 0.99))
            )
        else:
            per_part = total

        received = 0.0
        for part in range(parts):
            when = arrival + timedelta(seconds=float(b.rng.uniform(5, 90)) * part)
            b.emit(funder, mule, per_part, when)
            received += per_part

        # Exit behaviour is not uniform. An earlier version had every mule do
        # exactly one thing — receive and cash out in full — which made a ring
        # structurally unambiguous and drove classification to a meaningless
        # ROC-AUC of 1.0000. Real participants differ: some take the whole
        # amount, some take part and leave a remainder, and some never touch
        # an ATM at all because their role is to pass the money on.
        exit_style = float(b.rng.random())
        delay = b.lognormal(MULE_DELAY_MEDIAN_MIN, MULE_DELAY_SIGMA, 0.4, 240.0)
        moment = arrival + timedelta(minutes=delay)

        if exit_style < 0.15:
            # Pure relay: hands the money to the next account in the chain
            # rather than to a machine. Nothing about it looks like cash-out.
            onward = mules[(position + 1) % len(mules)]
            b.emit(mule, onward, received * float(b.rng.uniform(0.90, 0.99)), moment)
        elif exit_style < 0.35:
            # Partial cash-out; the remainder stays and is spent normally.
            taken = received * float(b.rng.uniform(0.35, 0.75))
            b.cash_out(mule, atm, taken, moment)
            _spend_down(b, mule, received - taken, moment, merchants, terminals)
        else:
            b.cash_out(mule, atm, received * float(b.rng.uniform(0.95, 1.0)), moment)

    return SimulatedNetwork(
        network_id=f"NET{index:04d}",
        kind="mule_fast",
        scale="fast",
        account_ids=tuple(mules),
        organizer_ids=tuple(funders),
    )


def _gen_pyramid(b: _Builder, terminals: list[str], index: int) -> SimulatedNetwork:
    """A slow scheme: payouts funded by inflow.

    The same invariant as a mule network — transit without own income —
    stretched over months. Its referral tree is expressed through money:
    whoever recruited a depositor receives a bonus shortly afterwards, and
    that bonus is visible in the event stream. Nothing about the tree is
    written down for the detector; it has to be reconstructed.
    """
    core = b.new_account("PYR", account_type=TYPE_COMPANY, opened_days_ago=(30, 400))
    investors: list[str] = []
    referrals: dict[str, str] = {}
    balance = 0.0

    for day in range(b.config.days):
        # recruitment accelerates over time
        for _ in range(int(b.rng.poisson(1.0 + 5.0 * day / max(1, b.config.days)))):
            investor = b.new_account("INV")
            investors.append(investor)
            b.world.populations[investor] = "pyramid_investor"

            if len(investors) > 1 and b.rng.random() < 0.82:
                pool = investors[-40:-1] if len(investors) > 40 else investors[:-1]
                referrals[investor] = str(b.rng.choice(pool))
            else:
                referrals[investor] = core

            amount = b.uniform(PYRAMID_DEPOSIT_RANGE)
            when = b.moment(day, (9, 20))
            b.emit(investor, core, amount, when)
            balance += amount

            parent = referrals[investor]
            if parent != core and balance > 0:
                bonus = min(amount * float(b.rng.uniform(0.05, 0.18)), balance)
                b.emit(
                    core, parent, bonus,
                    when + timedelta(hours=float(b.rng.uniform(1, 48))),
                )
                balance -= bonus

        # repeat deposits: a participant who saw a payout brings more
        if investors:
            for _ in range(int(b.rng.poisson(1.2))):
                who = str(b.rng.choice(investors))
                amount = b.uniform(PYRAMID_DEPOSIT_RANGE) * float(b.rng.uniform(0.3, 1.0))
                b.emit(who, core, amount, b.moment(day, (9, 20)))
                balance += amount

        # payouts to early participants out of the same inflow
        if investors and balance > 0 and b.rng.random() < 0.75:
            for _ in range(min(int(b.rng.integers(1, 6)), len(investors))):
                who = str(b.rng.choice(investors[: max(1, len(investors) // 2)]))
                amount = min(balance * float(b.rng.uniform(0.02, 0.09)), balance)
                if amount <= 0:
                    continue
                b.emit(core, who, amount, b.moment(day, (10, 20)))
                balance -= amount

        # the organiser regularly takes a cut in cash
        if day % 7 == 0 and balance > 200_000:
            take = balance * float(b.rng.uniform(0.05, 0.15))
            b.cash_out(core, str(b.rng.choice(terminals)), take, b.moment(day, (11, 19)))
            balance -= take

    # Investors are VICTIMS, not participants. Only the core belongs to the
    # network: counting investors as fraud accounts inflated the fraud share
    # threefold and made the base rate unrealistic.
    return SimulatedNetwork(
        network_id=f"NET{index:04d}",
        kind="pyramid_slow",
        scale="slow",
        account_ids=(core,),
        organizer_ids=(core,),
        referrals=referrals,
    )


# ==========================================================================
# Entry point
# ==========================================================================


def generate_world(config: SimulationConfig | None = None) -> SimulatedWorld:
    """Build a full synthetic world and return it with its ground truth."""
    b = _Builder(config or SimulationConfig())

    terminals = np.array([b.new_terminal() for _ in range(b.config.terminals)])
    merchants = np.array([
        b.new_account("MER", account_type=TYPE_MERCHANT) for _ in range(b.config.merchants)
    ])

    # ASSUMED: few large employers rather than many small ones. With a
    # hundred tiny companies nobody accumulates a fan-out and the honest
    # counterpart of "handing money to many people" never appears.
    employers = np.array([
        b.new_account("EMP", account_type=TYPE_COMPANY) for _ in range(b.config.employers)
    ])
    for employer in employers:
        b.world.populations[employer] = "employer"

    b.mark(_gen_salary_earners(b, merchants, terminals, employers), "salary")
    b.mark(_gen_freelancers(b, merchants, terminals), "freelancer")
    b.mark(_gen_traders(b, merchants, terminals), "trader")
    b.mark(_gen_fast_spenders(b, terminals, employers), "fast_spender")
    b.mark(_gen_marketplace_sellers(b, terminals), "marketplace_seller")
    b.mark(_gen_family_circles(b, merchants, terminals), "family_circle")
    _gen_crowd_collections(b, merchants, terminals)

    index = 1
    for _ in range(b.config.mule_networks):
        network = _gen_mule_network(b, merchants, terminals, employers, index)
        b.world.networks.append(network)
        b.mark(list(network.account_ids), "mule")
        index += 1
    for _ in range(b.config.pyramids):
        network = _gen_pyramid(b, terminals, index)
        b.world.networks.append(network)
        b.mark(list(network.account_ids), "pyramid")
        index += 1

    b.world.events.sort(key=lambda e: e.ts)
    return b.world
