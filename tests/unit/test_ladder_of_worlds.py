"""Tests for the ladder of worlds.

These pin the DESIGN of the ladder, not its results. A rung set that drifts
into varying the class balance instead of the kind of negative produces a
figure that looks like a difficulty curve and is not one, and no test of the
numbers would catch it — the numbers would be internally consistent and
answer the wrong question.
"""

from __future__ import annotations

import pytest

from apris.cheops.infrastructure.experiments.ladder_of_worlds import (
    _BASE,
    _HONEST_TARGET,
    LADDER,
    DETECTOR_VERSION,
    Rung,
    UnitResult,
)
from apris.cheops.infrastructure.simulation.config import EvasionKnobs, SimulationConfig

HONEST_POPULATIONS = (
    "salary_earners",
    "freelancers",
    "traders",
    "fast_spenders",
    "family_circles",
    "crowd_collections",
    "marketplace_sellers",
)


def _honest_total(rung: Rung) -> int:
    return sum(int(rung.config.get(name, 0)) for name in HONEST_POPULATIONS)


# ==========================================================================
# The rule the first run taught
# ==========================================================================


def test_every_rung_carries_a_comparable_honest_population():
    """Difficulty must change the KIND of negative, not the number of them.

    The first version of this ladder zeroed every honest population at W1.
    The world came out 96.3 % fraud: ROC-AUC 1.0000 with nothing to confuse,
    recall at a ten-per-cent budget structurally capped at 0.10, and no
    network metric at all on four rungs out of five because every candidate
    was a real ring and one class has no ROC curve.

    That ladder measured class imbalance and called it difficulty.
    """
    for rung in LADDER:
        total = _honest_total(rung)
        assert 0.5 * _HONEST_TARGET <= total <= 1.5 * _HONEST_TARGET, (
            f"{rung.key} carries {total} honest accounts against a target of "
            f"{_HONEST_TARGET}; a rung must vary the kind of negative, not the count"
        )


def test_the_fraud_side_is_identical_across_the_first_three_rungs():
    """The thing being detected must not change while the negatives do."""
    for rung in LADDER[:3]:
        assert rung.config["mule_networks"] == _BASE["mule_networks"]
        assert rung.config.get("pyramids", 0) == 0


def test_only_the_last_rung_turns_on_evasion():
    """Evasion is one rung's subject. Mixing it into an earlier rung would
    make two changes at once and neither would be measurable."""
    for rung in LADDER[:-1]:
        assert rung.evasion == {}
    assert LADDER[-1].evasion, "the last rung is the one that costs the organiser"


def test_the_last_two_rungs_differ_only_in_evasion():
    """W5 is W4 plus hiding. If the populations also moved, the rung would
    measure both and attribute it to evasion."""
    assert LADDER[-1].config == LADDER[-2].config


# ==========================================================================
# Every rung must be able to say why it exists
# ==========================================================================


def test_each_rung_states_its_purpose_in_prose():
    keys = [r.key for r in LADDER]
    assert keys == sorted(keys), "rungs must read in order"
    assert len(set(keys)) == len(keys)
    for rung in LADDER:
        assert len(rung.why.split()) >= 15, f"{rung.key}: a one-line reason is required"
        assert rung.title and not rung.title.endswith("."), rung.key


def test_every_rung_builds_a_valid_config():
    """A knob the config does not have must fail here, not three minutes into
    a run that has already generated two worlds."""
    for rung in LADDER:
        config = SimulationConfig(
            seed=1,
            evasion=EvasionKnobs(**rung.evasion) if rung.evasion else EvasionKnobs(),
            **rung.config,
        )
        assert config.mule_networks > 0, rung.key


# ==========================================================================
# Absence is reported as absence
# ==========================================================================


def test_a_unit_with_no_measurement_reports_none_not_zero():
    """A rung where the fit was impossible and a detector that found nothing
    are different facts. Writing 0.0 for the first is the mistake this repo
    keeps catching in other people's code."""
    blank = UnitResult("network", rows=3, positives=3, base_rate=1.0,
                       coverage=1.0, roc_auc=None, average_precision=None,
                       recall_at_budget=None)
    assert blank.roc_auc is None
    assert blank.average_precision is None


def test_the_detector_version_is_a_plain_stamp():
    """It is what makes two runs comparable, so it must not be derived from
    the git hash — a documentation edit would then look like a new detector."""
    assert DETECTOR_VERSION
    assert " " not in DETECTOR_VERSION
    assert not DETECTOR_VERSION.startswith("0x")


@pytest.mark.parametrize("rung", LADDER, ids=[r.key for r in LADDER])
def test_no_rung_silently_drops_the_terminals_a_cash_exit_needs(rung: Rung):
    assert rung.config.get("terminals", 0) >= 10, rung.key
