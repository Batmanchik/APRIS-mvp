"""The named worlds everything demonstrable runs on.

One definition, in one place, on purpose. Before this module the interface
carried its own two presets and every script built a config inline, so
"the demo world" meant a different world depending on which entry point you
came through — and two screens quoting different numbers for the same run is
the failure mode that costs a demo its credibility.

A preset is a world plus the sentence that says what it may be used for.
The quick one exists because an honest world takes twenty seconds and
clicking around should not; it carries a proportionally larger share of
fraud, so a metric read off it is not a measurement, and its ``note`` says
so wherever it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

from apris.cheops.infrastructure.simulation.config import SimulationConfig


@dataclass(frozen=True)
class WorldPreset:
    key: str
    label: str
    note: str
    config: SimulationConfig


PRESETS: dict[str, WorldPreset] = {
    "quick": WorldPreset(
        key="quick",
        label="Быстрый (демо)",
        note=(
            "Несколько секунд. Мошеннических кандидатов здесь непропорционально "
            "много, и задача разделяется почти идеально — для метрики берите "
            "полный мир."
        ),
        config=SimulationConfig(
            seed=17,
            days=60,
            salary_earners=400,
            freelancers=60,
            traders=40,
            fast_spenders=200,
            family_circles=20,
            crowd_collections=25,
            marketplace_sellers=100,
            employers=8,
            mule_networks=40,
            pyramids=8,
            terminals=24,
            merchants=120,
        ),
    ),
    "full": WorldPreset(
        key="full",
        label="Полный мир",
        note=(
            "Около двадцати секунд. Та же конфигурация, на которой считался "
            "аудит, и единственная, по которой стоит судить о качестве."
        ),
        config=SimulationConfig(),
    ),
}

DEFAULT_PRESET = "quick"
DEFAULT_SEED = 17


def preset_config(key: str, seed: int) -> SimulationConfig:
    """The preset's world, at the seed the caller asked for.

    The seed is an argument rather than part of the preset because a run
    that cannot be repeated at another seed is a picture, not a result.
    """
    if key not in PRESETS:
        raise KeyError(f"unknown preset {key!r}; have {sorted(PRESETS)}")
    return SimulationConfig(**{**PRESETS[key].config.__dict__, "seed": int(seed)})


__all__ = ["DEFAULT_PRESET", "DEFAULT_SEED", "PRESETS", "WorldPreset", "preset_config"]
