"""Shared, cached state for the interface.

Every page works on the same three objects — a world, the candidates
discovery proposed from its events, and the labels attached afterwards. They
are expensive enough to be worth caching (a full world takes about fifteen
seconds) and important enough that all pages must see the same one, or two
screens would quietly disagree.

The scale presets exist because the honest full world is slow. The small one
is for clicking around; it is labelled as such wherever a number depends on
it, because a run too thin to measure must not be read as a measurement.
"""

from __future__ import annotations

import streamlit as st

from apris.cheops.infrastructure.ml.case_pipeline import CaseDataset, build_case_dataset
from apris.cheops.infrastructure.simulation.generator import SimulatedWorld, generate_world
from apris.cheops.infrastructure.simulation.presets import (
    DEFAULT_PRESET,
    PRESETS,
    WorldPreset,
    preset_config,
)
from apris.cheops.infrastructure.simulation.presets import DEFAULT_SEED as PRESET_SEED

WORLD_KEY = "cheops_world"
DATASET_KEY = "cheops_dataset"
SCALE_KEY = "cheops_scale"


#: The presets live in the simulation layer, not here. They used to be
#: defined in this module, which meant the pipeline script and the interface
#: could disagree about what "the demo world" is — and two screens quoting
#: different numbers for the same run is how a demo loses the room.
Scale = WorldPreset
SCALES: dict[str, Scale] = PRESETS

DEFAULT_SCALE = DEFAULT_PRESET
DEFAULT_SEED = PRESET_SEED


def build_world(scale_key: str, seed: int) -> SimulatedWorld:
    return generate_world(preset_config(scale_key, seed))


def build_dataset(world: SimulatedWorld) -> CaseDataset:
    return build_case_dataset(world)


@st.cache_data(show_spinner=False)
def cached_world(scale_key: str, seed: int) -> SimulatedWorld:
    return build_world(scale_key, seed)


@st.cache_data(show_spinner=False)
def cached_dataset(scale_key: str, seed: int) -> CaseDataset:
    return build_dataset(cached_world(scale_key, seed))


def ensure_state(
    scale_key: str = DEFAULT_SCALE, seed: int = DEFAULT_SEED
) -> tuple[SimulatedWorld, CaseDataset]:
    """Give the caller a world and its candidates, building them if needed.

    No page is a dead end. Opening any of them directly — a bookmark, a
    refresh, a link handed to somebody — used to land on "go to the other
    page first", which is a worse answer than simply doing the work: the
    results are cached, so the second page pays nothing.
    """
    key = (scale_key, int(seed))
    if st.session_state.get(SCALE_KEY) != key or WORLD_KEY not in st.session_state:
        st.session_state[WORLD_KEY] = cached_world(scale_key, int(seed))
        st.session_state[DATASET_KEY] = cached_dataset(scale_key, int(seed))
        st.session_state[SCALE_KEY] = key
    return st.session_state[WORLD_KEY], st.session_state[DATASET_KEY]


def current_state() -> tuple[SimulatedWorld, CaseDataset]:
    """The world already on screen, or the default one."""
    key = st.session_state.get(SCALE_KEY)
    if key is None:
        return ensure_state()
    return ensure_state(key[0], key[1])


def world_summary_rows(world: SimulatedWorld) -> list[tuple[str, str]]:
    summary = world.summary()
    return [
        ("Событий", f"{int(summary['events']):,}".replace(",", " ")),
        ("Счетов", f"{int(summary['accounts']):,}".replace(",", " ")),
        ("Реальных сетей", f"{int(summary['networks'])}"),
        ("Доля мошеннических счетов", f"{summary['fraud_share_of_personal']:.3f}"),
    ]
