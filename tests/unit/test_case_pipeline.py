"""Tests for the case pipeline.

The pipeline exists so that a number on a screen can be re-derived rather
than quoted. So what is pinned here is not "the score is high" — it is that
the path from a world to that score cannot quietly cheat, and that a run too
thin to measure says so instead of returning a confident figure.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import numpy as np
import pytest

from apris.cheops.infrastructure.ml.case_pipeline import (
    CASE_FEATURE_COLUMNS,
    DEFAULT_MIN_TRAIN,
    DEFAULT_PURGE,
    DEFAULT_SPLITS,
    build_case_dataset,
    case_features,
    fit_case_model,
    run_case_validation,
)
from apris.cheops.infrastructure.ml.validation_v2 import purged_walk_forward_splits
from apris.cheops.infrastructure.simulation.config import SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world

# Small enough to run in a second, and too small to validate on — which is
# itself one of the things under test.
TINY = SimulationConfig(
    seed=13,
    days=40,
    salary_earners=120,
    freelancers=20,
    traders=12,
    fast_spenders=60,
    family_circles=6,
    crowd_collections=10,
    marketplace_sellers=30,
    employers=3,
    mule_networks=10,
    pyramids=3,
    terminals=10,
    merchants=50,
    crypto_layering=0,
    crypto_traders=0,
)

# Large enough that walk-forward has something to work with.
MEASURABLE = SimulationConfig(
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
)


@pytest.fixture(scope="module")
def world():
    return generate_world(TINY)


@pytest.fixture(scope="module")
def dataset(world):
    return build_case_dataset(world)


@pytest.fixture(scope="module")
def measurable_dataset():
    return build_case_dataset(generate_world(MEASURABLE))


# ==========================================================================
# The guarantee: the answer file is not an input
# ==========================================================================


def test_removing_every_network_leaves_the_candidates_unchanged(world, dataset):
    """Discovery proposes from events, so the answers cannot be reaching it.

    The dataset layer is where a leak would be easiest to introduce by
    accident — one convenience lookup of ``world.networks`` while building
    features and the ceiling on recall becomes fictional again.
    """
    stripped = dataclasses.replace(world, networks=[])
    blind = build_case_dataset(stripped)

    assert [c.candidate_id for c in blind.candidates] == [
        c.candidate_id for c in dataset.candidates
    ]
    assert [c.member_ids for c in blind.candidates] == [
        c.member_ids for c in dataset.candidates
    ]
    # Labels, however, must collapse: with no networks nothing is fraud.
    assert blind.labels.sum() == 0


def test_coverage_is_a_ceiling_not_a_constant(dataset):
    """Coverage is measured against networks discovery never proposed."""
    assert 0.0 <= dataset.coverage <= 1.0
    missed = len(dataset.discovery.missed_network_ids)
    assert dataset.discovery.networks_covered + missed == dataset.discovery.networks_total


# ==========================================================================
# Features
# ==========================================================================


def test_features_are_the_ten_event_columns_and_stay_bounded(dataset):
    assert list(dataset.features.columns) == list(CASE_FEATURE_COLUMNS)
    assert len(CASE_FEATURE_COLUMNS) == 10
    values = dataset.features.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_a_candidate_with_one_event_yields_zeros_not_a_crash(dataset):
    """Defect found the expensive way: an empty graph used to raise."""
    candidate = dataset.candidates[0]
    single = dataclasses.replace(candidate, events=candidate.events[:1])
    features = case_features(single)
    assert set(features) == set(CASE_FEATURE_COLUMNS)
    assert all(value == 0.0 for value in features.values())


def test_a_candidate_is_placed_at_its_last_event(dataset):
    """Time ordering decides what trains on what, so it must be the end."""
    for candidate, stamp in zip(dataset.candidates, dataset.timestamps):
        assert stamp == max(event.ts for event in candidate.events)


# ==========================================================================
# Validation refuses to produce a number it cannot support
# ==========================================================================


def test_an_empty_dataset_reports_unscorable_rather_than_raising(world):
    empty = build_case_dataset(world, min_size=10_000)
    assert empty.size == 0
    report = run_case_validation(empty)
    assert report.roc_auc is None
    assert "no candidates" in report.describe()


def test_too_few_candidates_is_reported_as_such(dataset):
    """The distinction that matters: unmeasurable, not broken.

    A tiny world yields a handful of candidates, which walk-forward cannot
    split at all. Returning a blank score with no reason would read as a
    failure; the reason is that there was nothing to measure.
    """
    assert dataset.size <= DEFAULT_MIN_TRAIN
    report = run_case_validation(dataset)
    assert report.roc_auc is None
    assert "min_train" in report.note


def test_an_over_wide_purge_says_so_instead_of_blaming_the_data(measurable_dataset):
    """A purge wider than the period starves training, and names itself.

    The two causes of a blank score look identical from the outside — no
    folds either way — and lead to opposite actions: gather more candidates,
    or narrow the purge.
    """
    assert measurable_dataset.size > DEFAULT_MIN_TRAIN
    report = run_case_validation(measurable_dataset, purge=timedelta(days=400))
    assert report.roc_auc is None
    assert "purge" in report.note


def test_every_split_leaves_a_fold_row_behind(measurable_dataset):
    """Folds are never dropped silently, scorable or not.

    A fold whose training block holds one class produces no AUC. Discarding
    it would leave a score with no way to see how few folds made it.
    """
    splits = purged_walk_forward_splits(
        list(measurable_dataset.timestamps),
        n_splits=DEFAULT_SPLITS,
        purge=DEFAULT_PURGE,
        min_train=DEFAULT_MIN_TRAIN,
    )
    report = run_case_validation(measurable_dataset)
    assert len(report.folds) == len(splits)
    assert all(fold.test_size > 0 for fold in report.folds)
    assert report.scored_folds <= len(report.folds)


def test_a_scorable_run_reports_folds_and_a_ladder(measurable_dataset):
    report = run_case_validation(measurable_dataset)
    assert report.roc_auc is not None, report.note
    assert 0.0 <= report.roc_auc <= 1.0
    assert report.ladder is not None
    assert len(report.ladder.bucket_rates) == 5
    assert report.scored_folds >= 1
    assert set(report.single_feature_auc) == set(CASE_FEATURE_COLUMNS)


def test_the_interface_model_is_withheld_when_one_class_is_missing(world):
    stripped = dataclasses.replace(world, networks=[])
    blind = build_case_dataset(stripped)
    assert fit_case_model(blind) is None
