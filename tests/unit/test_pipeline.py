"""Тесты конвейера: порог, очередь и то, что очередь обещает.

Здесь проверяются две вещи, каждая из которых ломается тихо.

**Порог выбирается на прошлом.** Если калибровка подсмотрит в тот блок, на
котором потом отчитываются, цифры отчёта станут недостижимыми в понедельник,
и ни один тест на диапазон этого не заметит: числа останутся правдоподобными.

**Пустая очередь — это ответ, а не сбой.** Порог, никого не пропустивший,
обязан вернуть пустой список и не выдумывать точность. Ноль в этом месте
читается как «все дела оказались пустышками», то есть ровно наоборот.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from apris.cheops.infrastructure.experiments.ladder import Row
from apris.cheops.infrastructure.pipeline import (
    PipelineReport,
    QueueOutcome,
    build_queue,
    read_queue,
    threshold_for_recall,
    write_queue,
)
from apris.cheops.infrastructure.simulation.presets import (
    DEFAULT_PRESET,
    PRESETS,
    preset_config,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _rows(count: int, positives_every: int = 4) -> list[Row]:
    """Строки, у которых признак почти совпадает с меткой.

    Задача теста — не качество модели, а механика отреза, поэтому сигнал
    здесь заведомо есть: иначе тест на пороге падал бы из-за леса, а
    выглядел бы как ошибка конвейера.
    """
    rows: list[Row] = []
    for i in range(count):
        label = 1 if i % positives_every == 0 else 0
        rows.append(
            Row(
                key=f"obj-{i}",
                ts=START + timedelta(hours=i),
                features={"signal": float(label) + (i % 3) * 0.01},
                label=label,
                events=(),
                members=(f"obj-{i}",),
            )
        )
    return rows


# ==========================================================================
# Порог
# ==========================================================================


def test_threshold_catches_the_share_it_was_asked_for():
    scores = np.linspace(0.0, 1.0, 101)
    truth = np.array([1 if s >= 0.5 else 0 for s in scores])

    threshold = threshold_for_recall(scores, truth, 0.5)
    caught = int(((scores >= threshold) & (truth == 1)).sum())
    assert caught / int(truth.sum()) >= 0.5


def test_a_higher_target_never_raises_the_threshold():
    """Ловить больше — значит опускать планку. Обратное означало бы, что
    полнота и порог связаны не тем знаком, и очередь под 80 % оказалась бы
    короче очереди под 50 %."""
    scores = np.linspace(0.0, 1.0, 101)
    truth = np.array([1 if s >= 0.4 else 0 for s in scores])

    thresholds = [threshold_for_recall(scores, truth, r) for r in (0.5, 0.8, 0.95)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_history_without_fraud_gives_an_empty_queue_not_a_low_bar():
    """Прошлое без единого мошенника не даёт оснований звать аналитика.
    Низкий порог здесь выдал бы человеку список из всех подряд."""
    scores = np.array([0.1, 0.2, 0.3])
    truth = np.zeros(3, dtype=int)
    assert threshold_for_recall(scores, truth, 0.5) == float("inf")


# ==========================================================================
# Очередь
# ==========================================================================


def test_queue_is_cut_on_a_block_the_model_never_trained_on():
    rows = _rows(400)
    outcome = build_queue(rows, "счета", ceiling=1.0, target_recall=0.5)

    # Последний блок walk-forward меньше всего набора: если бы отрез шёл по
    # всем строкам, размер блока совпал бы с их числом.
    assert 0 < outcome.block_rows < len(rows)
    assert outcome.queued <= outcome.block_rows


def test_queue_items_are_ordered_by_score():
    rows = _rows(400)
    outcome = build_queue(rows, "счета", ceiling=1.0, target_recall=0.8)
    scores = [item.score for item in outcome.items]

    assert scores == sorted(scores, reverse=True)
    assert [item.rank for item in outcome.items] == list(range(1, len(scores) + 1))


def test_a_queue_that_asks_for_more_recall_is_not_shorter():
    rows = _rows(400)
    half = build_queue(rows, "счета", 1.0, target_recall=0.5)
    most = build_queue(rows, "счета", 1.0, target_recall=0.9)
    assert most.queued >= half.queued


def test_empty_queue_reports_no_precision_rather_than_zero():
    """Никого не позвали — значит и ошибиться было негде."""
    outcome = QueueOutcome(
        unit="сети", block_rows=18, block_positives=2, block_prevalence=0.11,
        threshold=0.9, target_recall=0.5, queued=0, caught=0, precision=0.0,
        recall=0.0, reviews_per_catch=None, unit_ceiling=1.0, items=(),
    )
    report = PipelineReport(
        run_id="test", detector="forest-v1", generated_at="2026-09-06T00:00:00+00:00",
        preset="quick", seed=17, world={}, seconds=1.0, outcomes=(outcome,),
    )
    line = [row for row in report.table().splitlines() if row.startswith("сети")][0]
    # Два прочерка: точность и «проверок на находку». Полнота при этом
    # остаётся числом — ноль пойманных из двух настоящих это правда о блоке,
    # а не отсутствие ответа.
    assert line.count("—") == 2
    assert "0.000" in line


def test_too_few_rows_returns_a_blank_outcome_not_an_exception():
    """Мир, в котором нечего делить, обязан сказать это словами."""
    outcome = build_queue(_rows(10), "счета", ceiling=0.5)
    assert outcome.queued == 0
    assert outcome.items == ()
    assert outcome.unit_ceiling == 0.5


# ==========================================================================
# Файл, который читает интерфейс
# ==========================================================================


def test_queue_survives_a_round_trip_through_the_file(tmp_path):
    rows = _rows(400)
    report = PipelineReport(
        run_id="abc123", detector="forest-v1",
        generated_at="2026-09-06T00:00:00+00:00", preset="quick", seed=17,
        world={"events": 100.0}, seconds=2.0,
        outcomes=(build_queue(rows, "счета", 1.0),),
    )
    path = write_queue(report, tmp_path / "queue.json")
    loaded = read_queue(path)

    assert loaded is not None
    assert loaded["run_id"] == "abc123"
    assert loaded["outcomes"][0]["unit"] == "счета"


def test_a_missing_queue_file_is_not_an_error(tmp_path):
    """Страница должна сказать «конвейер ещё не запускался», а не упасть."""
    assert read_queue(tmp_path / "нет-такого.json") is None


# ==========================================================================
# Пресеты миров
# ==========================================================================


def test_the_interface_and_the_pipeline_share_one_definition_of_the_world():
    """Раньше пресеты жили в интерфейсе, и скрипт строил свой мир. Два экрана
    с разными числами про «тот же прогон» — то, ради чего это съехалось."""
    from apris.frontend import session

    assert session.SCALES is PRESETS
    assert session.DEFAULT_SCALE == DEFAULT_PRESET


def test_seed_is_applied_and_the_rest_of_the_preset_is_not_touched():
    config = preset_config("quick", seed=999)
    assert config.seed == 999
    assert config.days == PRESETS["quick"].config.days
    assert config.mule_networks == PRESETS["quick"].config.mule_networks


def test_an_unknown_preset_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="quick"):
        preset_config("нет-такого", seed=1)
