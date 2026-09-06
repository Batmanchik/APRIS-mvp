"""Полный конвейер одной командой: мир → поиск → признаки → очередь дел.

    python scripts/run_pipeline.py                     # быстрый мир, демо
    python scripts/run_pipeline.py --preset full       # полный мир
    python scripts/run_pipeline.py --target-recall 0.8 # ловить 4 из 5

Пишет artifacts/analyst_queue.json — то, что читает страница «Очередь
аналитика» в интерфейсе.

Порог берётся не из воздуха и не из бюджета: он читается по прошлым блокам
walk-forward под заданную полноту, а очередь режется на последнем блоке,
который модель не видела. Длина очереди — результат, а не настройка.
"""

from __future__ import annotations

import argparse

from apris.cheops.infrastructure.pipeline import (
    DEFAULT_TARGET_RECALL,
    run_pipeline,
    write_queue,
)
from apris.cheops.infrastructure.simulation.presets import (
    DEFAULT_PRESET,
    DEFAULT_SEED,
    PRESETS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL,
                        help="какую долю мошенников очередь должна ловить")
    args = parser.parse_args()

    preset = PRESETS[args.preset]
    print(f"Мир: {preset.label}, сид {args.seed}")
    print(f"      {preset.note}\n")

    report = run_pipeline(args.preset, args.seed, args.target_recall)
    path = write_queue(report)

    print(report.table())
    print()
    print("блок      — строки последнего отрезка времени, модель их не видела")
    print("порог     — выбран по прошлым отрезкам под полноту "
          f"{args.target_recall:.0%}, а не по бюджету")
    print("на находку — сколько дел аналитик открывает на одну настоящую находку")
    print()
    for outcome in report.outcomes:
        print(f"  {outcome.unit}: потолок самого уровня {outcome.unit_ceiling:.3f} "
              f"(выше него не прыгнет никакая модель)")
        if outcome.queued == 0:
            print(f"      очередь пустая: порог {outcome.threshold:.3f} не пропустил "
                  f"никого из {outcome.block_rows} строк блока, где настоящих "
                  f"{outcome.block_positives}. Это ответ «на этой неделе тихо», "
                  f"а не ошибка — но на таком блоке он и не мог быть уверенным.")
    print()
    print(f"детектор {report.detector}, прогон {report.run_id}, "
          f"{report.seconds:.0f} с")
    print(f"{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
