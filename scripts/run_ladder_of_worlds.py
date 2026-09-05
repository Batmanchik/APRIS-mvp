"""Run the ladder of worlds — five worlds from clean schema to hiding organiser.

    python scripts/run_ladder_of_worlds.py                 # 3 seeds
    python scripts/run_ladder_of_worlds.py --seeds 5

Writes artifacts/ladder_of_worlds.json plus a stamped archive copy, so a
later run can be drawn against this one rather than replacing it.
"""

from __future__ import annotations

import argparse
import time

from apris.cheops.infrastructure.experiments.ladder_of_worlds import (
    LADDER,
    run_ladder_of_worlds,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3, help="how many seeds per rung")
    parser.add_argument("--first-seed", type=int, default=20261005)
    args = parser.parse_args()

    seeds = [args.first_seed + i for i in range(args.seeds)]
    print(f"{len(LADDER)} rungs x {len(seeds)} seeds\n")
    for rung in LADDER:
        print(f"  {rung.key}  {rung.title}")
        print(f"       {rung.why}")
    print()

    started = time.time()
    report = run_ladder_of_worlds(seeds)
    path = write_report(report)

    print(report.table())
    print()
    print("ceil = what the UNIT can see at all, before any model:")
    print("  account  share of fraudulent accounts with enough history to judge")
    print("  network  share of real networks discovery proposed at all")
    print()
    print(f"detector {report.detector}, run {report.run_id}")
    print(f"{path} written in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
