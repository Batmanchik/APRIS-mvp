"""Run the evasion curve — where hiding starts to work, and what it costs.

    python scripts/run_evasion_curve.py               # 3 seeds
    python scripts/run_evasion_curve.py --seeds 5

Writes artifacts/evasion_curve.json plus a stamped archive copy.

One dial at a time from the naive organiser (one funder, one ATM) up to the
W5 setting, plus the joint point. Every point reports the network unit's own
ceiling and the overlap profile behind it, so a ring that discovery split can
never be confused with a ring discovery never saw.
"""

from __future__ import annotations

import argparse
import time

from apris.cheops.infrastructure.experiments.evasion_curve import (
    CURVE,
    run_evasion_curve,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3, help="how many seeds per point")
    parser.add_argument("--first-seed", type=int, default=20261005)
    args = parser.parse_args()

    seeds = [args.first_seed + i for i in range(args.seeds)]
    print(f"{len(CURVE)} points x {len(seeds)} seeds, W4 populations throughout\n")
    for point in CURVE:
        print(f"  {point.key:<6} funders={point.funders} terminals={point.terminals}"
              f"   costs the organiser: {point.cost}")
    print()

    started = time.time()
    report = run_evasion_curve(seeds)
    path = write_report(report)

    print(report.table())
    print()
    print("cover   = share of real rings discovery proposed at all (>=50 % of one")
    print("          ring inside one candidate)")
    print("ovl med = median over rings of the best single-candidate overlap")
    print("grip    = share of rings with at least 10 % of themselves in one")
    print("          candidate; a low median with a high grip is fragmentation,")
    print("          both near zero is blindness")
    print()
    print(f"detector {report.detector}, run {report.run_id}")
    print(f"{path} written in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
