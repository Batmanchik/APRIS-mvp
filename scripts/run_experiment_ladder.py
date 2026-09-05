"""Run E1/E2 — the detector ladder crossed with the unit of analysis.

    python scripts/run_experiment_ladder.py            # default world
    python scripts/run_experiment_ladder.py --days 120 --mule-networks 40

Writes artifacts/experiment_ladder.json and prints the table.
"""

from __future__ import annotations

import argparse
import time

from apris.cheops.infrastructure.experiments.ladder import run_ladder, write_report
from apris.cheops.infrastructure.simulation.config import SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--mule-networks", type=int, default=30)
    parser.add_argument("--pyramids", type=int, default=8)
    parser.add_argument("--crowd-collections", type=int, default=40)
    parser.add_argument("--family-circles", type=int, default=70)
    parser.add_argument("--employers", type=int, default=25)
    parser.add_argument("--terminals", type=int, default=60)
    args = parser.parse_args()

    config = SimulationConfig(
        seed=args.seed,
        days=args.days,
        mule_networks=args.mule_networks,
        pyramids=args.pyramids,
        crowd_collections=args.crowd_collections,
        family_circles=args.family_circles,
        employers=args.employers,
        terminals=args.terminals,
    )

    started = time.time()
    report = run_ladder(config)
    path = write_report(report)

    print(report.table())
    print()
    print(
        f"discovery coverage {report.coverage:.3f} "
        f"({report.networks_covered}/{report.networks_total} networks proposed) "
        "- a recall ceiling no model above can lift"
    )
    c = report.account_ceiling
    print(
        f"account unit coverage  {c.coverage:.3f} "
        f"({c.fraud_scored}/{c.fraud_total} mules have enough history to be judged; "
        f"{c.accounts_scored}/{c.accounts_total} accounts scored) "
        "- the same kind of ceiling, at the other unit"
    )
    print(f"{path} written in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
