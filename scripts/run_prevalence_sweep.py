"""Run the prevalence sweep — what the detector is worth when fraud is rare.

    python scripts/run_prevalence_sweep.py            # 3 seeds
    python scripts/run_prevalence_sweep.py --seeds 5

Writes artifacts/prevalence_sweep.json plus a stamped archive copy.

Two arms, printed together on purpose:

  measured   positives thinned to the target share BEFORE the split, so the
             forest trains and is tested where fraud is that rare;
  projected  the same ROC curve from the 6.87 % world, re-priced by
             arithmetic at each target share, with no refit.

Where the two agree, the loss is arithmetic and no amount of modelling gets
it back. Where the measured arm falls below the projection, the loss is
learning under rarity, and that one has known remedies.
"""

from __future__ import annotations

import argparse
import time

from apris.cheops.infrastructure.experiments.prevalence import (
    HONEST_SCALE,
    PREVALENCE_TARGETS,
    run_prevalence_sweep,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3, help="how many seeds per cell")
    parser.add_argument("--first-seed", type=int, default=20261005)
    parser.add_argument("--scale", type=int, default=HONEST_SCALE,
                        help="multiplier on the honest populations")
    args = parser.parse_args()

    seeds = [args.first_seed + i for i in range(args.seeds)]
    targets = ", ".join(f"{t:.1%}" for t in PREVALENCE_TARGETS)
    print(f"W4 world, honest side x{args.scale}, {len(seeds)} seeds")
    print(f"targets below the world's own share: {targets}\n")

    started = time.time()
    report = run_prevalence_sweep(seeds, scale=args.scale)
    path = write_report(report)

    print(report.table())
    print()
    print("AP/base = average precision over its own floor; the floor IS the")
    print("          prevalence, so the raw AP falls even for a perfect model.")
    print("per catch = honest accounts reviewed for each fraud found, at a")
    print("          ten-per-cent budget.")
    print()
    print(f"detector {report.detector}, run {report.run_id}")
    print(f"{path} written in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
