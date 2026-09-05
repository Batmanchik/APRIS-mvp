"""Re-derive the candidate-classification baseline.

    python -m scripts.case_baseline [--seed 42] [--splits 5]

Prints every number the write-up is allowed to quote about case-level
detection, and prints them from a run rather than from memory. The audit
previously carried a baseline that no committed code reproduced; this script
exists so that never happens again.
"""

from __future__ import annotations

import argparse
import time

from apris.cheops.infrastructure.ml.case_pipeline import (
    build_case_dataset,
    run_case_validation,
)
from apris.cheops.infrastructure.simulation.config import SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    world = generate_world(SimulationConfig(seed=args.seed))
    dataset = build_case_dataset(world)
    report = run_case_validation(dataset, n_splits=args.splits)
    elapsed = time.time() - started

    summary = world.summary()
    print(f"seed {args.seed}, {elapsed:.1f}s")
    print(f"  events                {int(summary['events'])}")
    print(f"  networks              {int(summary['networks'])}")
    print()
    print(f"  candidates            {report.candidates}")
    print(f"  of them networks      {report.positives}")
    print(f"  base rate             {report.base_rate:.4f}")
    print(f"  COVERAGE              {report.coverage:.4f}   <- ceiling on recall")
    print(f"  never proposed        {len(report.missed_networks)}")
    print()

    if report.roc_auc is None:
        print(f"  NOT SCORABLE: {report.note}")
        return 1

    print(f"  ROC-AUC (out-of-fold) {report.roc_auc:.4f}")
    print(f"  PR-AUC                {report.pr_auc:.4f}")
    print(f"  scored folds          {report.scored_folds} of {len(report.folds)}")
    if report.ladder is not None:
        rates = ", ".join(f"{rate:.2f}" for rate in report.ladder.bucket_rates)
        print(f"  quintile ladder       {report.ladder.describe()}")
        print(f"                        [{rates}]")
    print()
    print("  single-feature AUC, in-sample (a caveat, not a result):")
    for name, value in sorted(report.single_feature_auc.items(), key=lambda x: -x[1]):
        print(f"    {name:24s} {value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
