"""Run E1/E2 — the detector ladder crossed with the unit of analysis.

    python scripts/run_experiment_ladder.py            # default world, 10 seeds

Writes artifacts/experiment_ladder.json and prints the table.
"""

from __future__ import annotations

import argparse
import time
import statistics
import math
from collections import defaultdict

from apris.cheops.infrastructure.experiments.ladder import run_ladder, write_report
from apris.cheops.infrastructure.simulation.config import SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds to run")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--mule-networks", type=int, default=30)
    parser.add_argument("--pyramids", type=int, default=8)
    parser.add_argument("--crowd-collections", type=int, default=40)
    parser.add_argument("--family-circles", type=int, default=70)
    parser.add_argument("--employers", type=int, default=25)
    parser.add_argument("--terminals", type=int, default=60)
    args = parser.parse_args()

    started = time.time()
    
    metrics_per_cell = defaultdict(lambda: {"roc_auc": [], "average_precision": [], "rows": [], "pos": [], "base": []})
    last_report = None
    
    for i in range(args.seeds):
        current_seed = args.seed + i
        print(f"Running seed {current_seed} ({i+1}/{args.seeds})...", flush=True)
        config = SimulationConfig(
            seed=current_seed,
            days=args.days,
            mule_networks=args.mule_networks,
            pyramids=args.pyramids,
            crowd_collections=args.crowd_collections,
            family_circles=args.family_circles,
            employers=args.employers,
            terminals=args.terminals,
        )

        report = run_ladder(config)
        last_report = report
        
        for cell in report.cells:
            key = (cell.scope, cell.model)
            metrics_per_cell[key]["rows"].append(cell.rows)
            metrics_per_cell[key]["pos"].append(cell.positives)
            metrics_per_cell[key]["base"].append(cell.base_rate)
            if cell.roc_auc is not None:
                metrics_per_cell[key]["roc_auc"].append(cell.roc_auc)
            if cell.average_precision is not None:
                metrics_per_cell[key]["average_precision"].append(cell.average_precision)

    if last_report:
        write_report(last_report)

    header = (
        f"{'scope':<20}{'model':<10}{'rows':>7}{'pos':>6}{'base':>8}"
        f"{'ROC-AUC':>18}{'AP':>18}"
    )
    lines = ["", header, "-" * len(header)]
    
    def get_stats(vals):
        if not vals:
            return "—"
        mean = statistics.mean(vals)
        if len(vals) > 1:
            ci = 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))
            return f"{mean:.4f} \u00B1 {ci:.4f}"
        return f"{mean:.4f}"

    for (scope, model), data in metrics_per_cell.items():
        avg_rows = int(statistics.mean(data["rows"]))
        avg_pos = int(statistics.mean(data["pos"]))
        avg_base = statistics.mean(data["base"])
        
        auc_str = get_stats(data["roc_auc"])
        ap_str = get_stats(data["average_precision"])
        
        lines.append(
            f"{scope:<20}{model:<10}{avg_rows:>7}{avg_pos:>6}"
            f"{avg_base:>8.3f}{auc_str:>18}{ap_str:>18}"
        )

    print("\n".join(lines))
    print()
    
    if last_report:
        print(
            f"discovery coverage (last run) {last_report.coverage:.3f} "
            f"({last_report.networks_covered}/{last_report.networks_total} networks proposed) "
            "- a recall ceiling no model above can lift"
        )
        c = last_report.account_ceiling
        print(
            f"account unit coverage (last run)  {c.coverage:.3f} "
            f"({c.fraud_scored}/{c.fraud_total} mules have enough history to be judged; "
            f"{c.accounts_scored}/{c.accounts_total} accounts scored) "
            "- the same kind of ceiling, at the other unit"
        )
    print(f"Total run time: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
