"""Draw the five standard diagnostics on the account-level task."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from apris.cheops.infrastructure.reporting.figures import use_project_style
from apris.cheops.infrastructure.reporting.diagnostics import (
    plot_amount_distribution, plot_signal_stability, plot_quintile_ladder,
    plot_walk_forward, plot_importance,
)
from apris.cheops.infrastructure.ml.validation_v2 import (
    purged_walk_forward_splits, quintile_ladder, permutation_importance_with_noise_floor,
)
from apris.cheops.infrastructure.experiments.ladder import (
    ACCOUNT_FEATURE_COLUMNS, build_account_rows,
)
from apris.cheops.infrastructure.simulation.config import SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world

use_project_style()
cfg = SimulationConfig(seed=20261005, days=120, mule_networks=90, pyramids=30,
                       crowd_collections=220, family_circles=300, employers=80, terminals=120)
world = generate_world(cfg)
rows, ceiling = build_account_rows(world)
X = pd.DataFrame([r.features for r in rows], columns=list(ACCOUNT_FEATURE_COLUMNS)).astype(float).to_numpy()
y = np.array([r.label for r in rows])
ts = [r.ts for r in rows]

print("1", plot_amount_distribution([e.amount for e in world.events]))

splits = purged_walk_forward_splits(ts, n_splits=5, purge=timedelta(days=2))
print("4", plot_walk_forward(splits, len(rows)))

per_fold, pooled_s, pooled_y = [], [], []
for sp in splits:
    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                               class_weight="balanced", random_state=1, n_jobs=-1)
    m.fit(X[list(sp.train)], y[list(sp.train)])
    p = np.asarray(m.predict_proba(X[list(sp.test)]))[:, 1]
    t = y[list(sp.test)]
    per_fold.append(roc_auc_score(t, p) if len(set(t)) > 1 else float("nan"))
    pooled_s += [float(v) for v in p]; pooled_y += [int(v) for v in t]

print("2", plot_signal_stability([f"fold {i+1}" for i in range(len(per_fold))], per_fold))
print("3", plot_quintile_ladder(quintile_ladder(pooled_s, pooled_y)))

imp = permutation_importance_with_noise_floor(
    RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                           class_weight="balanced", random_state=1, n_jobs=-1),
    X, y, ACCOUNT_FEATURE_COLUMNS, repeats=8)
print("5", plot_importance(imp))
print("\nfolds:", " ".join(f"{v:.4f}" for v in per_fold))
print("above floor:", imp.above_floor())
