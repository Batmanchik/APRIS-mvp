"""Measure W against the noise floor."""

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from apris.cheops.infrastructure.simulation.config import SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world
from apris.cheops.infrastructure.experiments.ladder import (
    build_account_rows,
    _make_model,
    _pooled_out_of_fold,
    Row,
    ACCOUNT_FEATURE_COLUMNS,
)
from apris.cheops.infrastructure.ml.flow_weight import flow_weight, LAMBDA_FAST, LAMBDA_SLOW
from apris.cheops.infrastructure.ml.validation_v2 import permutation_importance_with_noise_floor

def main():
    print("Generating world...")
    config = SimulationConfig(seed=20261005)
    world = generate_world(config)
    
    print("Building account rows...")
    base_rows, ceiling = build_account_rows(world)
    
    # 1. Add W columns
    print("Calculating flow_weight features...")
    new_rows = []
    w_fast_values = []
    w_slow_values = []
    
    for row in base_rows:
        w_fast = flow_weight(row.key, row.events, lam_per_hour=LAMBDA_FAST)
        w_slow = flow_weight(row.key, row.events, lam_per_hour=LAMBDA_SLOW)
        
        # impute None with -1.0
        w_f = float(w_fast) if w_fast is not None else -1.0
        w_s = float(w_slow) if w_slow is not None else -1.0
        
        w_fast_values.append(w_f)
        w_slow_values.append(w_s)
        
        new_features = dict(row.features)
        new_features["w_fast"] = w_f
        new_features["w_slow"] = w_s
        
        new_rows.append(Row(
            key=row.key,
            ts=row.ts,
            features=new_features,
            label=row.label,
            events=row.events,
            members=row.members
        ))

    y = np.array([r.label for r in new_rows], dtype=int)
    
    # (a) ROC-AUC of each column independently
    auc_w_fast = roc_auc_score(y, w_fast_values)
    auc_w_slow = roc_auc_score(y, w_slow_values)
    
    print(f"Standalone ROC-AUC w_fast: {auc_w_fast:.4f}")
    print(f"Standalone ROC-AUC w_slow: {auc_w_slow:.4f}")
    
    # (b) ROC-AUC of the model forest with and without columns
    print("Evaluating forest without W...")
    base_cols = ACCOUNT_FEATURE_COLUMNS
    preds_base, truths_base, _ = _pooled_out_of_fold(base_rows, base_cols, "forest")
    auc_base = roc_auc_score(truths_base, preds_base)
    
    print("Evaluating forest with W...")
    enriched_cols = list(base_cols) + ["w_fast", "w_slow"]
    preds_enriched, truths_enriched, _ = _pooled_out_of_fold(new_rows, enriched_cols, "forest")
    auc_enriched = roc_auc_score(truths_enriched, preds_enriched)
    
    print(f"Forest base AUC:     {auc_base:.4f}")
    print(f"Forest enriched AUC: {auc_enriched:.4f}")
    print(f"AUC lift:            {auc_enriched - auc_base:+.4f}")
    
    # (c) permutation_importance_with_noise_floor against shuffled control
    print("Calculating permutation importance...")
    model = _make_model("forest")
    
    x = np.array([[r.features[c] for c in enriched_cols] for r in new_rows], dtype=float)
    
    importance = permutation_importance_with_noise_floor(
        fit_predict=model,
        x=x,
        y=y,
        feature_names=enriched_cols,
        repeats=10,
        seed=42,
    )
    
    above = importance.above_floor()
    w_fast_beats_noise = "w_fast" in above
    w_slow_beats_noise = "w_slow" in above
    
    idx_w_fast = enriched_cols.index("w_fast")
    idx_w_slow = enriched_cols.index("w_slow")
    
    w_fast_imp = importance.means[idx_w_fast]
    w_slow_imp = importance.means[idx_w_slow]
    
    floor = importance.noise_floor
    
    print(f"Noise floor: {floor:.4f}")
    print(f"w_fast importance: {w_fast_imp:.4f} (Beats noise: {w_fast_beats_noise})")
    print(f"w_slow importance: {w_slow_imp:.4f} (Beats noise: {w_slow_beats_noise})")
    
    beat_noise = w_fast_beats_noise or w_slow_beats_noise
    
    if beat_noise:
        verdict = "Прирост превышает шумовой пол. Можно подключать признак."
    else:
        verdict = "Прирост не превышает шумовой пол. Признак не подключается."
        
    print(f"Verdict: {verdict}")

    report = {
        "standalone_auc": {
            "w_fast": auc_w_fast,
            "w_slow": auc_w_slow,
        },
        "model_auc": {
            "base": auc_base,
            "enriched": auc_enriched,
            "lift": auc_enriched - auc_base,
        },
        "permutation_importance": {
            "w_fast": {
                "mean": w_fast_imp,
                "error": importance.errors[idx_w_fast],
                "beats_noise": w_fast_beats_noise,
            },
            "w_slow": {
                "mean": w_slow_imp,
                "error": importance.errors[idx_w_slow],
                "beats_noise": w_slow_beats_noise,
            },
            "noise_floor": floor,
            "noise_floor_error": importance.noise_floor_error,
        },
        "verdict": verdict,
    }
    
    out_path = Path("artifacts/flow_weight_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {out_path}")
    
if __name__ == "__main__":
    main()
