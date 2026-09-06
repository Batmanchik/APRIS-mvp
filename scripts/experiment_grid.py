import pandas as pd
from sklearn.metrics import roc_auc_score
from apris.cheops.infrastructure.simulation.config import SimulationConfig, EvasionKnobs
from apris.cheops.infrastructure.simulation.generator import generate_world
from apris.cheops.infrastructure.simulation.cases import build_cases
from apris.cheops.infrastructure.ml.event_features_v2 import build_graph_matrix_from_events, GRAPH_FEATURE_COLUMNS

def run_grid():
    results = []
    
    # Grid of evasion parameters to explore
    # Similar to the original ladder.py that was lost, scaling evasion
    time_spreads = [12.0, 60.0, 240.0]
    funders = [1, 4, 8]
    
    for ts in time_spreads:
        for f in funders:
            print(f"Generating world: time_spread={ts}m, funders={f}...")
            # We scale down some counts to make the simulation reasonably fast for a script
            config = SimulationConfig(
                seed=42,
                days=45,
                salary_earners=200,
                fast_spenders=100,
                mule_networks=15,
                pyramids=10,
                evasion=EvasionKnobs(time_spread_minutes=ts, funders=f)
            )
            world = generate_world(config)
            cases = build_cases(world)
            
            event_groups = [case.events for case in cases]
            labels = [case.label for case in cases]
            
            df = build_graph_matrix_from_events(event_groups)
            
            res = {
                'time_spread': ts,
                'funders': f,
            }
            
            # Compute ROC-AUC for each feature
            for col in GRAPH_FEATURE_COLUMNS:
                try:
                    auc = roc_auc_score(labels, df[col])
                    # If feature is anti-correlated with fraud, invert the AUC
                    if auc < 0.5:
                        auc = 1.0 - auc
                    res[f'auc_{col}'] = round(auc, 3)
                except Exception as e:
                    res[f'auc_{col}'] = None
                    
            # Compute heuristic AUC
            # Simple average of hub, fanout and relay
            heuristic = (df['graph_hub_share'] + df['graph_fanout_share'] + df['graph_relay_share']) / 3.0
            res['auc_heuristic'] = round(roc_auc_score(labels, heuristic), 3)
            
            results.append(res)
            
    df_results = pd.DataFrame(results)
    print("\n--- Grid Search Results ---")
    print(df_results.to_markdown(index=False))
    return df_results

if __name__ == "__main__":
    run_grid()
