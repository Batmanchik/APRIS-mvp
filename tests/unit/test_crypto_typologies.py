from apris.cheops.infrastructure.simulation.config import SimulationConfig
from apris.cheops.infrastructure.simulation.generator import generate_world
from apris.cheops.domain.contracts import map_events_to_typology_labels

def test_crypto_typologies() -> None:
    config = SimulationConfig(
        days=20, 
        crypto_layering=10, 
        crypto_traders=20, 
        mule_networks=0, 
        pyramids=0
    )
    world = generate_world(config)
    
    traders = [acc for acc, pop in world.populations.items() if pop == "crypto_trader"]
    
    traders_triggered = 0
    for trader in traders:
        events = [e for e in world.events if e.sender_id == trader or e.receiver_id == trader]
        labels = map_events_to_typology_labels(events)
        if any(labels.get(typ, 0) > 0 for typ in ["LEGAL_LAYERING", "LEGAL_TO_CRYPTO_BRIDGE", "CRYPTO_MIXING"]):
            traders_triggered += 1
            
    assert traders_triggered == 0, f"Honest traders triggered typologies!"

    networks = [net for net in world.networks if net.kind == "crypto_layering"]
    
    triggered_counts = {
        "LEGAL_LAYERING": 0,
        "LEGAL_TO_CRYPTO_BRIDGE": 0,
        "CRYPTO_MIXING": 0,
    }
    
    for net in networks:
        events = [e for e in world.events if e.sender_id in net.account_ids and e.receiver_id in net.account_ids]
        labels = map_events_to_typology_labels(events)
        
        for typ in triggered_counts.keys():
            if labels.get(typ, 0) > 0:
                triggered_counts[typ] += 1
                
    print(f"Trigger counts: {triggered_counts}")
    for typ, count in triggered_counts.items():
        assert count > 0, f"Typology {typ} did not trigger at all!"
