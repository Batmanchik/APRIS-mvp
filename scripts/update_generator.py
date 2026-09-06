import sys
from datetime import timedelta

filepath = r'C:\Users\alibe\.gemini\antigravity\worktrees\CHEOPS-AI-ACTIVE-github\analyze_project_architect_role\src\apris\cheops\infrastructure\simulation\generator.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

crypto_traders_code = '''
def _gen_crypto_traders(b: _Builder) -> list[str]:
    """HARD NEGATIVE 4 — people who honestly buy and sell cryptocurrency.
    
    Without them, a model learns "crypto = fraud".
    """
    ids: list[str] = []
    exchange = b.new_account("EXC", account_type=TYPE_COMPANY, opened_days_ago=(100, 1000))
    b.world.populations[exchange] = "crypto_exchange"
    for _ in range(b.config.crypto_traders):
        account = b.new_account("ACC")
        wallet = b.new_account("CRY")
        b.world.populations[wallet] = "crypto_trader_wallet"
        ids.extend([account, wallet])
        # Random honest buying of crypto
        for _ in range(int(b.rng.integers(1, 6))):
            when = b.moment(int(b.rng.integers(0, b.config.days)), (8, 23))
            amount = b.uniform((10_000.0, 500_000.0))
            # Fiat to exchange
            b.emit(account, exchange, amount, when)
            # Exchange gives crypto (same value for simplicity)
            b.emit(exchange, wallet, amount, when + timedelta(minutes=int(b.rng.integers(1, 15))), channel="crypto", asset_type="crypto")
    return ids
'''

crypto_layering_code = '''
def _gen_crypto_layering(b: _Builder, index: int) -> SimulatedNetwork:
    """A scheme combining LEGAL_LAYERING, LEGAL_TO_CRYPTO_BRIDGE and CRYPTO_MIXING.
    
    Legal entrance -> 2-4 layers of transfers between front accounts -> bridge to crypto
    -> splitting between several crypto addresses.
    """
    # Legal entrance
    funder = b.new_account("FND", account_type=TYPE_COMPANY)
    layers = []
    num_layers = int(b.rng.integers(2, 5))
    prev_layer = [funder]
    all_accounts = [funder]
    
    start_day = int(b.rng.integers(0, max(1, b.config.days - 10)))
    current_time = b.moment(start_day, (9, 12))
    
    amount = b.uniform((500_000.0, 3_000_000.0))
    current_amounts = {funder: amount}
    
    # Layering
    for i in range(num_layers):
        next_layer_size = int(b.rng.integers(2, 5))
        next_layer = [b.new_account("LYR") for _ in range(next_layer_size)]
        all_accounts.extend(next_layer)
        next_amounts = {str(acc): 0.0 for acc in next_layer}
        
        current_time += timedelta(hours=float(b.rng.uniform(1, 12)))
        for src in prev_layer:
            src_amt = current_amounts[str(src)]
            if src_amt <= 0: continue
            
            # Split to next layer
            parts = int(b.rng.integers(1, min(4, next_layer_size + 1)))
            chosen_dsts = b.rng.choice(next_layer, size=parts, replace=False)
            split_amt = src_amt / parts
            for dst in chosen_dsts:
                b.emit(str(src), str(dst), split_amt, current_time + timedelta(minutes=float(b.rng.uniform(1, 30))))
                next_amounts[str(dst)] += split_amt
                
        prev_layer = next_layer
        current_amounts = next_amounts
        
    # Bridge to crypto and splitting
    exchange = b.new_account("EXC", account_type=TYPE_COMPANY)
    b.world.populations[exchange] = "crypto_exchange"
    
    crypto_targets = [b.new_account("CRY") for _ in range(int(b.rng.integers(4, 10)))]
    all_accounts.extend(crypto_targets)
    all_accounts.append(exchange)
    
    current_time += timedelta(hours=float(b.rng.uniform(1, 12)))
    
    for src in prev_layer:
        src_amt = current_amounts[str(src)]
        if src_amt <= 0: continue
        
        # Fiat to exchange
        bridge_time = current_time + timedelta(minutes=float(b.rng.uniform(1, 30)))
        b.emit(str(src), exchange, src_amt, bridge_time)
        
        # Exchange to crypto targets (mixing)
        parts = int(b.rng.integers(3, len(crypto_targets) + 1))
        chosen_crypto = b.rng.choice(crypto_targets, size=parts, replace=False)
        split_amt = src_amt / parts
        
        for dst in chosen_crypto:
            b.emit(exchange, str(dst), split_amt, bridge_time + timedelta(minutes=float(b.rng.uniform(5, 60))), channel="crypto", asset_type="crypto")
            
    return SimulatedNetwork(
        network_id=f"NET{index:04d}",
        kind="crypto_layering",
        scale="fast",
        account_ids=tuple(all_accounts),
        organizer_ids=(funder,),
    )
'''

# Replace placeholders
marker1 = "# ==========================================================================\n# Fraudulent structures"
if "def _gen_crypto_traders" not in content:
    content = content.replace(marker1, crypto_traders_code + "\n" + marker1)

marker2 = "# ==========================================================================\n# Entry point"
if "def _gen_crypto_layering" not in content:
    content = content.replace(marker2, crypto_layering_code + "\n" + marker2)

call_traders = 'b.mark(_gen_crypto_traders(b), "crypto_trader")'
if call_traders not in content:
    content = content.replace(
        'b.mark(_gen_marketplace_sellers(b, terminals), "marketplace_seller")',
        'b.mark(_gen_marketplace_sellers(b, terminals), "marketplace_seller")\n    ' + call_traders
    )

call_layering = '''    for _ in range(b.config.crypto_layering):
        network = _gen_crypto_layering(b, index)
        b.world.networks.append(network)
        b.mark(list(network.account_ids), "crypto_layering")
        index += 1'''

if 'for _ in range(b.config.crypto_layering)' not in content:
    content = content.replace("index = 1", "index = 1\n" + call_layering)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated generator.py')
