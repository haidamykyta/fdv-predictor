"""
Build Target Tokens Dataset
Объединяет данные из DropsTab, DefiLlama, и ручные данные для целевых токенов
"""

import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"

# Ручные данные по токенам (то что ты скинул)
MANUAL_DATA = {
    "zksync": {
        "name": "zkSync",
        "symbol": "ZK",
        "fdv_day1": 1.47e9,  # $1.47B
        "fundraising_total": 458e6,  # $458M
        "fundraising_rounds": 8,
        "investors": ["a16z", "Blockchain Capital", "Dragonfly Capital", "Lightspeed", "Variant"],
        "initial_unlock_pct": 17.48,
        "airdrop_pct": 17.51,
        "category": "L2",
    },
    "scroll": {
        "name": "Scroll",
        "symbol": "SCR",
        "fdv_day1": 1.5e9,  # $1.5B (neutral estimate)
        "fundraising_total": 50e6,  # $50M at $1.8B valuation
        "investors": ["Polychain Capital", "Binance Labs"],
        "initial_unlock_pct": 19,
        "airdrop_pct": 7,
        "category": "L2",
    },
    "linea": {
        "name": "Linea",
        "symbol": "LINEA",
        "fdv_day1": None,  # September 2025 launch
        "fundraising_total": 0,  # No VC
        "investors": ["Consensys", "Eigen Labs", "ENS Domains"],  # Consortium
        "initial_unlock_pct": 10,
        "total_supply": 72e9,
        "category": "L2",
    },
    "taiko": {
        "name": "Taiko",
        "symbol": "TAIKO",
        "fdv_day1": 2e9,  # $2B
        "fundraising_total": 37e6,  # $37M in 3 rounds
        "investors": ["Lightspeed Faction", "Hashed", "Generative Ventures", "Token Bay Capital",
                     "Wintermute Ventures", "OKX Ventures", "GSR", "Flow Traders", "Amber Group"],
        "category": "L2",
    },
    "berachain": {
        "name": "Berachain",
        "symbol": "BERA",
        "fdv_day1": 4.25e9,  # ~$4.25B at $8.5 pre-market
        "fundraising_total": 142e6,  # $142M-$211M
        "investors": ["Framework Ventures", "Brevan Howard", "Polychain Capital", "Tribe Capital",
                     "Hack VC", "Arrington Capital", "HashKey Capital", "Samsung NEXT", "Shima Capital"],
        "initial_unlock_pct": 17.7,
        "airdrop_pct": 15.75,
        "total_supply": 500e6,
        "category": "L1",
    },
    "monad": {
        "name": "Monad",
        "symbol": "MON",
        "fdv_day1": 2.5e9,  # ~$2.5B at $0.025 public sale
        "fundraising_total": 244e6 + 269e6,  # $244M + $269M token sale
        "valuation": 3e9,  # $3B post-investment
        "investors": ["Paradigm", "Dragonfly", "Coinbase Ventures", "Animoca Brands"],
        "initial_unlock_pct": 7.5,  # Public sale
        "category": "L1",
    },
    "hyperliquid": {
        "name": "Hyperliquid",
        "symbol": "HYPE",
        "fdv_day1": 4e9,  # $4B at launch, grew to $27B
        "fundraising_total": 0,  # Bootstrap, no VC
        "investors": [],  # No investors - founder funded
        "initial_unlock_pct": 31,  # Airdrop
        "airdrop_pct": 31,
        "category": "DeFi",
        "notes": "Bootstrapped through trading profits",
    },
    "etherfi": {
        "name": "ether.fi",
        "symbol": "ETHFI",
        "fdv_day1": 3e9,  # $3B
        "fundraising_total": 27e6 + 23e6,  # $27M + $23M Series A
        "investors": ["Bullish Capital", "CoinFund", "Arrington Capital", "OKX Ventures",
                     "Foresight Ventures", "Consensys", "Amber", "Selini", "Draper Dragon", "Bankless Ventures"],
        "initial_unlock_pct": 11.52,
        "category": "DeFi",
    },
    "renzo": {
        "name": "Renzo Protocol",
        "symbol": "REZ",
        "fdv_day1": None,  # No exact data
        "fundraising_total": 3.2e6,  # $3.2M seed at $25M valuation
        "seed_price": 0.0025,
        "investors": [],
        "initial_unlock_pct": 12,  # Airdrops
        "category": "DeFi",
    },
    "pendle": {
        "name": "Pendle",
        "symbol": "PENDLE",
        "fdv_day1": 35e6,  # $35M
        "fundraising_total": 3.7e6,  # $3.7M seed
        "investors": [],
        "initial_unlock_pct": 56,  # By July 2023
        "category": "DeFi",
    },
}


def load_json(filename: str) -> dict:
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def build_combined_dataset():
    """Combine all data sources into unified dataset."""

    # Load DropsTab data
    dropstab_coins = load_json("dropstab_coins.json").get("coins", [])
    dropstab_investors = load_json("dropstab_investors.json").get("investors", [])

    # Load DefiLlama data
    defillama_raises = load_json("defillama_raises.json").get("raises", [])

    # Build investor tier lookup from DropsTab
    investor_tiers = {}
    for inv in dropstab_investors:
        name = (inv.get("name") or "").lower()
        tier = (inv.get("tier") or "Tier 3").lower()
        if "tier 1" in tier:
            investor_tiers[name] = 1
        elif "tier 2" in tier:
            investor_tiers[name] = 2
        else:
            investor_tiers[name] = 3

    # Build coin lookup from DropsTab
    coin_lookup = {}
    for coin in dropstab_coins:
        symbol = (coin.get("symbol") or "").lower()
        name = (coin.get("name") or "").lower()
        coin_lookup[symbol] = coin
        coin_lookup[name] = coin

    # Combine data for each target token
    combined = []

    for key, manual in MANUAL_DATA.items():
        token_data = {
            "id": key,
            "name": manual["name"],
            "symbol": manual["symbol"],
            "category": manual.get("category"),

            # FDV data
            "fdv_day1_usd": manual.get("fdv_day1"),

            # Fundraising
            "total_raised_usd": manual.get("fundraising_total"),
            "valuation_usd": manual.get("valuation"),

            # Token distribution
            "initial_unlock_pct": manual.get("initial_unlock_pct"),
            "airdrop_pct": manual.get("airdrop_pct"),
            "total_supply": manual.get("total_supply"),

            # Investors
            "investors": manual.get("investors", []),
            "investor_tier": 3,  # Default

            # Current data from DropsTab
            "current_fdv_usd": None,
            "current_mcap_usd": None,
            "current_circ_pct": None,
            "rank": None,
        }

        # Get current data from DropsTab
        symbol_lower = manual["symbol"].lower()
        name_lower = manual["name"].lower()

        ds_coin = coin_lookup.get(symbol_lower) or coin_lookup.get(name_lower)
        if ds_coin:
            token_data["current_fdv_usd"] = ds_coin.get("fullyDilutedValuation")
            token_data["current_mcap_usd"] = ds_coin.get("marketCap")
            token_data["rank"] = ds_coin.get("rank")

            circ = ds_coin.get("circulatingSupply") or 0
            total = ds_coin.get("totalSupply") or 0
            if total > 0:
                token_data["current_circ_pct"] = round(circ / total * 100, 1)

        # Determine best investor tier
        for inv in manual.get("investors", []):
            inv_lower = inv.lower()
            for known_inv, tier in investor_tiers.items():
                if inv_lower in known_inv or known_inv in inv_lower:
                    token_data["investor_tier"] = min(token_data["investor_tier"], tier)
                    break

        combined.append(token_data)

    return combined


def main():
    print("=" * 60)
    print("Building Target Tokens Dataset")
    print("=" * 60)

    dataset = build_combined_dataset()

    # Save
    output = {
        "generated_at": datetime.now().isoformat(),
        "count": len(dataset),
        "tokens": dataset
    }

    output_path = DATA_DIR / "target_tokens_combined.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_path}")
    print(f"Tokens: {len(dataset)}")

    # Print summary
    print("\n" + "=" * 100)
    print("TARGET TOKENS SUMMARY:")
    print("=" * 100)
    print(f"{'Name':20} | {'Symbol':8} | {'FDV Day1':>12} | {'Raised':>12} | {'Inv Tier':>8} | {'Category':>10}")
    print("-" * 100)

    for t in dataset:
        fdv = t.get("fdv_day1_usd")
        raised = t.get("total_raised_usd")

        fdv_str = f"${fdv/1e9:.2f}B" if fdv and fdv >= 1e9 else (f"${fdv/1e6:.0f}M" if fdv else "N/A")
        raised_str = f"${raised/1e6:.0f}M" if raised else "$0"

        print(f"{t['name']:20} | {t['symbol']:8} | {fdv_str:>12} | {raised_str:>12} | Tier {t['investor_tier']:>4} | {t.get('category', 'N/A'):>10}")


if __name__ == "__main__":
    main()
