"""
Build Training Data for FDV Predictor
Объединяет данные из DefiLlama и CoinGecko для обучения модели
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"


def load_json(filename: str) -> dict:
    """Load JSON file."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def build_sector_benchmarks():
    """Build sector benchmarks from DefiLlama raises."""
    raises_data = load_json("defillama_raises.json")
    raises = raises_data.get("raises", [])

    by_category = defaultdict(list)

    for r in raises:
        category = r.get("category") or "Unknown"
        amount = r.get("amount")

        if amount and isinstance(amount, (int, float)) and amount > 0:
            by_category[category].append(amount)

    # Calculate stats
    benchmarks = {}
    for cat, amounts in by_category.items():
        if len(amounts) < 3:
            continue

        sorted_amounts = sorted(amounts)
        n = len(sorted_amounts)

        benchmarks[cat] = {
            "count": n,
            "total_raised_m": sum(amounts),
            "avg_raise_m": sum(amounts) / n,
            "median_raise_m": sorted_amounts[n // 2],
            "min_m": sorted_amounts[0],
            "max_m": sorted_amounts[-1],
            "p25_m": sorted_amounts[n // 4],
            "p75_m": sorted_amounts[3 * n // 4],
        }

    return benchmarks


def build_investor_tiers():
    """Build investor tier data from DefiLlama."""
    raises_data = load_json("defillama_raises.json")
    raises = raises_data.get("raises", [])

    investor_stats = defaultdict(lambda: {"rounds": 0, "total_raised": 0, "categories": set()})

    for r in raises:
        amount = r.get("amount") or 0
        category = r.get("category") or "Unknown"

        # Lead investors
        for inv in r.get("leadInvestors", []):
            investor_stats[inv]["rounds"] += 1
            investor_stats[inv]["total_raised"] += amount
            investor_stats[inv]["categories"].add(category)

        # Other investors (half weight)
        for inv in r.get("otherInvestors", []):
            investor_stats[inv]["rounds"] += 0.5
            investor_stats[inv]["total_raised"] += amount * 0.5
            investor_stats[inv]["categories"].add(category)

    # Convert to list and sort
    investors = []
    for name, stats in investor_stats.items():
        if stats["rounds"] >= 3:  # At least 3 rounds
            investors.append({
                "name": name,
                "rounds": int(stats["rounds"]),
                "total_raised_m": round(stats["total_raised"], 1),
                "avg_deal_m": round(stats["total_raised"] / stats["rounds"], 1),
                "categories": list(stats["categories"])[:5],
            })

    # Sort by total raised
    investors.sort(key=lambda x: x["total_raised_m"], reverse=True)

    # Assign tiers
    tier1_cutoff = 1000  # $1B+ total
    tier2_cutoff = 100   # $100M+ total

    for inv in investors:
        if inv["total_raised_m"] >= tier1_cutoff:
            inv["tier"] = 1
        elif inv["total_raised_m"] >= tier2_cutoff:
            inv["tier"] = 2
        else:
            inv["tier"] = 3

    return investors[:100]  # Top 100


def build_fdv_training_set():
    """Build training set from CoinGecko data."""
    coins_data = load_json("coingecko_top_coins.json")
    coins = coins_data.get("coins", [])

    training_data = []

    for coin in coins:
        fdv = coin.get("fdv")
        mcap = coin.get("market_cap")

        if not fdv or not mcap:
            continue

        # Calculate metrics
        circulating = coin.get("circulating_supply") or 0
        total = coin.get("total_supply") or 0
        max_supply = coin.get("max_supply")

        if circulating > 0 and total > 0:
            circulating_ratio = circulating / total
        else:
            circulating_ratio = 1.0

        training_data.append({
            "symbol": coin.get("symbol"),
            "name": coin.get("name"),
            "fdv": fdv,
            "market_cap": mcap,
            "fdv_mcap_ratio": fdv / mcap if mcap > 0 else 1.0,
            "circulating_ratio": circulating_ratio,
            "price": coin.get("price"),
            "ath": coin.get("ath"),
            "rank": coin.get("rank"),
        })

    return training_data


def build_recent_launches():
    """Build list of recent token launches for backtesting."""
    recent_data = load_json("defillama_recent_raises.json")
    tokens = recent_data.get("tokens", [])

    launches = []

    for t in tokens:
        amount = t.get("amount") or 0
        if amount < 1:  # Skip tiny rounds
            continue

        launches.append({
            "name": t.get("name"),
            "category": t.get("category"),
            "round": t.get("round"),
            "raised_m": amount,
            "date": t.get("date"),
            "lead_investors": t.get("lead_investors", []),
            "chains": t.get("chains", []),
        })

    return launches


def main():
    print("=" * 60)
    print("Building Training Data for FDV Predictor")
    print("=" * 60)

    # 1. Sector benchmarks
    print("\n1. Building sector benchmarks...")
    benchmarks = build_sector_benchmarks()
    with open(DATA_DIR / "sector_benchmarks_real.json", 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "source": "DefiLlama",
            "sectors": benchmarks
        }, f, indent=2)
    print(f"   Saved {len(benchmarks)} sectors")

    # 2. Investor tiers
    print("\n2. Building investor tiers...")
    investors = build_investor_tiers()
    with open(DATA_DIR / "investor_tiers_real.json", 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "source": "DefiLlama",
            "investors": investors
        }, f, indent=2)
    tier1 = len([i for i in investors if i["tier"] == 1])
    tier2 = len([i for i in investors if i["tier"] == 2])
    print(f"   Saved {len(investors)} investors (Tier1: {tier1}, Tier2: {tier2})")

    # 3. FDV training data
    print("\n3. Building FDV training data...")
    training = build_fdv_training_set()
    with open(DATA_DIR / "fdv_training_data.json", 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "source": "CoinGecko",
            "count": len(training),
            "tokens": training
        }, f, indent=2)
    print(f"   Saved {len(training)} tokens with FDV data")

    # 4. Recent launches
    print("\n4. Building recent launches...")
    launches = build_recent_launches()
    with open(DATA_DIR / "recent_launches.json", 'w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "source": "DefiLlama",
            "count": len(launches),
            "launches": launches
        }, f, indent=2)
    print(f"   Saved {len(launches)} recent launches")

    print("\n" + "=" * 60)
    print("Done! Training data ready in knowledge_base/data/")
    print("=" * 60)


if __name__ == "__main__":
    main()
