"""
Extract FDV Training Data from DropsTab

Creates training dataset for FDV prediction model:
- Historical FDV values
- Token categories (L1, L2, DeFi, etc.)
- Market cap comparisons
"""

import json
from pathlib import Path
from datetime import datetime

# Known token categories for Polymarket events
TOKEN_CATEGORIES = {
    # Layer 1s
    "L1": [
        "BTC", "ETH", "SOL", "ADA", "AVAX", "DOT", "ATOM", "NEAR",
        "APT", "SUI", "SEI", "TIA", "INJ", "FTM", "ALGO", "XTZ",
        "EGLD", "ICP", "HBAR", "TON", "KAS", "MINA", "CFX"
    ],
    # Layer 2s
    "L2": [
        "MATIC", "ARB", "OP", "MNT", "STRK", "ZK", "IMX", "METIS",
        "MANTA", "BLAST", "BERA", "SCROLL", "LINEA", "TAIKO", "MODE"
    ],
    # DeFi
    "DeFi": [
        "UNI", "AAVE", "MKR", "CRV", "COMP", "SNX", "YFI", "SUSHI",
        "1INCH", "BAL", "LDO", "RPL", "FXS", "CVX", "PENDLE", "GMX",
        "DYDX", "JUP", "RAY", "ORCA", "PYTH", "JTO", "W"
    ],
    # Infrastructure
    "Infra": [
        "LINK", "GRT", "FIL", "AR", "RENDER", "RNDR", "AKT", "THETA",
        "LPT", "ZRO", "PYTH", "WLD", "TAO"
    ],
    # Gaming/Metaverse
    "Gaming": [
        "AXS", "SAND", "MANA", "ENJ", "GALA", "IMX", "RONIN", "PIXEL",
        "PRIME", "BIGTIME", "MAGIC"
    ],
    # Meme (высокий риск)
    "Meme": [
        "DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "MEME", "COQ"
    ]
}

# Flatten for lookup
SYMBOL_TO_CATEGORY = {}
for category, symbols in TOKEN_CATEGORIES.items():
    for symbol in symbols:
        SYMBOL_TO_CATEGORY[symbol.upper()] = category


def load_coins():
    """Load coins from DropsTab data."""
    data_file = Path(__file__).parent.parent / "data" / "dropstab_coins.json"

    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)

    return data.get('coins', data)


def extract_training_data(coins: list) -> list:
    """Extract relevant tokens for FDV training."""
    training_data = []

    for coin in coins:
        symbol = (coin.get('symbol') or '').upper()
        fdv = coin.get('fullyDilutedValuation')
        mcap = coin.get('marketCap')

        if not fdv or fdv <= 0:
            continue

        # Determine category
        category = SYMBOL_TO_CATEGORY.get(symbol, 'Other')

        # Calculate FDV/MCap ratio (indicates unlock schedule)
        fdv_mcap_ratio = fdv / mcap if mcap and mcap > 0 else 1.0

        training_data.append({
            'symbol': symbol,
            'name': coin.get('name'),
            'category': category,
            'fdv': fdv,
            'fdv_billions': fdv / 1e9,
            'mcap': mcap,
            'mcap_billions': mcap / 1e9 if mcap else 0,
            'fdv_mcap_ratio': fdv_mcap_ratio,
            'rank': coin.get('rank'),
            'price': coin.get('price'),
            'circulating_supply': coin.get('circulatingSupply'),
            'total_supply': coin.get('totalSupply'),
            'ath_usd': coin.get('athUsd'),
            'ath_date': coin.get('athUsdDate'),
        })

    return training_data


def create_fdv_buckets_analysis(training_data: list) -> dict:
    """Analyze FDV distribution by category."""
    analysis = {}

    for category in TOKEN_CATEGORIES.keys():
        tokens = [t for t in training_data if t['category'] == category]

        if not tokens:
            continue

        fdvs = [t['fdv'] for t in tokens]

        analysis[category] = {
            'count': len(tokens),
            'tokens': [t['symbol'] for t in tokens],
            'fdv_min': min(fdvs),
            'fdv_max': max(fdvs),
            'fdv_median': sorted(fdvs)[len(fdvs)//2],
            'fdv_avg': sum(fdvs) / len(fdvs),
            # Bucket distribution
            'bucket_0_1B': len([f for f in fdvs if f < 1e9]),
            'bucket_1_5B': len([f for f in fdvs if 1e9 <= f < 5e9]),
            'bucket_5_10B': len([f for f in fdvs if 5e9 <= f < 10e9]),
            'bucket_10_20B': len([f for f in fdvs if 10e9 <= f < 20e9]),
            'bucket_20B_plus': len([f for f in fdvs if f >= 20e9]),
        }

    return analysis


def create_polymarket_events_data(training_data: list) -> list:
    """
    Create data format for Polymarket FDV events.
    Focus on tokens that are likely to have FDV markets.
    """
    # Filter to interesting tokens (L1, L2, DeFi, Infra with FDV 500M-50B)
    interesting_categories = ['L1', 'L2', 'DeFi', 'Infra']

    events = []
    for t in training_data:
        if t['category'] not in interesting_categories:
            continue
        if t['fdv'] < 5e8 or t['fdv'] > 50e9:  # 500M to 50B range
            continue

        # Create bucket structure based on current FDV
        fdv = t['fdv']

        # Generate buckets around current FDV
        buckets = []
        if fdv < 2e9:
            buckets = [
                {"label": "<$500M", "low": 0, "high": 5e8},
                {"label": "$500M-1B", "low": 5e8, "high": 1e9},
                {"label": "$1-2B", "low": 1e9, "high": 2e9},
                {"label": "$2-3B", "low": 2e9, "high": 3e9},
                {"label": "$3-5B", "low": 3e9, "high": 5e9},
                {"label": ">$5B", "low": 5e9, "high": None},
            ]
        elif fdv < 10e9:
            buckets = [
                {"label": "<$2B", "low": 0, "high": 2e9},
                {"label": "$2-4B", "low": 2e9, "high": 4e9},
                {"label": "$4-6B", "low": 4e9, "high": 6e9},
                {"label": "$6-8B", "low": 6e9, "high": 8e9},
                {"label": "$8-10B", "low": 8e9, "high": 10e9},
                {"label": ">$10B", "low": 10e9, "high": None},
            ]
        else:
            buckets = [
                {"label": "<$5B", "low": 0, "high": 5e9},
                {"label": "$5-10B", "low": 5e9, "high": 10e9},
                {"label": "$10-15B", "low": 10e9, "high": 15e9},
                {"label": "$15-20B", "low": 15e9, "high": 20e9},
                {"label": "$20-30B", "low": 20e9, "high": 30e9},
                {"label": ">$30B", "low": 30e9, "high": None},
            ]

        # Determine which bucket current FDV falls into
        winning_bucket = 0
        for i, b in enumerate(buckets):
            low = b['low']
            high = b['high'] if b['high'] else float('inf')
            if low <= fdv < high:
                winning_bucket = i
                break

        events.append({
            'symbol': t['symbol'],
            'name': t['name'],
            'category': t['category'],
            'current_fdv': fdv,
            'current_fdv_billions': t['fdv_billions'],
            'buckets': buckets,
            'current_bucket': winning_bucket,
            'rank': t['rank'],
            'fdv_mcap_ratio': t['fdv_mcap_ratio'],
        })

    return events


def main():
    print("=" * 60)
    print("FDV TRAINING DATA EXTRACTION")
    print("=" * 60)

    # Load coins
    print("\n[1] Loading DropsTab coins...")
    coins = load_coins()
    print(f"    Loaded {len(coins)} coins")

    # Extract training data
    print("\n[2] Extracting training data...")
    training_data = extract_training_data(coins)
    print(f"    Extracted {len(training_data)} tokens with FDV")

    # Categorized tokens
    categorized = [t for t in training_data if t['category'] != 'Other']
    print(f"    Categorized: {len(categorized)} tokens")

    # Analyze by category
    print("\n[3] Analyzing by category...")
    analysis = create_fdv_buckets_analysis(training_data)

    for cat, stats in analysis.items():
        print(f"\n    {cat}: {stats['count']} tokens")
        print(f"      FDV Range: ${stats['fdv_min']/1e9:.2f}B - ${stats['fdv_max']/1e9:.2f}B")
        print(f"      Median FDV: ${stats['fdv_median']/1e9:.2f}B")

    # Create Polymarket events format
    print("\n[4] Creating Polymarket events format...")
    events = create_polymarket_events_data(training_data)
    print(f"    Created {len(events)} potential FDV events")

    # Save all data
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Save training data
    training_file = output_dir / "fdv_training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump({
            'extracted_at': datetime.now().isoformat(),
            'total_tokens': len(training_data),
            'categorized_tokens': len(categorized),
            'tokens': training_data
        }, f, indent=2)
    print(f"\n[5] Saved: {training_file}")

    # Save category analysis
    analysis_file = output_dir / "fdv_category_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analyzed_at': datetime.now().isoformat(),
            'categories': analysis
        }, f, indent=2)
    print(f"    Saved: {analysis_file}")

    # Save Polymarket events format
    events_file = output_dir / "fdv_polymarket_events.json"
    with open(events_file, 'w', encoding='utf-8') as f:
        json.dump({
            'created_at': datetime.now().isoformat(),
            'event_count': len(events),
            'events': events
        }, f, indent=2)
    print(f"    Saved: {events_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nTotal tokens analyzed: {len(training_data)}")
    print(f"Categorized tokens: {len(categorized)}")
    print(f"Potential Polymarket events: {len(events)}")

    print("\nBy Category:")
    for cat in ['L1', 'L2', 'DeFi', 'Infra', 'Gaming', 'Meme']:
        count = len([t for t in training_data if t['category'] == cat])
        print(f"  {cat}: {count} tokens")

    print("\nTop 10 tokens by FDV (interesting categories):")
    interesting = [t for t in training_data if t['category'] in ['L1', 'L2', 'DeFi', 'Infra']]
    interesting.sort(key=lambda x: -x['fdv'])
    for t in interesting[:10]:
        print(f"  {t['symbol']:>8}: ${t['fdv_billions']:>6.2f}B ({t['category']})")


if __name__ == "__main__":
    main()
