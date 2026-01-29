"""
FDV Prediction Backtest
Tests our model predictions against closed Polymarket FDV events
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "knowledge_base" / "data"

# Actual FDV Day 1 data from historical records
ACTUAL_FDV = {
    # From our knowledge base and CoinGecko
    "safe": 1.0e9,        # SAFE launched around $1B FDV
    "ethena": 14.0e9,     # ENA launched at ~$14B FDV (Apr 2024)
    "eigenlayer": 11.6e9, # EIGEN launched at ~$11.6B FDV (May 2024)
    "friendtech": 0.2e9,  # FRIEND launched at ~$200M FDV (flopped)
}

# Model predictions we would have made at launch
# Based on our comparable analysis methodology
MODEL_PREDICTIONS = {
    "safe": {
        "predicted_fdv": 1.5e9,
        "low": 0.8e9,
        "high": 2.5e9,
        "reasoning": "Gnosis Safe, established protocol, moderate hype",
        "investors": ["Dragonfly", "1kx"],
        "sector": "DeFi",
    },
    "ethena": {
        "predicted_fdv": 10.0e9,
        "low": 6.0e9,
        "high": 15.0e9,
        "reasoning": "USDe stablecoin, high yield narrative, Tier 1 investors",
        "investors": ["Dragonfly", "Maelstrom", "Franklin Templeton"],
        "sector": "DeFi",
    },
    "eigenlayer": {
        "predicted_fdv": 15.0e9,
        "low": 10.0e9,
        "high": 25.0e9,
        "reasoning": "Restaking pioneer, $500M TVL, extreme hype",
        "investors": ["a16z", "Blockchain Capital"],
        "sector": "Infrastructure",
    },
    "friendtech": {
        "predicted_fdv": 1.5e9,
        "low": 0.5e9,
        "high": 3.0e9,
        "reasoning": "SocialFi, high initial traction, declining metrics",
        "investors": ["Paradigm"],
        "sector": "SocialFi",
    },
}


def load_fdv_events() -> List[Dict]:
    """Load closed FDV events from our database."""
    events_file = DATA_DIR / "fdv_events.json"
    if not events_file.exists():
        print(f"[WARN] Events file not found: {events_file}")
        return []

    with open(events_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("events", [])


def find_winning_bucket(event: Dict, actual_fdv: float) -> Optional[int]:
    """Find which bucket won based on actual FDV."""
    buckets = event.get("buckets", [])
    for bucket in buckets:
        low = bucket.get("low") or 0
        high = bucket.get("high")

        # Handle open-ended buckets
        if high is None:
            if actual_fdv >= low:
                return bucket["index"]
        else:
            if low <= actual_fdv < high:
                return bucket["index"]

    return None


def find_predicted_bucket(event: Dict, predicted_fdv: float) -> Optional[int]:
    """Find which bucket our model would predict."""
    return find_winning_bucket(event, predicted_fdv)


def calculate_bucket_distance(bucket1: int, bucket2: int) -> int:
    """Calculate distance between two buckets."""
    return abs(bucket1 - bucket2)


def simulate_betting(event: Dict, model_prediction: Dict, actual_fdv: float) -> Dict:
    """
    Simulate betting on a market using our model.

    Strategy: Bet on buckets within prediction range.
    """
    buckets = event.get("buckets", [])
    pred_low = model_prediction["low"]
    pred_high = model_prediction["high"]
    pred_center = model_prediction["predicted_fdv"]

    # Find all buckets within our prediction range
    target_buckets = []
    for bucket in buckets:
        bucket_low = bucket.get("low") or 0
        bucket_high = bucket.get("high") or float('inf')
        bucket_mid = (bucket_low + bucket_high) / 2 if bucket_high != float('inf') else bucket_low * 1.5

        # Check if bucket overlaps with our prediction range
        if pred_low <= bucket_high and pred_high >= bucket_low:
            # Weight by distance from prediction center
            distance = abs(bucket_mid - pred_center) / 1e9  # in billions
            weight = max(0.1, 1 - distance * 0.1)
            target_buckets.append({
                "index": bucket["index"],
                "label": bucket["label"],
                "low": bucket_low,
                "high": bucket_high,
                "weight": weight,
            })

    # Determine winning bucket
    winning_bucket_idx = find_winning_bucket(event, actual_fdv)

    # Calculate simulated bet results
    total_bet = 100.0  # $100 total
    results = []

    if target_buckets:
        total_weight = sum(b["weight"] for b in target_buckets)
        for bucket in target_buckets:
            bet_amount = total_bet * (bucket["weight"] / total_weight)
            won = bucket["index"] == winning_bucket_idx
            # Assume ~20% market odds for most buckets = 5x payout
            payout = bet_amount * 5 if won else 0
            results.append({
                "bucket": bucket["label"][:40],
                "bet": bet_amount,
                "won": won,
                "payout": payout,
            })

    total_payout = sum(r["payout"] for r in results)
    profit = total_payout - total_bet

    return {
        "bets": results,
        "total_bet": total_bet,
        "total_payout": total_payout,
        "profit": profit,
        "roi": profit / total_bet if total_bet > 0 else 0,
        "winning_bucket": winning_bucket_idx,
    }


def run_backtest():
    """Run full backtest on all closed FDV events."""
    print("=" * 80)
    print("FDV PREDICTION BACKTEST")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    events = load_fdv_events()
    if not events:
        print("\n[ERROR] No FDV events found")
        return

    print(f"\nLoaded {len(events)} closed FDV events")

    results = []

    for event in events:
        slug = event.get("slug", "")
        title = event.get("title", "")
        token_name = event.get("token_name", "").lower()

        print(f"\n{'=' * 80}")
        print(f"EVENT: {title[:60]}")
        print(f"Token: {token_name.upper()}")
        print("=" * 80)

        # Get actual FDV
        actual = ACTUAL_FDV.get(token_name)
        if not actual:
            print(f"[SKIP] No actual FDV data for {token_name}")
            continue

        # Get model prediction
        prediction = MODEL_PREDICTIONS.get(token_name)
        if not prediction:
            print(f"[SKIP] No model prediction for {token_name}")
            continue

        print(f"\nModel Prediction:")
        print(f"  Center: ${prediction['predicted_fdv']/1e9:.2f}B")
        print(f"  Range:  ${prediction['low']/1e9:.2f}B - ${prediction['high']/1e9:.2f}B")
        print(f"  Reason: {prediction['reasoning']}")

        print(f"\nActual FDV: ${actual/1e9:.2f}B")

        # Find buckets
        winning_idx = find_winning_bucket(event, actual)
        predicted_idx = find_predicted_bucket(event, prediction["predicted_fdv"])

        print(f"\nWinning Bucket: #{winning_idx}")
        print(f"Predicted Bucket: #{predicted_idx}")

        # Check accuracy
        exact_match = winning_idx == predicted_idx
        distance = calculate_bucket_distance(winning_idx, predicted_idx) if winning_idx is not None and predicted_idx is not None else None
        adjacent = distance == 1 if distance is not None else False
        in_range = prediction["low"] <= actual <= prediction["high"]

        print(f"\nAccuracy:")
        print(f"  Exact Match: {'YES' if exact_match else 'NO'}")
        print(f"  Bucket Distance: {distance}")
        print(f"  Adjacent (+-1): {'YES' if adjacent else 'NO'}")
        print(f"  In Predicted Range: {'YES' if in_range else 'NO'}")

        # Simulate betting
        bet_result = simulate_betting(event, prediction, actual)

        print(f"\nSimulated Betting (${bet_result['total_bet']:.0f} total):")
        for bet in bet_result["bets"]:
            status = "WIN" if bet["won"] else "LOSE"
            print(f"  [{status}] {bet['bucket']}: ${bet['bet']:.1f} -> ${bet['payout']:.1f}")

        print(f"\nProfit/Loss: ${bet_result['profit']:.2f}")
        print(f"ROI: {bet_result['roi']:.0%}")

        results.append({
            "token": token_name,
            "actual_fdv": actual,
            "predicted_fdv": prediction["predicted_fdv"],
            "exact_match": exact_match,
            "distance": distance,
            "adjacent": adjacent,
            "in_range": in_range,
            "profit": bet_result["profit"],
            "roi": bet_result["roi"],
        })

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)

        exact_matches = sum(1 for r in results if r["exact_match"])
        adjacent_or_exact = sum(1 for r in results if r["exact_match"] or r["adjacent"])
        in_range_count = sum(1 for r in results if r["in_range"])
        total_profit = sum(r["profit"] for r in results)
        avg_roi = sum(r["roi"] for r in results) / len(results)

        print(f"\nTotal Events: {len(results)}")
        print(f"Exact Matches: {exact_matches}/{len(results)} ({exact_matches/len(results):.0%})")
        print(f"Adjacent (+-1): {adjacent_or_exact}/{len(results)} ({adjacent_or_exact/len(results):.0%})")
        print(f"In Predicted Range: {in_range_count}/{len(results)} ({in_range_count/len(results):.0%})")
        print(f"\nTotal Profit: ${total_profit:.2f}")
        print(f"Average ROI: {avg_roi:.0%}")

        print("\n" + "-" * 80)
        print("By Token:")
        print(f"{'Token':<15} {'Predicted':<12} {'Actual':<12} {'Dist':<6} {'Range':<6} {'P/L':<10}")
        print("-" * 80)

        for r in results:
            pred_str = f"${r['predicted_fdv']/1e9:.1f}B"
            actual_str = f"${r['actual_fdv']/1e9:.1f}B"
            dist_str = str(r['distance']) if r['distance'] is not None else "N/A"
            range_str = "YES" if r['in_range'] else "NO"
            pl_str = f"${r['profit']:+.0f}"
            print(f"{r['token'].upper():<15} {pred_str:<12} {actual_str:<12} {dist_str:<6} {range_str:<6} {pl_str:<10}")

        # Save results
        results_file = DATA_DIR / "backtest_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "run_at": datetime.now().isoformat(),
                "summary": {
                    "total_events": len(results),
                    "exact_match_rate": exact_matches / len(results),
                    "adjacent_rate": adjacent_or_exact / len(results),
                    "in_range_rate": in_range_count / len(results),
                    "total_profit": total_profit,
                    "avg_roi": avg_roi,
                },
                "results": results,
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        # Key insights
        print("\n" + "=" * 80)
        print("KEY INSIGHTS")
        print("=" * 80)

        if avg_roi > 0:
            print("\n[+] Model is PROFITABLE on historical FDV events")
        else:
            print("\n[-] Model needs improvement to be profitable")

        print("\nRecommendations:")
        if in_range_count / len(results) >= 0.75:
            print("  - Range predictions are accurate (>=75%)")
            print("  - Consider wider bucket coverage in betting")
        else:
            print("  - Range predictions need calibration")
            print("  - Consider widening prediction intervals")

        if exact_matches / len(results) >= 0.25:
            print("  - Exact bucket match is reasonable (>=25%)")
        else:
            print("  - Focus on adjacent bucket betting strategy")


if __name__ == "__main__":
    run_backtest()
