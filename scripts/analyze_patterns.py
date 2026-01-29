"""
Analyze price movement patterns in FDV markets
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from fdv_database import FDVDatabase


def analyze_price_evolution():
    """Analyze how prices evolve over time."""
    print("=" * 60)
    print("PRICE EVOLUTION ANALYSIS")
    print("=" * 60)

    db = FDVDatabase()

    # Get resolved events
    events = db.get_all_events()
    resolved = [e for e in events if e.get('is_resolved') and e.get('actual_fdv')]

    print(f"\nAnalyzing {len(resolved)} resolved events")

    for event in resolved:
        slug = event['slug']
        token = event.get('token_name', 'Unknown')
        actual_fdv = event.get('actual_fdv', 0)
        buckets = json.loads(event.get('buckets_json', '[]'))

        print(f"\n{'='*60}")
        print(f"{token} ({slug})")
        print(f"Actual FDV: ${actual_fdv/1e9:.2f}B")
        print("=" * 60)

        # Find winning bucket
        winning_idx = None
        for i, b in enumerate(buckets):
            low = b.get('low', 0) or 0
            high = b.get('high')
            if high is None:
                if actual_fdv >= low:
                    winning_idx = i
            else:
                if low <= actual_fdv < high:
                    winning_idx = i
                    break

        if winning_idx is None:
            winning_idx = len(buckets) - 1

        print(f"Winning bucket: #{winning_idx}")

        # Get price history
        snapshots = db.get_all_prices_for_event(slug)

        if not snapshots:
            print("  No price history available")
            continue

        # Group by timestamp
        by_time = defaultdict(dict)
        for s in snapshots:
            ts = s['timestamp']
            idx = s['bucket_idx']
            by_time[ts][idx] = s['price']

        timestamps = sorted(by_time.keys())
        if not timestamps:
            print("  No timestamps")
            continue

        # Analyze price at different time points
        total_duration = timestamps[-1] - timestamps[0]

        print(f"\nPrice history ({len(timestamps)} snapshots over {total_duration/3600:.1f} hours)")
        print("-" * 50)

        checkpoints = [0, 0.25, 0.5, 0.75, 0.9, 1.0]

        for cp in checkpoints:
            target_time = timestamps[0] + int(cp * total_duration)
            # Find closest timestamp
            closest_ts = min(timestamps, key=lambda t: abs(t - target_time))
            prices = by_time[closest_ts]

            # Find best bucket (highest price)
            if prices:
                best_idx = max(prices.keys(), key=lambda k: prices.get(k, 0))
                best_price = prices.get(best_idx, 0)
                winning_price = prices.get(winning_idx, 0)

                print(f"  {int(cp*100):3d}%: Winner #{winning_idx} @ {winning_price:.1%}, Best #{best_idx} @ {best_price:.1%}")

        # Calculate optimal betting time
        print("\nOptimal entry points:")
        best_ev = 0
        best_time = None

        for ts in timestamps:
            prices = by_time[ts]
            winning_price = prices.get(winning_idx, 0)

            if winning_price > 0 and winning_price < 1:
                # Expected value: win = 1/price, lose = -1
                ev = (1 / winning_price) - 1  # simplified, assuming 100% win probability
                if ev > best_ev:
                    best_ev = ev
                    best_time = ts
                    best_winning_price = winning_price

        if best_time:
            pct_into_market = (best_time - timestamps[0]) / total_duration if total_duration > 0 else 0
            print(f"  Best entry: {pct_into_market:.0%} into market")
            print(f"  Winning bucket price: {best_winning_price:.1%}")
            print(f"  Potential return: {1/best_winning_price:.1f}x")

    db.close()


def analyze_confidence_vs_accuracy():
    """Analyze relationship between market confidence and accuracy."""
    print("\n" + "=" * 60)
    print("CONFIDENCE VS ACCURACY ANALYSIS")
    print("=" * 60)

    db = FDVDatabase()

    events = db.get_all_events()
    resolved = [e for e in events if e.get('is_resolved') and e.get('actual_fdv')]

    results = []

    for event in resolved:
        slug = event['slug']
        token = event.get('token_name', 'Unknown')
        actual_fdv = event.get('actual_fdv', 0)
        buckets = json.loads(event.get('buckets_json', '[]'))

        # Find winning bucket
        winning_idx = None
        for i, b in enumerate(buckets):
            low = b.get('low', 0) or 0
            high = b.get('high')
            if high is None:
                if actual_fdv >= low:
                    winning_idx = i
            else:
                if low <= actual_fdv < high:
                    winning_idx = i
                    break

        if winning_idx is None:
            winning_idx = len(buckets) - 1

        # Get final prices
        snapshots = db.get_all_prices_for_event(slug)
        if not snapshots:
            continue

        # Get latest snapshot for each bucket
        latest = {}
        for s in snapshots:
            idx = s['bucket_idx']
            if idx not in latest or s['timestamp'] > latest[idx]['timestamp']:
                latest[idx] = s

        if not latest:
            continue

        # Find predicted bucket (highest price)
        predicted_idx = max(latest.keys(), key=lambda k: latest[k]['price'])
        predicted_conf = latest[predicted_idx]['price']
        winning_conf = latest.get(winning_idx, {}).get('price', 0)

        is_correct = predicted_idx == winning_idx

        results.append({
            'token': token,
            'predicted': predicted_idx,
            'actual': winning_idx,
            'correct': is_correct,
            'confidence': predicted_conf,
            'winning_price': winning_conf,
        })

        print(f"\n{token}:")
        print(f"  Predicted: #{predicted_idx} @ {predicted_conf:.1%}")
        print(f"  Actual: #{winning_idx} @ {winning_conf:.1%}")
        print(f"  Correct: {'YES' if is_correct else 'NO'}")

    # Summary
    print("\n" + "-" * 40)
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # Confidence buckets
    high_conf = [r for r in results if r['confidence'] >= 0.7]
    med_conf = [r for r in results if 0.4 <= r['confidence'] < 0.7]
    low_conf = [r for r in results if r['confidence'] < 0.4]

    print(f"\nBy confidence level:")
    if high_conf:
        acc = sum(1 for r in high_conf if r['correct']) / len(high_conf)
        print(f"  High (>=70%): {acc:.0%} accurate ({len(high_conf)} events)")
    if med_conf:
        acc = sum(1 for r in med_conf if r['correct']) / len(med_conf)
        print(f"  Medium (40-70%): {acc:.0%} accurate ({len(med_conf)} events)")
    if low_conf:
        acc = sum(1 for r in low_conf if r['correct']) / len(low_conf)
        print(f"  Low (<40%): {acc:.0%} accurate ({len(low_conf)} events)")

    db.close()


def find_value_bets():
    """Find underpriced buckets that won."""
    print("\n" + "=" * 60)
    print("VALUE BET ANALYSIS")
    print("=" * 60)

    db = FDVDatabase()

    events = db.get_all_events()
    resolved = [e for e in events if e.get('is_resolved') and e.get('actual_fdv')]

    for event in resolved:
        slug = event['slug']
        token = event.get('token_name', 'Unknown')
        actual_fdv = event.get('actual_fdv', 0)
        buckets = json.loads(event.get('buckets_json', '[]'))

        # Find winning bucket
        winning_idx = None
        for i, b in enumerate(buckets):
            low = b.get('low', 0) or 0
            high = b.get('high')
            if high is None:
                if actual_fdv >= low:
                    winning_idx = i
            else:
                if low <= actual_fdv < high:
                    winning_idx = i
                    break

        if winning_idx is None:
            winning_idx = len(buckets) - 1

        # Get price history
        snapshots = db.get_all_prices_for_event(slug)
        if not snapshots:
            continue

        # Find lowest price for winning bucket
        winning_prices = [s['price'] for s in snapshots if s['bucket_idx'] == winning_idx]

        if not winning_prices:
            continue

        min_price = min(winning_prices)
        max_price = max(winning_prices)
        avg_price = sum(winning_prices) / len(winning_prices)

        print(f"\n{token}:")
        print(f"  Winning bucket: #{winning_idx}")
        print(f"  Price range: {min_price:.1%} - {max_price:.1%}")
        print(f"  Average price: {avg_price:.1%}")
        print(f"  Best entry ROI: {1/min_price:.1f}x")
        print(f"  Worst entry ROI: {1/max_price:.1f}x")

    db.close()


if __name__ == "__main__":
    analyze_price_evolution()
    analyze_confidence_vs_accuracy()
    find_value_bets()
