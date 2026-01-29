"""
Collect current prices for active FDV events
"""
import sys
import time
import requests
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from fdv_database import FDVDatabase

GAMMA_API = "https://gamma-api.polymarket.com"


def fetch_event_prices(slug):
    """Fetch current prices for an event."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={'slug': slug},
            timeout=30
        )
        events = resp.json()
        if not events:
            return None

        event = events[0]
        markets = event.get('markets', [])

        prices = []
        for idx, market in enumerate(markets):
            question = market.get('question', '') or market.get('groupItemTitle', '')

            # Parse outcome prices
            outcome_prices_raw = market.get('outcomePrices', [])
            if isinstance(outcome_prices_raw, str):
                try:
                    outcome_prices = json.loads(outcome_prices_raw)
                except:
                    outcome_prices = []
            else:
                outcome_prices = outcome_prices_raw or []

            price = float(outcome_prices[0]) if outcome_prices else 0

            # Get token ID
            clob_tokens_raw = market.get('clobTokenIds', [])
            if isinstance(clob_tokens_raw, str):
                try:
                    clob_tokens = json.loads(clob_tokens_raw)
                except:
                    clob_tokens = []
            else:
                clob_tokens = clob_tokens_raw or []

            token_id = clob_tokens[0] if clob_tokens else None

            prices.append({
                'bucket_idx': idx,
                'bucket_label': question,
                'price': price,
                'token_id': token_id,
            })

        return prices
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("=" * 60)
    print("COLLECT CURRENT PRICES FOR FDV EVENTS")
    print("=" * 60)

    db = FDVDatabase()
    events = db.get_all_events()

    # Filter active events
    active_events = [e for e in events if not e.get('is_resolved')]
    print(f"\nActive events: {len(active_events)}")

    timestamp = int(datetime.now().timestamp())
    total_snapshots = 0

    for event in active_events:
        slug = event['slug']
        token_name = event.get('token_name', 'Unknown')

        print(f"\n{token_name} ({slug})")

        prices = fetch_event_prices(slug)
        if not prices:
            print("  [X] Could not fetch prices")
            continue

        print(f"  Buckets: {len(prices)}")

        # Show top predictions
        sorted_prices = sorted(prices, key=lambda x: x['price'], reverse=True)
        print("  Top predictions:")
        for p in sorted_prices[:3]:
            print(f"    {p['price']:.1%} - {p['bucket_label'][:50]}")

        # Save to database using batch insert
        snapshots = [{
            'event_slug': slug,
            'timestamp': timestamp,
            'bucket_idx': p['bucket_idx'],
            'bucket_label': p['bucket_label'],
            'price': p['price'],
            'volume': None,
        } for p in prices]

        saved = db.insert_price_snapshots_batch(snapshots)
        total_snapshots += saved

        time.sleep(0.3)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Events processed: {len(active_events)}")
    print(f"Price snapshots saved: {total_snapshots}")

    stats = db.get_stats()
    print(f"Total price snapshots in DB: {stats['price_snapshots']}")

    db.close()


if __name__ == "__main__":
    main()
