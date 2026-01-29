"""
Step 2: Collect Actual FDV from CoinGecko
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coingecko_client import CoinGeckoClient
from fdv_database import FDVDatabase


def main():
    print("=" * 60)
    print("Collect Actual FDV - Step 2")
    print("=" * 60)

    # Initialize
    client = CoinGeckoClient()
    db = FDVDatabase()

    # Get resolved events
    events = db.get_all_events(resolved_only=True)
    print(f"\nResolved events to process: {len(events)}")

    matched = 0
    failed = 0

    for event in events:
        slug = event['slug']
        token_name = event.get('token_name', '')
        token_symbol = event.get('token_symbol', '')

        print(f"\nProcessing: {token_name} ({token_symbol or 'N/A'})...")

        # Skip if already has FDV
        if event.get('actual_fdv'):
            print(f"  Already has FDV: ${event['actual_fdv']:,.0f}")
            matched += 1
            continue

        # Try to match to CoinGecko
        coin_id = client.match_token_to_coingecko(token_name, token_symbol)

        if not coin_id:
            print(f"  [X] Could not match to CoinGecko")
            failed += 1
            continue

        print(f"  Matched to: {coin_id}")

        # Get FDV data
        fdv_data = client.get_fdv_at_listing(coin_id)

        if fdv_data and fdv_data.get('fdv'):
            fdv = fdv_data['fdv']
            print(f"  FDV: ${fdv:,.0f}")

            # Update database
            db.update_event_fdv(slug, fdv)

            # Also save token metadata
            coin_data = client.get_coin_data(coin_id)
            if coin_data:
                db.insert_token_metadata({
                    'token_symbol': token_symbol or coin_data.symbol,
                    'token_name': token_name or coin_data.name,
                    'coingecko_id': coin_id,
                })

            matched += 1
        else:
            print(f"  [X] Could not get FDV data")
            failed += 1

        # Rate limiting
        time.sleep(1)

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)

    print(f"\nProcessed: {len(events)} events")
    print(f"  Matched with FDV: {matched}")
    print(f"  Failed to match: {failed}")

    # Show events with FDV
    events_with_fdv = db.get_resolved_events()
    print(f"\nEvents ready for backtest: {len(events_with_fdv)}")

    if events_with_fdv:
        print(f"\n{'='*60}")
        print("EVENTS WITH ACTUAL FDV")
        print("=" * 60)

        for event in events_with_fdv[:15]:
            print(f"\n• {event['token_name']} ({event.get('token_symbol', 'N/A')})")
            print(f"  Resolved bucket: {event.get('resolved_bucket_label', 'N/A')}")
            print(f"  Actual FDV: ${event['actual_fdv']:,.0f}")

    db.close()

    print(f"\n[OK] Done! Next step: python scripts/run_backtest.py")


if __name__ == "__main__":
    main()
