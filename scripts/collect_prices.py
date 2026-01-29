"""
Step 1.5: Collect Historical Prices from Polymarket
Uses both CLOB API and GraphQL Subgraph
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fdv_database import FDVDatabase
from fdv_price_collector import FDVPriceCollector

# Polymarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


def get_event_markets(event_slug: str) -> dict:
    """Fetch full event data with markets from Gamma API."""
    try:
        # First get the event
        resp = requests.get(f"{GAMMA_API}/events", params={"slug": event_slug}, timeout=30)
        resp.raise_for_status()
        events = resp.json()

        if events and len(events) > 0:
            return events[0]
        return None
    except Exception as e:
        print(f"Error fetching event {event_slug}: {e}")
        return None


def extract_token_ids(event_data: dict) -> dict:
    """Extract actual token IDs from event markets."""
    import json as json_module

    token_map = {}
    markets = event_data.get('markets', [])

    for idx, market in enumerate(markets):
        question = market.get('question', '') or market.get('groupItemTitle', '')
        clob_token_ids_raw = market.get('clobTokenIds', [])

        # clobTokenIds can be a JSON string or an actual array
        if isinstance(clob_token_ids_raw, str):
            try:
                clob_token_ids = json_module.loads(clob_token_ids_raw)
            except:
                clob_token_ids = []
        else:
            clob_token_ids = clob_token_ids_raw or []

        # clobTokenIds is usually [YES_token, NO_token]
        if clob_token_ids and len(clob_token_ids) >= 1:
            yes_token_id = clob_token_ids[0]  # YES token
            token_map[yes_token_id] = {
                'bucket_idx': idx,
                'bucket_label': question,
                'market_id': market.get('id'),
            }

            # Also add outcomes info
            outcome_prices = market.get('outcomePrices', [])
            if outcome_prices:
                try:
                    # outcomePrices can also be a JSON string
                    if isinstance(outcome_prices, str):
                        outcome_prices = json_module.loads(outcome_prices)
                    token_map[yes_token_id]['current_price'] = float(outcome_prices[0])
                except:
                    pass

    return token_map


def collect_clob_prices(token_id: str, bucket_idx: int, bucket_label: str) -> list:
    """Collect price history from CLOB API."""
    try:
        url = f"{CLOB_API}/prices-history"
        params = {
            'market': token_id,
            'interval': '1d',  # daily
            'fidelity': 100,   # up to 100 data points
        }

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        history = data.get('history', [])

        prices = []
        for point in history:
            prices.append({
                'timestamp': int(point.get('t', 0)),
                'bucket_idx': bucket_idx,
                'bucket_label': bucket_label,
                'price': float(point.get('p', 0)),
                'volume': None,
            })

        return prices

    except Exception as e:
        print(f"    CLOB API error: {e}")
        return []


# GraphQL endpoint for historical trades
GRAPHQL_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"


def collect_graphql_prices(token_ids: list, start_ts: int, end_ts: int) -> list:
    """Collect price history from GraphQL Subgraph."""
    from decimal import Decimal

    USDC_ASSET_ID = "0"
    AMT_DECIMALS = 6

    all_trades = []
    ids_literal = "[" + ",".join(f'"{t}"' for t in token_ids) + "]"

    print(f"    Querying GraphQL for {len(token_ids)} tokens...")

    # Fetch trades where token is makerAssetID or takerAssetID
    for asset_key in ["makerAssetID", "takerAssetID"]:
        last_ts = start_ts - 1

        for _ in range(50):  # Max 50 pages
            query = f"""
            query {{
              ordersMatchedEvents(
                first: 1000,
                where: {{
                  {asset_key}_in: {ids_literal},
                  timestamp_gte: "{start_ts}",
                  timestamp_lt: "{end_ts}",
                  timestamp_gt: "{last_ts}"
                }},
                orderBy: timestamp,
                orderDirection: asc
              ) {{
                id
                timestamp
                makerAssetID
                takerAssetID
                makerAmountFilled
                takerAmountFilled
              }}
            }}
            """

            try:
                resp = requests.post(GRAPHQL_URL, json={"query": query}, timeout=60)
                if resp.status_code != 200:
                    break

                data = resp.json()
                if "errors" in data:
                    print(f"    GraphQL error: {data['errors']}")
                    break

                page = data.get("data", {}).get("ordersMatchedEvents", [])
                if not page:
                    break

                all_trades.extend(page)
                last_ts = int(page[-1]["timestamp"])

                if len(page) < 1000:
                    break

                time.sleep(0.1)

            except Exception as e:
                print(f"    GraphQL error: {e}")
                break

    print(f"    Found {len(all_trades)} trades")

    # Convert trades to prices
    prices = []
    token_set = set(token_ids)

    for trade in all_trades:
        maker_id = str(trade["makerAssetID"])
        taker_id = str(trade["takerAssetID"])
        maker_amt = int(trade["makerAmountFilled"])
        taker_amt = int(trade["takerAmountFilled"])

        # Determine token and USDC amounts
        if maker_id == USDC_ASSET_ID and taker_id in token_set:
            token_id = taker_id
            usdc_raw = maker_amt
            token_raw = taker_amt
        elif taker_id == USDC_ASSET_ID and maker_id in token_set:
            token_id = maker_id
            usdc_raw = taker_amt
            token_raw = maker_amt
        else:
            continue

        if token_raw == 0:
            continue

        # Calculate price
        token_amt = Decimal(token_raw) / (Decimal(10) ** AMT_DECIMALS)
        usdc_amt = Decimal(usdc_raw) / (Decimal(10) ** AMT_DECIMALS)
        price = float(usdc_amt / token_amt) if token_amt > 0 else 0

        prices.append({
            'timestamp': int(trade["timestamp"]),
            'token_id': token_id,
            'price': price,
            'volume': float(usdc_amt),
        })

    return prices


def main():
    print("=" * 60)
    print("Collect Historical Prices - Step 1.5")
    print("=" * 60)

    db = FDVDatabase()

    # Get events
    events = db.get_all_events()
    print(f"\nEvents to process: {len(events)}")

    total_prices = 0

    for event in events:
        slug = event['slug']
        print(f"\n{'='*60}")
        print(f"Event: {slug}")
        print("=" * 60)

        # Fetch fresh event data from Gamma API
        event_data = get_event_markets(slug)
        if not event_data:
            print("  [X] Could not fetch event data")
            continue

        # Extract real token IDs
        token_map = extract_token_ids(event_data)
        print(f"  Found {len(token_map)} tokens with IDs")

        if not token_map:
            print("  [X] No token IDs found")
            continue

        # Collect prices for each token
        all_prices = []

        # First try CLOB API for each token
        clob_found = False
        for token_id, info in token_map.items():
            bucket_idx = info['bucket_idx']
            bucket_label = info['bucket_label']

            print(f"\n  Bucket {bucket_idx}: {bucket_label[:50]}...")
            print(f"    Token ID: {token_id[:30]}...")

            # Get prices from CLOB API
            prices = collect_clob_prices(token_id, bucket_idx, bucket_label)

            if prices:
                print(f"    Got {len(prices)} price points from CLOB")
                all_prices.extend(prices)
                clob_found = True
            else:
                print(f"    No CLOB history")

            time.sleep(0.3)

        # If CLOB API didn't work, try GraphQL for all tokens
        if not clob_found:
            print(f"\n  Trying GraphQL Subgraph...")

            # Get event date range
            start_date = event.get('start_date')
            end_date = event.get('end_date')

            try:
                from datetime import datetime
                if start_date:
                    start_ts = int(datetime.fromisoformat(start_date.replace('Z', '+00:00')).timestamp())
                else:
                    start_ts = int(datetime(2024, 1, 1).timestamp())

                if end_date:
                    end_ts = int(datetime.fromisoformat(end_date.replace('Z', '+00:00')).timestamp())
                else:
                    end_ts = int(datetime.now().timestamp())

                # Add buffer
                start_ts -= 30 * 24 * 3600  # 30 days before
                end_ts += 7 * 24 * 3600     # 7 days after

            except Exception as e:
                print(f"    Date parsing error: {e}")
                start_ts = int(datetime(2024, 1, 1).timestamp())
                end_ts = int(datetime.now().timestamp())

            # Collect from GraphQL
            token_ids = list(token_map.keys())
            graphql_prices = collect_graphql_prices(token_ids, start_ts, end_ts)

            if graphql_prices:
                # Map token_id to bucket info
                for price in graphql_prices:
                    tid = price['token_id']
                    if tid in token_map:
                        info = token_map[tid]
                        all_prices.append({
                            'timestamp': price['timestamp'],
                            'bucket_idx': info['bucket_idx'],
                            'bucket_label': info['bucket_label'],
                            'price': price['price'],
                            'volume': price['volume'],
                        })

                print(f"    Got {len(all_prices)} prices from GraphQL")

        # Save to database using batch insert
        if all_prices:
            snapshots = [{
                'event_slug': slug,
                'timestamp': price['timestamp'],
                'bucket_idx': price['bucket_idx'],
                'bucket_label': price['bucket_label'],
                'price': price['price'],
                'volume': price.get('volume'),
            } for price in all_prices]

            saved = db.insert_price_snapshots_batch(snapshots)
            print(f"\n  [OK] Saved {saved} price points to database")
            total_prices += saved
        else:
            print(f"\n  No prices collected")

        # Update event's token_map in database
        import json as json_module
        db.conn.execute(
            "UPDATE fdv_events SET token_map_json = ? WHERE slug = ?",
            [json_module.dumps(token_map), slug]
        )
        db.conn.commit()

        time.sleep(1)  # Rate limiting between events

    # Summary
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)

    stats = db.get_stats()
    print(f"\nTotal price snapshots: {stats['price_snapshots']}")
    print(f"Events processed: {len(events)}")

    db.close()

    print(f"\n[OK] Done! Now re-run: python scripts/run_backtest.py")


if __name__ == "__main__":
    main()
