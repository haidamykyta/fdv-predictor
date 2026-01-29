"""
Add newly discovered FDV events to the database
"""
import sys
import time
import requests
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fdv_database import FDVDatabase
from fdv_event_discovery import FDVEventDiscovery

# New FDV events discovered
NEW_FDV_SLUGS = [
    # Active FDV events
    'megaeth-market-cap-fdv-one-day-after-launch',
    'opensea-fdv-above-one-day-after-launch',
    'edgex-fdv-above-one-day-after-launch',
    'based-fdv-above-one-day-after-launch',
    'trove-fdv-above-one-day-after-launch',
    'opinion-fdv-above-one-day-after-launch',
    'infinex-fdv-above-one-day-after-launch',
    'zama-fdv-above-one-day-after-launch',
    'spacex-ipo-closing-market-cap',

    # Price prediction events (also multi-bucket)
    'what-price-will-bitcoin-hit-in-january-2026',
    'what-price-will-ethereum-hit-in-january-2026',
    'what-price-will-solana-hit-in-january-2026',
    'what-price-will-xrp-hit-in-january-2026',

    # Token launch events
    'will-metamask-launch-a-token-in-2025',
    'will-base-launch-a-token-in-2025-341',
]

GAMMA_API = "https://gamma-api.polymarket.com"


def parse_fdv_range(text):
    """Parse FDV range from bucket text."""
    import re
    text = text.lower()

    # Pattern: "$X-Y billion/million" or "$Xb-$Yb"
    range_match = re.search(r'\$?([\d.]+)\s*[mb]?\s*-\s*\$?([\d.]+)\s*([mb])?', text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        unit = range_match.group(3) or ''

        mult = 1e9 if 'b' in unit or 'billion' in text else 1e6 if 'm' in unit or 'million' in text else 1e9
        return low * mult, high * mult

    # Pattern: "above $X billion"
    above_match = re.search(r'above\s*\$?([\d.]+)\s*([mb])?', text)
    if above_match:
        low = float(above_match.group(1))
        unit = above_match.group(2) or ''
        mult = 1e9 if 'b' in unit else 1e6 if 'm' in unit else 1e9
        return low * mult, None

    # Pattern: "below $X billion" or "<$X"
    below_match = re.search(r'(?:below|under|<)\s*\$?([\d.]+)\s*([mb])?', text)
    if below_match:
        high = float(below_match.group(1))
        unit = below_match.group(2) or ''
        mult = 1e9 if 'b' in unit else 1e6 if 'm' in unit else 1e9
        return 0, high * mult

    return None, None


def fetch_event_details(slug):
    """Fetch event details from Gamma API."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={'slug': slug},
            timeout=30
        )
        events = resp.json()
        if events and len(events) > 0:
            return events[0]
    except Exception as e:
        print(f"  Error: {e}")
    return None


def extract_token_name(title):
    """Extract token name from event title."""
    import re

    # Common patterns
    # "MegaETH market cap" -> "MegaETH"
    # "OpenSea FDV above" -> "OpenSea"
    # "What price will Bitcoin hit" -> "Bitcoin"

    patterns = [
        r'^(\w+)\s+(?:market cap|fdv|FDV)',
        r'^(\w+)\s+FDV',
        r'price will (\w+) hit',
        r'^Will (\w+) launch',
    ]

    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(1)

    # Fallback: first capitalized word
    words = title.split()
    for word in words:
        if word[0].isupper() and len(word) > 2:
            return word.strip('?!.,')

    return title.split()[0] if title else 'Unknown'


def main():
    print("=" * 60)
    print("ADD NEW FDV EVENTS TO DATABASE")
    print("=" * 60)

    db = FDVDatabase()

    # Get existing events
    existing = db.get_all_events()
    existing_slugs = {e['slug'] for e in existing}
    print(f"\nExisting events: {len(existing_slugs)}")

    added = 0
    skipped = 0

    for slug in NEW_FDV_SLUGS:
        print(f"\n{slug}")

        if slug in existing_slugs:
            print("  [SKIP] Already in database")
            skipped += 1
            continue

        # Fetch event details
        event_data = fetch_event_details(slug)
        if not event_data:
            print("  [X] Could not fetch event")
            continue

        title = event_data.get('title', '')
        markets = event_data.get('markets', [])
        is_closed = event_data.get('closed', False)

        print(f"  Title: {title}")
        print(f"  Markets: {len(markets)} | Closed: {is_closed}")

        if len(markets) < 3:
            print("  [SKIP] Not enough markets (need 3+)")
            skipped += 1
            continue

        # Extract token name
        token_name = extract_token_name(title)
        print(f"  Token: {token_name}")

        # Parse buckets
        buckets = []
        token_map = {}

        for idx, market in enumerate(markets):
            question = market.get('question', '') or market.get('groupItemTitle', '')
            low, high = parse_fdv_range(question)

            # Get token IDs
            clob_tokens_raw = market.get('clobTokenIds', [])
            if isinstance(clob_tokens_raw, str):
                try:
                    clob_tokens = json.loads(clob_tokens_raw)
                except:
                    clob_tokens = []
            else:
                clob_tokens = clob_tokens_raw or []

            token_id = clob_tokens[0] if clob_tokens else None

            bucket = {
                'index': idx,
                'label': question,
                'low': low,
                'high': high,
                'token_id': token_id,
            }
            buckets.append(bucket)

            if token_id:
                token_map[token_id] = {
                    'bucket_idx': idx,
                    'bucket_label': question,
                    'low': low,
                    'high': high,
                }

        # Determine market type
        market_type = 'ipo' if 'ipo' in slug.lower() or 'spacex' in slug.lower() else 'crypto'

        # Insert into database
        event_record = {
            'slug': slug,
            'title': title,
            'token_name': token_name,
            'token_symbol': token_name.upper()[:10] if token_name else None,
            'condition_id': event_data.get('conditionId'),
            'buckets_json': json.dumps(buckets),
            'start_date': event_data.get('startDate') or event_data.get('createdAt'),
            'end_date': event_data.get('endDate'),
            'is_resolved': is_closed,
            'total_volume': event_data.get('volume', 0),
            'token_map_json': json.dumps(token_map),
        }

        db.insert_event(event_record)
        print(f"  [OK] Added to database")
        added += 1

        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Added: {added}")
    print(f"Skipped: {skipped}")
    print(f"Total events now: {len(existing_slugs) + added}")

    # Show all events
    all_events = db.get_all_events()
    print(f"\nAll FDV events in database:")
    for e in all_events:
        status = '[X]' if e.get('is_resolved') else '[O]'
        buckets = json.loads(e.get('buckets_json', '[]'))
        print(f"  {status} {len(buckets)} mkts | {e['token_name']}: {e['title'][:40]}")

    db.close()


if __name__ == "__main__":
    main()
