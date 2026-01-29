"""
Search for more FDV events on Polymarket
Extended search with pagination and broader keywords
"""

import requests
import re
import time


def fetch_all_events():
    """Fetch all events with pagination."""
    all_events = []

    # Fetch active events
    print("Fetching active events...")
    resp = requests.get(
        'https://gamma-api.polymarket.com/events',
        params={'limit': 500, 'active': 'true'},
        timeout=60
    )
    active = resp.json()
    all_events.extend(active)
    print(f"  Active: {len(active)}")

    time.sleep(0.5)

    # Fetch closed events with pagination
    print("Fetching closed events...")
    offset = 0
    while True:
        resp = requests.get(
            'https://gamma-api.polymarket.com/events',
            params={'limit': 500, 'closed': 'true', 'offset': offset},
            timeout=60
        )
        batch = resp.json()
        if not batch:
            break
        all_events.extend(batch)
        print(f"  Closed batch {offset}: {len(batch)}")
        offset += 500
        if len(batch) < 500:
            break
        time.sleep(0.5)

    return all_events


def search_fdv_events():
    print("=" * 60)
    print("EXTENDED FDV EVENT SEARCH")
    print("=" * 60)

    events = fetch_all_events()
    print(f"\nTotal events fetched: {len(events)}")

    # Strong FDV indicators (high confidence)
    fdv_strong = [
        'fdv', 'fully diluted', 'tge', 'token generation',
        'day after launch', 'day after listing', 'day after airdrop',
        'week after launch', 'one day after',
    ]

    # Medium FDV indicators
    fdv_medium = [
        'market cap', 'mcap', 'valuation', 'listing price',
        'launch price', 'ipo price', 'opening price',
    ]

    # Known crypto tokens for FDV events
    known_tokens = [
        'eigenlayer', 'ethena', 'safe', 'friend.tech', 'friendtech',
        'jupiter', 'jup', 'starknet', 'strk', 'celestia', 'tia',
        'dymension', 'dym', 'manta', 'alt layer', 'altlayer',
        'pyth', 'wormhole', 'portal', 'ondo', 'zksync', 'scroll',
        'linea', 'blast', 'mode', 'merlin', 'renzo', 'eigenpie',
        'ether.fi', 'etherfi', 'pendle', 'layerzero', 'zro',
        'monad', 'berachain', 'movement', 'sui', 'aptos', 'sei',
        'injective', 'osmosis', 'axelar', 'saga', 'omni',
    ]

    # Exclude keywords
    exclude_keywords = [
        'box office', 'movie', 'film', 'gross', 'opening weekend',
        'election', 'president', 'vote', 'trump', 'biden', 'democrat', 'republican',
        'covid', 'coronavirus', 'vaccine',
        'sports', 'nfl', 'nba', 'mlb', 'nhl', 'soccer', 'football',
        'super bowl', 'world cup', 'olympics', 'championship',
        'weather', 'temperature', 'hurricane',
        'emmy', 'grammy', 'oscar', 'golden globe',
    ]

    fdv_events = []

    for e in events:
        title = e.get('title', '').lower()
        desc = e.get('description', '').lower()
        slug = e.get('slug', '').lower()
        text = f"{title} {desc} {slug}"
        markets = e.get('markets', [])

        # Skip excluded
        if any(kw in text for kw in exclude_keywords):
            continue

        # Check indicators
        has_strong = any(kw in text for kw in fdv_strong)
        has_medium = any(kw in text for kw in fdv_medium)
        has_token = any(kw in text for kw in known_tokens)
        has_value = bool(re.search(r'\$\d+.*[MBKmb]|billion|million', text))
        has_crypto = any(kw in text for kw in ['token', 'crypto', 'airdrop', 'chain', 'defi', 'layer'])

        # Multi-bucket check (3+ markets = likely FDV buckets)
        is_multi_bucket = len(markets) >= 3

        # Score the event
        score = 0
        if has_strong:
            score += 3
        if has_medium and (has_crypto or has_token):
            score += 2
        if has_token:
            score += 2
        if has_value:
            score += 1
        if is_multi_bucket:
            score += 2

        if score >= 3:
            fdv_events.append({
                'slug': e.get('slug'),
                'title': e.get('title'),
                'closed': e.get('closed', False),
                'markets': len(markets),
                'volume': e.get('volume', 0),
                'score': score,
                'has_buckets': is_multi_bucket,
            })

    # Print found events
    print(f"\nFound {len(fdv_events)} potential FDV events:")
    print("-" * 80)

    # Sort by score then volume
    fdv_events.sort(key=lambda x: (x['score'], x['volume']), reverse=True)

    # Separate multi-bucket and single-market events
    multi_bucket = [e for e in fdv_events if e['has_buckets']]
    single = [e for e in fdv_events if not e['has_buckets']]

    print(f"\n=== MULTI-BUCKET FDV EVENTS ({len(multi_bucket)}) ===")
    for ev in multi_bucket[:20]:
        status = '[X]' if ev['closed'] else '[O]'
        vol = f"${ev['volume']:,.0f}" if ev['volume'] else "N/A"
        print(f"\n{status} {ev['title'][:65]}")
        print(f"  Slug: {ev['slug']}")
        print(f"  Markets: {ev['markets']} | Volume: {vol} | Score: {ev['score']}")

    print(f"\n=== SINGLE-MARKET FDV EVENTS ({len(single)}) ===")
    for ev in single[:10]:
        status = '[X]' if ev['closed'] else '[O]'
        vol = f"${ev['volume']:,.0f}" if ev['volume'] else "N/A"
        print(f"{status} {ev['title'][:60]} | {vol}")

    return fdv_events


def search_multi_bucket_events():
    """Search for events with multiple markets (bucket-style)."""
    print("\n" + "=" * 60)
    print("ALL MULTI-BUCKET CRYPTO EVENTS")
    print("=" * 60)

    events = fetch_all_events()
    print(f"Total events: {len(events)}")

    # Skip patterns
    skip = [
        'election', 'oscar', 'super bowl', 'mlb', 'nba', 'nfl', 'ncaa',
        'world cup', 'grammy', 'emmy', 'president', 'congress', 'senate',
        'box office', 'movie', 'film',
    ]

    # Crypto indicators
    crypto_words = [
        'token', 'crypto', 'coin', 'chain', 'defi', 'airdrop', 'fdv',
        'market cap', 'listing', 'launch', 'tge', 'eth', 'btc', 'sol',
    ]

    multi = []
    for e in events:
        markets = e.get('markets', [])
        if len(markets) < 3:  # Lowered from 5 to 3
            continue

        title = e.get('title', '').lower()
        desc = e.get('description', '').lower()
        text = title + ' ' + desc

        if any(x in text for x in skip):
            continue

        # Check if crypto-related
        is_crypto = any(x in text for x in crypto_words)
        has_value = bool(re.search(r'\$\d+|billion|million', text))

        multi.append({
            'slug': e.get('slug'),
            'title': e.get('title'),
            'markets': len(markets),
            'vol': e.get('volume', 0),
            'closed': e.get('closed', False),
            'is_crypto': is_crypto,
            'has_value': has_value,
        })

    # Separate crypto and other
    crypto_events = [e for e in multi if e['is_crypto'] or e['has_value']]
    other_events = [e for e in multi if not e['is_crypto'] and not e['has_value']]

    print(f"\n=== CRYPTO MULTI-BUCKET ({len(crypto_events)}) ===")
    print("-" * 80)

    for ev in sorted(crypto_events, key=lambda x: x['vol'], reverse=True)[:30]:
        status = '[X]' if ev['closed'] else '[O]'
        print(f"{status} {ev['markets']:2d} mkts | ${ev['vol']:>12,.0f} | {ev['title'][:50]}")

    print(f"\n=== OTHER MULTI-BUCKET ({len(other_events)}) ===")
    for ev in sorted(other_events, key=lambda x: x['vol'], reverse=True)[:10]:
        status = '[X]' if ev['closed'] else '[O]'
        print(f"{status} {ev['markets']:2d} mkts | ${ev['vol']:>12,.0f} | {ev['title'][:50]}")


if __name__ == "__main__":
    search_fdv_events()
    search_multi_bucket_events()
