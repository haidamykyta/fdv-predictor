"""Quick FDV event search - search by slug patterns"""
import requests
import re
import time

print('=' * 60)
print('POLYMARKET FDV EVENT SEARCH')
print('=' * 60)

# Known FDV event slugs (to verify they exist)
known_slugs = [
    'safe-market-cap-fdv-one-week-after-launch',
    'ethena-market-cap-fdv-one-day-after-airdrop',
    'eigenlayer-market-cap-fdv-one-day-after-launch',
    'friendtech-fdv-one-day-after-launch',
]

# Check known events
print('\n=== CHECKING KNOWN FDV EVENTS ===')
for slug in known_slugs:
    resp = requests.get(
        'https://gamma-api.polymarket.com/events',
        params={'slug': slug},
        timeout=30
    )
    events = resp.json()
    if events:
        e = events[0]
        markets = e.get('markets', [])
        print(f"[OK] {slug}")
        print(f"     Markets: {len(markets)} | Closed: {e.get('closed')}")
    else:
        print(f"[X] {slug} - NOT FOUND")
    time.sleep(0.3)

# Search for events with specific keywords in slug
print('\n=== SEARCHING BY SLUG PATTERNS ===')

slug_patterns = [
    'fdv', 'market-cap', 'tge', 'launch', 'airdrop', 'listing',
    'valuation', 'token', 'price'
]

found_events = {}

# First fetch a larger batch of events
print('\nFetching events...')
for offset in range(0, 3000, 500):
    for status in ['true', 'false']:
        resp = requests.get(
            'https://gamma-api.polymarket.com/events',
            params={'limit': 500, 'closed': status, 'offset': offset},
            timeout=60
        )
        batch = resp.json()
        if not batch:
            continue

        for e in batch:
            slug = e.get('slug', '')
            markets = e.get('markets', [])

            # Check if slug matches any pattern
            for pattern in slug_patterns:
                if pattern in slug.lower():
                    if slug not in found_events:
                        found_events[slug] = {
                            'title': e.get('title'),
                            'markets': len(markets),
                            'volume': e.get('volume', 0),
                            'closed': e.get('closed', False),
                            'pattern': pattern,
                        }
                    break

        if len(batch) < 500:
            break
        time.sleep(0.3)
    print(f"  Offset {offset}: total found {len(found_events)}")

# Filter for multi-bucket FDV-like events
print(f'\n=== FOUND {len(found_events)} EVENTS WITH SLUG PATTERNS ===')

# Sort by markets count and volume
multi_bucket = [(k, v) for k, v in found_events.items() if v['markets'] >= 3]
single_market = [(k, v) for k, v in found_events.items() if v['markets'] < 3]

print(f'\nMulti-bucket (3+ markets): {len(multi_bucket)}')
print('-' * 70)

for slug, info in sorted(multi_bucket, key=lambda x: x[1]['volume'], reverse=True)[:20]:
    status = '[X]' if info['closed'] else '[O]'
    print(f"{status} {info['markets']:2d} mkts | ${info['volume']:>12,.0f} | {info['title'][:40]}")
    print(f"   {slug}")

print(f'\nSingle-market (with FDV keywords): {len(single_market)}')
for slug, info in sorted(single_market, key=lambda x: x[1]['volume'], reverse=True)[:10]:
    status = '[X]' if info['closed'] else '[O]'
    print(f"{status} {info['pattern']}: {info['title'][:50]}")
