"""Find closed FDV events by searching specific slugs"""
import requests
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from fdv_database import FDVDatabase

# Known FDV event slugs to check
KNOWN_FDV_SLUGS = [
    # Original 4 from backtest
    'safe-market-cap-fdv-one-week-after-launch',
    'ethena-market-cap-fdv-one-day-after-airdrop',
    'eigenlayer-market-cap-fdv-one-day-after-launch',
    'friendtech-fdv-one-day-after-launch',

    # Potential additional FDV events (token launches that happened)
    'jupiter-jup-fdv',
    'starknet-strk-fdv',
    'celestia-tia-fdv',
    'dymension-dym-fdv',
    'pyth-fdv',
    'wormhole-fdv',
    'ondo-fdv',
    'layerzero-zro-fdv',
    'blast-fdv',
    'renzo-fdv',

    # Try variations
    'jupiter-market-cap',
    'starknet-market-cap',
    'celestia-market-cap',
    'pyth-market-cap',
    'wormhole-market-cap',
    'layerzero-market-cap',
]

print("=" * 60)
print("CHECKING KNOWN FDV EVENT SLUGS")
print("=" * 60)

found = []
not_found = []

for slug in KNOWN_FDV_SLUGS:
    resp = requests.get(
        'https://gamma-api.polymarket.com/events',
        params={'slug': slug},
        timeout=30
    )
    events = resp.json()

    if events:
        e = events[0]
        markets = e.get('markets', [])
        is_closed = e.get('closed', False)

        status = '[X]' if is_closed else '[O]'
        print(f"{status} {slug}")
        print(f"    {e.get('title', '')[:50]}")
        print(f"    Markets: {len(markets)} | Closed: {is_closed}")

        if is_closed and len(markets) >= 3:
            found.append({
                'slug': slug,
                'title': e.get('title'),
                'markets': len(markets),
                'volume': e.get('volume', 0),
            })
    else:
        not_found.append(slug)

    time.sleep(0.2)

print(f"\n\nFound {len(found)} closed FDV events")
print(f"Not found: {len(not_found)}")

# Check what's in database
print("\n" + "=" * 60)
print("DATABASE STATUS")
print("=" * 60)

db = FDVDatabase()
existing = db.get_all_events()
existing_slugs = {e['slug'] for e in existing}

print(f"Events in DB: {len(existing)}")
print("\nExisting events:")
for e in existing:
    status = '[X]' if e.get('is_resolved') else '[O]'
    print(f"  {status} {e['token_name']}: {e['slug']}")

# Find new closed events to add
new_to_add = [e for e in found if e['slug'] not in existing_slugs]
print(f"\nNew closed events to add: {len(new_to_add)}")

db.close()
