"""
Step 1: Discover FDV Events on Polymarket
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fdv_event_discovery import FDVEventDiscovery, FDVEvent
from fdv_database import FDVDatabase
from config import EVENT_SCOPE


def main():
    print("=" * 60)
    print("FDV Event Discovery - Step 1")
    print("=" * 60)

    # Initialize
    discovery = FDVEventDiscovery()
    db = FDVDatabase()

    # Discover events
    include_resolved = EVENT_SCOPE in ('resolved', 'both')
    include_active = EVENT_SCOPE in ('active', 'both')

    print(f"\nScope: {EVENT_SCOPE}")
    print(f"  Include resolved: {include_resolved}")
    print(f"  Include active: {include_active}")

    events = discovery.discover_fdv_events(
        include_resolved=include_resolved,
        include_active=include_active
    )

    # Save to database
    print(f"\nSaving {len(events)} events to database...")

    for event in events:
        db.insert_event(event.to_dict())

    # Save to JSON
    discovery.save_events(events)

    # Print summary
    print("\n" + "=" * 60)
    print("DISCOVERY SUMMARY")
    print("=" * 60)

    resolved = [e for e in events if e.is_resolved]
    active = [e for e in events if not e.is_resolved]

    print(f"\nTotal events found: {len(events)}")
    print(f"  Resolved: {len(resolved)}")
    print(f"  Active: {len(active)}")

    # List resolved events (needed for backtest)
    if resolved:
        print(f"\n{'='*60}")
        print("RESOLVED EVENTS (for backtest)")
        print("=" * 60)

        for event in resolved[:20]:
            print(f"\n• {event.title[:60]}...")
            print(f"  Token: {event.token_name} ({event.token_symbol or 'N/A'})")
            print(f"  Buckets: {len(event.buckets)}")
            print(f"  Resolved bucket: {event.resolved_bucket_label or 'N/A'}")
            print(f"  Volume: ${event.total_volume:,.0f}")

        if len(resolved) > 20:
            print(f"\n... and {len(resolved) - 20} more resolved events")

    # List active events (for monitoring)
    if active:
        print(f"\n{'='*60}")
        print("ACTIVE EVENTS (for monitoring)")
        print("=" * 60)

        for event in active[:10]:
            print(f"\n• {event.title[:60]}...")
            print(f"  Token: {event.token_name} ({event.token_symbol or 'N/A'})")
            print(f"  Buckets: {len(event.buckets)}")
            print(f"  End date: {event.end_date or 'N/A'}")

        if len(active) > 10:
            print(f"\n... and {len(active) - 10} more active events")

    # Database stats
    print(f"\n{'='*60}")
    print("DATABASE STATS")
    print("=" * 60)
    print(db.get_stats())

    db.close()

    print(f"\n[OK] Done! Next step: python scripts/collect_actual_fdv.py")


if __name__ == "__main__":
    main()
