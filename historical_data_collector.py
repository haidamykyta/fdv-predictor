"""
Historical Data Collector for FDV Markets

Collects and stores price snapshots from Polymarket for future backtesting.

Data collected:
- Market metadata (title, buckets, resolution date)
- Price snapshots at regular intervals
- Volume and liquidity data
- Final resolution (actual outcome)

Storage format: JSON + Parquet for efficient analysis

Usage:
    python historical_data_collector.py --scan     # Scan and save current markets
    python historical_data_collector.py --collect  # Continuous collection
    python historical_data_collector.py --status   # Show collection status
    python historical_data_collector.py --export   # Export to parquet

Author: haidamykyta@gmail.com
"""

import requests
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from pathlib import Path
import sys

# Try to import pandas for parquet export
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

# Polymarket API
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

# Collection settings
DEFAULT_INTERVAL_MINUTES = 60  # Collect every hour
MAX_MARKETS_PER_SCAN = 100


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class MarketMetadata:
    """Market metadata."""
    market_id: str
    condition_id: str
    question: str
    description: str
    outcomes: List[str]
    end_date: Optional[str]
    category: str
    tags: List[str]
    created_at: str

    # FDV specific
    is_fdv_market: bool = False
    token_symbol: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PriceSnapshot:
    """Single price snapshot."""
    market_id: str
    timestamp: str
    prices: Dict[str, float]  # outcome -> price
    volume: float
    liquidity: float
    spread: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MarketResolution:
    """Market resolution data."""
    market_id: str
    resolved_at: str
    winning_outcome: str
    final_prices: Dict[str, float]

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# POLYMARKET CLIENT
# =============================================================================

class PolymarketHistoricalClient:
    """Client for collecting historical data from Polymarket."""

    FDV_KEYWORDS = [
        "FDV", "fully diluted", "market cap", "valuation",
        "TGE", "listing", "billion", "million"
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'FDVHistoricalCollector/1.0'
        })

    def get_all_markets(self, limit: int = 500, active_only: bool = True) -> List[Dict]:
        """Fetch all markets."""
        try:
            params = {"limit": limit}
            if active_only:
                params["active"] = "true"

            resp = self.session.get(f"{GAMMA_API_URL}/markets", params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[!] Error fetching markets: {e}")
            return []

    def get_market_details(self, market_id: str) -> Optional[Dict]:
        """Get detailed market info."""
        try:
            resp = self.session.get(f"{GAMMA_API_URL}/markets/{market_id}")
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def get_market_prices(self, condition_id: str) -> Dict[str, float]:
        """Get current prices for market outcomes."""
        try:
            resp = self.session.get(
                f"{CLOB_API_URL}/book",
                params={"token_id": condition_id}
            )
            if resp.status_code == 200:
                book = resp.json()
                bids = book.get('bids', [])
                asks = book.get('asks', [])

                if bids and asks:
                    best_bid = float(bids[0]['price']) if bids else 0
                    best_ask = float(asks[0]['price']) if asks else 1
                    return {
                        "bid": best_bid,
                        "ask": best_ask,
                        "mid": (best_bid + best_ask) / 2,
                        "spread": best_ask - best_bid
                    }
        except:
            pass
        return {}

    def is_fdv_market(self, market: Dict) -> bool:
        """Check if market is FDV-related."""
        question = (market.get('question') or '').lower()
        description = (market.get('description') or '').lower()
        tags = [t.lower() for t in market.get('tags', [])]

        # Check keywords
        text = question + " " + description
        has_fdv_keyword = any(kw.lower() in text for kw in self.FDV_KEYWORDS)

        # Check tags
        has_crypto_tag = any(t in tags for t in ['crypto', 'defi', 'blockchain', 'token'])

        return has_fdv_keyword or (has_crypto_tag and '$' in question)

    def extract_token_symbol(self, market: Dict) -> str:
        """Try to extract token symbol from market title."""
        question = market.get('question', '')

        # Common token patterns
        known_tokens = [
            "BTC", "ETH", "SOL", "LINK", "ARB", "OP", "SUI", "APT", "SEI",
            "TAO", "WLD", "RENDER", "FIL", "ZRO", "PYTH", "JUP", "JTO",
            "UNI", "AAVE", "MKR", "CRV", "STRK", "ZK", "TIA", "INJ"
        ]

        question_upper = question.upper()
        for token in known_tokens:
            if token in question_upper:
                return token

        return ""


# =============================================================================
# DATA STORAGE
# =============================================================================

class HistoricalDataStore:
    """Storage for historical market data."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "historical"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.markets_dir = self.data_dir / "markets"
        self.snapshots_dir = self.data_dir / "snapshots"
        self.markets_dir.mkdir(exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)

        # Index files
        self.markets_index_file = self.data_dir / "markets_index.json"
        self.collection_log_file = self.data_dir / "collection_log.json"

    def save_market_metadata(self, metadata: MarketMetadata):
        """Save market metadata."""
        file_path = self.markets_dir / f"{metadata.market_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def save_price_snapshot(self, snapshot: PriceSnapshot):
        """Append price snapshot to market's snapshot file."""
        file_path = self.snapshots_dir / f"{snapshot.market_id}_snapshots.json"

        # Load existing snapshots
        snapshots = []
        if file_path.exists():
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    snapshots = data.get('snapshots', [])
            except:
                pass

        # Append new snapshot
        snapshots.append(snapshot.to_dict())

        # Save
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'market_id': snapshot.market_id,
                'snapshot_count': len(snapshots),
                'last_updated': datetime.now().isoformat(),
                'snapshots': snapshots
            }, f, indent=2)

    def save_resolution(self, resolution: MarketResolution):
        """Save market resolution."""
        file_path = self.markets_dir / f"{resolution.market_id}_resolution.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(resolution.to_dict(), f, indent=2)

    def update_markets_index(self, markets: List[MarketMetadata]):
        """Update markets index."""
        index = {
            'updated_at': datetime.now().isoformat(),
            'market_count': len(markets),
            'fdv_markets': sum(1 for m in markets if m.is_fdv_market),
            'markets': [
                {
                    'market_id': m.market_id,
                    'question': m.question[:100],
                    'is_fdv': m.is_fdv_market,
                    'token': m.token_symbol,
                }
                for m in markets
            ]
        }

        with open(self.markets_index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

    def log_collection(self, stats: Dict):
        """Log collection run."""
        logs = []
        if self.collection_log_file.exists():
            try:
                with open(self.collection_log_file, encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                pass

        logs.append({
            'timestamp': datetime.now().isoformat(),
            **stats
        })

        # Keep last 1000 entries
        logs = logs[-1000:]

        with open(self.collection_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)

    def get_market_snapshots(self, market_id: str) -> List[Dict]:
        """Get all snapshots for a market."""
        file_path = self.snapshots_dir / f"{market_id}_snapshots.json"
        if file_path.exists():
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('snapshots', [])
            except:
                pass
        return []

    def export_to_parquet(self, output_file: str = None):
        """Export all snapshots to parquet format."""
        if not PANDAS_AVAILABLE:
            print("[!] pandas not installed. Cannot export to parquet.")
            return

        output_file = output_file or str(self.data_dir / "fdv_historical_data.parquet")

        all_snapshots = []

        # Load all snapshot files
        for file_path in self.snapshots_dir.glob("*_snapshots.json"):
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    for snap in data.get('snapshots', []):
                        all_snapshots.append(snap)
            except:
                continue

        if not all_snapshots:
            print("[!] No snapshots to export")
            return

        df = pd.DataFrame(all_snapshots)
        df.to_parquet(output_file, index=False)
        print(f"[OK] Exported {len(all_snapshots)} snapshots to {output_file}")

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        market_files = list(self.markets_dir.glob("*.json"))
        snapshot_files = list(self.snapshots_dir.glob("*_snapshots.json"))

        total_snapshots = 0
        for file_path in snapshot_files:
            try:
                with open(file_path, encoding='utf-8') as f:
                    data = json.load(f)
                    total_snapshots += data.get('snapshot_count', 0)
            except:
                pass

        return {
            'markets_tracked': len([f for f in market_files if '_resolution' not in f.name]),
            'snapshot_files': len(snapshot_files),
            'total_snapshots': total_snapshots,
            'resolutions': len([f for f in market_files if '_resolution' in f.name]),
        }


# =============================================================================
# COLLECTOR
# =============================================================================

class HistoricalDataCollector:
    """
    Collects historical data from Polymarket.

    Runs periodically to capture price snapshots.
    """

    def __init__(self):
        self.client = PolymarketHistoricalClient()
        self.store = HistoricalDataStore()
        self.tracked_markets: Dict[str, MarketMetadata] = {}

    def scan_markets(self) -> List[MarketMetadata]:
        """Scan and catalog all FDV-related markets."""
        print("\n" + "=" * 60)
        print("SCANNING MARKETS")
        print("=" * 60)

        markets = self.client.get_all_markets(limit=MAX_MARKETS_PER_SCAN)
        print(f"\nFetched {len(markets)} markets")

        fdv_markets = []
        for market in markets:
            is_fdv = self.client.is_fdv_market(market)

            if is_fdv:
                metadata = MarketMetadata(
                    market_id=market.get('id', ''),
                    condition_id=market.get('condition_id', ''),
                    question=market.get('question', ''),
                    description=market.get('description', ''),
                    outcomes=market.get('outcomes', ['Yes', 'No']),
                    end_date=market.get('end_date_iso'),
                    category=market.get('category', ''),
                    tags=market.get('tags', []),
                    created_at=datetime.now().isoformat(),
                    is_fdv_market=True,
                    token_symbol=self.client.extract_token_symbol(market),
                )
                fdv_markets.append(metadata)
                self.store.save_market_metadata(metadata)
                self.tracked_markets[metadata.market_id] = metadata

        self.store.update_markets_index(fdv_markets)
        print(f"Found {len(fdv_markets)} FDV-related markets")

        return fdv_markets

    def collect_snapshot(self, market: MarketMetadata) -> Optional[PriceSnapshot]:
        """Collect single price snapshot."""
        prices = self.client.get_market_prices(market.condition_id)

        if not prices:
            return None

        snapshot = PriceSnapshot(
            market_id=market.market_id,
            timestamp=datetime.now().isoformat(),
            prices=prices,
            volume=0,  # Would need separate API call
            liquidity=0,
            spread=prices.get('spread', 0),
        )

        self.store.save_price_snapshot(snapshot)
        return snapshot

    def collect_all_snapshots(self) -> Dict:
        """Collect snapshots for all tracked markets."""
        if not self.tracked_markets:
            # Load from index
            self.scan_markets()

        collected = 0
        failed = 0

        for market_id, metadata in self.tracked_markets.items():
            snapshot = self.collect_snapshot(metadata)
            if snapshot:
                collected += 1
            else:
                failed += 1
            time.sleep(0.2)  # Rate limit

        stats = {
            'collected': collected,
            'failed': failed,
            'total_markets': len(self.tracked_markets),
        }

        self.store.log_collection(stats)
        return stats

    def run_continuous(self, interval_minutes: int = DEFAULT_INTERVAL_MINUTES):
        """Run continuous collection."""
        print("\n" + "=" * 60)
        print("HISTORICAL DATA COLLECTOR")
        print("=" * 60)
        print(f"\nCollection interval: {interval_minutes} minutes")
        print("Press Ctrl+C to stop")

        # Initial scan
        self.scan_markets()

        try:
            while True:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Collecting snapshots...")
                stats = self.collect_all_snapshots()
                print(f"  Collected: {stats['collected']}/{stats['total_markets']}")

                # Show storage stats
                storage_stats = self.store.get_stats()
                print(f"  Total snapshots: {storage_stats['total_snapshots']}")

                print(f"\nNext collection in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\nStopping collector...")
            self.print_status()

    def print_status(self):
        """Print collection status."""
        stats = self.store.get_stats()

        print("\n" + "=" * 60)
        print("COLLECTION STATUS")
        print("=" * 60)

        print(f"\nMarkets tracked: {stats['markets_tracked']}")
        print(f"Snapshot files: {stats['snapshot_files']}")
        print(f"Total snapshots: {stats['total_snapshots']}")
        print(f"Resolutions: {stats['resolutions']}")

        print(f"\nData directory: {self.store.data_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Historical Data Collector")
    parser.add_argument('--scan', action='store_true', help='Scan and save market metadata')
    parser.add_argument('--collect', action='store_true', help='Run continuous collection')
    parser.add_argument('--snapshot', action='store_true', help='Collect single snapshot')
    parser.add_argument('--status', action='store_true', help='Show collection status')
    parser.add_argument('--export', action='store_true', help='Export to parquet')
    parser.add_argument('--interval', type=int, default=60, help='Collection interval (minutes)')

    args = parser.parse_args()

    collector = HistoricalDataCollector()

    if args.scan:
        markets = collector.scan_markets()
        print(f"\nScanned {len(markets)} FDV markets")
        for m in markets[:10]:
            print(f"  - {m.question[:60]}...")
    elif args.collect:
        collector.run_continuous(args.interval)
    elif args.snapshot:
        collector.scan_markets()
        stats = collector.collect_all_snapshots()
        print(f"\nCollected {stats['collected']} snapshots")
    elif args.export:
        collector.store.export_to_parquet()
    elif args.status:
        collector.print_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
