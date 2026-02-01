"""
Live Paper Trading Bot for FDV Markets

Connects to real Polymarket API for FDV market data.
Executes paper trades using Combined Strategy + Risk Manager.

Features:
- Real-time FDV market discovery
- Paper trade execution (no real money)
- PnL tracking and reporting
- Strategy performance monitoring

Usage:
    python live_paper_trader.py --scan       # Scan for FDV markets
    python live_paper_trader.py --trade      # Start paper trading
    python live_paper_trader.py --status     # Show portfolio status

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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from combined_strategy import CombinedStrategy, MarketSignal, TradeDecision, StrategyType
from risk_manager import RiskManager


# =============================================================================
# POLYMARKET FDV CLIENT
# =============================================================================

class PolymarketFDVClient:
    """Client for fetching FDV markets from Polymarket."""

    BASE_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    # FDV market keywords
    FDV_KEYWORDS = [
        "FDV", "fully diluted", "market cap", "valuation",
        "TGE", "listing", "launch price"
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'FDVPredictor/1.0'
        })
        self.cache_dir = Path(__file__).parent / "data" / "live_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_all_markets(self, limit: int = 500) -> List[Dict]:
        """Fetch all active markets."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/markets",
                params={"limit": limit, "active": "true"}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def find_fdv_markets(self) -> List[Dict]:
        """Find FDV-related markets."""
        all_markets = self.get_all_markets()
        fdv_markets = []

        for market in all_markets:
            question = (market.get('question') or '').lower()
            description = (market.get('description') or '').lower()
            tags = [t.lower() for t in market.get('tags', [])]

            # Check for FDV keywords
            is_fdv = any(
                kw.lower() in question or kw.lower() in description
                for kw in self.FDV_KEYWORDS
            )

            # Also check for crypto token names with price/cap
            has_crypto_valuation = (
                ('billion' in question or '$' in question) and
                any(t in tags for t in ['crypto', 'defi', 'blockchain', 'token'])
            )

            if is_fdv or has_crypto_valuation:
                fdv_markets.append(market)

        return fdv_markets

    def get_market_prices(self, condition_id: str) -> Dict[str, float]:
        """Get current prices for market outcomes."""
        try:
            resp = self.session.get(
                f"{self.CLOB_URL}/prices",
                params={"token_ids": condition_id}
            )
            if resp.status_code == 200:
                return resp.json()
        except:
            pass

        # Fallback: estimate from orderbook
        try:
            resp = self.session.get(
                f"{self.CLOB_URL}/book",
                params={"token_id": condition_id}
            )
            if resp.status_code == 200:
                book = resp.json()
                bids = book.get('bids', [])
                asks = book.get('asks', [])

                if bids and asks:
                    best_bid = float(bids[0]['price']) if bids else 0
                    best_ask = float(asks[0]['price']) if asks else 1
                    return {"mid": (best_bid + best_ask) / 2}
        except:
            pass

        return {}

    def parse_fdv_buckets(self, market: Dict) -> List[Dict]:
        """Parse FDV bucket structure from market."""
        # Multi-outcome markets have outcomes array
        outcomes = market.get('outcomes', [])

        if outcomes:
            buckets = []
            for i, outcome in enumerate(outcomes):
                # Try to parse bucket range from outcome name
                name = outcome if isinstance(outcome, str) else outcome.get('name', f'Bucket {i}')
                buckets.append({
                    "label": name,
                    "idx": i,
                    "outcome_id": outcome.get('id') if isinstance(outcome, dict) else None
                })
            return buckets

        # Binary market (Yes/No)
        return [
            {"label": "Yes", "idx": 0},
            {"label": "No", "idx": 1}
        ]


# =============================================================================
# PAPER PORTFOLIO
# =============================================================================

@dataclass
class PaperPosition:
    """Paper trading position."""
    market_id: str
    market_title: str
    bucket_idx: int
    bucket_label: str

    entry_price: float
    shares: float
    entry_time: datetime
    strategy: str

    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    # Exit
    exit_price: float = 0.0
    exit_time: datetime = None
    realized_pnl: float = 0.0
    is_closed: bool = False


@dataclass
class PaperPortfolio:
    """Paper trading portfolio."""
    initial_balance: float = 1000.0
    current_balance: float = 1000.0

    positions: List[PaperPosition] = field(default_factory=list)
    closed_positions: List[PaperPosition] = field(default_factory=list)

    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0

    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_position(self, position: PaperPosition):
        """Add new position."""
        cost = position.shares * position.entry_price
        self.current_balance -= cost
        self.positions.append(position)
        self.total_trades += 1
        self.last_updated = datetime.now()

    def close_position(self, position: PaperPosition, exit_price: float):
        """Close position at given price."""
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.realized_pnl = position.shares * (exit_price - position.entry_price)
        position.is_closed = True

        # Update portfolio
        self.current_balance += position.shares * exit_price
        self.total_pnl += position.realized_pnl

        if position.realized_pnl > 0:
            self.winning_trades += 1

        # Move to closed
        self.positions.remove(position)
        self.closed_positions.append(position)
        self.last_updated = datetime.now()

    def update_unrealized(self, market_prices: Dict[str, float]):
        """Update unrealized PnL for open positions."""
        for pos in self.positions:
            if pos.market_id in market_prices:
                pos.current_price = market_prices[pos.market_id]
                pos.unrealized_pnl = pos.shares * (pos.current_price - pos.entry_price)

    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        open_value = sum(p.shares * p.current_price for p in self.positions)
        return self.current_balance + open_value

    @property
    def win_rate(self) -> float:
        """Win rate of closed trades."""
        if not self.closed_positions:
            return 0.0
        return self.winning_trades / len(self.closed_positions)

    def to_dict(self) -> Dict:
        """Convert to dict for saving."""
        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl": self.total_pnl,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "positions": [asdict(p) for p in self.positions],
            "closed_positions": [asdict(p) for p in self.closed_positions],
        }


# =============================================================================
# LIVE PAPER TRADER
# =============================================================================

class LivePaperTrader:
    """
    Live paper trading bot for FDV markets.

    Connects to real Polymarket data but executes paper trades only.
    """

    def __init__(
        self,
        bankroll: float = 1000,
        data_dir: str = None
    ):
        self.client = PolymarketFDVClient()
        self.strategy = CombinedStrategy(bankroll=bankroll)
        self.portfolio = PaperPortfolio(initial_balance=bankroll, current_balance=bankroll)

        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data" / "paper_trading"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.portfolio_file = self.data_dir / "portfolio.json"
        self.trades_file = self.data_dir / "trades.json"

        self._load_state()

    def _load_state(self):
        """Load saved state."""
        if self.portfolio_file.exists():
            try:
                with open(self.portfolio_file, encoding='utf-8') as f:
                    data = json.load(f)
                self.portfolio.current_balance = data.get('current_balance', self.portfolio.initial_balance)
                self.portfolio.total_trades = data.get('total_trades', 0)
                self.portfolio.total_pnl = data.get('total_pnl', 0)
                print(f"[OK] Loaded portfolio: ${self.portfolio.current_balance:.0f}")
            except:
                pass

    def _save_state(self):
        """Save current state."""
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(self.portfolio.to_dict(), f, indent=2, default=str)

    def scan_markets(self) -> List[Dict]:
        """Scan for FDV markets."""
        print("\n" + "=" * 60)
        print("SCANNING FOR FDV MARKETS")
        print("=" * 60)

        fdv_markets = self.client.find_fdv_markets()

        print(f"\nFound {len(fdv_markets)} potential FDV markets:")

        for i, market in enumerate(fdv_markets[:20]):  # Show top 20
            title = market.get('question', 'Unknown')[:60]
            try:
                volume = float(market.get('volume', 0) or 0)
                liquidity = float(market.get('liquidity', 0) or 0)
            except (ValueError, TypeError):
                volume = 0
                liquidity = 0

            print(f"\n{i+1}. {title}")
            print(f"   Volume: ${volume:,.0f} | Liquidity: ${liquidity:,.0f}")

            # Get prices if available
            cond_id = market.get('condition_id')
            if cond_id:
                prices = self.client.get_market_prices(cond_id)
                if prices:
                    print(f"   Prices: {prices}")

        return fdv_markets

    def analyze_market(self, market: Dict) -> Optional[TradeDecision]:
        """Analyze single market for trade opportunity."""
        # Parse market into signal
        buckets = self.client.parse_fdv_buckets(market)

        # Get prices for each bucket
        bucket_prices = []
        for b in buckets:
            # Default prices based on number of outcomes
            default_price = 1 / len(buckets) if buckets else 0.5
            bucket_prices.append(default_price)

        # Try to determine token info from title
        title = market.get('question', '')
        token_symbol = "UNKNOWN"
        category = "Other"

        # Check for known tokens
        known_tokens = {
            "LINK": "Infra", "TAO": "Infra", "WLD": "Infra",
            "RENDER": "Infra", "FIL": "Infra", "ZRO": "Infra", "PYTH": "Infra",
            "BTC": "L1", "ETH": "L1", "SOL": "L1", "SUI": "L1",
            "ARB": "L2", "OP": "L2", "STRK": "L2",
            "UNI": "DeFi", "AAVE": "DeFi", "JUP": "DeFi",
        }

        for symbol, cat in known_tokens.items():
            if symbol.lower() in title.lower():
                token_symbol = symbol
                category = cat
                break

        # Create signal
        signal = MarketSignal(
            token_symbol=token_symbol,
            token_name=title[:30],
            category=category,
            buckets=buckets,
            current_bucket=len(buckets) // 2,  # Assume middle
            fdv_mcap_ratio=1.5,  # Default
            bucket_prices=bucket_prices,
        )

        return self.strategy.analyze_market(signal)

    def execute_paper_trade(self, market: Dict, decision: TradeDecision):
        """Execute paper trade."""
        position = PaperPosition(
            market_id=market.get('condition_id', 'unknown'),
            market_title=market.get('question', 'Unknown')[:50],
            bucket_idx=decision.bucket_idx,
            bucket_label=f"Bucket {decision.bucket_idx}",
            entry_price=decision.entry_price,
            shares=decision.size_usd / decision.entry_price,
            entry_time=datetime.now(),
            strategy=decision.strategy.value,
            current_price=decision.entry_price,
        )

        self.portfolio.add_position(position)
        self._save_state()

        print(f"\n[PAPER TRADE EXECUTED]")
        print(f"  Market: {position.market_title}")
        print(f"  Strategy: {decision.strategy.value}")
        print(f"  Entry: {decision.entry_price:.0%}")
        print(f"  Size: ${decision.size_usd:.0f} ({position.shares:.1f} shares)")
        print(f"  Reason: {decision.reason}")

        return position

    def run_single_scan(self):
        """Run single scan and trade cycle."""
        print("\n" + "=" * 60)
        print(f"PAPER TRADING SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)

        # Check if we can trade
        if not self.strategy.risk_manager.can_trade():
            print("\n[!] Trading paused by risk manager")
            self.strategy.risk_manager.print_status()
            return

        # Scan markets
        fdv_markets = self.client.find_fdv_markets()
        print(f"\nFound {len(fdv_markets)} FDV markets")

        trades_made = 0
        max_trades = 3  # Max trades per scan

        for market in fdv_markets:
            if trades_made >= max_trades:
                break

            # Check if we already have position
            market_id = market.get('condition_id')
            has_position = any(p.market_id == market_id for p in self.portfolio.positions)

            if has_position:
                continue

            # Analyze
            decision = self.analyze_market(market)

            if decision and decision.action == "BUY":
                self.execute_paper_trade(market, decision)
                trades_made += 1

        print(f"\nMade {trades_made} paper trades this scan")

    def run_continuous(self, interval_minutes: int = 60):
        """Run continuous paper trading."""
        print("\n" + "=" * 60)
        print("STARTING CONTINUOUS PAPER TRADING")
        print(f"Scan interval: {interval_minutes} minutes")
        print("Press Ctrl+C to stop")
        print("=" * 60)

        try:
            while True:
                self.run_single_scan()
                self.print_status()

                print(f"\nNext scan in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\nStopping paper trader...")
            self._save_state()
            self.print_status()

    def print_status(self):
        """Print portfolio status."""
        print("\n" + "=" * 60)
        print("PAPER PORTFOLIO STATUS")
        print("=" * 60)

        print(f"\nBalance: ${self.portfolio.current_balance:,.0f}")
        print(f"Total Value: ${self.portfolio.total_value:,.0f}")
        print(f"Total PnL: ${self.portfolio.total_pnl:+,.0f}")

        print(f"\nTrades: {self.portfolio.total_trades}")
        print(f"Win Rate: {self.portfolio.win_rate:.1%}")

        # Open positions
        if self.portfolio.positions:
            print(f"\n{'='*40}")
            print("OPEN POSITIONS")
            print(f"{'='*40}")

            for pos in self.portfolio.positions:
                print(f"\n  {pos.market_title[:40]}")
                print(f"    Strategy: {pos.strategy}")
                print(f"    Entry: {pos.entry_price:.0%} | Current: {pos.current_price:.0%}")
                print(f"    Shares: {pos.shares:.1f}")
                print(f"    Unrealized: ${pos.unrealized_pnl:+,.0f}")

        # Strategy breakdown
        self.strategy.print_status()


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="FDV Live Paper Trader")
    parser.add_argument('--scan', action='store_true', help='Scan for FDV markets')
    parser.add_argument('--trade', action='store_true', help='Run single trade cycle')
    parser.add_argument('--continuous', action='store_true', help='Run continuous trading')
    parser.add_argument('--status', action='store_true', help='Show portfolio status')
    parser.add_argument('--bankroll', type=float, default=1000, help='Initial bankroll')
    parser.add_argument('--interval', type=int, default=60, help='Scan interval (minutes)')

    args = parser.parse_args()

    trader = LivePaperTrader(bankroll=args.bankroll)

    if args.scan:
        trader.scan_markets()
    elif args.trade:
        trader.run_single_scan()
    elif args.continuous:
        trader.run_continuous(args.interval)
    elif args.status:
        trader.print_status()
    else:
        # Default: show status and scan
        trader.print_status()
        print("\n" + "-" * 60)
        trader.scan_markets()


if __name__ == "__main__":
    main()
