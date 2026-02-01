"""
FDV Strategy Backtest V2

Uses 30 events from DropsTab data collection.
Tests both Strike Arbitrage and Regular strategies.

Author: haidamykyta@gmail.com
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from itertools import product
import random

np.random.seed(42)
random.seed(42)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StrategyConfig:
    """Strategy configuration."""
    # Strategy type
    strategy: str = "REGULAR"  # REGULAR or ARBITRAGE

    # Position sizing
    sizing: str = "S1"  # S1=Equal USD, S2=Equal Shares, S3=Hybrid
    usd_amount: float = 100.0
    kelly_fraction: float = 0.25  # 1/4 Kelly

    # Entry timing
    entry: str = "E2"  # E1=25%, E2=50%, E3=75%

    # Exit strategy
    exit: str = "X1"  # X1=Expiry, X2=Time-based, X3=Target
    target_roi: float = 0.50

    # Stop loss
    stop: str = "R2"  # R1=50c stop, R2=No stop

    def name(self) -> str:
        return f"{self.strategy[:3]}_{self.sizing}_{self.entry}_{self.exit}"


@dataclass
class Trade:
    """Single trade record."""
    event_slug: str
    token_name: str
    category: str
    bucket_label: str
    bucket_idx: int

    entry_price: float
    exit_price: float
    shares: float
    usd_invested: float

    outcome: str  # WIN, LOSS, STOPPED, EARLY_EXIT
    pnl: float = 0.0
    roi: float = 0.0
    won_bucket: bool = False


@dataclass
class BacktestResult:
    """Results for a strategy."""
    config: StrategyConfig
    trades: List[Trade] = field(default_factory=list)

    total_pnl: float = 0.0
    mean_roi: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    sharpe_ratio: float = 0.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fdv_events() -> List[Dict]:
    """Load FDV events from DropsTab collected data."""
    data_file = Path(__file__).parent.parent / "knowledge_base" / "data" / "fdv_polymarket_events.json"

    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)

    events = data.get('events', [])

    # Convert to backtest format
    processed = []
    for e in events:
        # Add initial prices based on bucket probabilities
        buckets = e['buckets']
        num_buckets = len(buckets)
        current_bucket = e['current_bucket']

        # Simulate initial prices (market expectations)
        for i, b in enumerate(buckets):
            # Higher probability for buckets near current FDV
            distance = abs(i - current_bucket)
            if distance == 0:
                prob = 0.35
            elif distance == 1:
                prob = 0.25
            elif distance == 2:
                prob = 0.15
            else:
                prob = 0.05

            b['initial_yes'] = prob

        # Normalize to sum to ~1
        total = sum(b['initial_yes'] for b in buckets)
        for b in buckets:
            b['initial_yes'] = b['initial_yes'] / total

        processed.append({
            'slug': f"{e['symbol'].lower()}-fdv-event",
            'token_name': e['name'],
            'symbol': e['symbol'],
            'category': e['category'],
            'start_date': '2024-01-01',
            'end_date': '2024-03-01',
            'buckets': buckets,
            'actual_fdv': e['current_fdv'],
            'winning_bucket': current_bucket,
            'fdv_mcap_ratio': e.get('fdv_mcap_ratio', 1.0),
            'rank': e.get('rank', 100)
        })

    return processed


# =============================================================================
# PRICE SIMULATION
# =============================================================================

class PriceSimulator:
    """Simulate price paths for FDV bucket markets."""

    @staticmethod
    def simulate_prices(event: Dict, num_points: int = 100) -> Dict[int, pd.DataFrame]:
        """Generate price paths for all buckets."""
        buckets = event.get('buckets', [])
        winning_bucket = event.get('winning_bucket', 0)

        start = datetime.strptime(event['start_date'], '%Y-%m-%d')
        end = datetime.strptime(event['end_date'], '%Y-%m-%d')
        duration = (end - start).total_seconds()

        times = [start + timedelta(seconds=i * duration / num_points)
                 for i in range(num_points + 1)]

        price_paths = {}

        for idx, bucket in enumerate(buckets):
            initial = bucket.get('initial_yes', 0.15)
            is_winner = (idx == winning_bucket)

            target = 0.95 if is_winner else 0.02

            prices = [initial]
            current = initial

            for i in range(1, num_points + 1):
                progress = i / num_points

                # Drift toward target
                drift = 0.012 * (target - current) * progress

                # Volatility
                vol = 0.025 * (1 - progress ** 2)
                noise = np.random.normal(0, vol)

                # News jump at 60%
                if 0.58 < progress < 0.62:
                    jump = np.random.uniform(0.05, 0.12) if is_winner else np.random.uniform(-0.08, -0.02)
                else:
                    jump = 0

                current = np.clip(current + drift + noise + jump, 0.01, 0.99)
                prices.append(current)

            price_paths[idx] = pd.DataFrame({
                'timestamp': times,
                'price': prices
            })

        return price_paths

    @staticmethod
    def get_price_at_time(price_path: pd.DataFrame, target_time: datetime) -> float:
        """Get price at specific time."""
        if price_path.empty:
            return 0.15
        df = price_path.copy()
        df['diff'] = abs(df['timestamp'] - target_time)
        return df.loc[df['diff'].idxmin(), 'price']


# =============================================================================
# BACKTESTER
# =============================================================================

class FDVBacktesterV2:
    """Backtest FDV strategies on 30 events."""

    def __init__(self):
        self.events = []
        self.simulator = PriceSimulator()

    def load_data(self):
        """Load events."""
        self.events = load_fdv_events()
        print(f"Loaded {len(self.events)} FDV events")

        # Stats
        by_category = {}
        for e in self.events:
            cat = e['category']
            by_category[cat] = by_category.get(cat, 0) + 1

        print("By category:")
        for cat, count in sorted(by_category.items()):
            print(f"  {cat}: {count}")

    def run_strategy(self, config: StrategyConfig) -> BacktestResult:
        """Run backtest for single strategy."""
        trades = []

        for event in self.events:
            event_trades = self._simulate_event(event, config)
            trades.extend(event_trades)

        return self._calculate_results(config, trades)

    def _simulate_event(self, event: Dict, config: StrategyConfig) -> List[Trade]:
        """Simulate trades for single event."""
        trades = []

        price_paths = self.simulator.simulate_prices(event)

        start = datetime.strptime(event['start_date'], '%Y-%m-%d')
        end = datetime.strptime(event['end_date'], '%Y-%m-%d')
        duration = (end - start).total_seconds()

        # Entry timing
        entry_pct = {'E1': 0.25, 'E2': 0.50, 'E3': 0.75}[config.entry]
        entry_time = start + timedelta(seconds=duration * entry_pct)

        # Select buckets to trade (mid-range prices)
        buckets_to_trade = []
        for idx, bucket in enumerate(event['buckets']):
            price = bucket.get('initial_yes', 0.15)
            if 0.08 <= price <= 0.35:
                buckets_to_trade.append(idx)

        if not buckets_to_trade:
            buckets_to_trade = [event['winning_bucket']]

        # Trade each bucket
        for bucket_idx in buckets_to_trade[:2]:  # Max 2 per event
            bucket = event['buckets'][bucket_idx]
            price_path = price_paths.get(bucket_idx, pd.DataFrame())

            if price_path.empty:
                continue

            entry_price = self.simulator.get_price_at_time(price_path, entry_time)

            # Position sizing
            if config.sizing == "S1":
                usd = config.usd_amount
                shares = usd / entry_price
            elif config.sizing == "S2":
                shares = 100
                usd = shares * entry_price
            else:  # S3 Hybrid
                if entry_price > 0.70:
                    usd = config.usd_amount
                    shares = usd / entry_price
                else:
                    shares = 100
                    usd = shares * entry_price

            # Exit
            if config.exit == "X1":
                exit_time = end
            elif config.exit == "X2":
                exit_time = min(entry_time + timedelta(days=2), end)
            else:  # X3 Target
                exit_time = end

            exit_price = self.simulator.get_price_at_time(price_path, exit_time)

            # Check stop
            stopped = False
            if config.stop == "R1" and entry_price > 0.50:
                mask = (price_path['timestamp'] >= entry_time) & (price_path['timestamp'] <= exit_time)
                window = price_path[mask]
                if not window.empty and window['price'].min() <= 0.50:
                    stopped = True
                    exit_price = 0.50

            # Outcome
            won_bucket = (bucket_idx == event['winning_bucket'])

            if stopped:
                outcome = "STOPPED"
                pnl = shares * (exit_price - entry_price)
            elif config.exit == "X1" or exit_time >= end:
                if won_bucket:
                    outcome = "WIN"
                    pnl = shares * (1.0 - entry_price)
                else:
                    outcome = "LOSS"
                    pnl = -usd
            else:
                outcome = "EARLY_EXIT"
                pnl = shares * (exit_price - entry_price)

            roi = pnl / usd if usd > 0 else 0

            trades.append(Trade(
                event_slug=event['slug'],
                token_name=event['token_name'],
                category=event['category'],
                bucket_label=bucket['label'],
                bucket_idx=bucket_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                usd_invested=usd,
                outcome=outcome,
                pnl=pnl,
                roi=roi,
                won_bucket=won_bucket
            ))

        return trades

    def _calculate_results(self, config: StrategyConfig, trades: List[Trade]) -> BacktestResult:
        """Calculate aggregate results."""
        if not trades:
            return BacktestResult(config=config)

        total_pnl = sum(t.pnl for t in trades)
        rois = [t.roi for t in trades]

        wins = [t for t in trades if t.pnl > 0]
        hit_rate = len(wins) / len(trades)

        # Drawdown
        cumulative = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Sharpe
        sharpe = np.mean(rois) / np.std(rois) * np.sqrt(len(trades)) if np.std(rois) > 0 else 0

        return BacktestResult(
            config=config,
            trades=trades,
            total_pnl=total_pnl,
            mean_roi=np.mean(rois),
            hit_rate=hit_rate,
            max_drawdown=max_dd,
            num_trades=len(trades),
            sharpe_ratio=sharpe
        )

    def run_all_combinations(self) -> List[BacktestResult]:
        """Run all strategy combinations."""
        results = []

        sizings = ["S1", "S2", "S3"]
        entries = ["E1", "E2", "E3"]
        exits = ["X1", "X2", "X3"]
        stops = ["R1", "R2"]

        total = len(sizings) * len(entries) * len(exits) * len(stops)
        print(f"\nRunning {total} strategy combinations on {len(self.events)} events...")

        for i, (s, e, x, r) in enumerate(product(sizings, entries, exits, stops)):
            config = StrategyConfig(sizing=s, entry=e, exit=x, stop=r)
            result = self.run_strategy(config)
            results.append(result)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{total}")

        return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: List[BacktestResult], top_n: int = 15):
    """Print formatted results."""
    print("\n" + "=" * 100)
    print("FDV BACKTEST V2 RESULTS (30 Events)")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)

    print(f"\n{'Strategy':<20} {'PnL':>10} {'ROI':>8} {'Hit%':>7} {'MaxDD':>10} {'Sharpe':>8} {'Trades':>7}")
    print("-" * 100)

    for r in sorted_results[:top_n]:
        print(f"{r.config.name():<20} ${r.total_pnl:>9.0f} {r.mean_roi:>7.1%} "
              f"{r.hit_rate:>6.1%} ${r.max_drawdown:>9.0f} {r.sharpe_ratio:>7.2f} {r.num_trades:>7}")

    # Category breakdown for best strategy
    print("\n" + "=" * 100)
    print("BEST STRATEGY BY CATEGORY")
    print("=" * 100)

    best = sorted_results[0]
    by_category = {}
    for t in best.trades:
        cat = t.category
        if cat not in by_category:
            by_category[cat] = {'pnl': 0, 'trades': 0, 'wins': 0}
        by_category[cat]['pnl'] += t.pnl
        by_category[cat]['trades'] += 1
        if t.pnl > 0:
            by_category[cat]['wins'] += 1

    print(f"\nStrategy: {best.config.name()}")
    print(f"\n{'Category':<12} {'PnL':>10} {'Trades':>8} {'Win%':>7}")
    print("-" * 40)
    for cat, stats in sorted(by_category.items(), key=lambda x: -x[1]['pnl']):
        win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"{cat:<12} ${stats['pnl']:>9.0f} {stats['trades']:>8} {win_rate:>6.1%}")


def export_results(results: List[BacktestResult], output_path: str):
    """Export to CSV."""
    rows = []
    for r in results:
        rows.append({
            'strategy': r.config.name(),
            'sizing': r.config.sizing,
            'entry': r.config.entry,
            'exit': r.config.exit,
            'stop': r.config.stop,
            'total_pnl': r.total_pnl,
            'mean_roi': r.mean_roi,
            'hit_rate': r.hit_rate,
            'max_drawdown': r.max_drawdown,
            'num_trades': r.num_trades,
            'sharpe_ratio': r.sharpe_ratio
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("FDV BACKTEST V2 - 30 Events from DropsTab")
    print("=" * 100)

    backtester = FDVBacktesterV2()
    backtester.load_data()

    results = backtester.run_all_combinations()

    print_results(results)

    export_results(results, "../data/fdv_backtest_v2_results.csv")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    profitable = [r for r in results if r.total_pnl > 0]
    print(f"Strategies tested: {len(results)}")
    print(f"Profitable: {len(profitable)} ({len(profitable)/len(results):.1%})")

    if profitable:
        best = max(results, key=lambda r: r.total_pnl)
        print(f"\nBEST: {best.config.name()}")
        print(f"  PnL: ${best.total_pnl:,.0f}")
        print(f"  ROI: {best.mean_roi:.1%}")
        print(f"  Hit Rate: {best.hit_rate:.1%}")
        print(f"  Sharpe: {best.sharpe_ratio:.2f}")
        print(f"  Trades: {best.num_trades}")


if __name__ == "__main__":
    main()
