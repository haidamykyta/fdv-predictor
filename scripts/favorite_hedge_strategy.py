"""
Favorite + Hedge Strategy for Polymarket FDV Markets

Based on Polymarket bots analysis - CS:GO inspired strategy.

Concept:
1. Buy the FAVORITE (>80c) at halfway point (50% of event duration)
2. Wait for TIME DECAY - favorite price increases as time passes
3. Buy HEDGE on underdog near expiry (85-90% of duration)
4. Lock in profit regardless of outcome

Example:
  Start:  Bucket A (favorite) = 82c, Bucket B = 18c
  +50%:   Buy 100 shares of A @ 82c = $82 cost

  Time decay occurs...

  +85%:   A = 91c, B = 9c
          Buy hedge: 100 shares of B @ 9c = $9 cost
          Total cost: $82 + $9 = $91

  Expiry: Either A or B wins -> payout $100
          Profit: $100 - $91 = $9 (9.9% ROI)

Key insight: Time decay makes favorites stronger near expiry.

Author: haidamykyta@gmail.com
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import random

np.random.seed(42)
random.seed(42)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FavoriteHedgeConfig:
    """Configuration for Favorite + Hedge strategy."""
    # Entry timing (% of event duration)
    favorite_entry_pct: float = 0.50  # Buy favorite at 50%
    hedge_entry_pct: float = 0.85     # Buy hedge at 85%

    # Thresholds
    min_favorite_price: float = 0.75  # Only buy favorites >75c
    max_hedge_price: float = 0.25     # Only hedge if underdog <25c

    # Position sizing
    favorite_usd: float = 100.0       # $100 on favorite
    hedge_mode: str = "EQUAL_SHARES"  # EQUAL_SHARES or BREAKEVEN

    # Risk management
    stop_loss_pct: float = 0.15       # Exit if favorite drops 15%
    min_profit_target: float = 0.05   # Need at least 5% profit potential

    # Dynamic hedge timing
    dynamic_hedge: bool = True        # Hedge when ROI target hit
    target_roi: float = 0.08          # Target 8% ROI

    def name(self) -> str:
        return f"FH_E{int(self.favorite_entry_pct*100)}_H{int(self.hedge_entry_pct*100)}"


@dataclass
class FavoriteHedgeTrade:
    """Single Favorite + Hedge trade."""
    event_slug: str
    token_name: str
    category: str

    # Favorite leg
    favorite_bucket: str
    favorite_entry_time: datetime
    favorite_entry_price: float
    favorite_shares: float
    favorite_cost: float

    # Hedge leg
    hedge_bucket: str
    hedge_entry_time: datetime
    hedge_entry_price: float
    hedge_shares: float
    hedge_cost: float

    # Total
    total_cost: float
    total_payout: float  # Always $1 per share if either wins

    # Outcome
    outcome: str  # PROFIT, LOSS, STOPPED, NO_HEDGE
    pnl: float = 0.0
    roi: float = 0.0
    favorite_won: bool = False


@dataclass
class FavoriteHedgeResult:
    """Results for Favorite + Hedge strategy."""
    config: FavoriteHedgeConfig
    trades: List[FavoriteHedgeTrade] = field(default_factory=list)

    total_pnl: float = 0.0
    mean_roi: float = 0.0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    num_trades: int = 0
    num_stopped: int = 0
    num_no_hedge: int = 0
    sharpe_ratio: float = 0.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_fdv_events() -> List[Dict]:
    """Load FDV events from DropsTab data."""
    data_file = Path(__file__).parent.parent / "knowledge_base" / "data" / "fdv_polymarket_events.json"

    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)

    events = []
    for e in data.get('events', []):
        buckets = e['buckets']
        current_bucket = e['current_bucket']

        # Generate initial prices
        for i, b in enumerate(buckets):
            distance = abs(i - current_bucket)
            if distance == 0:
                prob = 0.40
            elif distance == 1:
                prob = 0.25
            else:
                prob = 0.10 / max(1, distance)
            b['initial_yes'] = prob

        # Normalize
        total = sum(b['initial_yes'] for b in buckets)
        for b in buckets:
            b['initial_yes'] = b['initial_yes'] / total

        events.append({
            'slug': f"{e['symbol'].lower()}-fdv",
            'token_name': e['name'],
            'symbol': e['symbol'],
            'category': e['category'],
            'start_date': '2024-01-01',
            'end_date': '2024-03-01',
            'buckets': buckets,
            'actual_fdv': e['current_fdv'],
            'winning_bucket': current_bucket,
        })

    return events


# =============================================================================
# PRICE SIMULATION WITH TIME DECAY
# =============================================================================

class TimeDecaySimulator:
    """Simulate price paths with time decay effect."""

    @staticmethod
    def simulate_with_decay(event: Dict, num_points: int = 100) -> Dict[int, pd.DataFrame]:
        """
        Generate price paths with time decay.
        Favorites strengthen near expiry, underdogs weaken.
        """
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
            is_favorite = (initial > 0.30)  # Favorite if >30c initially

            target = 0.95 if is_winner else 0.02

            prices = [initial]
            current = initial

            for i in range(1, num_points + 1):
                progress = i / num_points

                # TIME DECAY EFFECT
                # Favorites drift UP, underdogs drift DOWN as time passes
                if is_favorite:
                    # Favorite strengthens due to time decay
                    time_decay_drift = 0.0015 * progress * (1 - current)
                else:
                    # Underdog weakens
                    time_decay_drift = -0.001 * progress * current

                # Trend toward actual outcome
                outcome_drift = 0.012 * (target - current) * progress

                # Total drift
                drift = time_decay_drift + outcome_drift

                # Volatility (lower near expiry)
                vol = 0.02 * (1 - progress ** 1.5)
                noise = np.random.normal(0, vol)

                # News jump at 60%
                if 0.58 < progress < 0.62:
                    if is_winner:
                        jump = np.random.uniform(0.05, 0.12)
                    else:
                        jump = np.random.uniform(-0.08, -0.02)
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
# STRATEGY IMPLEMENTATION
# =============================================================================

class FavoriteHedgeStrategy:
    """Favorite + Hedge strategy implementation."""

    def __init__(self, config: FavoriteHedgeConfig = None):
        self.config = config or FavoriteHedgeConfig()
        self.simulator = TimeDecaySimulator()

    def find_favorite(self, event: Dict, price_paths: Dict, entry_time: datetime) -> Optional[Tuple[int, float]]:
        """Find the favorite bucket at entry time."""
        best_idx = None
        best_price = 0

        for idx, bucket in enumerate(event['buckets']):
            price = self.simulator.get_price_at_time(price_paths[idx], entry_time)
            if price > best_price:
                best_price = price
                best_idx = idx

        if best_price >= self.config.min_favorite_price:
            return best_idx, best_price
        return None

    def find_best_hedge(self, event: Dict, price_paths: Dict, hedge_time: datetime,
                        favorite_idx: int) -> Optional[Tuple[int, float]]:
        """Find best hedge bucket (cheapest underdog)."""
        best_idx = None
        best_price = 1.0

        for idx, bucket in enumerate(event['buckets']):
            if idx == favorite_idx:
                continue
            price = self.simulator.get_price_at_time(price_paths[idx], hedge_time)
            if price < best_price and price <= self.config.max_hedge_price:
                best_price = price
                best_idx = idx

        if best_idx is not None:
            return best_idx, best_price
        return None

    def execute_trade(self, event: Dict, price_paths: Dict) -> Optional[FavoriteHedgeTrade]:
        """Execute Favorite + Hedge trade on event."""
        start = datetime.strptime(event['start_date'], '%Y-%m-%d')
        end = datetime.strptime(event['end_date'], '%Y-%m-%d')
        duration = (end - start).total_seconds()

        # Favorite entry time
        fav_entry_time = start + timedelta(seconds=duration * self.config.favorite_entry_pct)

        # Find favorite
        favorite = self.find_favorite(event, price_paths, fav_entry_time)
        if not favorite:
            return None

        fav_idx, fav_price = favorite
        fav_bucket = event['buckets'][fav_idx]

        # Calculate favorite position
        fav_shares = self.config.favorite_usd / fav_price
        fav_cost = self.config.favorite_usd

        # Check for stop loss before hedge
        stopped = False
        stop_price = fav_price * (1 - self.config.stop_loss_pct)

        fav_path = price_paths[fav_idx]
        hedge_time = start + timedelta(seconds=duration * self.config.hedge_entry_pct)

        # Check if stopped
        mask = (fav_path['timestamp'] >= fav_entry_time) & (fav_path['timestamp'] <= hedge_time)
        window = fav_path[mask]
        if not window.empty and window['price'].min() <= stop_price:
            stopped = True
            exit_price = stop_price
            pnl = fav_shares * (exit_price - fav_price)

            return FavoriteHedgeTrade(
                event_slug=event['slug'],
                token_name=event['token_name'],
                category=event['category'],
                favorite_bucket=fav_bucket['label'],
                favorite_entry_time=fav_entry_time,
                favorite_entry_price=fav_price,
                favorite_shares=fav_shares,
                favorite_cost=fav_cost,
                hedge_bucket="N/A",
                hedge_entry_time=hedge_time,
                hedge_entry_price=0,
                hedge_shares=0,
                hedge_cost=0,
                total_cost=fav_cost,
                total_payout=0,
                outcome="STOPPED",
                pnl=pnl,
                roi=pnl / fav_cost,
                favorite_won=False
            )

        # Dynamic hedge timing
        if self.config.dynamic_hedge:
            # Check if we can hedge earlier with good ROI
            for _, row in window.iterrows():
                current_fav_price = row['price']
                # Find hedge at this time
                hedge_result = self.find_best_hedge(event, price_paths, row['timestamp'], fav_idx)
                if hedge_result:
                    hedge_idx, hedge_price = hedge_result
                    potential_cost = fav_cost + fav_shares * hedge_price
                    potential_profit = fav_shares - potential_cost
                    potential_roi = potential_profit / potential_cost

                    if potential_roi >= self.config.target_roi:
                        hedge_time = row['timestamp']
                        break

        # Find hedge
        hedge = self.find_best_hedge(event, price_paths, hedge_time, fav_idx)
        if not hedge:
            # No good hedge available
            # Hold to expiry
            final_fav_price = self.simulator.get_price_at_time(fav_path, end)
            favorite_won = (fav_idx == event['winning_bucket'])

            if favorite_won:
                pnl = fav_shares * (1.0 - fav_price)
            else:
                pnl = -fav_cost

            return FavoriteHedgeTrade(
                event_slug=event['slug'],
                token_name=event['token_name'],
                category=event['category'],
                favorite_bucket=fav_bucket['label'],
                favorite_entry_time=fav_entry_time,
                favorite_entry_price=fav_price,
                favorite_shares=fav_shares,
                favorite_cost=fav_cost,
                hedge_bucket="N/A",
                hedge_entry_time=hedge_time,
                hedge_entry_price=0,
                hedge_shares=0,
                hedge_cost=0,
                total_cost=fav_cost,
                total_payout=fav_shares if favorite_won else 0,
                outcome="NO_HEDGE",
                pnl=pnl,
                roi=pnl / fav_cost,
                favorite_won=favorite_won
            )

        hedge_idx, hedge_price = hedge
        hedge_bucket = event['buckets'][hedge_idx]

        # Calculate hedge position
        if self.config.hedge_mode == "EQUAL_SHARES":
            hedge_shares = fav_shares
        else:  # BREAKEVEN
            # Hedge enough to break even
            hedge_shares = fav_shares

        hedge_cost = hedge_shares * hedge_price
        total_cost = fav_cost + hedge_cost
        total_payout = fav_shares  # One side always wins -> $1 per share

        pnl = total_payout - total_cost
        roi = pnl / total_cost

        # Check minimum profit
        if roi < self.config.min_profit_target:
            # Not worth hedging
            favorite_won = (fav_idx == event['winning_bucket'])
            if favorite_won:
                pnl = fav_shares * (1.0 - fav_price)
            else:
                pnl = -fav_cost

            return FavoriteHedgeTrade(
                event_slug=event['slug'],
                token_name=event['token_name'],
                category=event['category'],
                favorite_bucket=fav_bucket['label'],
                favorite_entry_time=fav_entry_time,
                favorite_entry_price=fav_price,
                favorite_shares=fav_shares,
                favorite_cost=fav_cost,
                hedge_bucket="N/A (low ROI)",
                hedge_entry_time=hedge_time,
                hedge_entry_price=hedge_price,
                hedge_shares=0,
                hedge_cost=0,
                total_cost=fav_cost,
                total_payout=fav_shares if favorite_won else 0,
                outcome="NO_HEDGE",
                pnl=pnl,
                roi=pnl / fav_cost,
                favorite_won=favorite_won
            )

        return FavoriteHedgeTrade(
            event_slug=event['slug'],
            token_name=event['token_name'],
            category=event['category'],
            favorite_bucket=fav_bucket['label'],
            favorite_entry_time=fav_entry_time,
            favorite_entry_price=fav_price,
            favorite_shares=fav_shares,
            favorite_cost=fav_cost,
            hedge_bucket=hedge_bucket['label'],
            hedge_entry_time=hedge_time,
            hedge_entry_price=hedge_price,
            hedge_shares=hedge_shares,
            hedge_cost=hedge_cost,
            total_cost=total_cost,
            total_payout=total_payout,
            outcome="PROFIT" if pnl > 0 else "LOSS",
            pnl=pnl,
            roi=roi,
            favorite_won=(fav_idx == event['winning_bucket'])
        )


# =============================================================================
# BACKTESTER
# =============================================================================

class FavoriteHedgeBacktester:
    """Backtest Favorite + Hedge strategy."""

    def __init__(self):
        self.events = []
        self.simulator = TimeDecaySimulator()

    def load_data(self):
        """Load events."""
        self.events = load_fdv_events()
        print(f"Loaded {len(self.events)} FDV events")

    def run_strategy(self, config: FavoriteHedgeConfig) -> FavoriteHedgeResult:
        """Run backtest for config."""
        strategy = FavoriteHedgeStrategy(config)
        trades = []

        for event in self.events:
            price_paths = self.simulator.simulate_with_decay(event)
            trade = strategy.execute_trade(event, price_paths)
            if trade:
                trades.append(trade)

        return self._calculate_results(config, trades)

    def _calculate_results(self, config: FavoriteHedgeConfig,
                           trades: List[FavoriteHedgeTrade]) -> FavoriteHedgeResult:
        """Calculate aggregate results."""
        if not trades:
            return FavoriteHedgeResult(config=config)

        total_pnl = sum(t.pnl for t in trades)
        rois = [t.roi for t in trades]

        profits = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        stopped = [t for t in trades if t.outcome == "STOPPED"]
        no_hedge = [t for t in trades if t.outcome == "NO_HEDGE"]

        win_rate = len(profits) / len(trades) if trades else 0
        avg_profit = np.mean([t.pnl for t in profits]) if profits else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0

        sharpe = np.mean(rois) / np.std(rois) * np.sqrt(len(trades)) if np.std(rois) > 0 else 0

        return FavoriteHedgeResult(
            config=config,
            trades=trades,
            total_pnl=total_pnl,
            mean_roi=np.mean(rois),
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            num_trades=len(trades),
            num_stopped=len(stopped),
            num_no_hedge=len(no_hedge),
            sharpe_ratio=sharpe
        )

    def run_parameter_sweep(self) -> List[FavoriteHedgeResult]:
        """Run parameter sweep."""
        results = []

        fav_entries = [0.40, 0.50, 0.60]
        hedge_entries = [0.80, 0.85, 0.90]
        target_rois = [0.05, 0.08, 0.10]
        dynamic_options = [True, False]

        total = len(fav_entries) * len(hedge_entries) * len(target_rois) * len(dynamic_options)
        print(f"\nRunning {total} parameter combinations...")

        count = 0
        for fe in fav_entries:
            for he in hedge_entries:
                for tr in target_rois:
                    for dyn in dynamic_options:
                        config = FavoriteHedgeConfig(
                            favorite_entry_pct=fe,
                            hedge_entry_pct=he,
                            target_roi=tr,
                            dynamic_hedge=dyn
                        )
                        result = self.run_strategy(config)
                        results.append(result)
                        count += 1

                        if count % 10 == 0:
                            print(f"  Progress: {count}/{total}")

        return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: List[FavoriteHedgeResult]):
    """Print formatted results."""
    print("\n" + "=" * 100)
    print("FAVORITE + HEDGE STRATEGY RESULTS")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)

    print(f"\n{'Config':<30} {'PnL':>10} {'ROI':>8} {'Win%':>7} {'Trades':>7} {'Stopped':>8} {'NoHedge':>8} {'Sharpe':>8}")
    print("-" * 100)

    for r in sorted_results[:15]:
        cfg = r.config
        name = f"E{int(cfg.favorite_entry_pct*100)}_H{int(cfg.hedge_entry_pct*100)}_T{int(cfg.target_roi*100)}{'_D' if cfg.dynamic_hedge else ''}"
        print(f"{name:<30} ${r.total_pnl:>9.0f} {r.mean_roi:>7.1%} "
              f"{r.win_rate:>6.1%} {r.num_trades:>7} {r.num_stopped:>8} {r.num_no_hedge:>8} {r.sharpe_ratio:>7.2f}")

    # Best result details
    print("\n" + "=" * 100)
    print("BEST STRATEGY DETAILS")
    print("=" * 100)

    best = sorted_results[0]
    cfg = best.config
    print(f"\nConfiguration:")
    print(f"  Favorite Entry: {cfg.favorite_entry_pct:.0%} of duration")
    print(f"  Hedge Entry: {cfg.hedge_entry_pct:.0%} of duration")
    print(f"  Target ROI: {cfg.target_roi:.0%}")
    print(f"  Dynamic Hedge: {cfg.dynamic_hedge}")

    print(f"\nResults:")
    print(f"  Total PnL: ${best.total_pnl:.0f}")
    print(f"  Mean ROI: {best.mean_roi:.1%}")
    print(f"  Win Rate: {best.win_rate:.1%}")
    print(f"  Avg Profit: ${best.avg_profit:.0f}")
    print(f"  Avg Loss: ${best.avg_loss:.0f}")
    print(f"  Sharpe: {best.sharpe_ratio:.2f}")

    print(f"\nTrade Breakdown:")
    print(f"  Total Trades: {best.num_trades}")
    print(f"  Stopped: {best.num_stopped}")
    print(f"  No Hedge: {best.num_no_hedge}")

    # Sample trades
    print(f"\nSample Trades:")
    for t in best.trades[:8]:
        status = "WIN" if t.pnl > 0 else "LOSS"
        print(f"  {t.token_name:<12} Fav:{t.favorite_entry_price:.0%} Hedge:{t.hedge_entry_price:.0%} "
              f"Cost:${t.total_cost:.0f} PnL:${t.pnl:>6.0f} ({status})")


def export_results(results: List[FavoriteHedgeResult], output_path: str):
    """Export to CSV."""
    rows = []
    for r in results:
        rows.append({
            'fav_entry': r.config.favorite_entry_pct,
            'hedge_entry': r.config.hedge_entry_pct,
            'target_roi': r.config.target_roi,
            'dynamic': r.config.dynamic_hedge,
            'total_pnl': r.total_pnl,
            'mean_roi': r.mean_roi,
            'win_rate': r.win_rate,
            'num_trades': r.num_trades,
            'num_stopped': r.num_stopped,
            'sharpe': r.sharpe_ratio
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
    print("FAVORITE + HEDGE STRATEGY BACKTEST")
    print("Time Decay Harvesting on FDV Markets")
    print("=" * 100)

    backtester = FavoriteHedgeBacktester()
    backtester.load_data()

    results = backtester.run_parameter_sweep()

    print_results(results)

    export_results(results, "../data/favorite_hedge_results.csv")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    profitable = [r for r in results if r.total_pnl > 0]
    print(f"Configurations tested: {len(results)}")
    print(f"Profitable: {len(profitable)} ({len(profitable)/len(results):.1%})")

    if profitable:
        best = max(results, key=lambda r: r.total_pnl)
        print(f"\nBEST CONFIGURATION:")
        print(f"  Favorite Entry: {best.config.favorite_entry_pct:.0%}")
        print(f"  Hedge Entry: {best.config.hedge_entry_pct:.0%}")
        print(f"  Target ROI: {best.config.target_roi:.0%}")
        print(f"  Dynamic Hedge: {best.config.dynamic_hedge}")
        print(f"\n  Total PnL: ${best.total_pnl:.0f}")
        print(f"  Mean ROI: {best.mean_roi:.1%}")
        print(f"  Sharpe: {best.sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()
