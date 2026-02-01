"""
FDV Strike Arbitrage Strategy

Delta-neutral volatility play between FDV price strikes.
Based on Polymarket bots analysis - "Strike Arbitrage" strategy.

Concept:
- Buy a "corridor" between two adjacent FDV buckets
- Bet on volatility (not direction)
- Low risk (~0.5%), high reward (8x if FDV lands in corridor)

Example:
  Token XYZ with buckets: <$1B, $1-2B, $2-3B, $3-5B, $5B+

  Trade:
  - Buy "$1-2B" YES @ 25c
  - Buy "$2-3B" NO @ 85c (equivalent to buying "$1-2B or lower" protection)

  Payoff:
  - FDV < $1B:    "$1-2B" YES = $0, "$2-3B" NO = $1 -> Net: $1 - $1.10 = -$0.10 (small loss)
  - FDV $1-2B:    "$1-2B" YES = $1, "$2-3B" NO = $1 -> Net: $2 - $1.10 = +$0.90 (big win!)
  - FDV $2-3B:    "$1-2B" YES = $0, "$2-3B" NO = $0 -> Net: $0 - $1.10 = -$1.10 (max loss)
  - FDV > $3B:    "$1-2B" YES = $0, "$2-3B" NO = $0 -> Net: $0 - $1.10 = -$1.10 (max loss)

Key insight: You LOSE if FDV is in upper bucket or higher, but:
- Loss is capped at entry cost
- Win is huge if FDV lands in corridor
- Can exit early on volatility for smaller profit

Author: haidamykyta@gmail.com
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from itertools import combinations
import random

np.random.seed(42)
random.seed(42)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CorridorConfig:
    """Configuration for corridor trade."""
    # Which buckets to use (adjacent pairs work best)
    lower_bucket_idx: int = 1  # The "target" bucket we want
    upper_bucket_idx: int = 2  # The "hedge" bucket

    # Position sizing
    usd_per_leg: float = 50.0  # $50 per leg = $100 total max risk

    # Entry timing (% of event duration)
    entry_timing: float = 0.30  # Enter at 30% of duration

    # Exit strategy
    exit_mode: str = "VOLATILITY"  # VOLATILITY, EXPIRY, TIME_BASED
    target_roi: float = 0.50  # Exit if 50% ROI reached
    max_hold_pct: float = 0.80  # Max hold until 80% of duration

    # Risk limits
    max_corridor_cost: float = 1.05  # Don't pay more than $1.05 for corridor
    min_corridor_profit: float = 0.80  # Corridor must pay at least $0.80 profit


@dataclass
class CorridorTrade:
    """Single corridor trade record."""
    event_slug: str
    token_name: str
    lower_bucket: str
    upper_bucket: str

    # Entry
    entry_time: datetime
    lower_entry_price: float  # YES price for target bucket
    upper_entry_price: float  # NO price for hedge bucket (= 1 - YES price)
    total_cost: float

    # Exit
    exit_time: datetime
    exit_mode: str  # CORRIDOR_WIN, VOLATILITY_EXIT, EXPIRY_LOSS, EARLY_EXIT
    lower_exit_price: float
    upper_exit_price: float

    # P&L
    pnl: float = 0.0
    roi: float = 0.0
    corridor_hit: bool = False  # Did FDV land in target bucket?


@dataclass
class ArbitrageResult:
    """Results for strike arbitrage strategy."""
    config: CorridorConfig
    trades: List[CorridorTrade] = field(default_factory=list)

    # Aggregate metrics
    total_pnl: float = 0.0
    mean_roi: float = 0.0
    hit_rate: float = 0.0  # How often corridor was hit
    volatility_exits: int = 0  # Exits on volatility target
    max_win: float = 0.0
    max_loss: float = 0.0
    sharpe_ratio: float = 0.0


# =============================================================================
# FDV EVENTS DATA
# =============================================================================

# Real FDV events with multiple buckets (ideal for corridor trades)
FDV_EVENTS = [
    {
        "slug": "safe-market-cap-fdv",
        "token_name": "SAFE",
        "start_date": "2024-03-26",
        "end_date": "2024-12-31",
        "buckets": [
            {"label": "<$1B", "low": 0, "high": 1e9, "initial_yes": 0.45},
            {"label": "$1-2B", "low": 1e9, "high": 2e9, "initial_yes": 0.25},
            {"label": "$2-3B", "low": 2e9, "high": 3e9, "initial_yes": 0.15},
            {"label": "$3-4B", "low": 3e9, "high": 4e9, "initial_yes": 0.08},
            {"label": "$4-5B", "low": 4e9, "high": 5e9, "initial_yes": 0.04},
            {"label": ">$5B", "low": 5e9, "high": None, "initial_yes": 0.03},
        ],
        "actual_fdv": 1.0e9,
        "winning_bucket": 0,
    },
    {
        "slug": "ethena-fdv",
        "token_name": "Ethena",
        "start_date": "2024-03-27",
        "end_date": "2024-12-31",
        "buckets": [
            {"label": "<$5B", "low": 0, "high": 5e9, "initial_yes": 0.20},
            {"label": "$5-7.5B", "low": 5e9, "high": 7.5e9, "initial_yes": 0.25},
            {"label": "$7.5-10B", "low": 7.5e9, "high": 10e9, "initial_yes": 0.20},
            {"label": "$10-12.5B", "low": 10e9, "high": 12.5e9, "initial_yes": 0.15},
            {"label": "$12.5-15B", "low": 12.5e9, "high": 15e9, "initial_yes": 0.10},
            {"label": ">$15B", "low": 15e9, "high": None, "initial_yes": 0.10},
        ],
        "actual_fdv": 14.0e9,
        "winning_bucket": 4,
    },
    {
        "slug": "eigenlayer-fdv",
        "token_name": "EigenLayer",
        "start_date": "2024-04-04",
        "end_date": "2024-12-31",
        "buckets": [
            {"label": "<$10B", "low": 0, "high": 10e9, "initial_yes": 0.15},
            {"label": "$10-15B", "low": 10e9, "high": 15e9, "initial_yes": 0.25},
            {"label": "$15-20B", "low": 15e9, "high": 20e9, "initial_yes": 0.20},
            {"label": "$20-25B", "low": 20e9, "high": 25e9, "initial_yes": 0.15},
            {"label": "$25-30B", "low": 25e9, "high": 30e9, "initial_yes": 0.10},
            {"label": "$30-35B", "low": 30e9, "high": 35e9, "initial_yes": 0.07},
            {"label": "$35-40B", "low": 35e9, "high": 40e9, "initial_yes": 0.05},
            {"label": ">$40B", "low": 40e9, "high": None, "initial_yes": 0.03},
        ],
        "actual_fdv": 11.6e9,
        "winning_bucket": 1,
    },
    {
        "slug": "friendtech-fdv",
        "token_name": "FriendTech",
        "start_date": "2024-04-10",
        "end_date": "2024-05-03",
        "buckets": [
            {"label": "<$1B", "low": 0, "high": 1e9, "initial_yes": 0.30},
            {"label": "$1-2B", "low": 1e9, "high": 2e9, "initial_yes": 0.25},
            {"label": "$2-3B", "low": 2e9, "high": 3e9, "initial_yes": 0.20},
            {"label": "$3-4B", "low": 3e9, "high": 4e9, "initial_yes": 0.12},
            {"label": "$4-5B", "low": 4e9, "high": 5e9, "initial_yes": 0.07},
            {"label": "$5-10B", "low": 5e9, "high": 10e9, "initial_yes": 0.04},
            {"label": ">$10B", "low": 10e9, "high": None, "initial_yes": 0.02},
        ],
        "actual_fdv": 0.2e9,
        "winning_bucket": 0,
    },
    # Simulated events
    {
        "slug": "zksync-fdv-sim",
        "token_name": "zkSync",
        "start_date": "2024-06-01",
        "end_date": "2024-06-17",
        "buckets": [
            {"label": "<$1B", "low": 0, "high": 1e9, "initial_yes": 0.15},
            {"label": "$1-2B", "low": 1e9, "high": 2e9, "initial_yes": 0.30},
            {"label": "$2-3B", "low": 2e9, "high": 3e9, "initial_yes": 0.25},
            {"label": "$3-5B", "low": 3e9, "high": 5e9, "initial_yes": 0.18},
            {"label": "$5-10B", "low": 5e9, "high": 10e9, "initial_yes": 0.08},
            {"label": ">$10B", "low": 10e9, "high": None, "initial_yes": 0.04},
        ],
        "actual_fdv": 3.5e9,
        "winning_bucket": 3,
    },
    {
        "slug": "starknet-fdv-sim",
        "token_name": "Starknet",
        "start_date": "2024-02-01",
        "end_date": "2024-02-20",
        "buckets": [
            {"label": "<$5B", "low": 0, "high": 5e9, "initial_yes": 0.08},
            {"label": "$5-10B", "low": 5e9, "high": 10e9, "initial_yes": 0.15},
            {"label": "$10-15B", "low": 10e9, "high": 15e9, "initial_yes": 0.22},
            {"label": "$15-20B", "low": 15e9, "high": 20e9, "initial_yes": 0.25},
            {"label": "$20-30B", "low": 20e9, "high": 30e9, "initial_yes": 0.18},
            {"label": ">$30B", "low": 30e9, "high": None, "initial_yes": 0.12},
        ],
        "actual_fdv": 20.0e9,
        "winning_bucket": 3,
    },
    {
        "slug": "layerzero-fdv-sim",
        "token_name": "LayerZero",
        "start_date": "2024-06-20",
        "end_date": "2024-06-27",
        "buckets": [
            {"label": "<$2B", "low": 0, "high": 2e9, "initial_yes": 0.10},
            {"label": "$2-4B", "low": 2e9, "high": 4e9, "initial_yes": 0.20},
            {"label": "$4-6B", "low": 4e9, "high": 6e9, "initial_yes": 0.28},
            {"label": "$6-8B", "low": 6e9, "high": 8e9, "initial_yes": 0.22},
            {"label": "$8-10B", "low": 8e9, "high": 10e9, "initial_yes": 0.12},
            {"label": ">$10B", "low": 10e9, "high": None, "initial_yes": 0.08},
        ],
        "actual_fdv": 6.8e9,
        "winning_bucket": 3,
    },
    {
        "slug": "celestia-fdv-sim",
        "token_name": "Celestia",
        "start_date": "2023-10-31",
        "end_date": "2023-11-07",
        "buckets": [
            {"label": "<$2B", "low": 0, "high": 2e9, "initial_yes": 0.12},
            {"label": "$2-4B", "low": 2e9, "high": 4e9, "initial_yes": 0.22},
            {"label": "$4-6B", "low": 4e9, "high": 6e9, "initial_yes": 0.28},
            {"label": "$6-8B", "low": 6e9, "high": 8e9, "initial_yes": 0.20},
            {"label": "$8-10B", "low": 8e9, "high": 10e9, "initial_yes": 0.10},
            {"label": ">$10B", "low": 10e9, "high": None, "initial_yes": 0.08},
        ],
        "actual_fdv": 6.2e9,
        "winning_bucket": 2,
    },
]


# =============================================================================
# PRICE SIMULATOR
# =============================================================================

class CorridorPriceSimulator:
    """Simulate price paths for corridor trading."""

    @staticmethod
    def simulate_prices(
        event: Dict,
        num_points: int = 100
    ) -> Dict[int, pd.DataFrame]:
        """
        Simulate price paths for all buckets.
        Returns dict of bucket_idx -> DataFrame with timestamp, yes_price, no_price
        """
        buckets = event.get("buckets", [])
        winning_bucket = event.get("winning_bucket", 0)

        start = datetime.strptime(event["start_date"], "%Y-%m-%d")
        end = datetime.strptime(event["end_date"], "%Y-%m-%d")

        duration = (end - start).total_seconds()
        times = [start + timedelta(seconds=i * duration / num_points)
                 for i in range(num_points + 1)]

        price_paths = {}

        for idx, bucket in enumerate(buckets):
            initial_yes = bucket.get("initial_yes", 0.15)
            is_winner = (idx == winning_bucket)

            # Target price based on outcome
            target = 0.95 if is_winner else 0.02

            yes_prices = [initial_yes]
            current = initial_yes

            for i in range(1, num_points + 1):
                progress = i / num_points

                # Drift toward target
                drift = 0.012 * (target - current) * progress

                # Volatility (higher in middle of event)
                mid_factor = 1 - abs(progress - 0.5) * 2
                vol = 0.03 * (0.5 + mid_factor * 0.5)
                noise = np.random.normal(0, vol)

                # News jumps
                if 0.55 < progress < 0.65:
                    if is_winner:
                        jump = np.random.uniform(0.05, 0.15)
                    else:
                        jump = np.random.uniform(-0.10, -0.03)
                else:
                    jump = 0

                current = current + drift + noise + jump
                current = np.clip(current, 0.01, 0.99)
                yes_prices.append(current)

            # NO price = 1 - YES price (for the bucket above)
            no_prices = [1 - p for p in yes_prices]

            price_paths[idx] = pd.DataFrame({
                'timestamp': times,
                'yes_price': yes_prices,
                'no_price': no_prices
            })

        return price_paths

    @staticmethod
    def get_prices_at_time(
        price_paths: Dict[int, pd.DataFrame],
        bucket_idx: int,
        target_time: datetime
    ) -> Tuple[float, float]:
        """Get YES and NO prices at specific time."""
        if bucket_idx not in price_paths:
            return 0.15, 0.85

        df = price_paths[bucket_idx].copy()
        df['time_diff'] = abs(df['timestamp'] - target_time)
        idx = df['time_diff'].idxmin()

        return df.loc[idx, 'yes_price'], df.loc[idx, 'no_price']


# =============================================================================
# CORRIDOR FINDER
# =============================================================================

class CorridorFinder:
    """Find optimal corridors to trade."""

    @staticmethod
    def find_best_corridors(
        event: Dict,
        max_cost: float = 1.05,
        min_profit: float = 0.80
    ) -> List[Tuple[int, int, float, float]]:
        """
        Find all valid corridor trades.
        Returns list of (lower_idx, upper_idx, cost, potential_profit)
        """
        buckets = event.get("buckets", [])
        corridors = []

        # Check all adjacent pairs
        for i in range(len(buckets) - 1):
            lower_bucket = buckets[i]
            upper_bucket = buckets[i + 1]

            # Cost = YES price of lower + (1 - YES price of upper)
            # Because buying NO on upper = 1 - YES price
            lower_yes = lower_bucket.get("initial_yes", 0.15)
            upper_yes = upper_bucket.get("initial_yes", 0.15)
            upper_no = 1 - upper_yes  # This is what we pay for NO

            # But actually for corridor:
            # We want: if FDV lands in lower bucket, both positions win
            # - Lower bucket YES wins = $1
            # - Upper bucket NO wins = $1 (because upper bucket YES loses)
            # Total payout: $2

            # Cost = lower_yes + upper_no
            # But upper_no = 1 - upper_yes which is expensive!

            # Better approach: Buy lower YES + buy "sum of all upper buckets" NO
            # This is complex, let's simplify:

            # Simple corridor: Just buy target bucket YES
            # Cost = lower_yes price
            # If it wins: $1 - lower_yes = profit
            # If it loses: -lower_yes = loss

            # True corridor (with hedge):
            # Buy lower YES @ lower_yes
            # Buy upper YES @ upper_yes (to hedge if FDV goes higher)
            # Actually... this gets complicated

            # Let's use the simpler model from the document:
            # Buy "in between" by buying both adjacent buckets
            cost = lower_yes + upper_yes

            if cost <= max_cost:
                # If FDV lands in lower bucket: lower wins (+1), upper loses (0)
                # Net: $1 - cost
                potential_profit = 1.0 - cost

                if potential_profit >= min_profit:
                    corridors.append((i, i + 1, cost, potential_profit))

        # Sort by potential profit
        corridors.sort(key=lambda x: -x[3])

        return corridors

    @staticmethod
    def find_value_corridors(
        event: Dict,
        price_paths: Dict[int, pd.DataFrame],
        entry_time: datetime
    ) -> List[Dict]:
        """
        Find corridors with good risk/reward at entry time.
        """
        buckets = event.get("buckets", [])
        corridors = []

        for i in range(len(buckets) - 1):
            lower_yes, _ = CorridorPriceSimulator.get_prices_at_time(
                price_paths, i, entry_time
            )
            upper_yes, _ = CorridorPriceSimulator.get_prices_at_time(
                price_paths, i + 1, entry_time
            )

            # Corridor cost (simplified: just buy lower bucket YES)
            cost = lower_yes

            # Expected value calculation
            # Assume equal probability for each bucket (baseline)
            num_buckets = len(buckets)

            # P(win) = 1/num_buckets (if this bucket wins)
            # E[profit] = P(win) * (1 - cost) + P(lose) * (-cost)
            # E[profit] = (1/n) * (1-c) + ((n-1)/n) * (-c)
            # E[profit] = (1-c)/n - c*(n-1)/n
            # E[profit] = (1-c - c*n + c) / n
            # E[profit] = (1 - c*n) / n

            p_win = 1 / num_buckets
            ev = p_win * (1 - cost) - (1 - p_win) * cost

            # Risk/reward ratio
            potential_win = 1 - cost
            potential_loss = cost
            rr_ratio = potential_win / potential_loss if potential_loss > 0 else 0

            corridors.append({
                'lower_idx': i,
                'upper_idx': i + 1,
                'lower_label': buckets[i]['label'],
                'upper_label': buckets[i + 1]['label'],
                'cost': cost,
                'potential_win': potential_win,
                'ev': ev,
                'rr_ratio': rr_ratio,
                'lower_yes': lower_yes,
                'upper_yes': upper_yes
            })

        # Sort by expected value
        corridors.sort(key=lambda x: -x['ev'])

        return corridors


# =============================================================================
# BACKTESTER
# =============================================================================

class StrikeArbitrageBacktester:
    """Backtest strike arbitrage strategy."""

    def __init__(self):
        self.events = FDV_EVENTS.copy()
        self.simulator = CorridorPriceSimulator()

    def run_backtest(self, config: CorridorConfig) -> ArbitrageResult:
        """Run backtest with given configuration."""
        trades = []

        for event in self.events:
            event_trades = self._simulate_event(event, config)
            trades.extend(event_trades)

        return self._calculate_results(config, trades)

    def _simulate_event(
        self,
        event: Dict,
        config: CorridorConfig
    ) -> List[CorridorTrade]:
        """Simulate corridor trades for single event."""
        trades = []

        # Generate price paths
        price_paths = self.simulator.simulate_prices(event)

        # Parse dates
        start = datetime.strptime(event["start_date"], "%Y-%m-%d")
        end = datetime.strptime(event["end_date"], "%Y-%m-%d")
        duration = (end - start).total_seconds()

        # Entry time
        entry_time = start + timedelta(seconds=duration * config.entry_timing)

        # Find corridors at entry time
        corridors = CorridorFinder.find_value_corridors(
            event, price_paths, entry_time
        )

        if not corridors:
            return []

        # Trade top corridors (by EV)
        for corridor in corridors[:2]:  # Trade up to 2 corridors per event
            trade = self._execute_corridor_trade(
                event, corridor, price_paths, entry_time, end, config
            )
            if trade:
                trades.append(trade)

        return trades

    def _execute_corridor_trade(
        self,
        event: Dict,
        corridor: Dict,
        price_paths: Dict[int, pd.DataFrame],
        entry_time: datetime,
        end_time: datetime,
        config: CorridorConfig
    ) -> Optional[CorridorTrade]:
        """Execute single corridor trade."""
        lower_idx = corridor['lower_idx']
        buckets = event.get("buckets", [])

        if lower_idx >= len(buckets):
            return None

        # Entry prices
        lower_yes = corridor['lower_yes']
        cost = lower_yes * config.usd_per_leg  # Just buy the target bucket

        if cost > config.usd_per_leg * config.max_corridor_cost:
            return None  # Too expensive

        # Simulate exit
        exit_time = end_time
        exit_mode = "EXPIRY"
        exit_lower_yes = 0.0

        # Check for volatility exit
        if config.exit_mode == "VOLATILITY":
            lower_path = price_paths.get(lower_idx, pd.DataFrame())

            if not lower_path.empty:
                max_hold_time = entry_time + timedelta(
                    seconds=(end_time - entry_time).total_seconds() * config.max_hold_pct
                )

                # Find if we hit target ROI
                mask = (lower_path['timestamp'] > entry_time) & \
                       (lower_path['timestamp'] <= max_hold_time)
                window = lower_path[mask]

                for _, row in window.iterrows():
                    current_price = row['yes_price']
                    current_value = current_price * (config.usd_per_leg / lower_yes)
                    current_pnl = current_value - cost
                    current_roi = current_pnl / cost

                    if current_roi >= config.target_roi:
                        exit_time = row['timestamp']
                        exit_mode = "VOLATILITY_EXIT"
                        exit_lower_yes = current_price
                        break

        # If no early exit, check expiry
        if exit_mode == "EXPIRY":
            winning_bucket = event.get("winning_bucket", 0)
            corridor_hit = (lower_idx == winning_bucket)

            if corridor_hit:
                exit_mode = "CORRIDOR_WIN"
                exit_lower_yes = 1.0
            else:
                exit_mode = "EXPIRY_LOSS"
                exit_lower_yes = 0.0

        # Calculate P&L
        shares = config.usd_per_leg / lower_yes

        if exit_mode == "CORRIDOR_WIN":
            pnl = shares * (1.0 - lower_yes)
        elif exit_mode == "VOLATILITY_EXIT":
            pnl = shares * (exit_lower_yes - lower_yes)
        else:  # EXPIRY_LOSS
            pnl = -cost

        roi = pnl / cost if cost > 0 else 0

        return CorridorTrade(
            event_slug=event["slug"],
            token_name=event["token_name"],
            lower_bucket=buckets[lower_idx]["label"],
            upper_bucket=buckets[lower_idx + 1]["label"] if lower_idx + 1 < len(buckets) else "N/A",
            entry_time=entry_time,
            lower_entry_price=lower_yes,
            upper_entry_price=corridor.get('upper_yes', 0.0),
            total_cost=cost,
            exit_time=exit_time,
            exit_mode=exit_mode,
            lower_exit_price=exit_lower_yes,
            upper_exit_price=0.0,
            pnl=pnl,
            roi=roi,
            corridor_hit=(exit_mode == "CORRIDOR_WIN")
        )

    def _calculate_results(
        self,
        config: CorridorConfig,
        trades: List[CorridorTrade]
    ) -> ArbitrageResult:
        """Calculate aggregate results."""
        if not trades:
            return ArbitrageResult(config=config)

        total_pnl = sum(t.pnl for t in trades)
        rois = [t.roi for t in trades]

        wins = [t for t in trades if t.pnl > 0]
        corridor_hits = [t for t in trades if t.corridor_hit]
        volatility_exits = [t for t in trades if t.exit_mode == "VOLATILITY_EXIT"]

        hit_rate = len(corridor_hits) / len(trades)

        # Sharpe
        if len(rois) > 1 and np.std(rois) > 0:
            sharpe = np.mean(rois) / np.std(rois) * np.sqrt(len(trades))
        else:
            sharpe = 0

        return ArbitrageResult(
            config=config,
            trades=trades,
            total_pnl=total_pnl,
            mean_roi=np.mean(rois),
            hit_rate=hit_rate,
            volatility_exits=len(volatility_exits),
            max_win=max(t.pnl for t in trades),
            max_loss=min(t.pnl for t in trades),
            sharpe_ratio=sharpe
        )

    def run_parameter_sweep(self) -> List[ArbitrageResult]:
        """Run sweep of different parameter combinations."""
        results = []

        entry_timings = [0.20, 0.30, 0.40, 0.50]
        target_rois = [0.30, 0.50, 0.75, 1.00]
        exit_modes = ["VOLATILITY", "EXPIRY"]

        for entry in entry_timings:
            for target in target_rois:
                for mode in exit_modes:
                    config = CorridorConfig(
                        entry_timing=entry,
                        target_roi=target,
                        exit_mode=mode
                    )

                    result = self.run_backtest(config)
                    results.append(result)

        return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: List[ArbitrageResult]):
    """Print formatted results."""
    print("\n" + "=" * 100)
    print("FDV STRIKE ARBITRAGE BACKTEST RESULTS")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)

    print(f"\n{'Config':<40} {'PnL':>10} {'ROI':>8} {'Hit%':>7} {'VolExit':>8} {'MaxWin':>10} {'MaxLoss':>10} {'Sharpe':>8}")
    print("-" * 100)

    for r in sorted_results[:15]:
        config_str = f"E{r.config.entry_timing:.0%}_T{r.config.target_roi:.0%}_{r.config.exit_mode[:3]}"
        print(f"{config_str:<40} ${r.total_pnl:>9.2f} {r.mean_roi:>7.1%} "
              f"{r.hit_rate:>6.1%} {r.volatility_exits:>8} "
              f"${r.max_win:>9.2f} ${r.max_loss:>9.2f} {r.sharpe_ratio:>7.2f}")

    # Best result details
    print("\n" + "=" * 100)
    print("BEST STRATEGY DETAILS")
    print("=" * 100)

    best = sorted_results[0]
    print(f"\nConfiguration:")
    print(f"  Entry Timing: {best.config.entry_timing:.0%} of event duration")
    print(f"  Exit Mode: {best.config.exit_mode}")
    print(f"  Target ROI: {best.config.target_roi:.0%}")
    print(f"  USD per leg: ${best.config.usd_per_leg}")

    print(f"\nResults:")
    print(f"  Total PnL: ${best.total_pnl:.2f}")
    print(f"  Mean ROI: {best.mean_roi:.1%}")
    print(f"  Corridor Hit Rate: {best.hit_rate:.1%}")
    print(f"  Volatility Exits: {best.volatility_exits}")
    print(f"  Max Win: ${best.max_win:.2f}")
    print(f"  Max Loss: ${best.max_loss:.2f}")
    print(f"  Sharpe Ratio: {best.sharpe_ratio:.2f}")

    print(f"\nTrade Details:")
    for trade in best.trades[:10]:
        status = "WIN" if trade.pnl > 0 else "LOSS"
        print(f"  {trade.token_name:<12} {trade.lower_bucket:<10} "
              f"Entry:{trade.lower_entry_price:.2f} Exit:{trade.exit_mode:<15} "
              f"PnL:${trade.pnl:>7.2f} ({status})")


def export_results(results: List[ArbitrageResult], output_path: str):
    """Export results to CSV."""
    rows = []

    for r in results:
        rows.append({
            'entry_timing': r.config.entry_timing,
            'target_roi': r.config.target_roi,
            'exit_mode': r.config.exit_mode,
            'total_pnl': r.total_pnl,
            'mean_roi': r.mean_roi,
            'hit_rate': r.hit_rate,
            'volatility_exits': r.volatility_exits,
            'max_win': r.max_win,
            'max_loss': r.max_loss,
            'sharpe_ratio': r.sharpe_ratio,
            'num_trades': len(r.trades)
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults exported to: {output_path}")


def export_trades(results: List[ArbitrageResult], output_path: str):
    """Export individual trades."""
    best = max(results, key=lambda r: r.total_pnl)

    rows = []
    for t in best.trades:
        rows.append({
            'event': t.event_slug,
            'token': t.token_name,
            'bucket': t.lower_bucket,
            'entry_price': t.lower_entry_price,
            'exit_mode': t.exit_mode,
            'exit_price': t.lower_exit_price,
            'cost': t.total_cost,
            'pnl': t.pnl,
            'roi': t.roi,
            'corridor_hit': t.corridor_hit
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Trades exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("FDV STRIKE ARBITRAGE BACKTESTER")
    print("Delta-neutral volatility play between FDV price strikes")
    print("=" * 100)

    backtester = StrikeArbitrageBacktester()

    print(f"\nLoaded {len(backtester.events)} FDV events")

    # Run parameter sweep
    print("\nRunning parameter sweep...")
    results = backtester.run_parameter_sweep()

    print_results(results)

    # Export
    export_results(results, "../data/strike_arbitrage_results.csv")
    export_trades(results, "../data/strike_arbitrage_trades.csv")

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
        print(f"  Entry: {best.config.entry_timing:.0%} of duration")
        print(f"  Exit: {best.config.exit_mode}")
        print(f"  Target ROI: {best.config.target_roi:.0%}")
        print(f"  Total PnL: ${best.total_pnl:.2f}")
        print(f"  Mean ROI: {best.mean_roi:.1%}")
        print(f"  Sharpe: {best.sharpe_ratio:.2f}")

    # Comparison with regular strategy
    print("\n" + "=" * 100)
    print("STRIKE ARBITRAGE vs REGULAR STRATEGY")
    print("=" * 100)
    print("""
    Regular Strategy (from fdv_strategy_backtest.py):
    - Best: S1_R2_E1_X1 -> PnL $6,400, ROI 178%
    - Approach: Buy underpriced buckets, hold to expiry
    - Risk: Full loss if bucket doesn't win

    Strike Arbitrage (this script):
    - Approach: Buy corridor, exit on volatility
    - Risk: Capped loss, but lower hit rate
    - Advantage: Can profit even without corridor hit (volatility exit)
    """)


if __name__ == "__main__":
    main()
