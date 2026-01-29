"""
Polymarket FDV Strategy Backtester

Backtest trading strategies for FDV (Fully Diluted Valuation) bucket markets.
Markets: MegaETH, Ethena, EigenLayer, SAFE, etc.

Strategies:
- Position Sizing: S1 (Equal USD), S2 (Equal Shares), S3 (Hybrid)
- Stop Logic: R1 (Stop at 50c), R2 (No Stop)
- Entry Timing: E1 (Early), E2 (Mid), E3 (Late)
- Exit Logic: X1 (Expiry), X2 (Time-based), X3 (Target)
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
    sizing: str = "S1"
    usd_amount: float = 100.0
    shares_amount: int = 100
    hybrid_threshold: float = 0.70

    stop: str = "R2"
    stop_price: float = 0.50

    entry: str = "E1"
    entry_pct: float = 0.25  # Early entry

    exit: str = "X1"
    exit_hours: int = 48
    target_profit_cents: float = 0.05

    def name(self) -> str:
        return f"{self.sizing}_{self.stop}_{self.entry}_{self.exit}"


@dataclass
class Trade:
    """Single trade record."""
    event_slug: str
    token_name: str
    bucket_label: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    shares: float
    usd_invested: float
    side: str
    outcome: str
    pnl: float = 0.0
    roi: float = 0.0
    won_bucket: bool = False


@dataclass
class BacktestResult:
    """Results for a single strategy."""
    config: StrategyConfig
    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    mean_roi: float = 0.0
    median_roi: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    num_wins: int = 0
    num_losses: int = 0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0


# =============================================================================
# FDV MARKET DATA
# =============================================================================

# Historical FDV events with actual outcomes
FDV_EVENTS = [
    {
        "slug": "safe-market-cap-fdv-one-week-after-launch",
        "token_name": "SAFE",
        "start_date": "2024-03-26",
        "end_date": "2024-12-31",
        "buckets": [
            {"label": "<$1B", "low": 0, "high": 1e9, "initial_price": 0.45},
            {"label": "$1-2B", "low": 1e9, "high": 2e9, "initial_price": 0.25},
            {"label": "$2-3B", "low": 2e9, "high": 3e9, "initial_price": 0.15},
            {"label": "$3-4B", "low": 3e9, "high": 4e9, "initial_price": 0.08},
            {"label": "$4-5B", "low": 4e9, "high": 5e9, "initial_price": 0.04},
            {"label": ">$5B", "low": 5e9, "high": None, "initial_price": 0.03},
        ],
        "actual_fdv": 1.0e9,
        "winning_bucket": 0,  # <$1B
        "volume": 116716
    },
    {
        "slug": "ethena-market-cap-fdv-one-day-after-airdrop",
        "token_name": "Ethena",
        "start_date": "2024-03-27",
        "end_date": "2024-12-31",
        "buckets": [
            {"label": "<$5B", "low": 0, "high": 5e9, "initial_price": 0.20},
            {"label": "$5-7.5B", "low": 5e9, "high": 7.5e9, "initial_price": 0.25},
            {"label": "$7.5-10B", "low": 7.5e9, "high": 10e9, "initial_price": 0.20},
            {"label": "$10-12.5B", "low": 10e9, "high": 12.5e9, "initial_price": 0.15},
            {"label": "$12.5-15B", "low": 12.5e9, "high": 15e9, "initial_price": 0.10},
            {"label": ">$15B", "low": 15e9, "high": None, "initial_price": 0.10},
        ],
        "actual_fdv": 14.0e9,
        "winning_bucket": 4,  # $12.5-15B
        "volume": 97997
    },
    {
        "slug": "eigenlayer-market-cap-fdv-one-day-after-launch",
        "token_name": "EigenLayer",
        "start_date": "2024-04-04",
        "end_date": "2024-12-31",
        "buckets": [
            {"label": "<$10B", "low": 0, "high": 10e9, "initial_price": 0.15},
            {"label": "$10-15B", "low": 10e9, "high": 15e9, "initial_price": 0.25},
            {"label": "$15-20B", "low": 15e9, "high": 20e9, "initial_price": 0.20},
            {"label": "$20-25B", "low": 20e9, "high": 25e9, "initial_price": 0.15},
            {"label": "$25-30B", "low": 25e9, "high": 30e9, "initial_price": 0.10},
            {"label": "$30-35B", "low": 30e9, "high": 35e9, "initial_price": 0.07},
            {"label": "$35-40B", "low": 35e9, "high": 40e9, "initial_price": 0.05},
            {"label": ">$40B", "low": 40e9, "high": None, "initial_price": 0.03},
        ],
        "actual_fdv": 11.6e9,
        "winning_bucket": 1,  # $10-15B
        "volume": 6594448
    },
    {
        "slug": "friendtech-fdv-one-day-after-launch",
        "token_name": "FriendTech",
        "start_date": "2024-04-10",
        "end_date": "2024-05-03",
        "buckets": [
            {"label": "<$1B", "low": 0, "high": 1e9, "initial_price": 0.30},
            {"label": "$1-2B", "low": 1e9, "high": 2e9, "initial_price": 0.25},
            {"label": "$2-3B", "low": 2e9, "high": 3e9, "initial_price": 0.20},
            {"label": "$3-4B", "low": 3e9, "high": 4e9, "initial_price": 0.12},
            {"label": "$4-5B", "low": 4e9, "high": 5e9, "initial_price": 0.07},
            {"label": "$5-10B", "low": 5e9, "high": 10e9, "initial_price": 0.04},
            {"label": ">$10B", "low": 10e9, "high": None, "initial_price": 0.02},
        ],
        "actual_fdv": 0.2e9,
        "winning_bucket": 0,  # <$1B (actually ~$200M)
        "volume": 156938
    },
]

# Simulated additional events based on L2 launches
SIMULATED_EVENTS = [
    {
        "token_name": "zkSync",
        "actual_fdv": 1.47e9,
        "buckets_pattern": "standard_l2",
        "initial_prices": [0.35, 0.25, 0.20, 0.12, 0.05, 0.03],
        "winning_bucket": 1,  # $1-2B
    },
    {
        "token_name": "Starknet",
        "actual_fdv": 20.0e9,
        "buckets_pattern": "large_l2",
        "initial_prices": [0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05],
        "winning_bucket": 5,  # $15-20B
    },
    {
        "token_name": "Arbitrum",
        "actual_fdv": 12.5e9,
        "buckets_pattern": "large_l2",
        "initial_prices": [0.08, 0.12, 0.20, 0.25, 0.18, 0.10, 0.05, 0.02],
        "winning_bucket": 3,  # $10-15B
    },
    {
        "token_name": "Optimism",
        "actual_fdv": 19.0e9,
        "buckets_pattern": "large_l2",
        "initial_prices": [0.05, 0.10, 0.12, 0.18, 0.22, 0.18, 0.10, 0.05],
        "winning_bucket": 5,  # $15-20B
    },
    {
        "token_name": "Celestia",
        "actual_fdv": 6.2e9,
        "buckets_pattern": "mid_l1",
        "initial_prices": [0.15, 0.20, 0.25, 0.20, 0.12, 0.08],
        "winning_bucket": 2,  # $5-7.5B
    },
    {
        "token_name": "Sui",
        "actual_fdv": 12.5e9,
        "buckets_pattern": "large_l2",
        "initial_prices": [0.10, 0.15, 0.20, 0.22, 0.18, 0.10, 0.03, 0.02],
        "winning_bucket": 3,  # $10-15B
    },
    {
        "token_name": "Aptos",
        "actual_fdv": 8.5e9,
        "buckets_pattern": "mid_l1",
        "initial_prices": [0.12, 0.18, 0.25, 0.22, 0.15, 0.08],
        "winning_bucket": 2,  # $7.5-10B (mapped to bucket 2)
    },
    {
        "token_name": "LayerZero",
        "actual_fdv": 6.8e9,
        "buckets_pattern": "mid_l1",
        "initial_prices": [0.15, 0.22, 0.28, 0.20, 0.10, 0.05],
        "winning_bucket": 2,  # $5-7.5B
    },
]


# =============================================================================
# PRICE SIMULATION
# =============================================================================

class FDVPriceSimulator:
    """Simulate price paths for FDV bucket markets."""

    @staticmethod
    def simulate_bucket_prices(
        event: Dict,
        num_points: int = 100
    ) -> Dict[int, pd.DataFrame]:
        """
        Simulate price paths for each bucket.
        Returns dict of bucket_idx -> DataFrame with timestamp, price
        """
        buckets = event.get("buckets", [])
        winning_bucket = event.get("winning_bucket", 0)

        # Parse dates
        if "start_date" in event:
            start = datetime.strptime(event["start_date"], "%Y-%m-%d")
            end = datetime.strptime(event["end_date"], "%Y-%m-%d")
        else:
            start = datetime.now() - timedelta(days=30)
            end = datetime.now()

        duration = (end - start).total_seconds()
        times = [start + timedelta(seconds=i * duration / num_points)
                 for i in range(num_points + 1)]

        price_paths = {}

        for idx, bucket in enumerate(buckets):
            initial_price = bucket.get("initial_price", 0.15)
            is_winner = (idx == winning_bucket)

            # Target price based on outcome
            target = 0.95 if is_winner else 0.02

            prices = [initial_price]
            current = initial_price

            for i in range(1, num_points + 1):
                progress = i / num_points

                # Drift toward target
                drift = 0.015 * (target - current) * progress

                # Volatility decreases near resolution
                vol = 0.025 * (1 - progress ** 2)
                noise = np.random.normal(0, vol)

                # News jumps at 60-70% progress
                if 0.58 < progress < 0.62:
                    if is_winner:
                        jump = np.random.uniform(0.05, 0.15)
                    else:
                        jump = np.random.uniform(-0.08, -0.02)
                else:
                    jump = 0

                current = current + drift + noise + jump
                current = np.clip(current, 0.01, 0.99)
                prices.append(current)

            price_paths[idx] = pd.DataFrame({
                'timestamp': times,
                'price': prices
            })

        return price_paths

    @staticmethod
    def get_price_at_time(
        price_path: pd.DataFrame,
        target_time: datetime
    ) -> float:
        """Get price at specific time."""
        if price_path.empty:
            return 0.15

        price_path = price_path.copy()
        price_path['time_diff'] = abs(price_path['timestamp'] - target_time)
        idx = price_path['time_diff'].idxmin()
        return price_path.loc[idx, 'price']


# =============================================================================
# POSITION SIZING
# =============================================================================

class PositionSizer:
    @staticmethod
    def calculate(config: StrategyConfig, price: float) -> Tuple[float, float]:
        if price <= 0 or price >= 1:
            return 0, 0

        if config.sizing == "S1":
            shares = config.usd_amount / price
            return shares, config.usd_amount
        elif config.sizing == "S2":
            usd = config.shares_amount * price
            return config.shares_amount, usd
        elif config.sizing == "S3":
            if price > config.hybrid_threshold:
                shares = config.usd_amount / price
                return shares, config.usd_amount
            else:
                usd = config.shares_amount * price
                return config.shares_amount, usd
        return 0, 0


# =============================================================================
# BUCKET SELECTION STRATEGY
# =============================================================================

class BucketSelector:
    """Select which buckets to bet on based on model prediction."""

    @staticmethod
    def select_underpriced_buckets(
        event: Dict,
        model_prediction_fdv: float,
        max_buckets: int = 3
    ) -> List[int]:
        """
        Select buckets that appear underpriced vs model prediction.
        Returns list of bucket indices to bet YES on.
        """
        buckets = event.get("buckets", [])
        selected = []

        for idx, bucket in enumerate(buckets):
            low = bucket.get("low", 0)
            high = bucket.get("high") or float('inf')
            price = bucket.get("initial_price", 0.15)

            # Check if model prediction falls in this bucket
            if low <= model_prediction_fdv < high:
                # Primary bucket - always include
                selected.append((idx, price, 1.0))
            elif abs((low + high) / 2 - model_prediction_fdv) < model_prediction_fdv * 0.3:
                # Adjacent bucket within 30%
                selected.append((idx, price, 0.5))

        # Sort by value (lower price = better value)
        selected.sort(key=lambda x: x[1])

        return [s[0] for s in selected[:max_buckets]]

    @staticmethod
    def select_by_price_range(
        event: Dict,
        min_price: float = 0.05,
        max_price: float = 0.25,
        max_buckets: int = 3
    ) -> List[int]:
        """Select buckets in target price range."""
        buckets = event.get("buckets", [])
        selected = []

        for idx, bucket in enumerate(buckets):
            price = bucket.get("initial_price", 0.15)
            if min_price <= price <= max_price:
                selected.append((idx, price))

        # Sort by price (lower = better odds potential)
        selected.sort(key=lambda x: x[1])

        return [s[0] for s in selected[:max_buckets]]


# =============================================================================
# BACKTESTER ENGINE
# =============================================================================

class FDVStrategyBacktester:
    """Main backtesting engine for FDV markets."""

    def __init__(self):
        self.events = []
        self.simulator = FDVPriceSimulator()

    def load_data(self):
        """Load FDV events data."""
        # Use real events
        self.events = FDV_EVENTS.copy()

        # Add simulated events
        for sim in SIMULATED_EVENTS:
            event = self._create_event_from_template(sim)
            self.events.append(event)

        print(f"Loaded {len(self.events)} FDV events")

        # Stats
        total_volume = sum(e.get("volume", 100000) for e in self.events)
        print(f"Total volume: ${total_volume:,.0f}")

    def _create_event_from_template(self, template: Dict) -> Dict:
        """Create full event from simulated template."""
        patterns = {
            "standard_l2": [
                {"label": "<$1B", "low": 0, "high": 1e9},
                {"label": "$1-2B", "low": 1e9, "high": 2e9},
                {"label": "$2-3B", "low": 2e9, "high": 3e9},
                {"label": "$3-5B", "low": 3e9, "high": 5e9},
                {"label": "$5-10B", "low": 5e9, "high": 10e9},
                {"label": ">$10B", "low": 10e9, "high": None},
            ],
            "large_l2": [
                {"label": "<$5B", "low": 0, "high": 5e9},
                {"label": "$5-7.5B", "low": 5e9, "high": 7.5e9},
                {"label": "$7.5-10B", "low": 7.5e9, "high": 10e9},
                {"label": "$10-15B", "low": 10e9, "high": 15e9},
                {"label": "$15-20B", "low": 15e9, "high": 20e9},
                {"label": "$20-30B", "low": 20e9, "high": 30e9},
                {"label": "$30-50B", "low": 30e9, "high": 50e9},
                {"label": ">$50B", "low": 50e9, "high": None},
            ],
            "mid_l1": [
                {"label": "<$2.5B", "low": 0, "high": 2.5e9},
                {"label": "$2.5-5B", "low": 2.5e9, "high": 5e9},
                {"label": "$5-7.5B", "low": 5e9, "high": 7.5e9},
                {"label": "$7.5-10B", "low": 7.5e9, "high": 10e9},
                {"label": "$10-15B", "low": 10e9, "high": 15e9},
                {"label": ">$15B", "low": 15e9, "high": None},
            ],
        }

        pattern = patterns.get(template["buckets_pattern"], patterns["standard_l2"])
        initial_prices = template.get("initial_prices", [0.15] * len(pattern))

        buckets = []
        for i, (b, p) in enumerate(zip(pattern, initial_prices)):
            buckets.append({**b, "initial_price": p})

        return {
            "slug": f"{template['token_name'].lower()}-fdv-simulated",
            "token_name": template["token_name"],
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "buckets": buckets,
            "actual_fdv": template["actual_fdv"],
            "winning_bucket": template["winning_bucket"],
            "volume": 500000,
        }

    def run_single_strategy(self, config: StrategyConfig) -> BacktestResult:
        """Run backtest for single strategy."""
        trades = []

        for event in self.events:
            event_trades = self._simulate_event_trades(event, config)
            trades.extend(event_trades)

        return self._calculate_results(config, trades)

    def _simulate_event_trades(
        self,
        event: Dict,
        config: StrategyConfig
    ) -> List[Trade]:
        """Simulate trades for a single event."""
        trades = []

        # Generate price paths
        price_paths = self.simulator.simulate_bucket_prices(event)

        # Select buckets to trade
        buckets_to_trade = BucketSelector.select_by_price_range(
            event,
            min_price=0.05,
            max_price=0.30,
            max_buckets=3
        )

        if not buckets_to_trade:
            return []

        # Parse dates
        start = datetime.strptime(event["start_date"], "%Y-%m-%d")
        end = datetime.strptime(event["end_date"], "%Y-%m-%d")

        # Entry timing
        if config.entry == "E1":
            entry_pct = 0.25
        elif config.entry == "E2":
            entry_pct = 0.50
        else:  # E3
            entry_pct = 0.75

        entry_time = start + (end - start) * entry_pct

        # Trade each selected bucket
        for bucket_idx in buckets_to_trade:
            if bucket_idx >= len(event["buckets"]):
                continue

            bucket = event["buckets"][bucket_idx]
            price_path = price_paths.get(bucket_idx, pd.DataFrame())

            if price_path.empty:
                continue

            entry_price = self.simulator.get_price_at_time(price_path, entry_time)

            # Position sizing
            shares, usd_invested = PositionSizer.calculate(config, entry_price)
            if shares <= 0:
                continue

            # Exit timing
            if config.exit == "X1":
                exit_time = end
            elif config.exit == "X2":
                exit_time = min(entry_time + timedelta(hours=config.exit_hours), end)
            else:  # X3 target
                exit_time = end

            exit_price = self.simulator.get_price_at_time(price_path, exit_time)

            # Check stop
            stopped = False
            if config.stop == "R1" and entry_price > config.stop_price:
                # Check for stop hit
                mask = (price_path['timestamp'] >= entry_time) & (price_path['timestamp'] <= exit_time)
                window = price_path[mask]
                if not window.empty and window['price'].min() <= config.stop_price:
                    stopped = True
                    exit_price = config.stop_price

            # Determine outcome
            won_bucket = (bucket_idx == event["winning_bucket"])

            if stopped:
                outcome = "STOPPED"
                pnl = shares * (exit_price - entry_price)
            elif config.exit == "X1" or exit_time >= end:
                # Held to expiry
                if won_bucket:
                    outcome = "WIN"
                    pnl = shares * (1.0 - entry_price)
                else:
                    outcome = "LOSS"
                    pnl = -usd_invested
            else:
                # Early exit
                outcome = "EARLY_EXIT"
                pnl = shares * (exit_price - entry_price)

            roi = pnl / usd_invested if usd_invested > 0 else 0

            trades.append(Trade(
                event_slug=event["slug"],
                token_name=event["token_name"],
                bucket_label=bucket["label"],
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                shares=shares,
                usd_invested=usd_invested,
                side="YES",
                outcome=outcome,
                pnl=pnl,
                roi=roi,
                won_bucket=won_bucket
            ))

        return trades

    def _calculate_results(
        self,
        config: StrategyConfig,
        trades: List[Trade]
    ) -> BacktestResult:
        """Calculate aggregate results."""
        if not trades:
            return BacktestResult(config=config)

        total_pnl = sum(t.pnl for t in trades)
        rois = [t.roi for t in trades]

        mean_roi = np.mean(rois)
        median_roi = np.median(rois)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        hit_rate = len(wins) / len(trades)

        # Max drawdown
        cumulative = np.cumsum([t.pnl for t in trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Sharpe
        if np.std(rois) > 0:
            sharpe = mean_roi / np.std(rois) * np.sqrt(len(trades))
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return BacktestResult(
            config=config,
            trades=trades,
            total_pnl=total_pnl,
            mean_roi=mean_roi,
            median_roi=median_roi,
            hit_rate=hit_rate,
            max_drawdown=max_drawdown,
            num_trades=len(trades),
            num_wins=len(wins),
            num_losses=len(losses),
            sharpe_ratio=sharpe,
            profit_factor=profit_factor
        )

    def run_all_combinations(self) -> List[BacktestResult]:
        """Run all strategy combinations."""
        results = []

        sizings = ["S1", "S2", "S3"]
        stops = ["R1", "R2"]
        entries = ["E1", "E2", "E3"]
        exits = ["X1", "X2", "X3"]

        total = len(sizings) * len(stops) * len(entries) * len(exits)
        print(f"\nRunning {total} strategy combinations...")

        for i, (s, r, e, x) in enumerate(product(sizings, stops, entries, exits)):
            config = StrategyConfig(
                sizing=s,
                stop=r,
                entry=e,
                exit=x
            )

            result = self.run_single_strategy(config)
            results.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{total}")

        return results


# =============================================================================
# REPORTING
# =============================================================================

def analyze_sensitivity(results: List[BacktestResult]) -> Dict:
    """Analyze parameter sensitivity."""
    sensitivity = {}

    for param in ["sizing", "stop", "entry", "exit"]:
        groups = {}
        for r in results:
            key = getattr(r.config, param)
            if key not in groups:
                groups[key] = {'pnl': [], 'roi': [], 'hit_rate': [], 'sharpe': []}
            groups[key]['pnl'].append(r.total_pnl)
            groups[key]['roi'].append(r.mean_roi)
            groups[key]['hit_rate'].append(r.hit_rate)
            groups[key]['sharpe'].append(r.sharpe_ratio)

        sensitivity[param] = {}
        for key, data in groups.items():
            sensitivity[param][key] = {
                'mean_pnl': np.mean(data['pnl']),
                'std_pnl': np.std(data['pnl']),
                'mean_roi': np.mean(data['roi']),
                'mean_hit_rate': np.mean(data['hit_rate']),
                'mean_sharpe': np.mean(data['sharpe'])
            }

    return sensitivity


def print_results(results: List[BacktestResult], top_n: int = 10):
    """Print formatted results."""
    print("\n" + "=" * 90)
    print("FDV STRATEGY BACKTEST RESULTS")
    print("=" * 90)

    sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)

    print(f"\nTOP {top_n} STRATEGIES BY PnL:")
    print("-" * 90)
    print(f"{'Strategy':<18} {'PnL':>10} {'ROI':>8} {'Hit%':>7} {'MaxDD':>10} {'Sharpe':>8} {'PF':>6} {'Trades':>7}")
    print("-" * 90)

    for r in sorted_results[:top_n]:
        print(f"{r.config.name():<18} ${r.total_pnl:>9.2f} {r.mean_roi:>7.1%} "
              f"{r.hit_rate:>6.1%} ${r.max_drawdown:>9.2f} {r.sharpe_ratio:>7.2f} "
              f"{r.profit_factor:>5.2f} {r.num_trades:>7}")

    print(f"\nBOTTOM 5:")
    print("-" * 90)
    for r in sorted_results[-5:]:
        print(f"{r.config.name():<18} ${r.total_pnl:>9.2f} {r.mean_roi:>7.1%} "
              f"{r.hit_rate:>6.1%} ${r.max_drawdown:>9.2f} {r.sharpe_ratio:>7.2f} "
              f"{r.profit_factor:>5.2f} {r.num_trades:>7}")

    # Sensitivity
    sensitivity = analyze_sensitivity(results)

    print(f"\n" + "=" * 90)
    print("SENSITIVITY ANALYSIS")
    print("=" * 90)

    for param, data in sensitivity.items():
        print(f"\n{param.upper()}:")
        for key, stats in sorted(data.items(), key=lambda x: -x[1]['mean_pnl']):
            print(f"  {key}: PnL ${stats['mean_pnl']:>8.2f} | ROI {stats['mean_roi']:>6.1%} | "
                  f"Hit {stats['mean_hit_rate']:>5.1%} | Sharpe {stats['mean_sharpe']:>5.2f}")


def export_results(results: List[BacktestResult], output_path: str):
    """Export to CSV."""
    rows = []
    for r in results:
        rows.append({
            'strategy': r.config.name(),
            'sizing': r.config.sizing,
            'stop': r.config.stop,
            'entry': r.config.entry,
            'exit': r.config.exit,
            'total_pnl': r.total_pnl,
            'mean_roi': r.mean_roi,
            'median_roi': r.median_roi,
            'hit_rate': r.hit_rate,
            'max_drawdown': r.max_drawdown,
            'num_trades': r.num_trades,
            'sharpe_ratio': r.sharpe_ratio,
            'profit_factor': r.profit_factor
        })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("POLYMARKET FDV STRATEGY BACKTEST")
    print("=" * 90)

    backtester = FDVStrategyBacktester()
    backtester.load_data()

    results = backtester.run_all_combinations()

    print_results(results)

    export_results(results, "../data/fdv_strategy_backtest_results.csv")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    profitable = [r for r in results if r.total_pnl > 0]
    print(f"Strategies tested: {len(results)}")
    print(f"Profitable: {len(profitable)} ({len(profitable)/len(results):.1%})")

    if profitable:
        best = max(results, key=lambda r: r.total_pnl)
        print(f"\nBEST: {best.config.name()}")
        print(f"  PnL: ${best.total_pnl:.2f}")
        print(f"  ROI: {best.mean_roi:.1%}")
        print(f"  Hit Rate: {best.hit_rate:.1%}")
        print(f"  Sharpe: {best.sharpe_ratio:.2f}")


if __name__ == "__main__":
    main()
