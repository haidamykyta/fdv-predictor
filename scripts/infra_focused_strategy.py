"""
Infra-Focused FDV Strategy

Backtest V2 showed Infra tokens generated $6,224 out of $7,499 total PnL (83%).
This strategy focuses exclusively on Infrastructure tokens.

Infra tokens: LINK, TAO, WLD, RENDER, FIL, ZRO, PYTH

Key characteristics:
- Higher FDV/MCap ratios (more unlocks = more volatility)
- Stronger correlation with BTC/ETH cycles
- Less memetic, more fundamentals-driven

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
# INFRA TOKEN CHARACTERISTICS
# =============================================================================

INFRA_TOKENS = {
    "LINK": {
        "name": "Chainlink",
        "fdv_billions": 9.75,
        "fdv_mcap_ratio": 1.41,
        "volatility_score": 0.7,  # Lower = more stable
        "unlock_risk": "medium",
    },
    "TAO": {
        "name": "Bittensor",
        "fdv_billions": 3.98,
        "fdv_mcap_ratio": 2.19,
        "volatility_score": 0.9,  # AI narrative = high vol
        "unlock_risk": "high",
    },
    "WLD": {
        "name": "World",
        "fdv_billions": 3.93,
        "fdv_mcap_ratio": 3.59,  # Highest ratio = most unlocks
        "volatility_score": 0.95,
        "unlock_risk": "very_high",
    },
    "RENDER": {
        "name": "Render",
        "fdv_billions": 0.80,
        "fdv_mcap_ratio": 1.03,  # Almost fully diluted
        "volatility_score": 0.6,
        "unlock_risk": "low",
    },
    "FIL": {
        "name": "Filecoin",
        "fdv_billions": 2.05,
        "fdv_mcap_ratio": 2.63,
        "volatility_score": 0.75,
        "unlock_risk": "high",
    },
    "ZRO": {
        "name": "LayerZero",
        "fdv_billions": 1.73,
        "fdv_mcap_ratio": 4.94,  # Second highest ratio
        "volatility_score": 0.85,
        "unlock_risk": "very_high",
    },
    "PYTH": {
        "name": "Pyth Network",
        "fdv_billions": 0.51,
        "fdv_mcap_ratio": 1.74,
        "volatility_score": 0.65,
        "unlock_risk": "medium",
    },
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InfraStrategyConfig:
    """Infra-focused strategy configuration."""

    # Position sizing
    base_usd: float = 100.0

    # Volatility adjustment
    vol_adjust: bool = True  # Size inversely to volatility

    # Entry timing
    entry_pct: float = 0.25  # Earlier entry for Infra (was 0.50)

    # FDV/MCap filter
    max_fdv_mcap_ratio: float = 5.0  # Skip extreme unlock risk
    min_fdv_mcap_ratio: float = 1.0

    # Bucket selection
    bucket_strategy: str = "ADJACENT"  # ADJACENT, CURRENT, HIGHER

    # Exit
    target_roi: float = 0.50

    def name(self) -> str:
        vol = "VOL" if self.vol_adjust else "FLAT"
        return f"INFRA_{vol}_{self.bucket_strategy}"


@dataclass
class Trade:
    """Single trade record."""
    token_symbol: str
    token_name: str
    bucket_label: str
    bucket_idx: int

    entry_price: float
    exit_price: float
    shares: float
    usd_invested: float

    fdv_mcap_ratio: float
    volatility_score: float

    outcome: str
    pnl: float = 0.0
    roi: float = 0.0
    won_bucket: bool = False


@dataclass
class BacktestResult:
    """Results for strategy."""
    config: InfraStrategyConfig
    trades: List[Trade] = field(default_factory=list)

    total_pnl: float = 0.0
    mean_roi: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    sharpe_ratio: float = 0.0

    # Infra-specific
    by_token: Dict[str, Dict] = field(default_factory=dict)
    by_unlock_risk: Dict[str, Dict] = field(default_factory=dict)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_infra_events() -> List[Dict]:
    """Load only Infra events from FDV data."""
    data_file = Path(__file__).parent.parent / "knowledge_base" / "data" / "fdv_polymarket_events.json"

    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)

    events = data.get('events', [])

    # Filter to Infra only
    infra_events = [e for e in events if e['category'] == 'Infra']

    # Enrich with characteristics
    for e in infra_events:
        symbol = e['symbol']
        if symbol in INFRA_TOKENS:
            chars = INFRA_TOKENS[symbol]
            e['volatility_score'] = chars['volatility_score']
            e['unlock_risk'] = chars['unlock_risk']
        else:
            e['volatility_score'] = 0.7
            e['unlock_risk'] = 'medium'

        # Add initial prices
        buckets = e['buckets']
        current_bucket = e['current_bucket']

        for i, b in enumerate(buckets):
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

        total = sum(b['initial_yes'] for b in buckets)
        for b in buckets:
            b['initial_yes'] = b['initial_yes'] / total

    return infra_events


# =============================================================================
# PRICE SIMULATION (Infra-specific)
# =============================================================================

class InfraPriceSimulator:
    """Simulate price paths for Infra tokens with volatility adjustment."""

    @staticmethod
    def simulate_prices(event: Dict, num_points: int = 100) -> Dict[int, pd.DataFrame]:
        """Generate price paths with Infra characteristics."""
        buckets = event.get('buckets', [])
        winning_bucket = event.get('current_bucket', 0)
        vol_score = event.get('volatility_score', 0.7)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)
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
                drift = 0.015 * (target - current) * progress

                # Volatility scaled by token characteristic
                base_vol = 0.03 * (1 - progress ** 2)
                vol = base_vol * vol_score
                noise = np.random.normal(0, vol)

                # News jump at 60% (stronger for high-vol tokens)
                if 0.58 < progress < 0.62:
                    jump_size = 0.08 * vol_score
                    jump = np.random.uniform(jump_size, jump_size * 1.5) if is_winner else np.random.uniform(-jump_size * 1.2, -jump_size * 0.5)
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
        if price_path.empty:
            return 0.15
        df = price_path.copy()
        df['diff'] = abs(df['timestamp'] - target_time)
        return df.loc[df['diff'].idxmin(), 'price']


# =============================================================================
# BACKTESTER
# =============================================================================

class InfraBacktester:
    """Backtest Infra-focused strategy."""

    def __init__(self):
        self.events = []
        self.simulator = InfraPriceSimulator()

    def load_data(self):
        """Load Infra events."""
        self.events = load_infra_events()
        print(f"Loaded {len(self.events)} Infra events")

        print("\nInfra tokens:")
        for e in self.events:
            print(f"  {e['symbol']:>6}: ${e['current_fdv']/1e9:.2f}B "
                  f"(FDV/MCap: {e['fdv_mcap_ratio']:.1f}x, Vol: {e.get('volatility_score', 0.7):.2f})")

    def run_strategy(self, config: InfraStrategyConfig) -> BacktestResult:
        """Run backtest for single strategy."""
        trades = []

        for event in self.events:
            # Skip if outside FDV/MCap filter
            ratio = event['fdv_mcap_ratio']
            if ratio < config.min_fdv_mcap_ratio or ratio > config.max_fdv_mcap_ratio:
                continue

            event_trades = self._simulate_event(event, config)
            trades.extend(event_trades)

        return self._calculate_results(config, trades)

    def _simulate_event(self, event: Dict, config: InfraStrategyConfig) -> List[Trade]:
        """Simulate trades for single Infra event."""
        trades = []

        price_paths = self.simulator.simulate_prices(event)

        start = datetime(2024, 1, 1)
        end = datetime(2024, 3, 1)
        duration = (end - start).total_seconds()

        entry_time = start + timedelta(seconds=duration * config.entry_pct)
        exit_time = end  # Hold to expiry

        winning_bucket = event['current_bucket']
        vol_score = event.get('volatility_score', 0.7)

        # Select bucket(s) to trade based on strategy
        if config.bucket_strategy == "CURRENT":
            buckets_to_trade = [winning_bucket]
        elif config.bucket_strategy == "ADJACENT":
            # Trade current and one adjacent bucket
            buckets_to_trade = [winning_bucket]
            if winning_bucket > 0:
                buckets_to_trade.append(winning_bucket - 1)
            elif winning_bucket < len(event['buckets']) - 1:
                buckets_to_trade.append(winning_bucket + 1)
        else:  # HIGHER
            # Trade buckets above current
            buckets_to_trade = [i for i in range(winning_bucket, min(winning_bucket + 2, len(event['buckets'])))]

        for bucket_idx in buckets_to_trade:
            if bucket_idx >= len(event['buckets']):
                continue

            bucket = event['buckets'][bucket_idx]
            price_path = price_paths.get(bucket_idx, pd.DataFrame())

            if price_path.empty:
                continue

            entry_price = self.simulator.get_price_at_time(price_path, entry_time)

            # Skip if entry price too high
            if entry_price > 0.40:
                continue

            # Position sizing with volatility adjustment
            if config.vol_adjust:
                # Size inversely to volatility
                size_mult = 1.5 - vol_score  # High vol = smaller size
                usd = config.base_usd * max(0.5, size_mult)
            else:
                usd = config.base_usd

            shares = usd / entry_price

            # Exit at expiry
            exit_price = self.simulator.get_price_at_time(price_path, exit_time)

            # Outcome
            won_bucket = (bucket_idx == winning_bucket)

            if won_bucket:
                outcome = "WIN"
                pnl = shares * (1.0 - entry_price)
            else:
                outcome = "LOSS"
                pnl = -usd

            roi = pnl / usd if usd > 0 else 0

            trades.append(Trade(
                token_symbol=event['symbol'],
                token_name=event.get('name', event['symbol']),
                bucket_label=bucket['label'],
                bucket_idx=bucket_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                usd_invested=usd,
                fdv_mcap_ratio=event['fdv_mcap_ratio'],
                volatility_score=vol_score,
                outcome=outcome,
                pnl=pnl,
                roi=roi,
                won_bucket=won_bucket
            ))

        return trades

    def _calculate_results(self, config: InfraStrategyConfig, trades: List[Trade]) -> BacktestResult:
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

        # By token
        by_token = {}
        for t in trades:
            sym = t.token_symbol
            if sym not in by_token:
                by_token[sym] = {'pnl': 0, 'trades': 0, 'wins': 0}
            by_token[sym]['pnl'] += t.pnl
            by_token[sym]['trades'] += 1
            if t.pnl > 0:
                by_token[sym]['wins'] += 1

        # By unlock risk
        by_risk = {}
        for t in trades:
            risk = INFRA_TOKENS.get(t.token_symbol, {}).get('unlock_risk', 'medium')
            if risk not in by_risk:
                by_risk[risk] = {'pnl': 0, 'trades': 0, 'wins': 0}
            by_risk[risk]['pnl'] += t.pnl
            by_risk[risk]['trades'] += 1
            if t.pnl > 0:
                by_risk[risk]['wins'] += 1

        return BacktestResult(
            config=config,
            trades=trades,
            total_pnl=total_pnl,
            mean_roi=np.mean(rois),
            hit_rate=hit_rate,
            max_drawdown=max_dd,
            num_trades=len(trades),
            sharpe_ratio=sharpe,
            by_token=by_token,
            by_unlock_risk=by_risk
        )

    def run_all_combinations(self) -> List[BacktestResult]:
        """Run all strategy combinations."""
        results = []

        vol_adjusts = [True, False]
        entry_pcts = [0.25, 0.50, 0.75]
        bucket_strategies = ["CURRENT", "ADJACENT", "HIGHER"]

        total = len(vol_adjusts) * len(entry_pcts) * len(bucket_strategies)
        print(f"\nRunning {total} strategy combinations...")

        for va, ep, bs in product(vol_adjusts, entry_pcts, bucket_strategies):
            config = InfraStrategyConfig(
                vol_adjust=va,
                entry_pct=ep,
                bucket_strategy=bs
            )
            result = self.run_strategy(config)
            results.append(result)

        return results


# =============================================================================
# REPORTING
# =============================================================================

def print_results(results: List[BacktestResult], top_n: int = 10):
    """Print formatted results."""
    print("\n" + "=" * 100)
    print("INFRA-FOCUSED STRATEGY RESULTS")
    print("=" * 100)

    sorted_results = sorted(results, key=lambda r: r.total_pnl, reverse=True)

    print(f"\n{'Strategy':<25} {'PnL':>10} {'ROI':>8} {'Hit%':>7} {'MaxDD':>10} {'Sharpe':>8} {'Trades':>7}")
    print("-" * 100)

    for r in sorted_results[:top_n]:
        print(f"{r.config.name():<25} ${r.total_pnl:>9.0f} {r.mean_roi:>7.1%} "
              f"{r.hit_rate:>6.1%} ${r.max_drawdown:>9.0f} {r.sharpe_ratio:>7.2f} {r.num_trades:>7}")

    # Best strategy breakdown
    print("\n" + "=" * 100)
    print("BEST STRATEGY BREAKDOWN")
    print("=" * 100)

    best = sorted_results[0]

    print(f"\nStrategy: {best.config.name()}")
    print(f"Vol Adjust: {best.config.vol_adjust}")
    print(f"Entry: {best.config.entry_pct:.0%}")
    print(f"Bucket Strategy: {best.config.bucket_strategy}")

    # By token
    print(f"\n{'Token':<8} {'PnL':>10} {'Trades':>8} {'Win%':>7}")
    print("-" * 35)
    for token, stats in sorted(best.by_token.items(), key=lambda x: -x[1]['pnl']):
        win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"{token:<8} ${stats['pnl']:>9.0f} {stats['trades']:>8} {win_rate:>6.1%}")

    # By unlock risk
    print(f"\n{'Unlock Risk':<12} {'PnL':>10} {'Trades':>8} {'Win%':>7}")
    print("-" * 40)
    for risk, stats in sorted(best.by_unlock_risk.items(), key=lambda x: -x[1]['pnl']):
        win_rate = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"{risk:<12} ${stats['pnl']:>9.0f} {stats['trades']:>8} {win_rate:>6.1%}")


def export_results(results: List[BacktestResult], output_path: str):
    """Export to CSV."""
    rows = []
    for r in results:
        rows.append({
            'strategy': r.config.name(),
            'vol_adjust': r.config.vol_adjust,
            'entry_pct': r.config.entry_pct,
            'bucket_strategy': r.config.bucket_strategy,
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
    print("INFRA-FOCUSED FDV STRATEGY")
    print("=" * 100)
    print("\nRationale: Backtest V2 showed Infra tokens = 83% of total PnL")
    print("Tokens: LINK, TAO, WLD, RENDER, FIL, ZRO, PYTH")

    backtester = InfraBacktester()
    backtester.load_data()

    results = backtester.run_all_combinations()

    print_results(results)

    export_results(results, "../data/infra_strategy_results.csv")

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

        # Top tokens
        print("\n  Top performing tokens:")
        for token, stats in sorted(best.by_token.items(), key=lambda x: -x[1]['pnl'])[:3]:
            print(f"    {token}: ${stats['pnl']:,.0f}")


if __name__ == "__main__":
    main()
