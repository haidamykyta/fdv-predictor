"""
Combined FDV Strategy

Integrates all tested strategies with Risk Manager:
1. Regular Strategy - high volume, moderate ROI
2. Infra-Focused - high ROI, 100% win rate
3. Favorite+Hedge - consistent, low risk
4. Strike Arbitrage - high ROI, low win rate

Strategy Selection Logic:
- Infra tokens -> Infra-Focused (best $/trade)
- High favorite (>75c) -> Favorite+Hedge
- High volatility -> Strike Arbitrage
- Default -> Regular

Risk Manager Integration:
- 1/4 Kelly sizing
- Bankroll Split (25/50/25)
- Consecutive Loss Stop
- Max Drawdown Protection

Author: haidamykyta@gmail.com
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum

# Import Risk Manager
import sys
sys.path.insert(0, str(Path(__file__).parent))
from risk_manager import RiskManager, RiskLimits, BankrollAllocation


class StrategyType(Enum):
    REGULAR = "regular"
    INFRA = "infra"
    FAVORITE_HEDGE = "favorite_hedge"
    ARBITRAGE = "arbitrage"


@dataclass
class MarketSignal:
    """Signal from market analysis."""
    token_symbol: str
    token_name: str
    category: str

    # Market data
    buckets: List[Dict]
    current_bucket: int
    fdv_mcap_ratio: float

    # Prices (YES prices for each bucket)
    bucket_prices: List[float]

    # Derived
    favorite_price: float = 0.0  # Highest bucket price
    favorite_bucket: int = 0
    volatility_score: float = 0.5

    def __post_init__(self):
        if self.bucket_prices:
            self.favorite_price = max(self.bucket_prices)
            self.favorite_bucket = self.bucket_prices.index(self.favorite_price)


@dataclass
class TradeDecision:
    """Decision output from strategy selector."""
    strategy: StrategyType
    action: str  # BUY, SELL, HOLD
    bucket_idx: int
    size_usd: float
    entry_price: float

    # Targets
    target_price: float = 0.0
    stop_price: float = 0.0

    # Reasoning
    reason: str = ""
    confidence: float = 0.5


class CombinedStrategy:
    """
    Combined strategy that selects optimal approach per market.

    Integrates with Risk Manager for position sizing and limits.
    """

    # Strategy performance from backtests
    STRATEGY_STATS = {
        StrategyType.REGULAR: {
            "roi": 1.25,
            "win_rate": 0.25,
            "sharpe": 1.14,
            "per_trade": 125,
        },
        StrategyType.INFRA: {
            "roi": 3.44,
            "win_rate": 1.00,
            "sharpe": 2.61,
            "per_trade": 344,
        },
        StrategyType.FAVORITE_HEDGE: {
            "roi": 0.22,
            "win_rate": 1.00,
            "sharpe": 13.86,
            "per_trade": 22,
        },
        StrategyType.ARBITRAGE: {
            "roi": 6.17,
            "win_rate": 0.06,
            "sharpe": 1.03,
            "per_trade": 98,
        },
    }

    # Infra tokens (best performers)
    INFRA_TOKENS = ["LINK", "TAO", "WLD", "RENDER", "FIL", "ZRO", "PYTH"]

    def __init__(
        self,
        risk_manager: RiskManager = None,
        bankroll: float = 1000,
    ):
        self.risk_manager = risk_manager or RiskManager(bankroll=bankroll)
        self.trades: List[Dict] = []

    def select_strategy(self, signal: MarketSignal) -> StrategyType:
        """Select optimal strategy based on market characteristics."""

        # Priority 1: Infra tokens -> Infra-Focused (highest $/trade)
        if signal.token_symbol in self.INFRA_TOKENS:
            return StrategyType.INFRA

        # Priority 2: Strong favorite -> Favorite+Hedge (100% win rate)
        if signal.favorite_price >= 0.75:
            return StrategyType.FAVORITE_HEDGE

        # Priority 3: High volatility / high FDV ratio -> Arbitrage
        if signal.fdv_mcap_ratio > 2.5 or signal.volatility_score > 0.8:
            return StrategyType.ARBITRAGE

        # Default: Regular strategy
        return StrategyType.REGULAR

    def analyze_market(self, signal: MarketSignal) -> Optional[TradeDecision]:
        """
        Analyze market and generate trade decision.

        Returns None if no trade should be made.
        """
        # Check if trading is allowed
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            return None

        # Select strategy
        strategy = self.select_strategy(signal)

        # Generate decision based on strategy
        if strategy == StrategyType.INFRA:
            return self._infra_decision(signal)
        elif strategy == StrategyType.FAVORITE_HEDGE:
            return self._favorite_hedge_decision(signal)
        elif strategy == StrategyType.ARBITRAGE:
            return self._arbitrage_decision(signal)
        else:
            return self._regular_decision(signal)

    def _infra_decision(self, signal: MarketSignal) -> Optional[TradeDecision]:
        """Infra-focused strategy decision."""
        # Buy current bucket at mid-range prices
        bucket_idx = signal.current_bucket

        if bucket_idx >= len(signal.bucket_prices):
            return None

        price = signal.bucket_prices[bucket_idx]

        # Skip if price too high
        if price > 0.40:
            return None

        # Calculate size with risk manager
        # Estimate probability slightly above price (we have edge)
        est_prob = min(price + 0.15, 0.60)
        size, shares, reason = self.risk_manager.get_position_size(
            probability=est_prob,
            price=price
        )

        if size < 10:  # Minimum $10 trade
            return None

        return TradeDecision(
            strategy=StrategyType.INFRA,
            action="BUY",
            bucket_idx=bucket_idx,
            size_usd=size,
            entry_price=price,
            target_price=1.0,  # Hold to resolution
            stop_price=0.0,  # No stop (hold to expiry)
            reason=f"Infra token {signal.token_symbol} - high $/trade expected",
            confidence=0.70
        )

    def _favorite_hedge_decision(self, signal: MarketSignal) -> Optional[TradeDecision]:
        """Favorite+Hedge strategy decision."""
        fav_idx = signal.favorite_bucket
        fav_price = signal.favorite_price

        if fav_price < 0.75:
            return None

        # Size based on risk manager
        size, shares, reason = self.risk_manager.get_position_size(
            probability=fav_price + 0.05,  # Slight edge
            price=fav_price
        )

        if size < 10:
            return None

        # Target ROI around 8% (from backtest)
        target = min(fav_price + 0.08, 0.99)

        return TradeDecision(
            strategy=StrategyType.FAVORITE_HEDGE,
            action="BUY",
            bucket_idx=fav_idx,
            size_usd=size,
            entry_price=fav_price,
            target_price=target,
            stop_price=fav_price - 0.10,  # 10c stop
            reason=f"Strong favorite at {fav_price:.0%} - time decay play",
            confidence=0.85
        )

    def _arbitrage_decision(self, signal: MarketSignal) -> Optional[TradeDecision]:
        """Strike Arbitrage strategy decision."""
        # Find two adjacent buckets for corridor trade
        prices = signal.bucket_prices
        best_corridor = None
        best_cost = 1.0

        for i in range(len(prices) - 1):
            # Corridor cost = sum of adjacent bucket prices
            corridor_cost = prices[i] + prices[i + 1]

            # Good corridor: costs less than sum of probabilities
            if corridor_cost < best_cost and corridor_cost < 0.60:
                best_corridor = i
                best_cost = corridor_cost

        if best_corridor is None:
            return None

        # Size conservatively (arb has low win rate)
        size, shares, reason = self.risk_manager.get_position_size(
            probability=0.40,  # Conservative
            price=best_cost / 2
        )

        size = size * 0.5  # Even more conservative for arb

        if size < 10:
            return None

        return TradeDecision(
            strategy=StrategyType.ARBITRAGE,
            action="BUY",
            bucket_idx=best_corridor,  # Buy both buckets
            size_usd=size,
            entry_price=best_cost,
            target_price=1.0,
            stop_price=0.0,
            reason=f"Corridor arb: buckets {best_corridor}-{best_corridor+1} at {best_cost:.0%}",
            confidence=0.40
        )

    def _regular_decision(self, signal: MarketSignal) -> Optional[TradeDecision]:
        """Regular strategy decision."""
        # Buy bucket with best value (mid-range price)
        best_idx = None
        best_value = 0

        for i, price in enumerate(signal.bucket_prices):
            # Look for 15-35c range (good value)
            if 0.15 <= price <= 0.35:
                # Prefer buckets closer to current
                distance = abs(i - signal.current_bucket)
                value = (0.35 - price) / (1 + distance * 0.2)

                if value > best_value:
                    best_value = value
                    best_idx = i

        if best_idx is None:
            return None

        price = signal.bucket_prices[best_idx]

        # Size with risk manager
        size, shares, reason = self.risk_manager.get_position_size(
            probability=price + 0.10,
            price=price
        )

        if size < 10:
            return None

        return TradeDecision(
            strategy=StrategyType.REGULAR,
            action="BUY",
            bucket_idx=best_idx,
            size_usd=size,
            entry_price=price,
            target_price=1.0,
            stop_price=0.0,
            reason=f"Value bucket at {price:.0%}",
            confidence=0.55
        )

    def execute_decision(self, decision: TradeDecision, actual_pnl: float = None):
        """
        Record trade execution.

        If actual_pnl is provided, records the result.
        """
        trade = {
            "timestamp": datetime.now().isoformat(),
            "strategy": decision.strategy.value,
            "action": decision.action,
            "bucket": decision.bucket_idx,
            "size": decision.size_usd,
            "entry": decision.entry_price,
            "target": decision.target_price,
            "reason": decision.reason,
        }

        if actual_pnl is not None:
            trade["pnl"] = actual_pnl
            trade["roi"] = actual_pnl / decision.size_usd if decision.size_usd > 0 else 0

            # Record with risk manager
            self.risk_manager.record_trade(actual_pnl)

        self.trades.append(trade)
        return trade

    def get_strategy_stats(self) -> Dict:
        """Get stats by strategy type."""
        stats = {}

        for strategy in StrategyType:
            strat_trades = [t for t in self.trades if t.get("strategy") == strategy.value and "pnl" in t]

            if strat_trades:
                pnls = [t["pnl"] for t in strat_trades]
                stats[strategy.value] = {
                    "trades": len(strat_trades),
                    "total_pnl": sum(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                    "avg_pnl": sum(pnls) / len(pnls),
                }

        return stats

    def print_status(self):
        """Print combined status."""
        print("\n" + "=" * 70)
        print("COMBINED STRATEGY STATUS")
        print("=" * 70)

        # Risk manager status
        self.risk_manager.print_status()

        # Strategy breakdown
        stats = self.get_strategy_stats()

        if stats:
            print("\n" + "-" * 70)
            print("STRATEGY BREAKDOWN")
            print("-" * 70)

            print(f"\n{'Strategy':<20} {'Trades':>8} {'PnL':>10} {'Win%':>8} {'Avg':>10}")
            print("-" * 60)

            for strat, s in stats.items():
                print(f"{strat:<20} {s['trades']:>8} ${s['total_pnl']:>9.0f} "
                      f"{s['win_rate']:>7.1%} ${s['avg_pnl']:>9.0f}")


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate combined strategy."""
    print("=" * 70)
    print("COMBINED STRATEGY DEMO")
    print("=" * 70)

    # Initialize with $1000 bankroll
    strategy = CombinedStrategy(bankroll=1000)

    # Test signals
    test_signals = [
        # Infra token (should use Infra strategy)
        MarketSignal(
            token_symbol="LINK",
            token_name="Chainlink",
            category="Infra",
            buckets=[{"label": f"${i}B"} for i in range(6)],
            current_bucket=2,
            fdv_mcap_ratio=1.4,
            bucket_prices=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05],
        ),
        # Strong favorite (should use Favorite+Hedge)
        MarketSignal(
            token_symbol="BTC",
            token_name="Bitcoin",
            category="L1",
            buckets=[{"label": f"${i}0K"} for i in range(6)],
            current_bucket=4,
            fdv_mcap_ratio=1.0,
            bucket_prices=[0.02, 0.05, 0.08, 0.10, 0.78, 0.05],
        ),
        # High volatility (should use Arbitrage)
        MarketSignal(
            token_symbol="WLD",
            token_name="World",
            category="Infra",
            buckets=[{"label": f"${i}B"} for i in range(6)],
            current_bucket=1,
            fdv_mcap_ratio=3.6,
            bucket_prices=[0.15, 0.25, 0.25, 0.15, 0.10, 0.10],
            volatility_score=0.95,
        ),
        # Regular token
        MarketSignal(
            token_symbol="UNI",
            token_name="Uniswap",
            category="DeFi",
            buckets=[{"label": f"${i}B"} for i in range(6)],
            current_bucket=2,
            fdv_mcap_ratio=1.4,
            bucket_prices=[0.10, 0.20, 0.30, 0.25, 0.10, 0.05],
        ),
    ]

    print("\nAnalyzing test signals...\n")

    for signal in test_signals:
        print(f"\n{'='*50}")
        print(f"Token: {signal.token_symbol} ({signal.category})")
        print(f"FDV/MCap: {signal.fdv_mcap_ratio:.1f}x")
        print(f"Prices: {[f'{p:.0%}' for p in signal.bucket_prices]}")

        decision = strategy.analyze_market(signal)

        if decision:
            print(f"\n[DECISION]")
            print(f"  Strategy: {decision.strategy.value}")
            print(f"  Action: {decision.action} bucket {decision.bucket_idx}")
            print(f"  Size: ${decision.size_usd:.0f}")
            print(f"  Entry: {decision.entry_price:.0%}")
            print(f"  Reason: {decision.reason}")
            print(f"  Confidence: {decision.confidence:.0%}")

            # Simulate execution with random outcome
            import random
            win = random.random() < decision.confidence
            if win:
                pnl = decision.size_usd * (1 / decision.entry_price - 1) * 0.5  # Partial win
            else:
                pnl = -decision.size_usd * 0.8  # Partial loss

            strategy.execute_decision(decision, pnl)
            print(f"  Simulated PnL: ${pnl:+.0f}")
        else:
            print(f"\n[NO TRADE]")

    # Final status
    strategy.print_status()


if __name__ == "__main__":
    demo()
