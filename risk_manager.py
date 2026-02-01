"""
Advanced Risk Manager for Polymarket Trading

Based on Polymarket bots analysis best practices:
- 1/4 Kelly position sizing (already in position_manager.py)
- Bankroll Split: 25% active / 50% hedge / 25% buffer
- Consecutive Loss Stop: pause after N consecutive losses
- Max exposure limits
- Drawdown protection

Usage:
    risk = RiskManager(bankroll=1000)

    # Check if we can trade
    if risk.can_trade():
        size = risk.get_position_size(probability=0.65, price=0.50)
        risk.record_trade(pnl=50)  # After trade completes

    # Check risk status
    risk.print_status()

Author: haidamykyta@gmail.com
"""

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum


class TradingState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"  # Consecutive loss stop
    STOPPED = "stopped"  # Max drawdown hit
    COOLDOWN = "cooldown"  # Waiting for cooldown period


@dataclass
class BankrollAllocation:
    """Bankroll split according to best practices."""
    total: float = 1000.0

    # Split percentages (from Polymarket bots analysis)
    active_pct: float = 0.25  # 25% for opening positions
    hedge_pct: float = 0.50   # 50% reserved for hedging
    buffer_pct: float = 0.25  # 25% emergency reserve

    @property
    def active(self) -> float:
        """Amount available for new trades."""
        return self.total * self.active_pct

    @property
    def hedge_reserve(self) -> float:
        """Amount reserved for hedging positions."""
        return self.total * self.hedge_pct

    @property
    def safety_buffer(self) -> float:
        """Emergency reserve - don't touch."""
        return self.total * self.buffer_pct

    def available_for_trade(self, current_exposure: float) -> float:
        """Calculate how much we can use for a new trade."""
        used = current_exposure
        available = self.active - used
        return max(0, available)


@dataclass
class RiskLimits:
    """Risk management limits."""
    # Position limits
    max_position_pct: float = 0.05  # Max 5% per trade
    max_total_exposure_pct: float = 0.20  # Max 20% total exposure

    # Loss limits
    max_consecutive_losses: int = 3  # Pause after 3 losses
    cooldown_hours: int = 24  # Wait 24h after consecutive loss stop
    max_drawdown_pct: float = 0.15  # Stop if down 15%

    # Daily limits
    max_trades_per_day: int = 10
    max_daily_loss_pct: float = 0.05  # Max 5% loss per day


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    pnl: float
    roi: float
    won: bool
    notes: str = ""


@dataclass
class RiskState:
    """Current risk management state."""
    state: TradingState = TradingState.ACTIVE
    consecutive_losses: int = 0
    consecutive_wins: int = 0

    # Today's stats
    trades_today: int = 0
    pnl_today: float = 0.0
    last_trade_date: str = ""

    # Overall stats
    total_trades: int = 0
    total_pnl: float = 0.0
    peak_value: float = 0.0
    current_drawdown: float = 0.0

    # Pause info
    pause_until: datetime = None
    pause_reason: str = ""

    # Trade history (last N trades)
    recent_trades: List[Dict] = field(default_factory=list)


class RiskManager:
    """
    Advanced risk management for Polymarket trading.

    Features:
    - Bankroll split (25/50/25)
    - 1/4 Kelly sizing
    - Consecutive loss stop
    - Drawdown protection
    - Daily limits
    """

    def __init__(
        self,
        bankroll: float = 1000,
        data_file: str = "risk_state.json",
        limits: RiskLimits = None,
        allocation: BankrollAllocation = None
    ):
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll
        self.data_file = data_file

        self.limits = limits or RiskLimits()
        self.allocation = allocation or BankrollAllocation(total=bankroll)
        self.state = RiskState(peak_value=bankroll)

        self.current_exposure = 0.0  # Current open positions value

        self._load()

    # ==================== Core Risk Checks ====================

    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check state
        if self.state.state == TradingState.STOPPED:
            return False, f"Trading stopped: {self.state.pause_reason}"

        if self.state.state == TradingState.PAUSED:
            if self.state.pause_until and datetime.now() < self.state.pause_until:
                remaining = self.state.pause_until - datetime.now()
                return False, f"Paused for {remaining.seconds // 3600}h: {self.state.pause_reason}"
            else:
                # Cooldown expired
                self._resume_trading()

        # Check daily limits
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.last_trade_date != today:
            self._reset_daily_stats()

        if self.state.trades_today >= self.limits.max_trades_per_day:
            return False, f"Daily trade limit reached ({self.limits.max_trades_per_day})"

        daily_loss_limit = self.initial_bankroll * self.limits.max_daily_loss_pct
        if self.state.pnl_today <= -daily_loss_limit:
            return False, f"Daily loss limit reached (${daily_loss_limit:.0f})"

        # Check drawdown
        if self.state.current_drawdown >= self.limits.max_drawdown_pct:
            self._stop_trading("Max drawdown limit reached")
            return False, "Max drawdown limit reached"

        # Check exposure
        max_exposure = self.current_bankroll * self.limits.max_total_exposure_pct
        if self.current_exposure >= max_exposure:
            return False, f"Max exposure reached (${max_exposure:.0f})"

        return True, "OK"

    def get_position_size(
        self,
        probability: float,
        price: float,
        kelly_fraction: float = 0.25
    ) -> Tuple[float, float, str]:
        """
        Calculate position size using 1/4 Kelly with risk limits.

        Args:
            probability: Our estimated win probability
            price: Current market price
            kelly_fraction: Kelly fraction (default 0.25 = 1/4 Kelly)

        Returns:
            Tuple of (dollars, shares, reasoning)
        """
        can_trade, reason = self.can_trade()
        if not can_trade:
            return 0, 0, f"Cannot trade: {reason}"

        # Check for edge
        edge = probability - price
        if edge <= 0:
            return 0, 0, f"No edge: prob {probability:.1%} <= price {price:.1%}"

        # Kelly calculation
        if price <= 0 or price >= 1:
            return 0, 0, "Invalid price"

        odds = 1 / price
        b = odds - 1
        p = probability
        q = 1 - p

        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0, 0, "Kelly says don't bet"

        # Apply fraction (1/4 Kelly by default)
        kelly = kelly * kelly_fraction

        # Get available capital from bankroll split
        available = self.allocation.available_for_trade(self.current_exposure)

        # Apply position limit
        max_position = self.current_bankroll * self.limits.max_position_pct
        available = min(available, max_position)

        # Calculate position
        position_dollars = min(kelly * self.current_bankroll, available)

        if position_dollars < 1:
            return 0, 0, "Position too small"

        shares = position_dollars / price

        reasoning = (
            f"Kelly: {kelly/kelly_fraction:.1%} -> {kelly:.1%} (1/{int(1/kelly_fraction)} Kelly)\n"
            f"Available: ${available:.0f} (from 25% active allocation)\n"
            f"Position: ${position_dollars:.0f} = {shares:.0f} shares @ {price:.1%}"
        )

        return position_dollars, shares, reasoning

    # ==================== Trade Recording ====================

    def record_trade(
        self,
        pnl: float,
        entry_cost: float = 0,
        notes: str = ""
    ):
        """
        Record a completed trade.

        Args:
            pnl: Profit/loss from trade
            entry_cost: Original investment (for ROI calculation)
            notes: Optional notes
        """
        roi = pnl / entry_cost if entry_cost > 0 else 0
        won = pnl > 0

        trade = TradeRecord(
            timestamp=datetime.now(),
            pnl=pnl,
            roi=roi,
            won=won,
            notes=notes
        )

        # Update state
        self.state.total_trades += 1
        self.state.total_pnl += pnl
        self.state.trades_today += 1
        self.state.pnl_today += pnl
        self.state.last_trade_date = datetime.now().strftime("%Y-%m-%d")

        # Update bankroll
        self.current_bankroll += pnl
        self.allocation.total = self.current_bankroll

        # Track peak and drawdown
        if self.current_bankroll > self.state.peak_value:
            self.state.peak_value = self.current_bankroll

        self.state.current_drawdown = (
            (self.state.peak_value - self.current_bankroll) / self.state.peak_value
        )

        # Track consecutive wins/losses
        if won:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

            # Check for consecutive loss stop
            if self.state.consecutive_losses >= self.limits.max_consecutive_losses:
                self._pause_trading(
                    hours=self.limits.cooldown_hours,
                    reason=f"{self.state.consecutive_losses} consecutive losses"
                )

        # Add to recent trades
        self.state.recent_trades.append(asdict(trade))
        self.state.recent_trades = self.state.recent_trades[-20:]  # Keep last 20

        self._save()

        # Print update
        emoji = "+" if won else "-"
        print(f"Trade recorded: {emoji}${abs(pnl):.2f} ({roi:+.1%})")
        if self.state.consecutive_losses > 0:
            print(f"  Consecutive losses: {self.state.consecutive_losses}/{self.limits.max_consecutive_losses}")

    # ==================== State Management ====================

    def _pause_trading(self, hours: int, reason: str):
        """Pause trading for specified hours."""
        self.state.state = TradingState.PAUSED
        self.state.pause_until = datetime.now() + timedelta(hours=hours)
        self.state.pause_reason = reason
        self._save()

        print(f"\n[!] TRADING PAUSED")
        print(f"  Reason: {reason}")
        print(f"  Resume: {self.state.pause_until.strftime('%Y-%m-%d %H:%M')}")

    def _stop_trading(self, reason: str):
        """Stop trading completely."""
        self.state.state = TradingState.STOPPED
        self.state.pause_reason = reason
        self._save()

        print(f"\n[X] TRADING STOPPED")
        print(f"  Reason: {reason}")
        print(f"  Manual intervention required")

    def _resume_trading(self):
        """Resume trading after cooldown."""
        self.state.state = TradingState.ACTIVE
        self.state.pause_until = None
        self.state.pause_reason = ""
        self.state.consecutive_losses = 0
        self._save()

        print("[OK] Trading resumed")

    def _reset_daily_stats(self):
        """Reset daily stats."""
        self.state.trades_today = 0
        self.state.pnl_today = 0.0
        self.state.last_trade_date = datetime.now().strftime("%Y-%m-%d")

    def force_resume(self):
        """Force resume trading (manual override)."""
        self._resume_trading()
        print("[!] Forced resume - use with caution!")

    # ==================== Reporting ====================

    def print_status(self):
        """Print current risk status."""
        print("\n" + "=" * 60)
        print("RISK MANAGER STATUS")
        print("=" * 60)

        # State
        state_emoji = {
            TradingState.ACTIVE: "[OK]",
            TradingState.PAUSED: "[!!]",
            TradingState.STOPPED: "[XX]",
            TradingState.COOLDOWN: "[..]"
        }
        print(f"\nState: {state_emoji.get(self.state.state, '?')} {self.state.state.value.upper()}")
        if self.state.pause_reason:
            print(f"  Reason: {self.state.pause_reason}")

        # Bankroll
        print(f"\n{'Bankroll:':<20} ${self.current_bankroll:,.0f}")
        print(f"  {'Active (25%):':<18} ${self.allocation.active:,.0f}")
        print(f"  {'Hedge (50%):':<18} ${self.allocation.hedge_reserve:,.0f}")
        print(f"  {'Buffer (25%):':<18} ${self.allocation.safety_buffer:,.0f}")

        # Exposure
        max_exposure = self.current_bankroll * self.limits.max_total_exposure_pct
        print(f"\n{'Current Exposure:':<20} ${self.current_exposure:,.0f} / ${max_exposure:,.0f}")

        # Drawdown
        dd_pct = self.state.current_drawdown * 100
        dd_emoji = "[OK]" if dd_pct < 5 else "[!!]" if dd_pct < 10 else "[XX]"
        print(f"\n{dd_emoji} Drawdown: {dd_pct:.1f}% (max {self.limits.max_drawdown_pct*100:.0f}%)")
        print(f"  Peak: ${self.state.peak_value:,.0f}")

        # Consecutive losses
        cl = self.state.consecutive_losses
        cl_max = self.limits.max_consecutive_losses
        cl_emoji = "[OK]" if cl == 0 else "[!!]" if cl < cl_max else "[XX]"
        print(f"\n{cl_emoji} Consecutive Losses: {cl}/{cl_max}")

        # Today
        print(f"\nToday: {self.state.trades_today} trades, ${self.state.pnl_today:+,.0f}")

        # Total
        print(f"\nTotal: {self.state.total_trades} trades, ${self.state.total_pnl:+,.0f}")

        # Recent trades
        if self.state.recent_trades:
            print("\n" + "-" * 60)
            print("RECENT TRADES (last 5)")
            print("-" * 60)
            for t in self.state.recent_trades[-5:]:
                emoji = "+" if t['won'] else "-"
                print(f"  {emoji}${abs(t['pnl']):.0f} ({t['roi']*100:+.0f}%)")

        print()

    def get_summary(self) -> Dict:
        """Get summary as dict."""
        can_trade, reason = self.can_trade()

        return {
            'state': self.state.state.value,
            'can_trade': can_trade,
            'trade_reason': reason,
            'bankroll': self.current_bankroll,
            'available_for_trade': self.allocation.available_for_trade(self.current_exposure),
            'current_exposure': self.current_exposure,
            'drawdown_pct': self.state.current_drawdown,
            'consecutive_losses': self.state.consecutive_losses,
            'total_trades': self.state.total_trades,
            'total_pnl': self.state.total_pnl
        }

    # ==================== Persistence ====================

    def _save(self):
        """Save state to file."""
        data = {
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'current_exposure': self.current_exposure,
            'state': {
                'state': self.state.state.value,
                'consecutive_losses': self.state.consecutive_losses,
                'consecutive_wins': self.state.consecutive_wins,
                'trades_today': self.state.trades_today,
                'pnl_today': self.state.pnl_today,
                'last_trade_date': self.state.last_trade_date,
                'total_trades': self.state.total_trades,
                'total_pnl': self.state.total_pnl,
                'peak_value': self.state.peak_value,
                'current_drawdown': self.state.current_drawdown,
                'pause_until': self.state.pause_until.isoformat() if self.state.pause_until else None,
                'pause_reason': self.state.pause_reason,
                'recent_trades': self.state.recent_trades
            },
            'limits': asdict(self.limits),
            'allocation': {
                'total': self.allocation.total,
                'active_pct': self.allocation.active_pct,
                'hedge_pct': self.allocation.hedge_pct,
                'buffer_pct': self.allocation.buffer_pct
            }
        }

        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load state from file."""
        if not os.path.exists(self.data_file):
            return

        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            self.initial_bankroll = data.get('initial_bankroll', self.initial_bankroll)
            self.current_bankroll = data.get('current_bankroll', self.current_bankroll)
            self.current_exposure = data.get('current_exposure', 0)

            state_data = data.get('state', {})
            self.state.state = TradingState(state_data.get('state', 'active'))
            self.state.consecutive_losses = state_data.get('consecutive_losses', 0)
            self.state.consecutive_wins = state_data.get('consecutive_wins', 0)
            self.state.trades_today = state_data.get('trades_today', 0)
            self.state.pnl_today = state_data.get('pnl_today', 0)
            self.state.last_trade_date = state_data.get('last_trade_date', '')
            self.state.total_trades = state_data.get('total_trades', 0)
            self.state.total_pnl = state_data.get('total_pnl', 0)
            self.state.peak_value = state_data.get('peak_value', self.initial_bankroll)
            self.state.current_drawdown = state_data.get('current_drawdown', 0)
            self.state.pause_reason = state_data.get('pause_reason', '')
            self.state.recent_trades = state_data.get('recent_trades', [])

            if state_data.get('pause_until'):
                self.state.pause_until = datetime.fromisoformat(state_data['pause_until'])

            # Load limits
            limits_data = data.get('limits', {})
            if limits_data:
                self.limits = RiskLimits(**limits_data)

            # Load allocation
            alloc_data = data.get('allocation', {})
            if alloc_data:
                self.allocation = BankrollAllocation(**alloc_data)

            print(f"Loaded risk state from {self.data_file}")

        except Exception as e:
            print(f"Error loading risk state: {e}")


# ==================== Demo ====================

def demo():
    """Demo the risk manager."""
    print("=" * 60)
    print("RISK MANAGER DEMO")
    print("=" * 60)

    # Create with $1000 bankroll
    risk = RiskManager(bankroll=1000, data_file="demo_risk.json")

    # Check if we can trade
    can_trade, reason = risk.can_trade()
    print(f"\nCan trade: {can_trade} ({reason})")

    # Get position size
    print("\n--- Position Sizing ---")
    dollars, shares, reasoning = risk.get_position_size(
        probability=0.65,
        price=0.50
    )
    print(f"Position: ${dollars:.0f}, {shares:.0f} shares")
    print(f"Reasoning:\n{reasoning}")

    # Simulate some trades
    print("\n--- Simulating Trades ---")

    # Win
    risk.record_trade(pnl=25, entry_cost=100, notes="Good trade")

    # Win
    risk.record_trade(pnl=15, entry_cost=80, notes="Another good one")

    # Loss
    risk.record_trade(pnl=-30, entry_cost=100, notes="Bad trade")

    # Loss
    risk.record_trade(pnl=-20, entry_cost=80, notes="Another loss")

    # Loss - should trigger pause!
    risk.record_trade(pnl=-25, entry_cost=100, notes="Third loss")

    # Check status
    risk.print_status()

    # Try to trade while paused
    can_trade, reason = risk.can_trade()
    print(f"\nCan trade after losses: {can_trade} ({reason})")


if __name__ == "__main__":
    demo()
