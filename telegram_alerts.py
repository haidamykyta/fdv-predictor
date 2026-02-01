"""
FDV Predictor Telegram Alerts

Sends notifications for:
- New FDV markets discovered
- Trade signals from Combined Strategy
- Paper trade executions
- Portfolio status updates
- Risk manager alerts (paused/stopped)

Setup:
1. Create bot via @BotFather on Telegram
2. Get your chat ID from @userinfobot
3. Set environment variables or edit config below

Usage:
    python telegram_alerts.py --test    # Test connection
    python telegram_alerts.py --monitor # Start monitoring

Author: haidamykyta@gmail.com
"""

import os
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Telegram library
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Local imports
from combined_strategy import CombinedStrategy, MarketSignal, TradeDecision, StrategyType
from live_paper_trader import LivePaperTrader, PolymarketFDVClient
from risk_manager import RiskManager, TradingState


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlertConfig:
    """Telegram alert configuration."""
    bot_token: str = ""
    chat_id: str = ""

    # Alert settings
    alert_on_new_market: bool = True
    alert_on_trade_signal: bool = True
    alert_on_trade_execution: bool = True
    alert_on_risk_event: bool = True

    # Thresholds
    min_signal_confidence: float = 0.60
    min_trade_size: float = 20.0

    # Monitoring
    scan_interval_minutes: int = 60


def load_config() -> AlertConfig:
    """Load config from environment or file."""
    config = AlertConfig()

    # Try environment variables first
    config.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    config.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    # Try config file
    config_path = Path(__file__).parent / "data" / "telegram_config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding='utf-8') as f:
                data = json.load(f)
            config.bot_token = data.get("bot_token", config.bot_token)
            config.chat_id = data.get("chat_id", config.chat_id)
            config.scan_interval_minutes = data.get("scan_interval", 60)
        except:
            pass

    return config


def save_config(config: AlertConfig):
    """Save config to file."""
    config_path = Path(__file__).parent / "data" / "telegram_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump({
            "bot_token": config.bot_token,
            "chat_id": config.chat_id,
            "scan_interval": config.scan_interval_minutes,
        }, f, indent=2)


# =============================================================================
# TELEGRAM CLIENT
# =============================================================================

class TelegramClient:
    """Simple Telegram bot client using HTTP API."""

    BASE_URL = "https://api.telegram.org/bot"

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"{self.BASE_URL}{bot_token}"

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send text message."""
        if not self.bot_token or not self.chat_id:
            print("[!] Telegram not configured")
            return False

        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            resp = requests.post(url, json=data, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"[!] Telegram error: {e}")
            return False

    def test_connection(self) -> bool:
        """Test bot connection."""
        try:
            url = f"{self.api_url}/getMe"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                bot_info = resp.json().get("result", {})
                print(f"[OK] Connected to bot: @{bot_info.get('username')}")
                return True
            return False
        except Exception as e:
            print(f"[!] Connection failed: {e}")
            return False


# =============================================================================
# ALERT FORMATTER
# =============================================================================

class AlertFormatter:
    """Format alerts for Telegram."""

    @staticmethod
    def new_market(market: Dict) -> str:
        """Format new market alert."""
        title = market.get('question', 'Unknown')[:80]
        volume = market.get('volume', 0)
        liquidity = market.get('liquidity', 0)

        try:
            volume = float(volume or 0)
            liquidity = float(liquidity or 0)
        except:
            volume = 0
            liquidity = 0

        return f"""
<b>NEW FDV MARKET</b>

<b>{title}</b>

Volume: ${volume:,.0f}
Liquidity: ${liquidity:,.0f}

<i>Analyzing for trade opportunity...</i>
"""

    @staticmethod
    def trade_signal(decision: TradeDecision, signal: MarketSignal) -> str:
        """Format trade signal alert."""
        strategy_emoji = {
            StrategyType.INFRA: "INFRA",
            StrategyType.FAVORITE_HEDGE: "FAV+HEDGE",
            StrategyType.ARBITRAGE: "ARBITRAGE",
            StrategyType.REGULAR: "REGULAR",
        }

        return f"""
<b>TRADE SIGNAL</b>

<b>{signal.token_symbol}</b> ({signal.category})

Strategy: {strategy_emoji.get(decision.strategy, decision.strategy.value)}
Action: {decision.action}
Bucket: {decision.bucket_idx}
Entry: {decision.entry_price:.0%}
Size: ${decision.size_usd:.0f}
Confidence: {decision.confidence:.0%}

<i>{decision.reason}</i>
"""

    @staticmethod
    def trade_executed(position_info: Dict) -> str:
        """Format trade execution alert."""
        return f"""
<b>PAPER TRADE EXECUTED</b>

Market: {position_info.get('market', 'Unknown')[:50]}
Strategy: {position_info.get('strategy', 'Unknown')}
Entry: {position_info.get('entry', 0):.0%}
Shares: {position_info.get('shares', 0):.1f}
Cost: ${position_info.get('cost', 0):.0f}

<i>Position opened at {datetime.now().strftime('%H:%M')}</i>
"""

    @staticmethod
    def risk_alert(state: TradingState, reason: str) -> str:
        """Format risk manager alert."""
        state_emoji = {
            TradingState.PAUSED: "PAUSED",
            TradingState.STOPPED: "STOPPED",
            TradingState.COOLDOWN: "COOLDOWN",
        }

        severity = "WARNING" if state == TradingState.PAUSED else "CRITICAL"

        return f"""
<b>{severity}: TRADING {state_emoji.get(state, state.value.upper())}</b>

Reason: {reason}

<i>Check risk manager status</i>
"""

    @staticmethod
    def portfolio_summary(portfolio_data: Dict) -> str:
        """Format portfolio summary."""
        return f"""
<b>PORTFOLIO STATUS</b>

Balance: ${portfolio_data.get('balance', 0):,.0f}
Total Value: ${portfolio_data.get('total_value', 0):,.0f}
Total PnL: ${portfolio_data.get('pnl', 0):+,.0f}

Trades: {portfolio_data.get('trades', 0)}
Win Rate: {portfolio_data.get('win_rate', 0):.1%}

Open Positions: {portfolio_data.get('open_positions', 0)}
"""

    @staticmethod
    def daily_summary(stats: Dict) -> str:
        """Format daily summary."""
        return f"""
<b>DAILY SUMMARY</b>
{datetime.now().strftime('%Y-%m-%d')}

Trades: {stats.get('trades', 0)}
PnL: ${stats.get('pnl', 0):+,.0f}
Win Rate: {stats.get('win_rate', 0):.1%}

Best: {stats.get('best_trade', 'N/A')}
Worst: {stats.get('worst_trade', 'N/A')}

<i>FDV Predictor Bot</i>
"""


# =============================================================================
# ALERT MONITOR
# =============================================================================

class FDVAlertMonitor:
    """Monitor FDV markets and send alerts."""

    def __init__(self, config: AlertConfig = None):
        self.config = config or load_config()
        self.telegram = TelegramClient(self.config.bot_token, self.config.chat_id)
        self.formatter = AlertFormatter()

        self.trader = LivePaperTrader(bankroll=1000)
        self.client = PolymarketFDVClient()

        # Track seen markets
        self.seen_markets = set()
        self.last_risk_state = TradingState.ACTIVE

    def check_new_markets(self) -> List[Dict]:
        """Check for new FDV markets."""
        new_markets = []

        try:
            fdv_markets = self.client.find_fdv_markets()

            for market in fdv_markets:
                market_id = market.get('condition_id', market.get('id', ''))
                if market_id and market_id not in self.seen_markets:
                    self.seen_markets.add(market_id)
                    new_markets.append(market)
        except Exception as e:
            print(f"[!] Error checking markets: {e}")

        return new_markets

    def check_trade_signals(self, markets: List[Dict]) -> List[tuple]:
        """Check markets for trade signals."""
        signals = []

        for market in markets:
            try:
                decision = self.trader.analyze_market(market)

                if decision and decision.confidence >= self.config.min_signal_confidence:
                    if decision.size_usd >= self.config.min_trade_size:
                        # Create basic signal for formatting
                        signal = MarketSignal(
                            token_symbol=market.get('question', '')[:10],
                            token_name=market.get('question', '')[:30],
                            category="FDV",
                            buckets=[],
                            current_bucket=0,
                            fdv_mcap_ratio=1.0,
                            bucket_prices=[],
                        )
                        signals.append((decision, signal, market))
            except:
                pass

        return signals

    def check_risk_state(self) -> Optional[tuple]:
        """Check if risk state changed."""
        current_state = self.trader.strategy.risk_manager.state.state
        reason = self.trader.strategy.risk_manager.state.pause_reason

        if current_state != self.last_risk_state:
            if current_state in [TradingState.PAUSED, TradingState.STOPPED]:
                self.last_risk_state = current_state
                return (current_state, reason)

            self.last_risk_state = current_state

        return None

    def send_new_market_alerts(self, markets: List[Dict]):
        """Send alerts for new markets."""
        if not self.config.alert_on_new_market:
            return

        for market in markets[:3]:  # Max 3 alerts per scan
            msg = self.formatter.new_market(market)
            self.telegram.send_message(msg)
            time.sleep(1)  # Rate limit

    def send_trade_signal_alerts(self, signals: List[tuple]):
        """Send alerts for trade signals."""
        if not self.config.alert_on_trade_signal:
            return

        for decision, signal, market in signals[:3]:
            msg = self.formatter.trade_signal(decision, signal)
            self.telegram.send_message(msg)
            time.sleep(1)

    def send_risk_alert(self, state: TradingState, reason: str):
        """Send risk manager alert."""
        if not self.config.alert_on_risk_event:
            return

        msg = self.formatter.risk_alert(state, reason)
        self.telegram.send_message(msg)

    def send_portfolio_summary(self):
        """Send portfolio summary."""
        portfolio = self.trader.portfolio

        data = {
            "balance": portfolio.current_balance,
            "total_value": portfolio.total_value,
            "pnl": portfolio.total_pnl,
            "trades": portfolio.total_trades,
            "win_rate": portfolio.win_rate,
            "open_positions": len(portfolio.positions),
        }

        msg = self.formatter.portfolio_summary(data)
        self.telegram.send_message(msg)

    def run_single_check(self):
        """Run single monitoring check."""
        print(f"\n[{datetime.now().strftime('%H:%M')}] Running check...")

        # Check new markets
        new_markets = self.check_new_markets()
        if new_markets:
            print(f"  Found {len(new_markets)} new markets")
            self.send_new_market_alerts(new_markets)

        # Check trade signals
        signals = self.check_trade_signals(new_markets)
        if signals:
            print(f"  Found {len(signals)} trade signals")
            self.send_trade_signal_alerts(signals)

        # Check risk state
        risk_change = self.check_risk_state()
        if risk_change:
            state, reason = risk_change
            print(f"  Risk state changed: {state.value}")
            self.send_risk_alert(state, reason)

        print("  Check complete")

    def run_continuous(self):
        """Run continuous monitoring."""
        print("\n" + "=" * 60)
        print("FDV ALERT MONITOR")
        print("=" * 60)

        if not self.telegram.test_connection():
            print("\n[!] Telegram connection failed")
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
            return

        print(f"\nScan interval: {self.config.scan_interval_minutes} minutes")
        print("Press Ctrl+C to stop")

        # Send startup message
        self.telegram.send_message("<b>FDV Alert Monitor Started</b>\n\nMonitoring for FDV markets...")

        try:
            while True:
                self.run_single_check()
                time.sleep(self.config.scan_interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\nStopping monitor...")
            self.telegram.send_message("<b>FDV Alert Monitor Stopped</b>")


# =============================================================================
# CLI
# =============================================================================

def setup_config():
    """Interactive config setup."""
    print("\n" + "=" * 60)
    print("TELEGRAM ALERT SETUP")
    print("=" * 60)

    print("\n1. Create a bot via @BotFather on Telegram")
    print("2. Copy the bot token")
    print("3. Get your chat ID from @userinfobot")

    token = input("\nBot Token: ").strip()
    chat_id = input("Chat ID: ").strip()

    config = AlertConfig(bot_token=token, chat_id=chat_id)

    # Test connection
    client = TelegramClient(token, chat_id)
    if client.test_connection():
        print("\n[OK] Connection successful!")
        client.send_message("<b>FDV Predictor Bot Connected!</b>\n\nYou will receive alerts for:\n- New FDV markets\n- Trade signals\n- Risk events")
        save_config(config)
        print(f"[OK] Config saved")
    else:
        print("\n[!] Connection failed. Check your token and chat ID.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FDV Telegram Alerts")
    parser.add_argument('--setup', action='store_true', help='Setup Telegram config')
    parser.add_argument('--test', action='store_true', help='Test Telegram connection')
    parser.add_argument('--monitor', action='store_true', help='Start monitoring')
    parser.add_argument('--status', action='store_true', help='Send portfolio status')
    parser.add_argument('--interval', type=int, default=60, help='Scan interval (minutes)')

    args = parser.parse_args()

    if args.setup:
        setup_config()
    elif args.test:
        config = load_config()
        client = TelegramClient(config.bot_token, config.chat_id)
        if client.test_connection():
            client.send_message("<b>Test Message</b>\n\nFDV Predictor Telegram Alerts working!")
            print("[OK] Test message sent")
        else:
            print("[!] Test failed")
    elif args.monitor:
        config = load_config()
        config.scan_interval_minutes = args.interval
        monitor = FDVAlertMonitor(config)
        monitor.run_continuous()
    elif args.status:
        config = load_config()
        monitor = FDVAlertMonitor(config)
        monitor.send_portfolio_summary()
        print("[OK] Status sent")
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
