"""
Token Unlock Calendar

Tracks token unlock events as price catalysts for FDV markets.
Integrates with DropsTab API for unlock schedules.

Unlock events can significantly impact FDV:
- Large unlocks (>5% supply) often cause price drops
- Cliff unlocks more impactful than linear vesting
- Team/investor unlocks more impactful than ecosystem

Usage:
    python token_unlock_calendar.py --fetch     # Fetch unlock data
    python token_unlock_calendar.py --upcoming  # Show upcoming unlocks
    python token_unlock_calendar.py --alerts    # Show high-impact unlocks

Author: haidamykyta@gmail.com
"""

import requests
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

# DropsTab API
DROPSTAB_BASE_URL = "https://public-api.dropstab.com/api/v1"
DROPSTAB_API_KEY = "b6fe2216-844f-4248-bac4-371f719d418f"

# Tokens to track (from FDV predictor)
TRACKED_TOKENS = [
    # Infra (highest ROI in backtest)
    {"symbol": "LINK", "name": "Chainlink", "category": "Infra"},
    {"symbol": "TAO", "name": "Bittensor", "category": "Infra"},
    {"symbol": "WLD", "name": "Worldcoin", "category": "Infra"},
    {"symbol": "RENDER", "name": "Render", "category": "Infra"},
    {"symbol": "FIL", "name": "Filecoin", "category": "Infra"},
    {"symbol": "ZRO", "name": "LayerZero", "category": "Infra"},
    {"symbol": "PYTH", "name": "Pyth Network", "category": "Infra"},

    # L1
    {"symbol": "SUI", "name": "Sui", "category": "L1"},
    {"symbol": "APT", "name": "Aptos", "category": "L1"},
    {"symbol": "SEI", "name": "Sei", "category": "L1"},
    {"symbol": "TIA", "name": "Celestia", "category": "L1"},
    {"symbol": "INJ", "name": "Injective", "category": "L1"},

    # L2
    {"symbol": "ARB", "name": "Arbitrum", "category": "L2"},
    {"symbol": "OP", "name": "Optimism", "category": "L2"},
    {"symbol": "STRK", "name": "StarkNet", "category": "L2"},
    {"symbol": "ZK", "name": "ZKsync", "category": "L2"},

    # DeFi
    {"symbol": "JUP", "name": "Jupiter", "category": "DeFi"},
    {"symbol": "JTO", "name": "Jito", "category": "DeFi"},
    {"symbol": "PENDLE", "name": "Pendle", "category": "DeFi"},
]


class UnlockType(Enum):
    CLIFF = "cliff"        # One-time large unlock
    LINEAR = "linear"      # Continuous vesting
    MILESTONE = "milestone"  # Based on conditions


class UnlockCategory(Enum):
    TEAM = "team"
    INVESTOR = "investor"
    ECOSYSTEM = "ecosystem"
    TREASURY = "treasury"
    COMMUNITY = "community"
    OTHER = "other"


@dataclass
class UnlockEvent:
    """Single unlock event."""
    token_symbol: str
    token_name: str
    unlock_date: datetime
    amount_tokens: float
    amount_usd: float
    percent_of_supply: float
    unlock_type: UnlockType
    category: UnlockCategory

    # Impact assessment
    impact_score: float = 0.0  # 0-1, higher = more impactful

    def __post_init__(self):
        # Calculate impact score
        self.impact_score = self._calculate_impact()

    def _calculate_impact(self) -> float:
        """Calculate impact score based on unlock characteristics."""
        score = 0.0

        # Size impact (larger = more impact)
        if self.percent_of_supply >= 10:
            score += 0.4
        elif self.percent_of_supply >= 5:
            score += 0.3
        elif self.percent_of_supply >= 2:
            score += 0.2
        elif self.percent_of_supply >= 1:
            score += 0.1

        # Type impact (cliff > milestone > linear)
        if self.unlock_type == UnlockType.CLIFF:
            score += 0.3
        elif self.unlock_type == UnlockType.MILESTONE:
            score += 0.2
        elif self.unlock_type == UnlockType.LINEAR:
            score += 0.1

        # Category impact (team/investor > others)
        if self.category in [UnlockCategory.TEAM, UnlockCategory.INVESTOR]:
            score += 0.3
        elif self.category == UnlockCategory.TREASURY:
            score += 0.2
        else:
            score += 0.1

        return min(score, 1.0)

    @property
    def is_high_impact(self) -> bool:
        """Check if this is a high-impact unlock."""
        return self.impact_score >= 0.5

    @property
    def days_until(self) -> int:
        """Days until unlock."""
        delta = self.unlock_date - datetime.now()
        return max(0, delta.days)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "token_symbol": self.token_symbol,
            "token_name": self.token_name,
            "unlock_date": self.unlock_date.isoformat(),
            "amount_tokens": self.amount_tokens,
            "amount_usd": self.amount_usd,
            "percent_of_supply": self.percent_of_supply,
            "unlock_type": self.unlock_type.value,
            "category": self.category.value,
            "impact_score": self.impact_score,
            "days_until": self.days_until,
        }


# =============================================================================
# DROPSTAB CLIENT
# =============================================================================

class DropsTabUnlockClient:
    """Client for fetching unlock data from DropsTab."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "*/*",
            "x-dropstab-api-key": DROPSTAB_API_KEY
        })
        self.data_dir = Path(__file__).parent / "data" / "unlocks"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, endpoint: str, params: dict = None) -> Optional[Dict]:
        """Make API request."""
        url = f"{DROPSTAB_BASE_URL}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            if isinstance(result, dict) and 'data' in result:
                return result['data']
            return result
        except Exception as e:
            print(f"[!] API error: {e}")
            return None

    def search_coin(self, query: str) -> Optional[int]:
        """Search for coin and return ID."""
        data = self._request("coins/search", {"query": query})
        if data:
            results = data if isinstance(data, list) else data.get('content', [])
            if results:
                return results[0].get('id')
        return None

    def get_coin_unlocks(self, coin_id: int) -> Optional[Dict]:
        """Get unlock schedule for coin."""
        return self._request(f"token-unlocks/{coin_id}")

    def get_upcoming_unlocks(self, days: int = 30) -> List[Dict]:
        """Get all upcoming unlocks."""
        data = self._request("token-unlocks", {
            "page": 0,
            "pageSize": 100
        })
        if data:
            unlocks = data.get('content', data) if isinstance(data, dict) else data
            return unlocks if unlocks else []
        return []

    def fetch_all_tracked_unlocks(self) -> List[UnlockEvent]:
        """Fetch unlocks for all tracked tokens."""
        events = []

        print("\nFetching unlock data for tracked tokens...")

        for token in TRACKED_TOKENS:
            symbol = token["symbol"]
            name = token["name"]

            print(f"  {symbol}...", end=" ")

            # Search for coin ID
            coin_id = self.search_coin(symbol)
            if not coin_id:
                coin_id = self.search_coin(name)

            if not coin_id:
                print("not found")
                continue

            # Get unlock data
            unlock_data = self.get_coin_unlocks(coin_id)
            if not unlock_data:
                print("no unlock data")
                continue

            # Parse unlock events
            schedules = unlock_data.get('schedules', unlock_data.get('unlocks', []))
            if not schedules:
                print("no schedules")
                continue

            token_events = self._parse_unlock_schedules(token, schedules)
            events.extend(token_events)
            print(f"{len(token_events)} events")

            # Rate limit
            import time
            time.sleep(0.5)

        return events

    def _parse_unlock_schedules(self, token: Dict, schedules: List) -> List[UnlockEvent]:
        """Parse unlock schedules into events."""
        events = []

        for schedule in schedules:
            try:
                # Parse date
                date_str = schedule.get('unlockDate') or schedule.get('date')
                if not date_str:
                    continue

                try:
                    unlock_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except:
                    unlock_date = datetime.strptime(date_str[:10], '%Y-%m-%d')

                # Skip past events
                if unlock_date < datetime.now():
                    continue

                # Parse amounts
                amount_tokens = float(schedule.get('amount', 0) or 0)
                amount_usd = float(schedule.get('amountUsd', 0) or 0)
                percent = float(schedule.get('percentOfSupply', 0) or schedule.get('percent', 0) or 0)

                # Parse type
                type_str = (schedule.get('type') or schedule.get('vestingType') or 'linear').lower()
                if 'cliff' in type_str:
                    unlock_type = UnlockType.CLIFF
                elif 'milestone' in type_str or 'condition' in type_str:
                    unlock_type = UnlockType.MILESTONE
                else:
                    unlock_type = UnlockType.LINEAR

                # Parse category
                cat_str = (schedule.get('category') or schedule.get('allocation') or 'other').lower()
                if 'team' in cat_str or 'founder' in cat_str:
                    category = UnlockCategory.TEAM
                elif 'investor' in cat_str or 'vc' in cat_str or 'seed' in cat_str:
                    category = UnlockCategory.INVESTOR
                elif 'ecosystem' in cat_str or 'growth' in cat_str:
                    category = UnlockCategory.ECOSYSTEM
                elif 'treasury' in cat_str or 'reserve' in cat_str:
                    category = UnlockCategory.TREASURY
                elif 'community' in cat_str or 'airdrop' in cat_str:
                    category = UnlockCategory.COMMUNITY
                else:
                    category = UnlockCategory.OTHER

                events.append(UnlockEvent(
                    token_symbol=token["symbol"],
                    token_name=token["name"],
                    unlock_date=unlock_date,
                    amount_tokens=amount_tokens,
                    amount_usd=amount_usd,
                    percent_of_supply=percent,
                    unlock_type=unlock_type,
                    category=category,
                ))

            except Exception as e:
                continue

        return events


# =============================================================================
# UNLOCK CALENDAR
# =============================================================================

class TokenUnlockCalendar:
    """
    Token unlock calendar for FDV prediction.

    Tracks upcoming unlocks and provides signals for trading.
    """

    def __init__(self):
        self.client = DropsTabUnlockClient()
        self.events: List[UnlockEvent] = []
        self.data_file = Path(__file__).parent / "data" / "unlock_calendar.json"

    def fetch_and_save(self):
        """Fetch unlock data and save to file."""
        print("=" * 60)
        print("FETCHING TOKEN UNLOCKS")
        print("=" * 60)

        self.events = self.client.fetch_all_tracked_unlocks()

        # Save to file
        self._save()

        print(f"\nTotal: {len(self.events)} upcoming unlock events")

    def load(self) -> bool:
        """Load from saved file."""
        if not self.data_file.exists():
            return False

        try:
            with open(self.data_file, encoding='utf-8') as f:
                data = json.load(f)

            self.events = []
            for e in data.get('events', []):
                try:
                    self.events.append(UnlockEvent(
                        token_symbol=e['token_symbol'],
                        token_name=e['token_name'],
                        unlock_date=datetime.fromisoformat(e['unlock_date']),
                        amount_tokens=e['amount_tokens'],
                        amount_usd=e['amount_usd'],
                        percent_of_supply=e['percent_of_supply'],
                        unlock_type=UnlockType(e['unlock_type']),
                        category=UnlockCategory(e['category']),
                    ))
                except:
                    continue

            return True
        except:
            return False

    def _save(self):
        """Save to file."""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "updated_at": datetime.now().isoformat(),
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events]
        }

        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_upcoming(self, days: int = 30) -> List[UnlockEvent]:
        """Get unlocks within next N days."""
        cutoff = datetime.now() + timedelta(days=days)
        return [e for e in self.events if e.unlock_date <= cutoff]

    def get_high_impact(self, days: int = 30) -> List[UnlockEvent]:
        """Get high-impact unlocks within next N days."""
        upcoming = self.get_upcoming(days)
        return [e for e in upcoming if e.is_high_impact]

    def get_by_token(self, symbol: str) -> List[UnlockEvent]:
        """Get unlocks for specific token."""
        return [e for e in self.events if e.token_symbol.upper() == symbol.upper()]

    def get_unlock_signal(self, symbol: str, days_lookahead: int = 7) -> Dict:
        """
        Get trading signal based on upcoming unlocks.

        Returns signal with adjustment factors for FDV prediction.
        """
        token_events = self.get_by_token(symbol)
        upcoming = [e for e in token_events if e.days_until <= days_lookahead]

        if not upcoming:
            return {
                "has_unlock": False,
                "signal": "neutral",
                "adjustment": 0.0,
                "reason": "No upcoming unlocks"
            }

        # Find most impactful event
        most_impactful = max(upcoming, key=lambda e: e.impact_score)

        # Generate signal
        if most_impactful.impact_score >= 0.7:
            signal = "bearish"
            adjustment = -0.15  # Reduce FDV estimate by 15%
        elif most_impactful.impact_score >= 0.5:
            signal = "cautious"
            adjustment = -0.08
        else:
            signal = "neutral"
            adjustment = -0.03

        return {
            "has_unlock": True,
            "signal": signal,
            "adjustment": adjustment,
            "days_until": most_impactful.days_until,
            "percent_unlock": most_impactful.percent_of_supply,
            "impact_score": most_impactful.impact_score,
            "reason": f"{most_impactful.percent_of_supply:.1f}% {most_impactful.category.value} unlock in {most_impactful.days_until} days"
        }

    def print_calendar(self, days: int = 30):
        """Print upcoming unlock calendar."""
        print("\n" + "=" * 80)
        print(f"TOKEN UNLOCK CALENDAR (Next {days} days)")
        print("=" * 80)

        upcoming = sorted(self.get_upcoming(days), key=lambda e: e.unlock_date)

        if not upcoming:
            print("\nNo upcoming unlocks found")
            return

        print(f"\n{'Date':<12} {'Token':<8} {'%Supply':>8} {'Type':<10} {'Category':<12} {'Impact':>8}")
        print("-" * 80)

        for e in upcoming[:30]:  # Show max 30
            impact_str = f"{e.impact_score:.0%}"
            if e.is_high_impact:
                impact_str = f"[!] {impact_str}"

            print(f"{e.unlock_date.strftime('%Y-%m-%d'):<12} "
                  f"{e.token_symbol:<8} "
                  f"{e.percent_of_supply:>7.1f}% "
                  f"{e.unlock_type.value:<10} "
                  f"{e.category.value:<12} "
                  f"{impact_str:>8}")

    def print_high_impact(self, days: int = 30):
        """Print high-impact unlocks only."""
        print("\n" + "=" * 80)
        print(f"HIGH-IMPACT UNLOCKS (Next {days} days)")
        print("=" * 80)

        high_impact = sorted(self.get_high_impact(days), key=lambda e: -e.impact_score)

        if not high_impact:
            print("\nNo high-impact unlocks found")
            return

        for e in high_impact[:10]:
            print(f"\n{e.token_symbol} ({e.token_name})")
            print(f"  Date: {e.unlock_date.strftime('%Y-%m-%d')} ({e.days_until} days)")
            print(f"  Amount: {e.percent_of_supply:.1f}% of supply (${e.amount_usd/1e6:.1f}M)")
            print(f"  Type: {e.unlock_type.value} | Category: {e.category.value}")
            print(f"  Impact Score: {e.impact_score:.0%}")
            print(f"  Signal: Consider BEARISH bias on FDV")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Token Unlock Calendar")
    parser.add_argument('--fetch', action='store_true', help='Fetch unlock data from DropsTab')
    parser.add_argument('--upcoming', action='store_true', help='Show upcoming unlocks')
    parser.add_argument('--alerts', action='store_true', help='Show high-impact unlocks')
    parser.add_argument('--signal', type=str, help='Get unlock signal for token (e.g., --signal ARB)')
    parser.add_argument('--days', type=int, default=30, help='Days to look ahead')

    args = parser.parse_args()

    calendar = TokenUnlockCalendar()

    if args.fetch:
        calendar.fetch_and_save()
        calendar.print_calendar(args.days)
    elif args.signal:
        if not calendar.load():
            print("[!] No saved data. Run --fetch first")
            return
        signal = calendar.get_unlock_signal(args.signal, args.days)
        print(f"\nUnlock Signal for {args.signal.upper()}:")
        for k, v in signal.items():
            print(f"  {k}: {v}")
    elif args.alerts:
        if not calendar.load():
            print("[!] No saved data. Run --fetch first")
            return
        calendar.print_high_impact(args.days)
    elif args.upcoming:
        if not calendar.load():
            print("[!] No saved data. Run --fetch first")
            return
        calendar.print_calendar(args.days)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
