"""
CoinGecko Data Collector
Бесплатный API без ключа - собирает данные о токенах

Лимиты: 10-30 запросов в минуту
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

BASE_URL = "https://api.coingecko.com/api/v3"


class CoinGeckoCollector:
    """Collect data from CoinGecko API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
        })
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.request_count = 0
        self.last_request = 0

    def _rate_limit(self):
        """Rate limiting - max 10 req/min."""
        now = time.time()
        if now - self.last_request < 6:  # 6 sec between requests
            time.sleep(6 - (now - self.last_request))
        self.last_request = time.time()
        self.request_count += 1

    def get_coins_list(self) -> list:
        """Get list of all coins."""
        self._rate_limit()
        resp = self.session.get(f"{BASE_URL}/coins/list", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_coin_data(self, coin_id: str) -> dict:
        """Get detailed coin data including FDV."""
        self._rate_limit()
        resp = self.session.get(
            f"{BASE_URL}/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "false",
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def get_coin_history(self, coin_id: str, date: str) -> dict:
        """Get historical data for coin on specific date (dd-mm-yyyy)."""
        self._rate_limit()
        resp = self.session.get(
            f"{BASE_URL}/coins/{coin_id}/history",
            params={"date": date, "localization": "false"},
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def get_markets(self, page: int = 1, per_page: int = 250) -> list:
        """Get top coins by market cap with FDV."""
        self._rate_limit()
        resp = self.session.get(
            f"{BASE_URL}/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def collect_top_coins(self, num_coins: int = 500) -> list:
        """Collect top coins with market data."""
        print(f"Collecting top {num_coins} coins...")
        all_coins = []

        pages = (num_coins // 250) + 1
        for page in range(1, pages + 1):
            try:
                coins = self.get_markets(page=page, per_page=250)
                all_coins.extend(coins)
                print(f"  Page {page}: {len(coins)} coins")

                if len(coins) < 250:
                    break
            except Exception as e:
                print(f"  Error: {e}")
                break

        return all_coins[:num_coins]

    def extract_fdv_data(self, coins: list) -> list:
        """Extract FDV-relevant data from coins."""
        fdv_data = []

        for coin in coins:
            fdv_data.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "market_cap": coin.get("market_cap"),
                "fdv": coin.get("fully_diluted_valuation"),
                "price": coin.get("current_price"),
                "circulating_supply": coin.get("circulating_supply"),
                "total_supply": coin.get("total_supply"),
                "max_supply": coin.get("max_supply"),
                "ath": coin.get("ath"),
                "ath_date": coin.get("ath_date"),
                "rank": coin.get("market_cap_rank"),
            })

        return fdv_data

    def collect_specific_tokens(self, token_ids: list) -> list:
        """Collect detailed data for specific tokens."""
        print(f"Collecting data for {len(token_ids)} specific tokens...")
        results = []

        for i, token_id in enumerate(token_ids):
            try:
                data = self.get_coin_data(token_id)

                market_data = data.get("market_data", {})

                token_info = {
                    "id": data.get("id"),
                    "symbol": data.get("symbol", "").upper(),
                    "name": data.get("name"),
                    "categories": data.get("categories", []),
                    "description": data.get("description", {}).get("en", "")[:500],

                    # Market data
                    "price_usd": market_data.get("current_price", {}).get("usd"),
                    "market_cap": market_data.get("market_cap", {}).get("usd"),
                    "fdv": market_data.get("fully_diluted_valuation", {}).get("usd"),
                    "volume_24h": market_data.get("total_volume", {}).get("usd"),

                    # Supply
                    "circulating_supply": market_data.get("circulating_supply"),
                    "total_supply": market_data.get("total_supply"),
                    "max_supply": market_data.get("max_supply"),

                    # ATH
                    "ath": market_data.get("ath", {}).get("usd"),
                    "ath_date": market_data.get("ath_date", {}).get("usd"),

                    # Social
                    "twitter_followers": data.get("community_data", {}).get("twitter_followers"),
                    "telegram_members": data.get("community_data", {}).get("telegram_channel_user_count"),

                    # Genesis
                    "genesis_date": data.get("genesis_date"),
                }

                results.append(token_info)
                print(f"  [{i+1}/{len(token_ids)}] {token_id}: FDV=${token_info['fdv']}")

            except Exception as e:
                print(f"  [{i+1}/{len(token_ids)}] {token_id}: ERROR - {e}")

        return results

    def save_data(self, data: any, filename: str):
        """Save to JSON."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved to {filepath}")

    def run_collection(self):
        """Run full data collection."""
        print("=" * 50)
        print("CoinGecko Data Collection")
        print("=" * 50)

        # Top coins
        coins = self.collect_top_coins(500)
        fdv_data = self.extract_fdv_data(coins)

        self.save_data({
            "collected_at": datetime.now().isoformat(),
            "count": len(fdv_data),
            "coins": fdv_data
        }, "coingecko_top_coins.json")

        # Specific tokens we care about
        target_tokens = [
            "eigenlayer", "ethena", "safe", "friend-tech",
            "jupiter", "starknet", "celestia", "pyth-network",
            "wormhole", "layerzero", "dymension", "manta-network",
            "ondo-finance", "blast", "mode", "renzo",
            "pendle", "ether-fi", "scroll",
        ]

        detailed = self.collect_specific_tokens(target_tokens)
        self.save_data({
            "collected_at": datetime.now().isoformat(),
            "count": len(detailed),
            "tokens": detailed
        }, "coingecko_fdv_tokens.json")

        print("\n" + "=" * 50)
        print("Done!")
        print(f"  Top coins: {len(fdv_data)}")
        print(f"  Detailed tokens: {len(detailed)}")


def main():
    import sys

    collector = CoinGeckoCollector()

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test
        print("Quick test - top 10 coins:")
        coins = collector.get_markets(page=1, per_page=10)
        for c in coins:
            fdv = c.get('fully_diluted_valuation')
            fdv_str = f"${fdv/1e9:.1f}B" if fdv else "N/A"
            print(f"  {c['symbol'].upper()}: FDV={fdv_str}")
    else:
        collector.run_collection()


if __name__ == "__main__":
    main()
