"""
DropsTab API Data Collector
Собирает данные о токенах, fundraising, investors, unlocks

API Key: Advanced Plan (500,000 requests/month)
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime

BASE_URL = "https://public-api.dropstab.com/api/v1"

# API Key - Advanced Plan
API_KEY = "b6fe2216-844f-4248-bac4-371f719d418f"

HEADERS = {
    "accept": "*/*",
    "x-dropstab-api-key": API_KEY
}

# Целевые токены для сбора детальных данных
TARGET_TOKENS = [
    "zksync", "scroll", "linea", "taiko", "berachain",
    "monad", "hyperliquid", "etherfi", "renzo", "pendle",
    # Дополнительно интересные
    "eigenlayer", "layerzero", "starknet", "arbitrum", "optimism",
    "celestia", "sui", "aptos", "sei", "injective",
    "jito", "jupiter", "pyth", "wormhole", "blur"
]


class DropsTabCollector:
    """Collect data from DropsTab API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with error handling."""
        url = f"{BASE_URL}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            # DropsTab wraps response in {status, data, ...}
            if isinstance(result, dict) and 'data' in result:
                return result['data']
            return result
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP Error {resp.status_code}: {endpoint}")
            if resp.status_code == 401:
                print("    [!] API Key invalid or expired")
            return None
        except Exception as e:
            print(f"    Error: {e}")
            return None

    # === COINS ===
    def get_coins(self, page: int = 0, page_size: int = 100) -> dict:
        """Get list of coins."""
        return self._request("coins", {
            "page": page,
            "pageSize": page_size,
            "trading": "ALL",
            "sortingField": "RANK",
            "sortingOrder": "ASC"
        })

    def get_coin_by_id(self, coin_id: str) -> dict:
        """Get detailed coin info by ID."""
        return self._request(f"coins/{coin_id}")

    def search_coin(self, query: str) -> dict:
        """Search for coin by name/symbol."""
        return self._request("coins/search", {"query": query})

    # === TOKEN UNLOCKS ===
    def get_unlocks_list(self, page: int = 0, page_size: int = 100) -> dict:
        """Get list of all token unlocks."""
        return self._request("token-unlocks", {
            "page": page,
            "pageSize": page_size
        })

    def get_coin_unlocks(self, coin_id: str) -> dict:
        """Get unlock schedule for specific coin."""
        return self._request(f"token-unlocks/{coin_id}")

    # === FUNDING ROUNDS ===
    def get_funding_rounds(self, page: int = 0, page_size: int = 100) -> dict:
        """Get funding rounds."""
        return self._request("funding-rounds", {
            "page": page,
            "pageSize": page_size
        })

    def get_coin_funding(self, coin_id: str) -> dict:
        """Get funding rounds for specific coin."""
        return self._request(f"coins/{coin_id}/funding-rounds")

    # === INVESTORS ===
    def get_investors(self, page: int = 0, page_size: int = 100) -> dict:
        """Get list of investors/VCs."""
        return self._request("investors", {
            "page": page,
            "pageSize": page_size
        })

    def get_investor_details(self, investor_id: str) -> dict:
        """Get investor portfolio."""
        return self._request(f"investors/{investor_id}")

    # === COLLECTION METHODS ===
    def collect_all_coins(self, max_pages: int = 20) -> list:
        """Collect all coins."""
        print("\n[1] Collecting coins...")
        all_coins = []

        for page in range(max_pages):
            data = self.get_coins(page=page, page_size=100)
            if not data:
                break

            coins = data.get('content', data) if isinstance(data, dict) else data
            if not coins:
                break

            all_coins.extend(coins)
            print(f"    Page {page}: {len(coins)} coins (total: {len(all_coins)})")

            if len(coins) < 100:
                break
            time.sleep(0.3)

        return all_coins

    def collect_all_unlocks(self, max_pages: int = 20) -> list:
        """Collect all token unlocks."""
        print("\n[2] Collecting token unlocks...")
        all_unlocks = []

        for page in range(max_pages):
            data = self.get_unlocks_list(page=page, page_size=100)
            if not data:
                break

            unlocks = data.get('content', data) if isinstance(data, dict) else data
            if not unlocks:
                break

            all_unlocks.extend(unlocks)
            print(f"    Page {page}: {len(unlocks)} unlocks (total: {len(all_unlocks)})")

            if len(unlocks) < 100:
                break
            time.sleep(0.3)

        return all_unlocks

    def collect_all_funding(self, max_pages: int = 30) -> list:
        """Collect all funding rounds."""
        print("\n[3] Collecting funding rounds...")
        all_rounds = []

        for page in range(max_pages):
            data = self.get_funding_rounds(page=page, page_size=100)
            if not data:
                break

            rounds = data.get('content', data) if isinstance(data, dict) else data
            if not rounds:
                break

            all_rounds.extend(rounds)
            print(f"    Page {page}: {len(rounds)} rounds (total: {len(all_rounds)})")

            if len(rounds) < 100:
                break
            time.sleep(0.3)

        return all_rounds

    def collect_all_investors(self, max_pages: int = 15) -> list:
        """Collect all investors."""
        print("\n[4] Collecting investors...")
        all_investors = []

        for page in range(max_pages):
            data = self.get_investors(page=page, page_size=100)
            if not data:
                break

            investors = data.get('content', data) if isinstance(data, dict) else data
            if not investors:
                break

            all_investors.extend(investors)
            print(f"    Page {page}: {len(investors)} investors (total: {len(all_investors)})")

            if len(investors) < 100:
                break
            time.sleep(0.3)

        return all_investors

    def collect_target_tokens_details(self, coins_data: list) -> list:
        """Collect detailed data for target tokens."""
        print("\n[5] Collecting details for target tokens...")

        # Build lookup by symbol/name
        coin_lookup = {}
        for coin in coins_data:
            symbol = (coin.get('symbol') or '').lower()
            name = (coin.get('name') or '').lower()
            coin_id = coin.get('id') or coin.get('slug')
            if symbol:
                coin_lookup[symbol] = coin_id
            if name:
                coin_lookup[name] = coin_id

        target_details = []

        for token in TARGET_TOKENS:
            token_lower = token.lower()
            coin_id = coin_lookup.get(token_lower)

            if not coin_id:
                # Try search
                search_result = self.search_coin(token)
                if search_result:
                    results = search_result if isinstance(search_result, list) else search_result.get('content', [])
                    if results:
                        coin_id = results[0].get('id') or results[0].get('slug')

            if coin_id:
                print(f"    {token}: found (id={coin_id})")

                # Get detailed info
                details = self.get_coin_by_id(coin_id)
                unlocks = self.get_coin_unlocks(coin_id)
                funding = self.get_coin_funding(coin_id)

                target_details.append({
                    "query": token,
                    "coin_id": coin_id,
                    "details": details,
                    "unlocks": unlocks,
                    "funding": funding,
                })
                time.sleep(0.5)
            else:
                print(f"    {token}: NOT FOUND")

        return target_details

    def save_data(self, data: any, filename: str):
        """Save to JSON."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved: {filepath}")

    def run_full_collection(self):
        """Run complete data collection."""
        print("=" * 60)
        print("DropsTab Full Data Collection")
        print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)

        # 1. Coins
        coins = self.collect_all_coins()
        if coins:
            self.save_data({
                "collected_at": datetime.now().isoformat(),
                "count": len(coins),
                "coins": coins
            }, "dropstab_coins.json")

        # 2. Token unlocks
        unlocks = self.collect_all_unlocks()
        if unlocks:
            self.save_data({
                "collected_at": datetime.now().isoformat(),
                "count": len(unlocks),
                "unlocks": unlocks
            }, "dropstab_unlocks.json")

        # 3. Funding rounds
        funding = self.collect_all_funding()
        if funding:
            self.save_data({
                "collected_at": datetime.now().isoformat(),
                "count": len(funding),
                "rounds": funding
            }, "dropstab_funding.json")

        # 4. Investors
        investors = self.collect_all_investors()
        if investors:
            self.save_data({
                "collected_at": datetime.now().isoformat(),
                "count": len(investors),
                "investors": investors
            }, "dropstab_investors.json")

        # 5. Target tokens details
        if coins:
            target_data = self.collect_target_tokens_details(coins)
            if target_data:
                self.save_data({
                    "collected_at": datetime.now().isoformat(),
                    "count": len(target_data),
                    "tokens": target_data
                }, "dropstab_target_tokens.json")

        # Summary
        print("\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print("=" * 60)
        print(f"  Coins: {len(coins) if coins else 0}")
        print(f"  Unlocks: {len(unlocks) if unlocks else 0}")
        print(f"  Funding rounds: {len(funding) if funding else 0}")
        print(f"  Investors: {len(investors) if investors else 0}")
        print(f"  Target tokens: {len(target_data) if 'target_data' in dir() else 0}")


def test_api():
    """Quick API test."""
    print("Testing DropsTab API...")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")

    collector = DropsTabCollector()

    # Test 1: Coins
    print("\n1. Testing /coins...")
    data = collector.get_coins(page=0, page_size=3)
    if data:
        coins = data.get('content', data) if isinstance(data, dict) else data
        print(f"   OK - {len(coins)} coins")
        if coins:
            c = coins[0]
            print(f"   Example: {c.get('name')} ({c.get('symbol')})")
            print(f"   Keys: {list(c.keys())[:10]}")
    else:
        print("   FAILED")

    # Test 2: Unlocks
    print("\n2. Testing /token-unlocks...")
    data = collector.get_unlocks_list(page=0, page_size=3)
    if data:
        unlocks = data.get('content', data) if isinstance(data, dict) else data
        print(f"   OK - {len(unlocks) if unlocks else 0} unlocks")
        if unlocks and len(unlocks) > 0:
            print(f"   Keys: {list(unlocks[0].keys())[:10]}")
    else:
        print("   FAILED")

    # Test 3: Funding
    print("\n3. Testing /funding-rounds...")
    data = collector.get_funding_rounds(page=0, page_size=3)
    if data:
        rounds = data.get('content', data) if isinstance(data, dict) else data
        print(f"   OK - {len(rounds) if rounds else 0} rounds")
    else:
        print("   FAILED")

    # Test 4: Investors
    print("\n4. Testing /investors...")
    data = collector.get_investors(page=0, page_size=3)
    if data:
        investors = data.get('content', data) if isinstance(data, dict) else data
        print(f"   OK - {len(investors) if investors else 0} investors")
    else:
        print("   FAILED")

    # Test 5: Search
    print("\n5. Testing search for 'ethereum'...")
    data = collector.search_coin("ethereum")
    if data:
        results = data if isinstance(data, list) else data.get('content', [])
        print(f"   OK - {len(results) if results else 0} results")
    else:
        print("   FAILED")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--test":
            test_api()
        elif cmd == "--full":
            collector = DropsTabCollector()
            collector.run_full_collection()
        elif cmd == "--coins":
            collector = DropsTabCollector()
            coins = collector.collect_all_coins()
            if coins:
                collector.save_data({"coins": coins}, "dropstab_coins.json")
        elif cmd == "--unlocks":
            collector = DropsTabCollector()
            unlocks = collector.collect_all_unlocks()
            if unlocks:
                collector.save_data({"unlocks": unlocks}, "dropstab_unlocks.json")
        else:
            print(f"Unknown command: {cmd}")
    else:
        print("DropsTab API Collector")
        print("=" * 40)
        print("Commands:")
        print("  --test    Quick API test")
        print("  --full    Full data collection")
        print("  --coins   Collect coins only")
        print("  --unlocks Collect unlocks only")
