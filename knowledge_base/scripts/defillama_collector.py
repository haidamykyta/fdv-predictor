"""
DefiLlama API Data Collector
Бесплатный API без ключа - собирает данные о протоколах, TVL, fundraising

API Docs: https://defillama.com/docs/api
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime

BASE_URL = "https://api.llama.fi"
COINS_URL = "https://coins.llama.fi"


class DefiLlamaCollector:
    """Collect data from DefiLlama API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "application/json",
        })
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def get_protocols(self) -> list:
        """Get all protocols with TVL data."""
        resp = self.session.get(f"{BASE_URL}/protocols", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_protocol(self, slug: str) -> dict:
        """Get detailed protocol data."""
        resp = self.session.get(f"{BASE_URL}/protocol/{slug}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_raises(self) -> list:
        """Get all fundraising rounds."""
        resp = self.session.get(f"{BASE_URL}/raises", timeout=30)
        resp.raise_for_status()
        return resp.json().get("raises", [])

    def get_tvl_history(self, protocol: str) -> list:
        """Get TVL history for protocol."""
        resp = self.session.get(f"{BASE_URL}/protocol/{protocol}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("tvl", [])

    def get_current_prices(self, coins: list) -> dict:
        """Get current prices for coins.

        coins format: ["coingecko:bitcoin", "coingecko:ethereum"]
        """
        coins_str = ",".join(coins)
        resp = self.session.get(
            f"{COINS_URL}/prices/current/{coins_str}",
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("coins", {})

    def collect_raises(self) -> list:
        """Collect all fundraising data."""
        print("Collecting fundraising rounds...")
        raises = self.get_raises()
        print(f"  Found {len(raises)} raises")
        return raises

    def collect_top_protocols(self, limit: int = 200) -> list:
        """Collect top protocols by TVL."""
        print(f"Collecting top {limit} protocols...")
        protocols = self.get_protocols()

        # Sort by TVL
        sorted_protocols = sorted(
            protocols,
            key=lambda x: x.get("tvl", 0) or 0,
            reverse=True
        )[:limit]

        print(f"  Found {len(sorted_protocols)} protocols")
        return sorted_protocols

    def extract_fdv_relevant_data(self, protocols: list) -> list:
        """Extract FDV-relevant data from protocols."""
        fdv_data = []

        for p in protocols:
            fdv_data.append({
                "id": p.get("slug"),
                "name": p.get("name"),
                "symbol": p.get("symbol", "").upper() if p.get("symbol") else None,
                "category": p.get("category"),
                "chains": p.get("chains", []),
                "tvl": p.get("tvl"),
                "mcap": p.get("mcap"),
                "fdv": p.get("fdv"),  # DefiLlama includes FDV!
                "change_1d": p.get("change_1d"),
                "change_7d": p.get("change_7d"),
            })

        return fdv_data

    def analyze_raises_by_category(self, raises: list) -> dict:
        """Analyze raises by category."""
        from collections import defaultdict

        by_category = defaultdict(list)

        for r in raises:
            category = r.get("category", "Unknown")
            amount = r.get("amount")
            if amount and isinstance(amount, (int, float)) and amount > 0:
                # Parse valuation - can be string or number
                val = r.get("valuation")
                if isinstance(val, str):
                    try:
                        val = float(val.replace(",", "").replace("$", ""))
                    except:
                        val = None

                by_category[category].append({
                    "name": r.get("name"),
                    "amount": amount,
                    "round": r.get("round"),
                    "date": r.get("date"),
                    "valuation": val,
                    "lead_investors": r.get("leadInvestors", []),
                    "other_investors": r.get("otherInvestors", []),
                })

        # Calculate stats per category
        stats = {}
        for cat, rounds in by_category.items():
            amounts = [r["amount"] for r in rounds if r["amount"]]
            valuations = [r["valuation"] for r in rounds if r.get("valuation") and isinstance(r["valuation"], (int, float))]

            stats[cat] = {
                "count": len(rounds),
                "total_raised": sum(amounts),
                "avg_raise": sum(amounts) / len(amounts) if amounts else 0,
                "median_raise": sorted(amounts)[len(amounts)//2] if amounts else 0,
                "avg_valuation": sum(valuations) / len(valuations) if valuations else None,
                "top_rounds": sorted(rounds, key=lambda x: x["amount"], reverse=True)[:5]
            }

        return stats

    def find_recent_tge_tokens(self, raises: list, months: int = 12) -> list:
        """Find tokens that had TGE in last N months."""
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=months * 30)
        recent = []

        for r in raises:
            date_str = r.get("date")
            if date_str:
                try:
                    # DefiLlama uses Unix timestamp
                    date = datetime.fromtimestamp(date_str)
                    if date > cutoff:
                        recent.append({
                            "name": r.get("name"),
                            "amount": r.get("amount"),
                            "round": r.get("round"),
                            "date": date.isoformat(),
                            "valuation": r.get("valuation"),
                            "category": r.get("category"),
                            "chains": r.get("chains", []),
                            "lead_investors": r.get("leadInvestors", []),
                        })
                except:
                    pass

        return sorted(recent, key=lambda x: x["date"], reverse=True)

    def save_data(self, data: any, filename: str):
        """Save to JSON."""
        filepath = self.data_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved to {filepath}")

    def run_collection(self):
        """Run full data collection."""
        print("=" * 50)
        print("DefiLlama Data Collection")
        print("=" * 50)

        # 1. Protocols with TVL/FDV
        protocols = self.collect_top_protocols(200)
        fdv_data = self.extract_fdv_relevant_data(protocols)

        # Filter only those with FDV
        with_fdv = [p for p in fdv_data if p.get("fdv")]
        print(f"  Protocols with FDV data: {len(with_fdv)}")

        self.save_data({
            "collected_at": datetime.now().isoformat(),
            "count": len(fdv_data),
            "with_fdv": len(with_fdv),
            "protocols": fdv_data
        }, "defillama_protocols.json")

        # 2. Fundraising rounds
        raises = self.collect_raises()

        self.save_data({
            "collected_at": datetime.now().isoformat(),
            "count": len(raises),
            "raises": raises
        }, "defillama_raises.json")

        # 3. Category analysis
        category_stats = self.analyze_raises_by_category(raises)

        self.save_data({
            "collected_at": datetime.now().isoformat(),
            "categories": category_stats
        }, "defillama_category_stats.json")

        # 4. Recent TGE tokens
        recent = self.find_recent_tge_tokens(raises, months=18)

        self.save_data({
            "collected_at": datetime.now().isoformat(),
            "count": len(recent),
            "tokens": recent
        }, "defillama_recent_raises.json")

        print("\n" + "=" * 50)
        print("Done!")
        print(f"  Protocols: {len(fdv_data)} ({len(with_fdv)} with FDV)")
        print(f"  Raises: {len(raises)}")
        print(f"  Categories: {len(category_stats)}")
        print(f"  Recent (18mo): {len(recent)}")


def main():
    import sys

    collector = DefiLlamaCollector()

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test
        print("Quick test - top 10 protocols:")
        protocols = collector.get_protocols()[:10]
        for p in protocols:
            fdv = p.get('fdv')
            fdv_str = f"${fdv/1e9:.1f}B" if fdv else "N/A"
            tvl = p.get('tvl', 0)
            tvl_str = f"${tvl/1e9:.1f}B" if tvl else "N/A"
            print(f"  {p.get('name', 'Unknown')}: TVL={tvl_str}, FDV={fdv_str}")

    elif len(sys.argv) > 1 and sys.argv[1] == "--raises":
        # Just raises
        print("Collecting raises only...")
        raises = collector.collect_raises()
        collector.save_data({
            "collected_at": datetime.now().isoformat(),
            "count": len(raises),
            "raises": raises
        }, "defillama_raises.json")

    else:
        collector.run_collection()


if __name__ == "__main__":
    main()
