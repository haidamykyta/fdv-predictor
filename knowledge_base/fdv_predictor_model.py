"""
FDV Predictor using Knowledge Base
Predicts token FDV at launch based on historical data and formulas
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TokenInput:
    """Input data for FDV prediction."""
    name: str
    sector: str
    total_raised_usd: float = 0
    last_valuation_usd: float = None
    fundraising_round: str = None  # seed, series_a, series_b
    top_investors: list = None
    initial_circ_pct: float = 15.0
    twitter_followers: int = 100000
    tvl_usd: float = None
    annual_revenue_usd: float = None


class FDVPredictor:
    """Predict FDV at token launch using knowledge base."""

    def __init__(self, kb_path: str = None):
        if kb_path is None:
            kb_path = Path(__file__).parent

        self.kb_path = Path(kb_path)

        # Try real data first, fall back to manual data
        self.sectors = self._load_json("data/sector_benchmarks_real.json")
        if not self.sectors:
            self.sectors = self._load_json("data/sector_benchmarks.json")

        self.investors = self._load_json("data/investor_tiers_real.json")
        if not self.investors:
            self.vc_data = self._load_json("data/vc_tiers.json")
        else:
            self.vc_data = None

        self.historical = self._load_json("data/token_launches.json")
        self.recent_launches = self._load_json("data/recent_launches.json")

    def _load_json(self, filename: str) -> dict:
        filepath = self.kb_path / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get_investor_tier(self, investors: list) -> Tuple[int, float]:
        """Get investor tier and multiplier."""
        if not investors:
            return 0, 0.7

        # Use real investor data if available
        if self.investors and 'investors' in self.investors:
            investor_list = self.investors['investors']

            # Build lookup by name
            tier_lookup = {}
            for inv in investor_list:
                name_lower = inv['name'].lower()
                tier_lookup[name_lower] = inv.get('tier', 3)

            # Check each investor
            best_tier = 3
            for inv in investors:
                inv_lower = inv.lower()
                # Exact match
                if inv_lower in tier_lookup:
                    best_tier = min(best_tier, tier_lookup[inv_lower])
                # Partial match
                else:
                    for known_name, tier in tier_lookup.items():
                        if inv_lower in known_name or known_name in inv_lower:
                            best_tier = min(best_tier, tier)
                            break

            multipliers = {1: 1.5, 2: 1.2, 3: 1.0}
            return best_tier, multipliers.get(best_tier, 1.0)

        # Fall back to old vc_data format
        if self.vc_data:
            tier_1_names = [v['name'].lower() for v in self.vc_data.get('investors', {}).get('tier_1', [])]
            tier_2_names = [v['name'].lower() for v in self.vc_data.get('investors', {}).get('tier_2', [])]

            for inv in investors:
                inv_lower = inv.lower()
                if inv_lower in tier_1_names:
                    return 1, 1.5
                if inv_lower in tier_2_names:
                    return 2, 1.2

        return 3, 1.0

    def get_sector_benchmark(self, sector: str) -> dict:
        """Get sector benchmark data."""
        sectors = self.sectors.get('sectors', {})

        # Direct match
        if sector in sectors:
            data = sectors[sector]
            # Convert from real data format (in millions) to full USD
            return {
                'avg_fdv_usd': data.get('avg_raise_m', 50) * 1e6 * 20,  # avg raise * 20x = estimated FDV
                'median_fdv_usd': data.get('median_raise_m', 30) * 1e6 * 15,  # median * 15x
                'avg_raise_m': data.get('avg_raise_m', 50),
                'median_raise_m': data.get('median_raise_m', 30),
                'count': data.get('count', 0),
            }

        # Fuzzy match
        sector_lower = sector.lower().replace(' ', '_').replace('-', '_')
        for key, data in sectors.items():
            key_lower = key.lower().replace(' ', '_').replace('-', '_') if key else ''
            if sector_lower in key_lower or key_lower in sector_lower:
                return {
                    'avg_fdv_usd': data.get('avg_raise_m', 50) * 1e6 * 20,
                    'median_fdv_usd': data.get('median_raise_m', 30) * 1e6 * 15,
                    'avg_raise_m': data.get('avg_raise_m', 50),
                    'median_raise_m': data.get('median_raise_m', 30),
                    'count': data.get('count', 0),
                }

        # Default
        return {
            'avg_fdv_usd': 1000000000,  # $1B default
            'median_fdv_usd': 500000000,  # $500M
            'avg_raise_m': 50,
            'median_raise_m': 30,
            'count': 0,
        }

    def get_historical_comparables(self, sector: str, limit: int = 5) -> list:
        """Get historical token launches in same sector."""
        tokens = self.historical.get('tokens', [])
        matches = []

        for token in tokens:
            if sector.lower() in token.get('sector', '').lower():
                matches.append(token)

        # Sort by launch date (newest first)
        matches.sort(key=lambda x: x.get('launch_date', ''), reverse=True)
        return matches[:limit]

    def predict(self, token: TokenInput, market_condition: str = "neutral") -> dict:
        """
        Predict FDV for token.

        Args:
            token: TokenInput with project data
            market_condition: "bull", "neutral", or "bear"

        Returns:
            dict with predicted FDV range and confidence
        """
        # Get sector benchmark
        sector_data = self.get_sector_benchmark(token.sector)
        sector_median = sector_data.get('median_fdv_usd', 2e9)
        sector_avg = sector_data.get('avg_fdv_usd', 3e9)

        # Method 1: Fundraising multiple
        fundraising_fdv = None
        if token.last_valuation_usd:
            multiples = {
                'seed': 30,
                'series_a': 8,
                'series_b': 4,
                'series_c': 2.5,
            }
            mult = multiples.get(token.fundraising_round, 5)
            fundraising_fdv = token.last_valuation_usd * mult

        # Method 2: TVL/Revenue multiple (for DeFi)
        metrics_fdv = None
        if token.tvl_usd:
            tvl_mult = 0.3  # Conservative
            metrics_fdv = token.tvl_usd * tvl_mult
        elif token.annual_revenue_usd:
            rev_mult = 100
            metrics_fdv = token.annual_revenue_usd * rev_mult

        # Get investor tier
        investor_tier, tier_mult = self.get_investor_tier(token.top_investors or [])

        # Market condition multiplier
        market_mults = {'bull': 1.5, 'neutral': 1.0, 'bear': 0.5}
        market_mult = market_mults.get(market_condition, 1.0)

        # Circulation premium (low float = higher initial price)
        circ_pct = max(token.initial_circ_pct, 1) / 100
        circ_premium = min(1 / math.sqrt(circ_pct), 5)

        # Hype factor
        followers = max(token.twitter_followers, 1000)
        hype_factor = min(max(math.log10(followers) / 5, 0.5), 1.5)

        # Calculate base FDV
        if fundraising_fdv:
            base_fdv = fundraising_fdv
        elif metrics_fdv:
            base_fdv = metrics_fdv
        else:
            base_fdv = sector_median

        # Apply factors
        predicted_fdv = base_fdv * tier_mult * market_mult * hype_factor

        # Apply circulation premium (dampened)
        predicted_fdv = predicted_fdv * (1 + (circ_premium - 1) * 0.3)

        # Calculate range
        low_fdv = predicted_fdv * 0.5
        high_fdv = predicted_fdv * 2.0

        # Confidence based on data quality
        confidence = 0.3  # Base
        if token.last_valuation_usd:
            confidence += 0.2
        if token.top_investors:
            confidence += 0.15
        if token.tvl_usd or token.annual_revenue_usd:
            confidence += 0.2
        confidence = min(confidence, 0.85)

        # Get comparables
        comparables = self.get_historical_comparables(token.sector)

        return {
            'token_name': token.name,
            'predicted_fdv_usd': predicted_fdv,
            'fdv_range': {
                'low': low_fdv,
                'high': high_fdv,
            },
            'confidence': confidence,
            'factors': {
                'base_fdv': base_fdv,
                'investor_tier': investor_tier,
                'tier_multiplier': tier_mult,
                'market_multiplier': market_mult,
                'hype_factor': hype_factor,
                'circ_premium': circ_premium,
            },
            'sector_benchmark': {
                'median': sector_median,
                'avg': sector_avg,
            },
            'comparables': [
                {
                    'name': c['name'],
                    'fdv_at_launch': c.get('fdv_at_launch_usd'),
                }
                for c in comparables[:3]
            ],
        }


def format_usd(value: float) -> str:
    """Format USD value."""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"


def main():
    """Example usage."""
    predictor = FDVPredictor()

    # Example: Predict FDV for a new L2
    token = TokenInput(
        name="ExampleL2",
        sector="l2_zk",
        total_raised_usd=100_000_000,
        last_valuation_usd=1_000_000_000,  # $1B valuation
        fundraising_round="series_a",
        top_investors=["a16z", "Paradigm"],
        initial_circ_pct=10.0,
        twitter_followers=500_000,
    )

    result = predictor.predict(token, market_condition="bull")

    print("=" * 60)
    print(f"FDV PREDICTION: {token.name}")
    print("=" * 60)
    print(f"\nPredicted FDV: {format_usd(result['predicted_fdv_usd'])}")
    print(f"Range: {format_usd(result['fdv_range']['low'])} - {format_usd(result['fdv_range']['high'])}")
    print(f"Confidence: {result['confidence']:.0%}")

    print("\nFactors:")
    for key, val in result['factors'].items():
        print(f"  {key}: {val}")

    print("\nSector Benchmark:")
    print(f"  Median: {format_usd(result['sector_benchmark']['median'])}")

    print("\nComparables:")
    for c in result['comparables']:
        if c['fdv_at_launch']:
            print(f"  {c['name']}: {format_usd(c['fdv_at_launch'])}")

    # Example 2: DeFi protocol
    print("\n" + "=" * 60)
    defi_token = TokenInput(
        name="NewDEX",
        sector="defi_dex",
        tvl_usd=500_000_000,
        annual_revenue_usd=10_000_000,
        top_investors=["Dragonfly"],
        initial_circ_pct=15.0,
        twitter_followers=100_000,
    )

    result2 = predictor.predict(defi_token, market_condition="neutral")

    print(f"FDV PREDICTION: {defi_token.name}")
    print("=" * 60)
    print(f"\nPredicted FDV: {format_usd(result2['predicted_fdv_usd'])}")
    print(f"Range: {format_usd(result2['fdv_range']['low'])} - {format_usd(result2['fdv_range']['high'])}")


if __name__ == "__main__":
    main()
