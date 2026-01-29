"""
Market Conditions Tracker
Fetches BTC/ETH price data to determine market sentiment for FDV predictions
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def get_crypto_prices():
    """Get current BTC/ETH prices and 30-day changes from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin,ethereum",
        "vs_currencies": "usd",
        "include_24hr_change": "true",
        "include_30d_change": "true",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching prices: {e}")
        return None


def get_historical_prices(coin_id: str, days: int = 30):
    """Get historical price data from CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("prices", [])
        if prices:
            return {
                "start_price": prices[0][1],
                "end_price": prices[-1][1],
                "change_pct": ((prices[-1][1] - prices[0][1]) / prices[0][1]) * 100,
                "high": max(p[1] for p in prices),
                "low": min(p[1] for p in prices),
            }
    except Exception as e:
        print(f"Error fetching historical for {coin_id}: {e}")
    return None


def calculate_market_condition(btc_change: float, eth_change: float) -> dict:
    """
    Determine market condition based on BTC/ETH performance.

    Returns:
        dict with condition ("bull", "neutral", "bear") and multiplier
    """
    avg_change = (btc_change + eth_change) / 2

    if avg_change > 20:
        return {"condition": "bull", "multiplier": 1.5, "description": "Strong bull market"}
    elif avg_change > 10:
        return {"condition": "bull", "multiplier": 1.3, "description": "Moderate bull market"}
    elif avg_change > 0:
        return {"condition": "neutral", "multiplier": 1.1, "description": "Slightly bullish"}
    elif avg_change > -10:
        return {"condition": "neutral", "multiplier": 1.0, "description": "Neutral/consolidation"}
    elif avg_change > -20:
        return {"condition": "bear", "multiplier": 0.8, "description": "Moderate bear market"}
    else:
        return {"condition": "bear", "multiplier": 0.6, "description": "Strong bear market"}


def get_fear_greed_index():
    """Get Crypto Fear & Greed Index."""
    url = "https://api.alternative.me/fng/"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            entry = data["data"][0]
            return {
                "value": int(entry.get("value", 50)),
                "classification": entry.get("value_classification", "Neutral"),
            }
    except Exception as e:
        print(f"Error fetching Fear & Greed: {e}")
    return {"value": 50, "classification": "Neutral"}


def analyze_market():
    """Full market analysis for FDV prediction adjustment."""
    print("=" * 60)
    print("MARKET CONDITIONS ANALYSIS")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Get historical data
    btc_data = get_historical_prices("bitcoin", 30)
    eth_data = get_historical_prices("ethereum", 30)

    if not btc_data or not eth_data:
        print("\nError: Could not fetch market data")
        return None

    print("\n30-DAY PERFORMANCE:")
    print("-" * 40)
    print(f"BTC: {btc_data['change_pct']:+.1f}%")
    print(f"     ${btc_data['start_price']:,.0f} -> ${btc_data['end_price']:,.0f}")
    print(f"     Range: ${btc_data['low']:,.0f} - ${btc_data['high']:,.0f}")
    print()
    print(f"ETH: {eth_data['change_pct']:+.1f}%")
    print(f"     ${eth_data['start_price']:,.0f} -> ${eth_data['end_price']:,.0f}")
    print(f"     Range: ${eth_data['low']:,.0f} - ${eth_data['high']:,.0f}")

    # Market condition
    condition = calculate_market_condition(btc_data['change_pct'], eth_data['change_pct'])

    print()
    print("MARKET CONDITION:")
    print("-" * 40)
    print(f"Status: {condition['condition'].upper()}")
    print(f"Description: {condition['description']}")
    print(f"FDV Multiplier: {condition['multiplier']}x")

    # Fear & Greed
    fng = get_fear_greed_index()
    print()
    print("FEAR & GREED INDEX:")
    print("-" * 40)
    print(f"Value: {fng['value']}/100")
    print(f"Classification: {fng['classification']}")

    # Combine into single score
    # FNG: 0-25 = Extreme Fear, 25-45 = Fear, 45-55 = Neutral, 55-75 = Greed, 75-100 = Extreme Greed
    fng_multiplier = 1.0
    if fng['value'] >= 75:
        fng_multiplier = 1.2  # Extreme greed = higher FDV expectations
    elif fng['value'] >= 55:
        fng_multiplier = 1.1
    elif fng['value'] <= 25:
        fng_multiplier = 0.8  # Extreme fear = lower FDV
    elif fng['value'] <= 45:
        fng_multiplier = 0.9

    combined_multiplier = (condition['multiplier'] + fng_multiplier) / 2

    print()
    print("COMBINED ANALYSIS:")
    print("-" * 40)
    print(f"Price-based multiplier: {condition['multiplier']:.2f}x")
    print(f"Sentiment multiplier: {fng_multiplier:.2f}x")
    print(f"Combined multiplier: {combined_multiplier:.2f}x")

    result = {
        "timestamp": datetime.now().isoformat(),
        "btc": {
            "price": btc_data['end_price'],
            "change_30d_pct": btc_data['change_pct'],
        },
        "eth": {
            "price": eth_data['end_price'],
            "change_30d_pct": eth_data['change_pct'],
        },
        "condition": condition['condition'],
        "price_multiplier": condition['multiplier'],
        "fear_greed": fng,
        "sentiment_multiplier": fng_multiplier,
        "combined_multiplier": combined_multiplier,
    }

    # Save to file
    output_path = DATA_DIR / "market_conditions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {output_path}")

    return result


def get_current_multiplier() -> float:
    """
    Get current market multiplier for FDV predictions.
    Loads from cached file or fetches fresh data.
    """
    cache_file = DATA_DIR / "market_conditions.json"

    # Check if cache exists and is fresh (< 1 hour old)
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=1):
                return data['combined_multiplier']
        except Exception:
            pass

    # Fetch fresh data
    result = analyze_market()
    if result:
        return result['combined_multiplier']

    return 1.0  # Default neutral


if __name__ == "__main__":
    result = analyze_market()

    if result:
        print()
        print("=" * 60)
        print("RECOMMENDATION FOR FDV PREDICTION:")
        print("=" * 60)

        mult = result['combined_multiplier']
        if mult >= 1.3:
            print("Apply BULL market adjustment (+30-50% to base FDV)")
            print("Token launches in this environment tend to see higher valuations")
        elif mult >= 1.1:
            print("Apply MODERATE BULL adjustment (+10-20% to base FDV)")
        elif mult <= 0.7:
            print("Apply STRONG BEAR adjustment (-30-40% to base FDV)")
            print("Token launches may struggle in current environment")
        elif mult <= 0.9:
            print("Apply BEAR market adjustment (-10-20% to base FDV)")
        else:
            print("Market is NEUTRAL - use base FDV prediction")

        print(f"\nUse multiplier: {mult:.2f}x on your base FDV prediction")
