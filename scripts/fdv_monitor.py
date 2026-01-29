"""
FDV Market Monitor
Automatically discovers and tracks new FDV markets on Polymarket
Generates predictions using our knowledge base
"""

import requests
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "knowledge_base" / "data"

# API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"

# Keywords for FDV market discovery
FDV_KEYWORDS = [
    "fdv", "fully diluted", "valuation",
    "market cap", "mcap", "fmv", "ipo", "tge"
]

# Known projects to track
TRACKED_PROJECTS = [
    # L2/Infrastructure
    "megaeth", "trove", "opensea", "monad", "berachain",
    "scroll", "linea", "taiko", "eclipse", "fuel", "movement",
    # DeFi
    "ethena", "eigenlayer", "renzo", "kelp", "ether.fi",
    "pendle", "morpho", "kamino", "marginfi",
    # AI/Gaming
    "worldcoin", "render", "akash", "io.net",
    # Meme/Social
    "farcaster", "lens", "friend.tech",
]


def fetch_all_events(active_only: bool = True, limit: int = 500) -> List[Dict]:
    """Fetch all events from Polymarket."""
    try:
        params = {"limit": limit}
        if active_only:
            params["closed"] = False

        response = requests.get(
            f"{GAMMA_API}/events",
            params=params,
            timeout=30
        )
        if response.ok:
            return response.json()
    except Exception as e:
        print(f"[ERROR] Failed to fetch events: {e}")

    return []


def search_events(query: str) -> List[Dict]:
    """Search for events by keyword."""
    try:
        response = requests.get(
            f"{GAMMA_API}/events",
            params={"limit": 100, "closed": False, "title_contains": query},
            timeout=15
        )
        if response.ok:
            return response.json()
    except Exception as e:
        print(f"[WARN] Search failed for '{query}': {e}")

    return []


def is_fdv_bucket_market(event: Dict) -> bool:
    """Check if event is an FDV bucket market."""
    title = event.get("title", "").lower()
    slug = event.get("slug", "").lower()
    combined = f"{title} {slug}"

    # Must have FDV-related keywords
    has_fdv = any(kw in combined for kw in FDV_KEYWORDS)
    if not has_fdv:
        return False

    # Must have bucket structure (multiple outcomes)
    markets = event.get("markets", [])
    if len(markets) < 3:
        return False

    # Check for bucket patterns
    bucket_count = 0
    for m in markets:
        outcome = m.get("groupItemTitle", "") or m.get("outcome", "")
        if parse_fdv_bucket(outcome):
            bucket_count += 1

    return bucket_count >= 2


def parse_fdv_bucket(bucket_name: str) -> Optional[Tuple[float, float]]:
    """Parse FDV range from bucket name. Returns (low, high) in USD."""
    name = bucket_name.strip().lower()
    name = name.replace("$", "").replace(",", "")

    # Handle "above X" or "> X"
    above_match = re.search(r"(?:above|over|>|greater\s+than)\s*(\d+(?:\.\d+)?)\s*b", name)
    if above_match:
        val = float(above_match.group(1)) * 1e9
        return (val, val * 2)

    # Handle "below X" or "< X"
    below_match = re.search(r"(?:below|under|<|less\s+than)\s*(\d+(?:\.\d+)?)\s*b", name)
    if below_match:
        val = float(below_match.group(1)) * 1e9
        return (0, val)

    # Handle "X-Y" range
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*b?\s*[-\u2013\u2014to]\s*(\d+(?:\.\d+)?)\s*b", name)
    if range_match:
        return (float(range_match.group(1)) * 1e9, float(range_match.group(2)) * 1e9)

    # Handle "between X and Y"
    between_match = re.search(r"between\s*(\d+(?:\.\d+)?)\s*(?:b|bn)?\s*(?:and|&)\s*(\d+(?:\.\d+)?)", name)
    if between_match:
        return (float(between_match.group(1)) * 1e9, float(between_match.group(2)) * 1e9)

    return None


def extract_project_name(event: Dict) -> str:
    """Extract project name from event."""
    title = event.get("title", "").lower()
    slug = event.get("slug", "").lower()
    combined = f"{title} {slug}"

    for project in TRACKED_PROJECTS:
        if project in combined:
            return project.title()

    # Fallback: first word
    words = title.split()
    if words:
        return words[0].title()

    return "Unknown"


def load_knowledge_base() -> Dict:
    """Load our token knowledge base."""
    kb = {
        "l2_launches": {},
        "token_launches": {},
    }

    # Load L2 launches
    l2_file = KB_DIR / "l2_launches_extended.json"
    if l2_file.exists():
        with open(l2_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for token in data.get("tokens", []):
                name = token.get("name", "").lower()
                kb["l2_launches"][name] = token

    # Load token launches
    token_file = KB_DIR / "token_launches.json"
    if token_file.exists():
        with open(token_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for token in data.get("tokens", []):
                name = token.get("name", "").lower()
                kb["token_launches"][name] = token

    return kb


def generate_prediction(project: str, event: Dict, kb: Dict) -> Dict:
    """Generate FDV prediction for a project using our model."""
    project_lower = project.lower()

    # Look up in knowledge base
    token_data = kb["l2_launches"].get(project_lower) or kb["token_launches"].get(project_lower)

    if token_data:
        # Use data from knowledge base
        total_raised = token_data.get("total_raised_usd", 0)
        last_val = token_data.get("valuation_last_round", 0)
        category = token_data.get("category", "Unknown")

        # Calculate prediction based on comparable analysis
        if total_raised > 400e6:
            # zkSync pattern: high raises = 3-5x
            mult = 3.5
        elif total_raised > 100e6:
            mult = 50
        else:
            mult = 100

        if last_val:
            base_fdv = last_val * 2  # Double last valuation
        else:
            base_fdv = total_raised * mult

        return {
            "project": project,
            "predicted_fdv": base_fdv,
            "low": base_fdv * 0.7,
            "high": base_fdv * 1.5,
            "method": "knowledge_base",
            "data": {
                "total_raised": total_raised,
                "last_valuation": last_val,
                "category": category,
            }
        }

    # Fallback: sector-based estimate
    title = event.get("title", "").lower()

    if "l2" in title or "layer" in title:
        base = 2e9
    elif "defi" in title or "protocol" in title:
        base = 1e9
    elif "ai" in title:
        base = 3e9
    elif "meme" in title:
        base = 0.2e9
    else:
        base = 1.5e9

    return {
        "project": project,
        "predicted_fdv": base,
        "low": base * 0.5,
        "high": base * 2,
        "method": "sector_estimate",
        "data": None,
    }


def analyze_market_opportunities(event: Dict, prediction: Dict) -> List[Dict]:
    """Find betting opportunities in a market."""
    opportunities = []
    markets = event.get("markets", [])

    pred_low = prediction["low"]
    pred_high = prediction["high"]

    for m in markets:
        outcome = m.get("groupItemTitle", "") or m.get("outcome", "")
        bucket_range = parse_fdv_bucket(outcome)

        if not bucket_range:
            continue

        bucket_low, bucket_high = bucket_range

        # Get current price
        price = 0
        outcome_prices = m.get("outcomePrices", "")
        if outcome_prices:
            try:
                prices_list = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                price = float(prices_list[0]) if prices_list else 0
            except:
                pass

        if price <= 0:
            price = float(m.get("lastTradePrice", 0) or 0)

        # Check if bucket overlaps with prediction
        overlaps = pred_low <= bucket_high and pred_high >= bucket_low

        # Calculate expected value
        # If our prediction range overlaps, we think probability is higher
        if overlaps:
            model_prob = 0.35  # Base probability for overlapping buckets
            # Closer to center = higher probability
            bucket_mid = (bucket_low + bucket_high) / 2
            pred_center = prediction["predicted_fdv"]
            distance_pct = abs(bucket_mid - pred_center) / pred_center
            model_prob = max(0.1, 0.5 - distance_pct * 0.3)
        else:
            model_prob = 0.05

        # Expected value calculation
        ev = model_prob * (1 - price) - (1 - model_prob) * price

        opportunities.append({
            "bucket": outcome[:40],
            "range": f"${bucket_low/1e9:.1f}B - ${bucket_high/1e9:.1f}B",
            "price": price,
            "model_prob": model_prob,
            "ev": ev,
            "overlaps": overlaps,
            "signal": "BUY" if ev > 0.05 else ("WATCH" if ev > 0 else "SKIP"),
        })

    # Sort by EV
    opportunities.sort(key=lambda x: -x["ev"])

    return opportunities


def discover_fdv_markets() -> List[Dict]:
    """Discover all active FDV bucket markets."""
    print("[*] Scanning Polymarket for FDV markets...")

    all_markets = []
    seen_slugs = set()

    # Method 1: Keyword search
    for keyword in FDV_KEYWORDS[:4]:  # Limit to avoid rate limits
        events = search_events(keyword)
        for event in events:
            slug = event.get("slug", "")
            if slug and slug not in seen_slugs and is_fdv_bucket_market(event):
                seen_slugs.add(slug)
                all_markets.append(event)

    # Method 2: Project search
    for project in TRACKED_PROJECTS[:10]:  # Limit
        events = search_events(project)
        for event in events:
            slug = event.get("slug", "")
            if slug and slug not in seen_slugs and is_fdv_bucket_market(event):
                seen_slugs.add(slug)
                all_markets.append(event)

    # Method 3: Full scan
    events = fetch_all_events(active_only=True)
    for event in events:
        slug = event.get("slug", "")
        if slug and slug not in seen_slugs and is_fdv_bucket_market(event):
            seen_slugs.add(slug)
            all_markets.append(event)

    print(f"[+] Found {len(all_markets)} FDV bucket markets")
    return all_markets


def run_monitor():
    """Run full market monitoring and analysis."""
    print("=" * 80)
    print("FDV MARKET MONITOR")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # Load knowledge base
    kb = load_knowledge_base()
    print(f"[+] Loaded {len(kb['l2_launches'])} L2 tokens")
    print(f"[+] Loaded {len(kb['token_launches'])} other tokens")

    # Discover markets
    markets = discover_fdv_markets()

    if not markets:
        print("\n[!] No active FDV markets found")
        return

    # Analyze each market
    all_results = []

    for event in markets:
        slug = event.get("slug", "")
        title = event.get("title", "")
        project = extract_project_name(event)

        print(f"\n{'=' * 80}")
        print(f"MARKET: {title[:60]}")
        print(f"Project: {project}")
        print(f"Slug: {slug}")
        print("=" * 80)

        # Generate prediction
        prediction = generate_prediction(project, event, kb)

        print(f"\nPrediction ({prediction['method']}):")
        print(f"  Center: ${prediction['predicted_fdv']/1e9:.2f}B")
        print(f"  Range:  ${prediction['low']/1e9:.2f}B - ${prediction['high']/1e9:.2f}B")

        if prediction.get("data"):
            data = prediction["data"]
            if data.get("total_raised"):
                print(f"  Raised: ${data['total_raised']/1e6:.0f}M")
            if data.get("category"):
                print(f"  Category: {data['category']}")

        # Find opportunities
        opportunities = analyze_market_opportunities(event, prediction)

        print(f"\nBetting Opportunities:")
        print(f"{'Bucket':<42} {'Price':>8} {'Model':>8} {'EV':>8} {'Signal':>8}")
        print("-" * 80)

        for opp in opportunities[:6]:  # Top 6
            print(f"{opp['bucket']:<42} {opp['price']:>7.0%} {opp['model_prob']:>7.0%} "
                  f"{opp['ev']:>+7.2f} {opp['signal']:>8}")

        # Store results
        result = {
            "slug": slug,
            "title": title,
            "project": project,
            "prediction": prediction,
            "opportunities": opportunities,
            "top_bets": [o for o in opportunities if o["signal"] == "BUY"][:3],
        }
        all_results.append(result)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - TOP OPPORTUNITIES")
    print("=" * 80)

    all_opportunities = []
    for result in all_results:
        for opp in result["opportunities"]:
            if opp["signal"] == "BUY":
                opp["project"] = result["project"]
                opp["slug"] = result["slug"]
                all_opportunities.append(opp)

    all_opportunities.sort(key=lambda x: -x["ev"])

    print(f"\n{'Project':<15} {'Bucket':<30} {'Price':>8} {'EV':>8}")
    print("-" * 70)

    for opp in all_opportunities[:10]:
        print(f"{opp['project']:<15} {opp['bucket']:<30} {opp['price']:>7.0%} {opp['ev']:>+7.2f}")

    # Save results
    output_file = DATA_DIR / "fdv_monitor_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "scan_time": datetime.now().isoformat(),
            "markets_found": len(markets),
            "opportunities_found": len(all_opportunities),
            "results": all_results,
        }, f, indent=2)

    print(f"\n[+] Results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    run_monitor()
