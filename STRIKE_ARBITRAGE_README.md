# FDV Strike Arbitrage Strategy

Delta-neutral volatility play between FDV price strikes for Polymarket.

## Concept

Buy a "corridor" between adjacent FDV buckets. Profit from volatility or if FDV lands in target bucket.

```
Example: Token XYZ

Buckets: <$1B | $1-2B | $2-3B | $3-5B | $5B+

Trade:
- Buy "$1-2B" YES @ 25c ($50)

Payoffs:
- FDV < $1B:   LOSS -$50
- FDV $1-2B:  WIN  +$150 (bought at 25c, wins $1)
- FDV > $2B:   LOSS -$50
```

## Backtest Results

**32 configurations tested, 68.8% profitable**

### Best Strategies

| Strategy | PnL | ROI | Hit Rate | Sharpe |
|----------|-----|-----|----------|--------|
| E40%_T100%_EXPIRY | $4,911 | 61781% | 6.2% | 1.03 |
| E30%_T50%_VOLATILITY | $473 | 6008% | 0% | **1.74** |
| E50%_T30%_VOLATILITY | $358 | 4304% | 0% | **1.80** |

### Key Findings

1. **EXPIRY mode** - Higher total PnL but depends on hitting corridor
   - One big win ($4,950 on Ethena) offsets many small losses
   - Lower Sharpe (more variance)

2. **VOLATILITY mode** - More consistent returns
   - Exit when target ROI hit (30-50%)
   - Higher Sharpe ratio (1.74-1.80)
   - 100% of trades exit via volatility target
   - Don't need to hit corridor to profit

## Strategy Parameters

| Parameter | Values Tested | Best |
|-----------|--------------|------|
| Entry Timing | 20%, 30%, 40%, 50% | 30-40% |
| Target ROI | 30%, 50%, 75%, 100% | 30-50% |
| Exit Mode | VOLATILITY, EXPIRY | VOLATILITY |

## Risk Profile

```
VOLATILITY MODE (Recommended):
- Max loss per trade: ~$20-40
- Typical win: ~$100-250
- Win rate: 80-100% (volatility exits)
- Sharpe: 1.5-1.8

EXPIRY MODE (High risk/reward):
- Max loss per trade: ~$10-15
- Typical win: ~$300-5000 (if corridor hit)
- Win rate: 6-12% (only corridor hits count)
- Sharpe: 0.8-1.0
```

## Usage

```bash
cd fdv_predictor/scripts
python fdv_strike_arbitrage.py
```

Results exported to:
- `data/strike_arbitrage_results.csv` - All configurations
- `data/strike_arbitrage_trades.csv` - Individual trades

## Comparison: Strike Arbitrage vs Regular Strategy

| Metric | Regular (S1_R2_E1_X1) | Strike Arbitrage |
|--------|----------------------|------------------|
| Best PnL | $6,400 | $4,911 |
| Approach | Buy underpriced buckets | Buy corridors |
| Risk | Full loss if wrong | Capped loss |
| Edge | Need to predict correctly | Profit from volatility |
| Sharpe | Variable | 1.5-1.8 |

## When to Use

**Use Strike Arbitrage when:**
- Market is volatile (high price swings)
- You don't have strong prediction on exact bucket
- Want consistent small gains over big occasional wins

**Use Regular Strategy when:**
- You have strong prediction for specific bucket
- Willing to accept higher variance
- Target maximum PnL over consistency

## Files

- `fdv_strike_arbitrage.py` - Main backtester
- `data/strike_arbitrage_results.csv` - Results
- `data/strike_arbitrage_trades.csv` - Trade details

## Author

haidamykyta@gmail.com
