# FDV Predictor Bot

Prediction model for Polymarket FDV (Fully Diluted Valuation) markets.

## Current Prediction: MegaETH

| Metric | Value |
|--------|-------|
| **Prediction** | $3.01B |
| **Range** | $2.1B - $4.2B |
| **Best Bet** | >$3B YES @ 9c |

## Model Performance (Backtest)

- **100%** predictions within range
- **67%** adjacent bucket accuracy
- **+20%** average ROI

## Quick Start

```bash
# Market conditions
python knowledge_base/scripts/market_conditions.py

# Run backtest
python scripts/fdv_backtest.py

# Monitor new markets
python scripts/fdv_monitor.py
```

## Files

- `MEGAETH_PREDICTION.md` - Full analysis
- `megaeth_prediction_data.json` - Raw data export
- `knowledge_base/data/` - 28 token comparables, investor data
- `scripts/` - Backtest and monitoring tools

## Author

haidamykyta@gmail.com
