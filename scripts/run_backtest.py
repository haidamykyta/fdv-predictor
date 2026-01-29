"""
Step 3: Run Walk-Forward Backtest
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fdv_backtest import FDVBacktest
from fdv_edge_simulation import run_simulation, FDVEdgeSimulator
from fdv_database import FDVDatabase
from config import BACKTEST_CHECKPOINTS


def main():
    parser = argparse.ArgumentParser(description='Run FDV backtest')
    parser.add_argument(
        '--checkpoints',
        type=str,
        default=','.join(map(str, BACKTEST_CHECKPOINTS)),
        help='Comma-separated checkpoint percentages (default: 0,25,50,75,90,95)'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Also run edge simulation'
    )
    args = parser.parse_args()

    checkpoints = [int(x) for x in args.checkpoints.split(',')]

    print("=" * 60)
    print("FDV Walk-Forward Backtest - Step 3")
    print("=" * 60)

    # Initialize
    db = FDVDatabase()

    # Check data availability
    stats = db.get_stats()
    print(f"\nDatabase stats:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Resolved events: {stats['resolved_events']}")
    print(f"  Events with actual FDV: {stats['events_with_actual_fdv']}")

    if stats['events_with_actual_fdv'] < 3:
        print("\n[!] Not enough events with actual FDV for meaningful backtest")
        print("  Run 'python scripts/collect_actual_fdv.py' first")
        db.close()
        return

    # Run backtest
    print(f"\nRunning backtest with checkpoints: {checkpoints}")

    backtest = FDVBacktest(db)
    results, summary = backtest.run_backtest(checkpoints=checkpoints)

    # Save results
    if results:
        output_path = Path('data/fdv_backtest_results.csv')

        import pandas as pd
        df = pd.DataFrame([{
            'slug': r.event_slug,
            'checkpoint': r.checkpoint_pct,
            'market_prediction': r.market_prediction_label,
            'actual': r.actual_bucket_label,
            'exact_match': r.exact_match,
            'adjacent_match': r.adjacent_match,
            'bucket_distance': r.bucket_distance,
            'fdv_distance': r.fdv_distance,
            'confidence': r.market_confidence,
        } for r in results])

        df.to_csv(output_path, index=False)
        print(f"\n[OK] Results saved to {output_path}")

    # Run edge simulation if requested
    if args.simulate:
        print("\n" + "=" * 60)
        print("EDGE SIMULATION")
        print("=" * 60)

        simulator = FDVEdgeSimulator(db)
        event_data = simulator.load_event_data()

        if event_data:
            run_simulation(event_data)
        else:
            print("No data for simulation - need price history")

    db.close()

    print(f"\n[OK] Backtest complete!")


if __name__ == "__main__":
    main()
