#!/usr/bin/env python3
"""
Compare Stage A+B backtest results across seasons.

Usage:
    uv run scripts/compare_backtests.py                        # compare stored results
    uv run scripts/compare_backtests.py --run 2324 2425        # re-run before comparing
    uv run scripts/compare_backtests.py --seasons 2324 2425    # compare specific seasons
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.ml.analysis import (
    backtest_summary_report,
    pick_accuracy_report,
    print_backtest_summary,
    print_pick_accuracy,
)
from epl_prediction_optimizer.paths import ARTIFACT_DIR, EXPORT_DIR


DEFAULT_SEASONS = ["2021", "2122", "2223", "2324", "2425"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest results across seasons")
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS, help="Seasons to compare"
    )
    parser.add_argument(
        "--run", nargs="+", metavar="SEASON", default=[],
        help="Re-run backtest for these seasons before comparing",
    )
    parser.add_argument("--picks", metavar="SEASON", help="Show pick breakdown for a season")
    args = parser.parse_args()

    if args.run:
        from epl_prediction_optimizer.pipeline import backtest_from_processed
        for season in args.run:
            print(f"Running backtest for {season}...")
            result = backtest_from_processed(season)
            print(f"  {season}: {result['optimized_points']} pts (winner: {result.get('winner_points', '—')})")

    print("\n=== Stage A+B Backtest Comparison ===")
    summary = backtest_summary_report(ARTIFACT_DIR, args.seasons)
    if summary.empty:
        print("No backtest results found. Run: uv run scripts/compare_backtests.py --run 2324 2425")
        return
    print_backtest_summary(summary)

    if args.picks:
        picks_path = EXPORT_DIR / f"{args.picks}_backtest_optimized_picks.csv"
        if picks_path.exists():
            picks = pd.read_csv(picks_path)
            print(f"\n=== Stage B Pick Breakdown: {args.picks} ===")
            print_pick_accuracy(pick_accuracy_report(picks))
        else:
            print(f"\nNo picks found for {args.picks}.")


if __name__ == "__main__":
    main()
