#!/usr/bin/env python3
"""Fetch official PL matchday (gameweek) numbers from openfootball GitHub.

Only fetches seasons not yet cached. Run this before rebuild_history.py for
accurate contest_week assignments on pre-2324 seasons (not covered by
football-data.org free tier).

Usage:
    uv run scripts/fetch_openfootball_gameweeks.py
    uv run scripts/fetch_openfootball_gameweeks.py --seasons 2122 2223
    uv run scripts/fetch_openfootball_gameweeks.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epl_prediction_optimizer.data.sources import (
    OPENFOOTBALL_SEASON_CODES,
    fetch_openfootball_gameweeks,
)
from epl_prediction_optimizer.paths import RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PL matchday data from openfootball")
    parser.add_argument(
        "--seasons", nargs="+",
        help="Season codes to fetch (e.g. 2122 2223). Default: all missing.",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    args = parser.parse_args()

    cache_dir = RAW_DIR / "openfootball"
    cache_dir.mkdir(parents=True, exist_ok=True)

    seasons = args.seasons or OPENFOOTBALL_SEASON_CODES
    targets = [
        s for s in seasons
        if args.force or not (cache_dir / f"{s}_PL_matchdays.csv").exists()
    ]

    if not targets:
        print("All seasons already cached. Use --force to re-fetch.")
        return

    print(f"Fetching {len(targets)} season(s) from openfootball GitHub...")
    for i, season in enumerate(targets):
        df = fetch_openfootball_gameweeks(season, cache_dir, force=args.force)
        if df.empty:
            print(f"  {season}: FAILED or no data")
        # Small delay between requests to be polite
        if i < len(targets) - 1:
            time.sleep(1)

    print("\nDone. Run 'uv run scripts/rebuild_history.py' to apply gameweeks.")


if __name__ == "__main__":
    main()
