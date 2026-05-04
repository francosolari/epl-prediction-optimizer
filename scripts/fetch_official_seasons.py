#!/usr/bin/env python3
"""Fetch football-data.org official matchday data for seasons not yet cached.

Only hits the API for seasons missing from data/raw/football-data-org/.
Run this before rebuild_history.py to get accurate contest_week assignments.

Usage:
    uv run scripts/fetch_official_seasons.py
    uv run scripts/fetch_official_seasons.py --seasons 2122 2223
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epl_prediction_optimizer.data.sources import get_cached_pl_matches
from epl_prediction_optimizer.paths import RAW_DIR

# Seasons the football-data.org free tier supports (start year → season code)
AVAILABLE_SEASONS = {
    2015: "1516",
    2016: "1617",
    2017: "1718",
    2018: "1819",
    2019: "1920",
    2020: "2021",
    2021: "2122",
    2022: "2223",
    2023: "2324",
    2024: "2425",
    2025: "2526",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons", nargs="+",
        help="Season codes to fetch (e.g. 2122 2223). Default: all missing.",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    args = parser.parse_args()

    cache_dir = RAW_DIR / "football-data-org"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.seasons:
        # Build start_year → code map for requested seasons
        code_to_year = {v: k for k, v in AVAILABLE_SEASONS.items()}
        targets = {code_to_year[s]: s for s in args.seasons if s in code_to_year}
    else:
        # Fetch all missing seasons
        targets = {
            year: code
            for year, code in AVAILABLE_SEASONS.items()
            if not (cache_dir / f"{code}_PL_matches.csv").exists() or args.force
        }

    if not targets:
        print("All seasons already cached. Use --force to re-fetch.")
        return

    print(f"Fetching {len(targets)} season(s) from football-data.org...")
    for i, (start_year, code) in enumerate(sorted(targets.items())):
        cache_path = cache_dir / f"{code}_PL_matches.csv"
        print(f"  {code} (start {start_year})...", end=" ", flush=True)
        try:
            df = get_cached_pl_matches(cache_path, season_start_year=start_year, force=args.force)
            print(f"{len(df)} matches")
        except Exception as e:
            print(f"FAILED: {e}")
        # Respect 10 req/min rate limit — wait between calls
        if i < len(targets) - 1:
            time.sleep(7)

    print("\nDone. Run 'uv run scripts/rebuild_history.py' to apply gameweeks.")


if __name__ == "__main__":
    main()
