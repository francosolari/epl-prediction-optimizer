#!/usr/bin/env python3
"""Re-parse all cached raw CSVs into historical_matches.csv with full column set.

Run this whenever read_historical_results is updated to extract new columns
(shots on target, betting odds, xG) without re-downloading source files.

Usage:
    uv run scripts/rebuild_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epl_prediction_optimizer.pipeline import rebuild_processed_history


def main() -> None:
    print("Rebuilding historical_matches.csv + SQLite from locally cached raw CSVs...")
    print("(No network calls — uses local CSV cache only. Run 'eplpo refresh' to fetch new data.)")
    result = rebuild_processed_history()
    print(f"Done: {result['seasons']} seasons, {result['matches']:,} matches written to CSV + SQLite.")


if __name__ == "__main__":
    main()
