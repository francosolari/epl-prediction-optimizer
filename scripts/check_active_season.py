#!/usr/bin/env python3
"""Audit the active season's data before trusting a pick.

The active season is the one input that cannot be re-derived later, and it is
also the most fragile: football-data.co.uk does not publish a season's CSV
until it is under way, so the current campaign is rebuilt from the official
fixtures API, which carries no Elo, no shot counts, and no prices. Anything
missing here degrades every prediction silently.

Exits non-zero when a check fails, so it can gate a refresh.

Usage:
    uv run scripts/check_active_season.py
    uv run scripts/check_active_season.py --season 2627
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.config.season import SeasonContext
from epl_prediction_optimizer.paths import PROCESSED_DIR
from epl_prediction_optimizer.pipeline import MARKET_BLEND_WEIGHT

DEFAULT_ELO = 1500.0
EXPECTED_MATCHES = 380
EXPECTED_TEAMS = 20
EXPECTED_ROUNDS = 38


def check(label: str, passed: bool, detail: str) -> bool:
    print(f"  [{'ok' if passed else 'FAIL'}] {label:<34}{detail}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=None)
    args = parser.parse_args()
    season = args.season or SeasonContext.current_season().code

    # The two files answer different questions and must be checked separately.
    # fixtures.csv is the full 380-match schedule. historical_matches.csv holds
    # results only: once football-data.co.uk publishes a season under way, the
    # active season's rows there are the played matches, not the whole calendar.
    fixtures = pd.read_csv(PROCESSED_DIR / "fixtures.csv")
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")

    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")
    played = matches[matches["season"].astype(str).str.zfill(4) == season].copy()
    played = played.dropna(subset=["home_goals", "away_goals"])
    played["date"] = pd.to_datetime(played["date"], errors="coerce")

    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
    results: list[bool] = []

    print(f"\nseason {season} — schedule (fixtures.csv)")
    results.append(check("fixtures", len(fixtures) == EXPECTED_MATCHES, f"{len(fixtures)}"))
    results.append(check("teams", len(teams) == EXPECTED_TEAMS, f"{len(teams)}"))
    results.append(
        check(
            "gameweeks",
            fixtures["contest_week"].nunique() == EXPECTED_ROUNDS,
            f"{fixtures['contest_week'].nunique()}",
        )
    )
    duplicates = fixtures.duplicated(subset=["home_team", "away_team"]).sum()
    results.append(check("no duplicate pairings", duplicates == 0, f"{duplicates} duplicated"))
    home_counts = fixtures["home_team"].value_counts()
    away_counts = fixtures["away_team"].value_counts()
    balanced = bool((home_counts == 19).all() and (away_counts == 19).all())
    results.append(
        check(
            "19 home + 19 away per team",
            balanced,
            f"{home_counts.min()}-{home_counts.max()} home",
        )
    )
    fixture_default_elo = int(
        ((fixtures["home_elo"] == DEFAULT_ELO) | (fixtures["away_elo"] == DEFAULT_ELO)).sum()
    )
    results.append(
        check(
            "Elo attached",
            fixture_default_elo == 0,
            f"{fixture_default_elo} rows on the 1500 default",
        )
    )

    print(f"\nseason {season} — results (historical_matches.csv)")
    now = pd.Timestamp(datetime.now(UTC)).tz_localize(None)
    kicked_off = fixtures[fixtures["date"] < now.normalize()]
    missing = len(kicked_off) - len(played)
    results.append(check("played matches recorded", len(played) > 0, f"{len(played)} with a score"))
    results.append(
        check("no missing past results", missing <= 0, f"{max(missing, 0)} not yet recorded")
    )
    for label, column in (("odds", "home_odds"), ("shots on target", "home_shots_ot")):
        if column in played:
            filled = int(played[column].notna().sum())
            print(f"  [--] {f'played rows with {label}':<34}{filled}/{len(played)}")

    print(f"\nseason {season} — market prices")
    ledger_path = PROCESSED_DIR / "market_odds.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    if not ledger.empty:
        ledger["date"] = pd.to_datetime(ledger["date"], errors="coerce")
        ledger = ledger[ledger["date"] >= fixtures["date"].min()]
    # Prices exist only for rounds the market has opened, so coverage of the
    # whole season is never expected. What matters is the round the next pick
    # comes from — picks are Saturday or Sunday fixtures that have not started.
    upcoming = fixtures[fixtures["date"] >= now.normalize()]
    weekend = upcoming[upcoming["date"].dt.dayofweek.isin([5, 6])]
    next_round = (
        weekend[weekend["contest_week"] == weekend["contest_week"].min()]
        if not weekend.empty
        else pd.DataFrame()
    )
    print(f"  [--] {'prices held in ledger':<34}{len(ledger)} for this season")
    if next_round.empty:
        print(f"  [--] {'next pick round':<34}no weekend rounds left")
    else:
        round_number = int(next_round["contest_week"].iloc[0])
        priced_next = 0
        if not ledger.empty:
            keys = set(zip(ledger["home_team"], ledger["away_team"], strict=True))
            priced_next = sum(
                1
                for row in next_round.itertuples(index=False)
                if (row.home_team, row.away_team) in keys
            )
        print(
            f"  [--] {f'round {round_number} priced':<34}"
            f"{priced_next}/{len(next_round)} (blend weight {MARKET_BLEND_WEIGHT})"
        )
        if priced_next < len(next_round):
            # Not a failure: bookmakers open a round a few days out. It is a
            # reason to refresh again before committing the pick.
            print(
                f"\n  NOTE: round {round_number} is not fully priced yet. The pick for that "
                f"round\n        is running on model probabilities alone. Run `uv run eplpo "
                f"refresh`\n        again closer to kickoff to pick up market prices."
            )

    if all(results):
        print("\nOK\n")
        return
    raise SystemExit("\nFAIL: active-season data is incomplete\n")


if __name__ == "__main__":
    main()
