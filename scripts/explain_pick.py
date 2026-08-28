#!/usr/bin/env python3
"""Explain the current round's choices in the terms that decide the contest.

The decision desk ranks candidates by full-season expected value. That is the
right ranking for scoring points, but the contest pays only for finishing
first, and a small expected-value gap can hide a meaningful difference in the
chance of winning. This adds the two things the table cannot show:

  win chance   probability the resulting season plan finishes top, scored
               against the finishing totals of a real completed field
  scarcity     where else that team-and-venue slot could be spent, so a slot
               being used at its best moment is visible rather than implied

Usage:
    uv run scripts/explain_pick.py
    uv run scripts/explain_pick.py --week 2 --top 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.config.season import SeasonContext
from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.optimizer.tournament import win_probability_against_field
from epl_prediction_optimizer.paths import EXPORT_DIR
from epl_prediction_optimizer.storage.database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--top", type=int, default=4)
    parser.add_argument("--field-season", default=None, help="Completed season to score against")
    args = parser.parse_args()

    season = args.season or SeasonContext.current_season().code
    database = Database()
    probabilities = pd.read_csv(EXPORT_DIR / "fixture_probabilities.csv")
    candidates = build_pick_candidates(probabilities)

    committed = database.list_actual_picks(season)
    banked = sum(int(p["actual_points"]) for p in committed if p.get("actual_points") is not None)
    locked = [
        {
            "contest_week": int(p["contest_week"]),
            "match_id": str(p["match_id"]),
            "team": p["team"],
            "venue": p["venue"],
        }
        for p in committed
    ]
    decided = {int(p["contest_week"]) for p in committed}
    week = args.week or min(
        (int(w) for w in candidates["contest_week"].unique() if int(w) not in decided),
        default=None,
    )
    if week is None:
        raise SystemExit("Every round already has a committed pick")

    field = _field_totals(database, args.field_season, season)
    round_options = candidates[candidates["contest_week"] == week].sort_values(
        "expected_points", ascending=False
    )

    rows = []
    for option in round_options.itertuples(index=False):
        lock = {
            "contest_week": week,
            "match_id": str(option.match_id),
            "team": option.team,
            "venue": option.venue,
        }
        try:
            plan = optimize_picks(candidates, [*locked, lock])
        except ValueError:
            continue
        remaining = plan[~plan["contest_week"].isin(decided)]
        chance = (
            win_probability_against_field(remaining, field, points_so_far=banked)["win_probability"]
            if field
            else float("nan")
        )
        rows.append(
            {
                "team": option.team,
                "opponent": option.opponent,
                "venue": option.venue,
                "p_win": float(option.p_win),
                "now": float(option.expected_points),
                "season_ev": float(plan["expected_points"].sum()),
                "win_chance": chance,
            }
        )

    frame = pd.DataFrame(rows).sort_values("season_ev", ascending=False).head(args.top)
    best_ev = frame["season_ev"].max()

    print(f"\nSeason {season} · round {week} · {banked} points banked")
    if field:
        print(f"Win chance scored against {len(field)} finishing totals from a completed season")
    header = (
        f"{'Choice':<26}{'Win':>6}{'Now':>7}{'Season EV':>11}{'Cost':>7}{'Win chance':>12}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for row in frame.itertuples(index=False):
        print(
            f"{row.team[:24]:<26}{row.p_win:>5.0%}{row.now:>7.2f}"
            f"{row.season_ev:>11.2f}{best_ev - row.season_ev:>7.2f}"
            f"{row.win_chance:>11.2%}"
        )

    print("\nWhere each slot could otherwise be spent")
    print("-" * 60)
    for row in frame.itertuples(index=False):
        print(f"  {row.team} ({row.venue}) — {_scarcity(candidates, row.team, row.venue, week)}")


def _scarcity(candidates: pd.DataFrame, team: str, venue: str, week: int) -> str:
    """Rank this round's opportunity among that team-and-venue's whole season."""
    slots = candidates[(candidates["team"] == team) & (candidates["venue"] == venue)]
    slots = slots.sort_values("expected_points", ascending=False).reset_index(drop=True)
    if slots.empty:
        return "no other fixtures in this slot"
    position = slots.index[slots["contest_week"] == week]
    rank = int(position[0]) + 1 if len(position) else None
    best_other = slots[slots["contest_week"] != week].head(1)
    if best_other.empty:
        return f"only {venue} fixture left this season"
    other = best_other.iloc[0]
    return (
        f"best {venue} fixture is round {int(other['contest_week'])} vs {other['opponent']} "
        f"({other['expected_points']:.2f} xpts); this round ranks "
        f"{rank} of {len(slots)}"
    )


def _field_totals(database: Database, requested: str | None, season: str) -> list[int]:
    for code in ([requested] if requested else ["2526", "2425", "2324"]):
        if code and code != season:
            totals = [
                entrant["points"]
                for entrant in database.list_league_entrants(code)
                if entrant["points"] is not None
            ]
            if totals:
                return totals
    return []


if __name__ == "__main__":
    main()
