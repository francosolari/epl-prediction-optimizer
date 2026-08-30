#!/usr/bin/env python3
"""Ask whether following the field's favourite is a good way to win the contest.

Taking the same team as everyone else is safe in the sense that a bad week is
a bad week for everyone — but it also cannot gain ground, and only gaining
ground wins a 65-entrant pool. This measures what actually happened in a
completed season, using the imported contest sheet:

  consensus     the most-taken team each round, its share of the field, and
                the points it returned
  follow rate   how much of an entrant's season matched the round consensus
  outcome       whether following the crowd went with finishing high or low

Usage:
    uv run scripts/analyze_field_strategy.py --season 2526
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from epl_prediction_optimizer.storage.database import Database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2526")
    parser.add_argument("--min-rounds", type=int, default=25)
    args = parser.parse_args()

    database = Database()
    picks = pd.DataFrame(database.list_league_picks(args.season))
    entrants = pd.DataFrame(database.list_league_entrants(args.season))
    if picks.empty or entrants.empty:
        raise SystemExit(f"No imported contest data for {args.season}; import the sheet first")

    rounds = picks.groupby("entrant").size()
    serious = rounds[rounds >= args.min_rounds].index
    picks = picks[picks["entrant"].isin(serious)]
    entrants = entrants[entrants["entrant"].isin(serious)]

    consensus = _consensus_by_round(picks)
    print(f"\nseason {args.season}: {len(entrants)} entrants who played "
          f"{args.min_rounds}+ rounds\n")

    print(f"{'Rd':>3}{'Most-taken team':>26}{'Share':>8}{'Pts':>6}{'Field avg':>11}")
    print("-" * 54)
    for row in consensus.itertuples(index=False):
        print(
            f"{row.contest_week:>3}{row.team:>26}{row.share:>7.0%}"
            f"{_fmt(row.consensus_points):>6}{row.field_average:>11.2f}"
        )

    print(
        f"\nfield concentration: mean {consensus['share'].mean():.0%} of entrants "
        f"on one team, max {consensus['share'].max():.0%}"
    )

    follow = _follow_rates(picks, consensus)
    merged = entrants.merge(follow, on="entrant", how="inner").dropna(subset=["points"])
    correlation = merged["follow_rate"].corr(merged["points"])
    print(f"\ncorrelation between follow rate and final points: {correlation:+.3f}")

    ranked = merged.sort_values("points", ascending=False)
    top = ranked.head(10)
    bottom = ranked.tail(10)
    print(f"  top 10 finishers    follow rate {top['follow_rate'].mean():.0%}  "
          f"points {top['points'].mean():.1f}")
    print(f"  bottom 10 finishers follow rate {bottom['follow_rate'].mean():.0%}  "
          f"points {bottom['points'].mean():.1f}")

    winner = ranked.iloc[0]
    print(f"\nwinner: {winner['entrant']} — {int(winner['points'])} pts, "
          f"followed the consensus in {winner['follow_rate']:.0%} of rounds")

    pure = consensus["consensus_points"].dropna()
    print(f"\nalways taking the most-taken team scores {pure.sum():.0f} over "
          f"{len(pure)} rounds ({pure.mean():.2f}/round)")
    print(f"the field averaged {merged['points'].mean():.1f} and the winner "
          f"scored {int(winner['points'])}")
    print("\nNote: that consensus line ignores the once-per-team and venue rules,")
    print("so it is an upper bound on crowd-following, not a legal season plan.")


def _consensus_by_round(picks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for week, group in picks.groupby("contest_week"):
        counts = group["team"].value_counts()
        top_team = counts.index[0]
        taken = group[group["team"] == top_team]
        points = taken["points"].dropna()
        rows.append(
            {
                "contest_week": int(week),
                "team": top_team,
                "share": float(counts.iloc[0] / len(group)),
                "consensus_points": float(points.iloc[0]) if len(points) else np.nan,
                "field_average": (
                    float(group["points"].mean()) if group["points"].notna().any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("contest_week").reset_index(drop=True)


def _follow_rates(picks: pd.DataFrame, consensus: pd.DataFrame) -> pd.DataFrame:
    """Share of an entrant's rounds spent on that round's most-taken team."""
    lookup = dict(zip(consensus["contest_week"], consensus["team"], strict=True))
    picks = picks.copy()
    picks["followed"] = [
        lookup.get(int(week)) == team
        for week, team in zip(picks["contest_week"], picks["team"], strict=True)
    ]
    grouped = picks.groupby("entrant")["followed"].mean().reset_index()
    return grouped.rename(columns={"followed": "follow_rate"})


def _fmt(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.0f}"


if __name__ == "__main__":
    main()
