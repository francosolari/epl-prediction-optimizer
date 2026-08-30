#!/usr/bin/env python3
"""Solve each season with perfect foresight to find the reachable ceiling.

Before tuning anything toward a winning score, it is worth knowing what the
contest rules allow at all. This replaces expected points with the points each
pick actually returned and re-solves under the same constraints, giving the
best season any entrant could have posted under this ruleset.

Reading it:
  ceiling  what perfect foresight scores — the hard cap
  winner   the recorded winning score for that season
  model    what the shipped expected-points plan scored

A winner above the ceiling means the benchmark and the encoded rules disagree,
and no amount of model work will close that gap.

Usage:
    uv run scripts/hindsight_ceiling.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from backtest_tournament_objective import load_season  # noqa: E402

from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.paths import PROCESSED_DIR
from epl_prediction_optimizer.pipeline import WINNER_BENCHMARKS
from epl_prediction_optimizer.scoring import score_picks

DEFAULT_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]


def realised_points(candidates: pd.DataFrame, evaluated: pd.DataFrame) -> pd.DataFrame:
    """Attach the points each candidate pick actually returned."""
    results = evaluated.drop_duplicates(subset="match_id").set_index("match_id")
    frame = candidates.copy()
    points = []
    for candidate in frame.itertuples(index=False):
        match = results.loc[candidate.match_id]
        if match.home_goals == match.away_goals:
            points.append(1)
        elif candidate.team == match.home_team and match.home_goals > match.away_goals:
            points.append(3)
        elif candidate.team == match.away_team and match.away_goals > match.home_goals:
            points.append(3)
        else:
            points.append(0)
    frame["realised_points"] = points
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")

    header = f"{'Season':<8}{'Rounds':>8}{'Ceiling':>9}{'Winner':>8}{'Model':>7}{'Headroom':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for season in args.seasons:
        evaluated = load_season(season, matches)
        candidates = build_pick_candidates(evaluated)
        scored_candidates = realised_points(candidates, evaluated)

        perfect = optimize_picks(scored_candidates, objective_column="realised_points")
        ceiling = int(perfect["realised_points"].sum())

        model_plan = optimize_picks(candidates)
        model_points = int(score_picks(model_plan, evaluated)["actual_points"].sum())

        winner = WINNER_BENCHMARKS.get(season)
        winner_text = str(winner) if winner is not None else "-"
        headroom = f"{ceiling - model_points:+d}"
        print(
            f"{season:<8}{len(perfect):>8}{ceiling:>9}{winner_text:>8}"
            f"{model_points:>7}{headroom:>10}"
        )


if __name__ == "__main__":
    main()
