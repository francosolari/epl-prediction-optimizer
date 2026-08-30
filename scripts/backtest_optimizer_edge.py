#!/usr/bin/env python3
"""Measure what the season optimizer adds over picking greedily each week.

The season optimizer exists to spend teams well: every club must be used once,
each at most twice and once per venue, so taking the best team available this
week can strand you with bad options later. That is a claim worth testing
rather than assuming.

Three strategies, same probabilities, same rules:

    greedy      each round, the highest expected-points eligible pick that
                still leaves a feasible remainder
    optimizer   the season-long integer program the product ships
    uncapped    the season program without the "use every team" rule, to price
                what that rule costs

Usage:
    uv run scripts/backtest_optimizer_edge.py
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
from epl_prediction_optimizer.scoring import score_picks

DEFAULT_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]


def greedy_plan(candidates: pd.DataFrame) -> pd.DataFrame:
    """Take the best pick each round that leaves the rest of the season solvable.

    This is the strategy a careful person plays without an optimizer: look at
    this weekend, take the best team you have not burned, and only reject it if
    it obviously breaks the season.
    """
    weeks = sorted(candidates["contest_week"].unique())
    chosen: list[dict[str, object]] = []
    for week in weeks:
        options = candidates[candidates["contest_week"] == week].sort_values(
            "expected_points", ascending=False
        )
        for option in options.itertuples(index=False):
            trial = [
                *chosen,
                {
                    "contest_week": int(option.contest_week),
                    "match_id": str(option.match_id),
                    "team": option.team,
                    "venue": option.venue,
                },
            ]
            try:
                optimize_picks(candidates, trial)
            except ValueError:
                # This pick breaks a rule outright, or strands a team that can
                # no longer be placed; try the next-best option for the round.
                continue
            chosen = trial
            break
        else:
            # Nothing keeps the season solvable; take the best available.
            best = options.iloc[0]
            chosen.append(
                {
                    "contest_week": int(best["contest_week"]),
                    "match_id": str(best["match_id"]),
                    "team": best["team"],
                    "venue": best["venue"],
                }
            )
    keys = {(pick["contest_week"], str(pick["match_id"]), pick["team"]) for pick in chosen}
    mask = candidates.apply(
        lambda row: (int(row["contest_week"]), str(row["match_id"]), row["team"]) in keys,
        axis=1,
    )
    return candidates[mask].sort_values("contest_week").reset_index(drop=True)


def uncapped_plan(candidates: pd.DataFrame) -> pd.DataFrame:
    """Season program with the use-every-team rule removed."""
    import pulp

    frame = candidates.reset_index(drop=True).copy()
    problem = pulp.LpProblem("uncapped", pulp.LpMaximize)
    variables = {
        index: pulp.LpVariable(f"pick_{index}", cat="Binary") for index in frame.index
    }
    problem += pulp.lpSum(
        frame.loc[index, "expected_points"] * variables[index] for index in frame.index
    )
    for week in sorted(frame["contest_week"].unique()):
        indexes = frame.index[frame["contest_week"] == week]
        problem += pulp.lpSum(variables[index] for index in indexes) == 1
    for team in sorted(frame["team"].unique()):
        team_indexes = frame.index[frame["team"] == team]
        problem += pulp.lpSum(variables[index] for index in team_indexes) <= 2
        for venue in ["home", "away"]:
            venue_indexes = frame.index[(frame["team"] == team) & (frame["venue"] == venue)]
            problem += pulp.lpSum(variables[index] for index in venue_indexes) <= 1
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [index for index, variable in variables.items() if variable.value() == 1]
    return frame.loc[selected].sort_values("contest_week").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")

    header = f"{'Season':<8}{'Strategy':<12}{'E[pts]':>9}{'Actual':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    totals: dict[str, list[int]] = {"greedy": [], "optimizer": [], "uncapped": []}
    expected: dict[str, list[float]] = {"greedy": [], "optimizer": [], "uncapped": []}

    for season in args.seasons:
        evaluated = load_season(season, matches)
        candidates = build_pick_candidates(evaluated)
        plans = {
            "greedy": greedy_plan(candidates),
            "optimizer": optimize_picks(candidates),
            "uncapped": uncapped_plan(candidates),
        }
        for label, plan in plans.items():
            points = int(score_picks(plan, evaluated)["actual_points"].sum())
            season_expected = float(plan["expected_points"].sum())
            totals[label].append(points)
            expected[label].append(season_expected)
            print(
                f"{season if label == 'greedy' else '':<8}{label:<12}"
                f"{season_expected:>9.1f}{points:>8}"
            )
        print("-" * len(header))

    print(f"\n{'':<20}{'mean E[pts]':>12}{'total actual':>14}")
    for label in ("greedy", "optimizer", "uncapped"):
        mean_expected = sum(expected[label]) / len(expected[label])
        print(f"{label:<20}{mean_expected:>12.1f}{sum(totals[label]):>14}")


if __name__ == "__main__":
    main()
