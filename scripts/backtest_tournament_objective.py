#!/usr/bin/env python3
"""Compare planning for expected points against planning to win.

The contest pays for finishing first, not for a good average season. This
scores both objectives on completed seasons: the expected-points plan the
product ships today, and the plan chosen to maximise the exact probability of
reaching that season's winning score.

Both objectives read the same probabilities, so any difference is the
objective, not the model.

``--endgame`` runs the same comparison over shortened horizons instead: with
only a handful of rounds left and a gap to close, risk-seeking has its best
possible case. It is reported because "take more variance when you are behind"
is the obvious intuition and it needs to be checked, not assumed.

Usage:
    uv run scripts/backtest_tournament_objective.py
    uv run scripts/backtest_tournament_objective.py --seasons 2324 2425
    uv run scripts/backtest_tournament_objective.py --endgame --seasons 2425
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.optimizer.tournament import optimize_for_target, plan_summary
from epl_prediction_optimizer.paths import EXPORT_DIR, PROCESSED_DIR
from epl_prediction_optimizer.pipeline import WINNER_BENCHMARKS
from epl_prediction_optimizer.scoring import actual_outcomes, score_picks

DEFAULT_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]
# Seasons without a confirmed contest winner still need a target to plan
# against; the confirmed benchmarks average close to this.
FALLBACK_TARGET = 83


def load_season(season: str, matches: pd.DataFrame) -> pd.DataFrame:
    """Read stored backtest probabilities and attach the real results."""
    path = EXPORT_DIR / f"{season}_backtest_probabilities.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run `uv run eplpo backtest --season {season}` first")
    predictions = pd.read_csv(path)
    predictions["match_id"] = predictions["match_id"].astype(str)

    actual = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    actual["match_id"] = actual["match_id"].astype(str)
    evaluated = predictions.merge(
        actual[["match_id", "home_goals", "away_goals"]], on="match_id", how="inner"
    )
    evaluated["actual"] = actual_outcomes(evaluated)
    return evaluated


HORIZONS = (4, 6, 8, 12)
GAPS = (2, 4, 6, 8)


def report_endgame(season: str, matches: pd.DataFrame) -> None:
    """Compare objectives over the final N rounds at several points behind."""
    evaluated = load_season(season, matches)
    candidates = build_pick_candidates(evaluated)
    weeks = sorted(candidates["contest_week"].unique())

    header = f"{'rounds':>7}{'gap':>6}{'EV-max P':>11}{'target P':>11}{'risk':>7}{'ratio':>8}"
    print(f"\nendgame horizons, season {season}\n")
    print(header)
    print("-" * len(header))
    for rounds in HORIZONS:
        tail = candidates[candidates["contest_week"].isin(weeks[-rounds:])]
        baseline = optimize_picks(tail)
        baseline_mean = plan_summary(baseline, 0)["expected_points"]
        for gap in GAPS:
            target = baseline_mean + gap
            planned = optimize_for_target(tail, target)
            p_baseline = plan_summary(baseline, target)["win_probability"]
            p_planned = planned.attrs["win_probability"]
            ratio = f"{p_planned / p_baseline:.3f}" if p_baseline > 0 else "-"
            print(
                f"{rounds:>7}{gap:>6}{p_baseline:>11.5f}{p_planned:>11.5f}"
                f"{planned.attrs['risk_weight']:>7g}{ratio:>8}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument(
        "--endgame",
        action="store_true",
        help="Compare objectives over the final rounds instead of full seasons",
    )
    args = parser.parse_args()

    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")

    if args.endgame:
        for season in args.seasons:
            report_endgame(season, matches)
        return

    header = (
        f"{'Season':<8}{'Target':>7}{'Objective':<14}"
        f"{'E[pts]':>8}{'SD':>7}{'P(win)':>9}{'Actual':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    totals = {"expected": 0, "tournament": 0}
    win_probabilities = {"expected": 0.0, "tournament": 0.0}

    for season in args.seasons:
        evaluated = load_season(season, matches)
        candidates = build_pick_candidates(evaluated)
        target = WINNER_BENCHMARKS.get(season, FALLBACK_TARGET)

        ev_plan = optimize_picks(candidates)
        tournament_plan = optimize_for_target(candidates, target)

        for label, plan in (("expected", ev_plan), ("tournament", tournament_plan)):
            summary = plan_summary(plan, target)
            actual_points = int(score_picks(plan, evaluated)["actual_points"].sum())
            totals[label] += actual_points
            win_probabilities[label] += summary["win_probability"]
            risk = plan.attrs.get("risk_weight")
            name = label if risk is None else f"{label} (r={risk:g})"
            print(
                f"{season if label == 'expected' else '':<8}"
                f"{target if label == 'expected' else '':>7}"
                f"{name:<14}{summary['expected_points']:>8.1f}{summary['std_dev']:>7.2f}"
                f"{summary['win_probability']:>9.4f}{actual_points:>8}"
            )
        print("-" * len(header))

    seasons = len(args.seasons)
    print(f"\n{'':<15}{'mean P(win)':>13}{'total actual':>14}")
    for label in ("expected", "tournament"):
        print(
            f"{label:<15}{win_probabilities[label] / seasons:>13.4f}{totals[label]:>14}"
        )


if __name__ == "__main__":
    main()
