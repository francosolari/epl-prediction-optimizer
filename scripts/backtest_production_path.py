#!/usr/bin/env python3
"""Backtest the path production actually runs, not the idealised one.

The existing season backtest predicts straight from the training frame, so it
never exercises build_fixture_features — the function the live predict step
depends on. This script replays each target season week by week: train on
earlier seasons only, then at every contest week build fixture features from
the history available at that moment and predict, exactly as the pipeline does.

It reports both fixture-feature modes so the cost of the neutral-default
fallback is measurable:

    history   fixture features accumulated from completed matches (production)
    neutral   form/streak/h2h features left at their defaults (the old path)

Usage:
    uv run scripts/backtest_production_path.py
    uv run scripts/backtest_production_path.py --seasons 2223 2324 2425 2526
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.ml.features import build_training_frame
from epl_prediction_optimizer.ml.model import season_decay_weights, train_model
from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.paths import PROCESSED_DIR
from epl_prediction_optimizer.scoring import (
    actual_outcomes,
    probability_metrics,
    score_picks,
)

DEFAULT_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
FIXTURE_COLUMNS = [
    "match_id",
    "contest_week",
    "date",
    "home_team",
    "away_team",
    "home_elo",
    "away_elo",
]


def season_start_year(season_code: str) -> int:
    code = str(season_code).zfill(4)
    yy = int(code[:2])
    return 1900 + yy if yy >= 50 else 2000 + yy


def replay_season(
    matches: pd.DataFrame,
    target_season: str,
    use_history: bool,
) -> pd.DataFrame:
    """Predict every target-season match using only data available beforehand."""
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    completed["season_code"] = completed["season"].astype(str).str.zfill(4)
    completed["date"] = pd.to_datetime(completed["date"]).dt.date

    target_year = season_start_year(target_season)
    season_year = completed["season_code"].map(season_start_year)
    prior = completed[season_year < target_year]
    target = completed[completed["season_code"] == target_season].sort_values("date")
    if prior.empty or target.empty:
        raise ValueError(f"No data to replay season {target_season}")

    training = build_training_frame(prior)
    weights = season_decay_weights(training["season"], reference_season=target_season)
    model = train_model(training, sample_weight=weights)

    predictions: list[pd.DataFrame] = []
    for week in sorted(target["contest_week"].unique()):
        week_matches = target[target["contest_week"] == week]
        played_earlier = target[target["contest_week"] < week]
        history = pd.concat([prior, played_earlier], ignore_index=True) if use_history else None
        fixtures = week_matches[
            [column for column in FIXTURE_COLUMNS if column in week_matches.columns]
        ].copy()
        predictions.append(
            model.predict(
                fixtures,
                history=history,
                fixture_season=target_season,
            )
        )

    predicted = pd.concat(predictions, ignore_index=True)
    predicted["match_id"] = predicted["match_id"].astype(str)
    target = target.copy()
    target["match_id"] = target["match_id"].astype(str)
    evaluated = predicted.merge(
        target[["match_id", "home_goals", "away_goals"]],
        on="match_id",
        how="inner",
    )
    evaluated["actual"] = actual_outcomes(evaluated)
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")

    print(f"\n{'Season':<8}{'Fixture features':<20}{'LogLoss':>9}{'RPS':>8}{'Acc':>7}{'Points':>8}")
    print("-" * 60)
    totals: dict[str, list[float]] = {"history": [], "neutral": []}
    points: dict[str, int] = {"history": 0, "neutral": 0}

    for season in args.seasons:
        for label, use_history in (("history", True), ("neutral", False)):
            evaluated = replay_season(matches, season, use_history=use_history)
            metrics = probability_metrics(evaluated)
            picks = optimize_picks(build_pick_candidates(evaluated))
            scored = score_picks(picks, evaluated)
            season_points = int(scored["actual_points"].sum())
            totals[label].append(metrics["log_loss"])
            points[label] += season_points
            print(
                f"{season if label == 'history' else '':<8}{label:<20}"
                f"{metrics['log_loss']:>9.4f}{metrics['rps']:>8.4f}"
                f"{metrics['accuracy']:>7.3f}{season_points:>8}"
            )
        print("-" * 60)

    print(f"\n{'mean':<8}{'':<20}{'LogLoss':>9}{'':>8}{'':>7}{'Points':>8}")
    for label in ("history", "neutral"):
        mean_loss = sum(totals[label]) / len(totals[label])
        print(f"{'':<8}{label:<20}{mean_loss:>9.4f}{'':>8}{'':>7}{points[label]:>8}")


if __name__ == "__main__":
    main()
