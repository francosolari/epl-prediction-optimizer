#!/usr/bin/env python3
"""Feature ablation test — measures which new feature groups help or hurt.

Runs backtests for 2324 and 2425 with each combination of new feature groups
removed, to identify which additions are net positive.

Usage:
    uv run scripts/ablation_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.ml.features import build_training_frame
from epl_prediction_optimizer.ml.model import season_decay_weights, train_model
from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.paths import PROCESSED_DIR
from epl_prediction_optimizer.pipeline import WINNER_BENCHMARKS, _score_optimized_picks, _actual_outcome
from sklearn.metrics import accuracy_score, log_loss

import numpy as np

BASE_FEATURES = [
    "elo_diff", "home_elo", "away_elo",
    "home_form_points", "away_form_points",
    "home_goal_diff_form", "away_goal_diff_form",
    "home_form_home_only", "away_form_away_only",
    "home_goal_diff_home_only", "away_goal_diff_away_only",
    "home_win_streak", "away_win_streak",
    "home_form_decayed", "away_form_decayed",
    "home_season_points", "away_season_points",
    "home_season_gd", "away_season_gd",
    "home_season_matches", "away_season_matches",
    "home_draw_tendency", "away_draw_tendency",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_meetings",
]

SHOTS_FEATURES = ["home_shots_ot_form", "away_shots_ot_form"]
XG_FEATURES = ["home_xg_form", "away_xg_form"]
ODDS_FEATURES = ["market_prob_home", "market_prob_draw", "market_prob_away"]

FEATURE_GROUPS = {
    "baseline (26 features)": BASE_FEATURES,
    "+ shots on target":      BASE_FEATURES + SHOTS_FEATURES,
    "+ xG":                   BASE_FEATURES + XG_FEATURES,
    "+ market odds":          BASE_FEATURES + ODDS_FEATURES,
    "+ shots + xG":           BASE_FEATURES + SHOTS_FEATURES + XG_FEATURES,
    "+ shots + odds":         BASE_FEATURES + SHOTS_FEATURES + ODDS_FEATURES,
    "+ all new features":     BASE_FEATURES + SHOTS_FEATURES + XG_FEATURES + ODDS_FEATURES,
}

TARGET_SEASONS = ["2324", "2425"]


def run_backtest(matches: pd.DataFrame, target_season: str, features: list[str]) -> dict:
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    completed["season_code"] = completed["season"].astype(str).str.zfill(4)
    all_features = build_training_frame(completed)
    all_features["season_code"] = all_features["season"].astype(str).str.zfill(4)

    train_f = all_features[all_features["season_code"] != target_season]
    target_f = all_features[all_features["season_code"] == target_season].reset_index(drop=True)
    target_raw = completed[completed["season_code"] == target_season]

    weights = season_decay_weights(train_f["season"], reference_season=target_season)
    model = train_model(train_f, sample_weight=weights, feature_columns=features)

    proba = model.estimator.predict_proba(target_f[features])
    classes = list(model.estimator.classes_)
    proba_df = pd.DataFrame(proba, columns=classes)
    predictions = target_f[["match_id", "contest_week", "date", "home_team", "away_team"]].copy()
    predictions["p_home_win"] = proba_df.get("HOME_WIN", 0.0)
    predictions["p_draw"] = proba_df.get("DRAW", 0.0)
    predictions["p_away_win"] = proba_df.get("AWAY_WIN", 0.0)
    totals = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    predictions[["p_home_win", "p_draw", "p_away_win"]] = predictions[
        ["p_home_win", "p_draw", "p_away_win"]
    ].div(totals, axis=0)

    actual = target_raw[["match_id", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    evaluated = predictions.merge(actual, on=["match_id", "home_team", "away_team"])
    evaluated["actual"] = evaluated.apply(_actual_outcome, axis=1)

    y_true = evaluated["actual"]
    y_pred = evaluated[["p_home_win", "p_draw", "p_away_win"]].idxmax(axis=1).map(
        {"p_home_win": "HOME_WIN", "p_draw": "DRAW", "p_away_win": "AWAY_WIN"}
    )

    picks = optimize_picks(build_pick_candidates(predictions))
    scored = _score_optimized_picks(picks, evaluated)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(
            y_true, evaluated[["p_away_win", "p_draw", "p_home_win"]],
            labels=["AWAY_WIN", "DRAW", "HOME_WIN"]
        )),
        "points": int(scored["actual_points"].sum()),
        "xpts": float(scored["expected_points"].sum()),
    }


def main() -> None:
    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")

    # Header
    print(f"\n{'Feature set':<28} ", end="")
    for s in TARGET_SEASONS:
        w = WINNER_BENCHMARKS.get(s)
        print(f"  {s}(w={w}) ", end="")
    print("  Total")
    print("-" * 80)

    for label, features in FEATURE_GROUPS.items():
        print(f"{label:<28} ", end="", flush=True)
        total = 0
        for season in TARGET_SEASONS:
            r = run_backtest(matches, season, features)
            total += r["points"]
            print(f"  {r['points']:>3}pts {r['accuracy']:.3f} ", end="", flush=True)
        print(f"  {total:>3}")


if __name__ == "__main__":
    main()
