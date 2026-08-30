"""Shared evaluation helpers for match probabilities and contest picks.

Backtest scripts and the pipeline score the same way through this module so a
number quoted in one place means the same thing in the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from epl_prediction_optimizer.ml.analysis import expected_calibration_error

PROBABILITY_COLUMNS = ["p_home_win", "p_draw", "p_away_win"]
# Ordered worst-to-best from the home side, which is what makes the ranked
# probability score meaningful for a three-outcome market.
ORDERED_COLUMNS = ["p_away_win", "p_draw", "p_home_win"]
ORDERED_CLASSES = ["AWAY_WIN", "DRAW", "HOME_WIN"]


def actual_outcome(row: pd.Series) -> str:
    """Classify a completed match into HOME_WIN / DRAW / AWAY_WIN."""
    if row["home_goals"] > row["away_goals"]:
        return "HOME_WIN"
    if row["home_goals"] < row["away_goals"]:
        return "AWAY_WIN"
    return "DRAW"


def actual_outcomes(evaluated: pd.DataFrame) -> pd.Series:
    """Vectorised outcome labels for a frame of completed matches."""
    home = pd.to_numeric(evaluated["home_goals"])
    away = pd.to_numeric(evaluated["away_goals"])
    return pd.Series(
        np.where(home > away, "HOME_WIN", np.where(home < away, "AWAY_WIN", "DRAW")),
        index=evaluated.index,
    )


def ranked_probability_score(probabilities: pd.DataFrame, actual: pd.Series) -> float:
    """Mean RPS over the ordered outcome scale (away < draw < home).

    RPS penalises a confident prediction less when it misses to an adjacent
    outcome than to the opposite one, which matches how a pick actually fails:
    backing a home win and getting a draw is a cheaper error than a defeat.
    """
    forecast = probabilities[ORDERED_COLUMNS].to_numpy(dtype=float)
    observed = np.zeros_like(forecast)
    for index, label in enumerate(ORDERED_CLASSES):
        observed[:, index] = (actual.to_numpy() == label).astype(float)
    cumulative_error = np.cumsum(forecast, axis=1) - np.cumsum(observed, axis=1)
    return float((cumulative_error[:, :-1] ** 2).sum(axis=1).mean())


def brier_score(probabilities: pd.DataFrame, actual: pd.Series) -> float:
    """Mean multi-class Brier score (0 is perfect, 2 is worst possible)."""
    forecast = probabilities[ORDERED_COLUMNS].to_numpy(dtype=float)
    observed = np.zeros_like(forecast)
    for index, label in enumerate(ORDERED_CLASSES):
        observed[:, index] = (actual.to_numpy() == label).astype(float)
    return float(((forecast - observed) ** 2).sum(axis=1).mean())


def probability_metrics(evaluated: pd.DataFrame) -> dict[str, float]:
    """Score a frame carrying probability columns and an ``actual`` column.

    Log loss, RPS, and Brier are the metrics that move on a full season of
    matches. Contest points move on 38 picks and are far too noisy to select
    features or weights with, so they are reported alongside, never instead.
    """
    actual = evaluated["actual"]
    predicted = (
        evaluated[PROBABILITY_COLUMNS]
        .idxmax(axis=1)
        .map({"p_home_win": "HOME_WIN", "p_draw": "DRAW", "p_away_win": "AWAY_WIN"})
    )
    return {
        "matches": float(len(evaluated)),
        "log_loss": float(
            log_loss(actual, evaluated[ORDERED_COLUMNS], labels=ORDERED_CLASSES)
        ),
        "rps": ranked_probability_score(evaluated, actual),
        "brier": brier_score(evaluated, actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "expected_calibration_error": expected_calibration_error(
            evaluated[PROBABILITY_COLUMNS].rename(
                columns={
                    "p_home_win": "HOME_WIN",
                    "p_draw": "DRAW",
                    "p_away_win": "AWAY_WIN",
                }
            ),
            actual.reset_index(drop=True),
        ),
    }


def score_picks(picks: pd.DataFrame, evaluated: pd.DataFrame) -> pd.DataFrame:
    """Attach contest points (3 win / 1 draw / 0 loss) and the final score to picks."""
    rows = []
    actual_by_match = evaluated.drop_duplicates(subset="match_id").set_index("match_id")
    for pick in picks.itertuples(index=False):
        match = actual_by_match.loc[pick.match_id]
        if match.home_goals == match.away_goals:
            points = 1
        elif pick.team == match.home_team and match.home_goals > match.away_goals:
            points = 3
        elif pick.team == match.away_team and match.away_goals > match.home_goals:
            points = 3
        else:
            points = 0
        row = pick._asdict()
        row["actual_points"] = points
        row["home_goals"] = int(match.home_goals)
        row["away_goals"] = int(match.away_goals)
        rows.append(row)
    return pd.DataFrame(rows)
