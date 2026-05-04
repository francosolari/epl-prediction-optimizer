from datetime import date

import pandas as pd

from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks


def test_build_pick_candidates_filters_weekends_and_scores_expected_points():
    probabilities = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "contest_week": 1,
                "date": date(2026, 8, 15),
                "home_team": "Arsenal",
                "away_team": "Everton",
                "p_home_win": 0.6,
                "p_draw": 0.2,
                "p_away_win": 0.2,
            },
            {
                "match_id": "m2",
                "contest_week": 1,
                "date": date(2026, 8, 17),
                "home_team": "Chelsea",
                "away_team": "Fulham",
                "p_home_win": 0.5,
                "p_draw": 0.25,
                "p_away_win": 0.25,
            },
        ]
    )

    candidates = build_pick_candidates(probabilities)

    assert set(candidates["match_id"]) == {"m1"}
    arsenal = candidates[candidates["team"] == "Arsenal"].iloc[0]
    everton = candidates[candidates["team"] == "Everton"].iloc[0]
    assert arsenal["venue"] == "home"
    assert arsenal["expected_points"] == 2.0
    assert everton["venue"] == "away"
    assert everton["expected_points"] == 0.8


def test_optimize_picks_enforces_weekly_team_and_venue_limits():
    rows = []
    teams = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    for week in range(1, 5):
        for index, team in enumerate(teams):
            rows.append(
                {
                    "contest_week": week,
                    "match_id": f"{week}-{team}",
                    "team": team,
                    "opponent": "Opponent",
                    "venue": "home" if week % 2 == index % 2 else "away",
                    "p_win": 0.4 + (0.01 * index),
                    "p_draw": 0.2,
                    "p_loss": 0.4 - (0.01 * index),
                    "expected_points": 1.4 + (0.03 * index),
                }
            )
    candidates = pd.DataFrame(rows)

    picks = optimize_picks(candidates)

    assert len(picks) == 4
    assert picks["contest_week"].nunique() == 4
    assert picks.groupby("team").size().min() >= 1
    assert picks.groupby("team").size().max() <= 2
    assert picks.groupby(["team", "venue"]).size().max() <= 1

