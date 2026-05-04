import pandas as pd

from epl_prediction_optimizer.ml.features import build_training_frame
from epl_prediction_optimizer.ml.model import train_model


def test_training_pipeline_outputs_normalized_probabilities():
    matches = pd.DataFrame(
        [
            {
                "date": f"2025-08-{day:02d}",
                "home_team": f"Home {day % 4}",
                "away_team": f"Away {day % 5}",
                "home_goals": 2 + (day % 3),
                "away_goals": day % 2,
                "home_elo": 1600 + day,
                "away_elo": 1500 - day,
            }
            for day in range(1, 24)
        ]
        + [
            {
                "date": f"2025-09-{day:02d}",
                "home_team": f"Home {day % 4}",
                "away_team": f"Away {day % 5}",
                "home_goals": day % 2,
                "away_goals": 2 + (day % 3),
                "home_elo": 1450 - day,
                "away_elo": 1580 + day,
            }
            for day in range(1, 24)
        ]
        + [
            {
                "date": f"2025-10-{day:02d}",
                "home_team": f"Home {day % 4}",
                "away_team": f"Away {day % 5}",
                "home_goals": 1,
                "away_goals": 1,
                "home_elo": 1500,
                "away_elo": 1500,
            }
            for day in range(1, 24)
        ]
    )
    training = build_training_frame(matches)
    model_run = train_model(training)
    fixtures = pd.DataFrame(
        [
            {
                "match_id": "future-1",
                "contest_week": 1,
                "date": "2026-08-15",
                "home_team": "Arsenal",
                "away_team": "Everton",
                "home_elo": 1650,
                "away_elo": 1520,
            }
        ]
    )

    predictions = model_run.predict(fixtures)

    assert list(predictions.columns) == [
        "match_id",
        "contest_week",
        "date",
        "home_team",
        "away_team",
        "p_home_win",
        "p_draw",
        "p_away_win",
    ]
    assert round(
        predictions.loc[0, ["p_home_win", "p_draw", "p_away_win"]].sum(), 8
    ) == 1.0
    assert model_run.metrics["log_loss"] >= 0

