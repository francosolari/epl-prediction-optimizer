import pandas as pd

from epl_prediction_optimizer.data.sources import apply_official_gameweeks


def test_apply_official_gameweeks_matches_by_date_and_teams():
    historical = pd.DataFrame(
        [
            {
                "date": "2025-08-15",
                "home_team": "Liverpool",
                "away_team": "Bournemouth",
                "contest_week": 1,
            },
            {
                "date": "2025-09-20",
                "home_team": "Manchester United",
                "away_team": "Chelsea",
                "contest_week": 6,
            },
        ]
    )
    official = pd.DataFrame(
        [
            {
                "date": "2025-08-15",
                "home_team": "Liverpool",
                "away_team": "Bournemouth",
                "contest_week": 1,
            },
            {
                "date": "2025-09-20",
                "home_team": "Manchester United",
                "away_team": "Chelsea",
                "contest_week": 5,
            },
        ]
    )

    merged = apply_official_gameweeks(historical, official)

    assert list(merged["contest_week"]) == [1, 5]
