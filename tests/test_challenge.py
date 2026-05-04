from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from epl_prediction_optimizer.app.main import create_app
from epl_prediction_optimizer.storage.database import Database


def test_challenge_manager_stores_manual_pick_and_scores_known_result(tmp_path: Path):
    data_dir = tmp_path / "data" / "exports"
    processed_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "match_id": "m1",
                "contest_week": 1,
                "date": "2025-08-16",
                "home_team": "Arsenal",
                "away_team": "Everton",
                "p_home_win": 0.51,
                "p_draw": 0.24,
                "p_away_win": 0.25,
            }
        ]
    ).to_csv(data_dir / "fixture_probabilities.csv", index=False)
    pd.DataFrame(
        [
            {
                "season": "2526",
                "match_id": "m1",
                "contest_week": 1,
                "date": "2025-08-16",
                "home_team": "Arsenal",
                "away_team": "Everton",
                "home_goals": 2,
                "away_goals": 1,
            }
        ]
    ).to_csv(processed_dir / "historical_matches.csv", index=False)
    database = Database(tmp_path / "state.sqlite")
    app = create_app(database=database, workdir=tmp_path, use_live_data=False)
    client = TestClient(app)

    response = client.post(
        "/api/manual-picks",
        data={
            "season": "2526",
            "contest_week": "1",
            "match_id": "m1",
            "team": "Everton",
            "venue": "away",
            "notes": "Late Arsenal injury, override recommendation.",
        },
    )
    page = client.get("/challenge?season=2526")

    assert response.status_code == 200
    assert response.json()["pick"]["team"] == "Everton"
    assert response.json()["pick"]["actual_points"] == 0
    assert page.status_code == 200
    assert "Late Arsenal injury" in page.text
    assert "Everton" in page.text
