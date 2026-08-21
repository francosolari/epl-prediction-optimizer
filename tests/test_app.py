from pathlib import Path

from fastapi.testclient import TestClient

from epl_prediction_optimizer.app.main import create_app
from epl_prediction_optimizer.storage.database import Database


def test_app_exposes_dashboard_and_status(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite")
    app = create_app(database=database, workdir=tmp_path, use_live_data=False)
    client = TestClient(app)

    home = client.get("/")
    status = client.get("/api/status")

    assert home.status_code == 200
    assert "EPL Prediction Optimizer" in home.text
    assert "Prepare season" in home.text
    assert "Choose this week’s line" in home.text
    assert "System" in home.text
    assert status.status_code == 200
    assert status.json()["status"] == "needs_refresh"
    readiness = client.get("/api/readiness")
    assert readiness.status_code == 200
    assert readiness.json()["source_mode"] == "offline"


def test_run_all_endpoint_refreshes_trains_predicts_and_optimizes(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite")
    app = create_app(database=database, workdir=tmp_path, use_live_data=False)
    client = TestClient(app)

    response = client.post("/api/run-all")
    predictions = client.get("/api/predictions")
    picks = client.get("/api/picks")
    status = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["status"] == "complete"
    assert len(predictions.json()) == 8
    assert len(picks.json()) == 4
    assert status.json()["last_action"] == "run-all"

    future_decision = client.get("/api/weeks/2").json()
    choice = future_decision["scenarios"][0]
    saved = client.put(
        "/api/picks/2",
        json={key: choice[key] for key in ("match_id", "team", "venue")},
    )
    assert saved.status_code == 200
    assert saved.json()["pick"]["pick_version"] == 1

    past_decision = client.get("/api/weeks/1").json()
    past_choice = past_decision["scenarios"][0]
    locked = client.put(
        "/api/picks/1",
        json={key: past_choice[key] for key in ("match_id", "team", "venue")},
    )
    assert locked.status_code == 409


def test_data_explorer_previews_stored_csv_files(tmp_path: Path):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    (data_dir / "historical_matches.csv").write_text(
        "date,home_team,away_team\n2025-08-16,Arsenal,Everton\n",
        encoding="utf-8",
    )
    database = Database(tmp_path / "state.sqlite")
    app = create_app(database=database, workdir=tmp_path, use_live_data=False)
    client = TestClient(app)

    page = client.get("/data")
    preview = client.get("/api/data/processed/historical_matches")

    assert page.status_code == 200
    assert "Stored Input Data" in page.text
    assert preview.status_code == 200
    assert preview.json()["rows"][0]["home_team"] == "Arsenal"
