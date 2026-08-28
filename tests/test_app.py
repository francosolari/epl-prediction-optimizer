import re
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from epl_prediction_optimizer.app.main import build_readiness, create_app
from epl_prediction_optimizer.config.season import SeasonContext
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


def test_dashboard_never_presents_cached_predictions_from_another_season(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite")
    database.set_json(
        "predictions",
        [
            {
                "match_id": "stale-1",
                "contest_week": 1,
                "date": "2025-08-16",
                "home_team": "West Ham United",
                "away_team": "Wolverhampton Wanderers",
                "p_home_win": 0.6,
                "p_draw": 0.2,
                "p_away_win": 0.2,
            }
        ],
    )
    client = TestClient(create_app(database=database, workdir=tmp_path, use_live_data=False))

    home = client.get("/")

    assert home.status_code == 200
    assert "West Ham United" not in home.text
    assert "No verified 2026–27 forecasts yet" in home.text


def test_all_analysis_surfaces_share_the_theme_control(tmp_path: Path):
    client = TestClient(
        create_app(
            database=Database(tmp_path / "state.sqlite"),
            workdir=tmp_path,
            use_live_data=False,
        )
    )

    for route in (
        "/",
        "/model",
        "/data",
        "/backtest",
        "/challenge?season=2627",
        "/scorecard?season=2627",
    ):
        response = client.get(route)
        assert response.status_code == 200
        assert 'data-theme="system"' in response.text
        assert "data-theme-toggle" in response.text
        assert "Decision" in response.text
        assert "Season" in response.text
        assert "Backtest" in response.text
        if route != "/":
            assert 'class="analysis-page"' in response.text
        # Every page must link to every other one with a real season code; an
        # empty one silently falls back and hides a broken template variable.
        assert 'href="/scorecard?season=2627"' in response.text
        assert 'href="/challenge?season=2627"' in response.text


def test_dashboard_offers_a_single_full_refresh_control(tmp_path: Path):
    client = TestClient(
        create_app(
            database=Database(tmp_path / "state.sqlite"),
            workdir=tmp_path,
            use_live_data=False,
        )
    )
    page = client.get("/").text
    assert 'formaction="/actions/run-all"' in page
    assert "Refresh everything" in page


def test_server_does_not_refresh_on_a_timer_by_default(tmp_path: Path):
    """A background loop would rewrite published forecasts without being asked."""
    import inspect

    from epl_prediction_optimizer.app.main import create_app as factory

    assert inspect.signature(factory).parameters["auto_refresh"].default is False


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

    dashboard = client.get("/?week=2")
    assert "data-chart-scale" in dashboard.text
    assert "data-week=" in dashboard.text
    assert "Season move tree" in dashboard.text


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


def test_scorecard_shows_committed_picks_with_points(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite")
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "historical_matches.csv").write_text(
        "match_id,season,contest_week,date,home_team,away_team,home_goals,away_goals\n"
        "m1,2627,1,2026-08-22,Hull City,Manchester United,2,0\n",
        encoding="utf-8",
    )
    database.upsert_actual_pick(
        {
            "season": "2627",
            "contest_week": 1,
            "match_id": "m1",
            "team": "Manchester United",
            "venue": "away",
            "actual_points": 0,
        }
    )
    client = TestClient(create_app(database=database, workdir=tmp_path, use_live_data=False))

    page = client.get("/scorecard?season=2627")

    assert page.status_code == 200
    assert "Manchester United" in page.text
    assert "Hull City 2-0" in page.text


def test_scorecard_ranks_the_user_within_an_imported_field(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite")
    (tmp_path / "data" / "processed").mkdir(parents=True)
    database.replace_league_data(
        "2526",
        [
            {"entrant": "Ann Lee", "place": "1", "points": 70, "goal_diff": 20},
            {"entrant": "Bo Ray", "place": "2", "points": 40, "goal_diff": 5},
        ],
        [{"entrant": "Ann Lee", "contest_week": 1, "team": "Arsenal", "venue": "home",
          "points": 3, "goal_diff": 2}],
    )
    database.upsert_actual_pick(
        {
            "season": "2526",
            "contest_week": 1,
            "match_id": "m1",
            "team": "Arsenal",
            "venue": "home",
            "actual_points": 3,
        }
    )
    client = TestClient(create_app(database=database, workdir=tmp_path, use_live_data=False))

    page = client.get("/scorecard?season=2526").text

    assert "Ann Lee" in page and "Bo Ray" in page
    # Three points puts the user behind both entrants.
    assert "3 / 2" in page or "3</strong>" in page


def test_importing_a_bad_sheet_url_reports_it_instead_of_failing(tmp_path: Path):
    client = TestClient(
        create_app(
            database=Database(tmp_path / "state.sqlite"),
            workdir=tmp_path,
            use_live_data=False,
        )
    )
    response = client.post(
        "/actions/import-league",
        data={"url": "https://example.com/not-a-sheet", "season": "2627"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Not a Google Sheets URL" in response.text


def test_dashboard_reports_what_fed_the_current_numbers(tmp_path: Path):
    """The pick is only trustworthy if its inputs are visible without a script."""
    import json

    processed = tmp_path / "data" / "processed"
    exports = tmp_path / "data" / "exports"
    artifacts = tmp_path / "data" / "artifacts"
    for directory in (processed, exports, artifacts):
        directory.mkdir(parents=True)

    (processed / "fixtures.csv").write_text(
        "match_id,contest_week,date,home_team,away_team,home_elo,away_elo\n"
        "m1,2,2099-01-03,Everton,Chelsea,1600,1700\n",
        encoding="utf-8",
    )
    (exports / "fixture_probabilities.csv").write_text(
        "match_id,contest_week,date,home_team,away_team,p_home_win,p_draw,p_away_win,market_weight\n"
        "m1,2,2099-01-03,Everton,Chelsea,0.4,0.25,0.35,0.9\n",
        encoding="utf-8",
    )
    (artifacts / "metrics.json").write_text(
        json.dumps({"trained_at": datetime.now(UTC).isoformat(), "rows": 11667}),
        encoding="utf-8",
    )

    client = TestClient(
        create_app(
            database=Database(tmp_path / "state.sqlite"),
            workdir=tmp_path,
            use_live_data=False,
        )
    )
    page = client.get("/").text

    assert "Model retrained" in page
    assert "11,667 matches" in page
    assert "market prices" in page
    assert "blended at weight 0.9" in page


def test_readiness_flags_a_model_older_than_its_data(tmp_path: Path):
    import json

    processed = tmp_path / "data" / "processed"
    artifacts = tmp_path / "data" / "artifacts"
    for directory in (processed, artifacts, tmp_path / "data" / "exports"):
        directory.mkdir(parents=True)
    (processed / "fixtures.csv").write_text("match_id,contest_week,date\nm1,2,2099-01-03\n")
    # Trained before the data it is supposed to have learned from.
    (artifacts / "metrics.json").write_text(
        json.dumps({"trained_at": "2020-01-01T00:00:00+00:00", "rows": 10}),
        encoding="utf-8",
    )

    readiness = build_readiness(
        Database(tmp_path / "state.sqlite"),
        tmp_path,
        SeasonContext.current_season(),
        0.9,
    )

    trained = next(c for c in readiness["checks"] if c["label"] == "Model retrained")
    assert trained["ok"] is False
    assert readiness["all_ok"] is False


def test_static_assets_are_versioned_so_a_stale_stylesheet_cannot_be_served(tmp_path: Path):
    """A cached stylesheet silently renders the page with the wrong layout."""
    client = TestClient(
        create_app(
            database=Database(tmp_path / "state.sqlite"),
            workdir=tmp_path,
            use_live_data=False,
        )
    )
    page = client.get("/").text
    match = re.search(r'href="/static/styles\.css\?v=(\d+)"', page)
    assert match, "stylesheet link carries no version stamp"
    assert client.get(f"/static/styles.css?v={match.group(1)}").status_code == 200


def test_committing_a_pick_offers_the_submission_email(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite")
    app = create_app(database=database, workdir=tmp_path, use_live_data=False)
    client = TestClient(app)
    client.post("/api/run-all")

    decision = client.get("/api/weeks/2").json()
    choice = decision["scenarios"][0]
    client.put(
        "/api/picks/2",
        json={key: choice[key] for key in ("match_id", "team", "venue")},
    )

    page = client.get("/?week=2").text
    assert "mail.google.com" in page
    assert "premierPicksCompetition" in page
    body = f"I pick {choice['team']} to beat {choice['opponent']}"
    assert body in page or body.replace(" ", "+") in page


def test_the_sending_google_account_can_be_set_and_pins_the_compose_link(tmp_path: Path):
    database = Database(tmp_path / "state.sqlite")
    client = TestClient(create_app(database=database, workdir=tmp_path, use_live_data=False))
    client.post("/api/run-all")
    decision = client.get("/api/weeks/2").json()
    choice = decision["scenarios"][0]
    client.put("/api/picks/2", json={k: choice[k] for k in ("match_id", "team", "venue")})

    before = client.get("/?week=2").text
    assert "Not set" in before

    client.post(
        "/actions/gmail-account",
        data={"account": "franco@gmail.com", "week": "2"},
        follow_redirects=True,
    )
    after = client.get("/?week=2").text
    assert "mail/u/franco%40gmail.com/" in after
    assert "Not set" not in after
