from pathlib import Path

import pandas as pd
import requests

from epl_prediction_optimizer.pipeline import backtest_season, refresh_data, run_all


def test_run_all_sample_pipeline_creates_feasible_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = run_all(mode="offline")

    assert result == {"predictions": 8, "picks": 4}


def test_offline_refresh_never_calls_requests_even_with_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FOOTBALL_DATA_ORG_API_KEY", "live-key-must-not-be-used")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("offline refresh must not call requests.get")

    monkeypatch.setattr(requests, "get", fail_if_called)

    result = refresh_data(mode="offline")

    assert Path(result["fixtures"]).exists()


def test_backtest_season_trains_on_prior_seasons_and_scores_completed_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    historical_rows = []
    teams = ["Arsenal", "Everton", "Chelsea", "Fulham"]
    for season, year in [("2324", 2023), ("2425", 2024)]:
        for week in range(1, 9):
            for index in range(0, len(teams), 2):
                historical_rows.append(
                    {
                        "season": season,
                        "match_id": f"{season}-{week}-{index}",
                        "contest_week": week,
                        "date": f"{year}-08-{week + index + 1:02d}",
                        "home_team": teams[(week + index) % len(teams)],
                        "away_team": teams[(week + index + 1) % len(teams)],
                        "home_goals": 2 if week % 3 == 0 else 1,
                        "away_goals": 1 if week % 3 == 0 else 1 if week % 3 == 1 else 2,
                        "home_elo": 1550 + week,
                        "away_elo": 1500 - week,
                    }
                )
    target_rows = []
    first_saturday = pd.Timestamp("2025-08-16")
    for week in range(1, 5):
        for index in range(0, len(teams), 2):
            target_rows.append(
                {
                    "season": "2526",
                    "match_id": f"2526-{week}-{index}",
                    "contest_week": week,
                    "date": first_saturday + pd.Timedelta(days=(week - 1) * 7),
                    "home_team": teams[(week + index) % len(teams)],
                    "away_team": teams[(week + index + 1) % len(teams)],
                    "home_goals": 2,
                    "away_goals": 1,
                    "home_elo": 1560 + week,
                    "away_elo": 1490 - week,
                }
            )
    matches = pd.DataFrame(historical_rows + target_rows)

    result = backtest_season(matches, target_season="2526")

    assert result["target_matches"] == 8
    assert result["optimized_picks"] == 4
    assert result["optimized_points"] >= 0
