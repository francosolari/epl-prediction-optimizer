"""Committed picks must settle against results, not sit pending forever."""

from __future__ import annotations

import pandas as pd
import pytest

from epl_prediction_optimizer.challenge import pick_points, score_committed_picks
from epl_prediction_optimizer.storage.database import Database


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "api-1",
                "season": "2627",
                "contest_week": 1,
                "date": "2026-08-22",
                "home_team": "Hull City",
                "away_team": "Manchester United",
                "home_goals": 2.0,
                "away_goals": 0.0,
            },
            {
                "match_id": "api-2",
                "season": "2627",
                "contest_week": 2,
                "date": "2026-08-29",
                "home_team": "Everton",
                "away_team": "Chelsea",
                "home_goals": 1.0,
                "away_goals": 1.0,
            },
            {
                "match_id": "api-3",
                "season": "2627",
                "contest_week": 3,
                "date": "2026-09-05",
                "home_team": "Arsenal",
                "away_team": "Leeds United",
                "home_goals": 3.0,
                "away_goals": 0.0,
            },
        ]
    )


@pytest.mark.parametrize(
    ("team", "home_goals", "away_goals", "expected"),
    [
        ("Hull City", 2, 0, 3),
        ("Manchester United", 2, 0, 0),
        ("Hull City", 1, 1, 1),
        ("Manchester United", 0, 2, 3),
    ],
)
def test_pick_points_scores_from_the_picked_side(
    team: str, home_goals: int, away_goals: int, expected: int
) -> None:
    assert pick_points(team, "Hull City", home_goals, away_goals) == expected


def _database(tmp_path) -> Database:
    database = Database(tmp_path / "state.sqlite")
    database.upsert_actual_pick(
        {
            "season": "2627",
            "contest_week": 1,
            "match_id": "api-1",
            "team": "Manchester United",
            "venue": "away",
        }
    )
    database.upsert_actual_pick(
        {
            "season": "2627",
            "contest_week": 2,
            "match_id": "api-2",
            "team": "Everton",
            "venue": "home",
        }
    )
    return database


def test_committed_picks_are_settled(tmp_path) -> None:
    database = _database(tmp_path)
    assert score_committed_picks(database, "2627", _matches()) == {"scored": 2, "pending": 0}
    points = {p["contest_week"]: p["actual_points"] for p in database.list_actual_picks("2627")}
    assert points == {1: 0, 2: 1}


def test_scoring_is_idempotent_and_leaves_pick_version_alone(tmp_path) -> None:
    """Settling a result is not a new decision by the user."""
    database = _database(tmp_path)
    score_committed_picks(database, "2627", _matches())
    versions = {p["contest_week"]: p["pick_version"] for p in database.list_actual_picks("2627")}
    assert set(versions.values()) == {1}
    assert score_committed_picks(database, "2627", _matches())["scored"] == 0


def test_picks_settle_when_the_stored_match_id_is_from_another_source(tmp_path) -> None:
    """Fixture ids differ between the official API and football-data.co.uk."""
    database = Database(tmp_path / "state.sqlite")
    database.upsert_actual_pick(
        {
            "season": "2627",
            "contest_week": 3,
            "match_id": "2627-0021",  # not the id carried in the match record
            "team": "Arsenal",
            "venue": "home",
        }
    )
    assert score_committed_picks(database, "2627", _matches())["scored"] == 1
    assert database.list_actual_picks("2627")[0]["actual_points"] == 3


def test_an_unplayed_match_stays_pending(tmp_path) -> None:
    database = Database(tmp_path / "state.sqlite")
    database.upsert_actual_pick(
        {
            "season": "2627",
            "contest_week": 9,
            "match_id": "api-9",
            "team": "Arsenal",
            "venue": "home",
        }
    )
    assert score_committed_picks(database, "2627", _matches()) == {"scored": 0, "pending": 1}
    assert database.list_actual_picks("2627")[0]["actual_points"] is None
