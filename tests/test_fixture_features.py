"""Fixture features must match the values training produced for the same match."""

from __future__ import annotations

import pandas as pd

from epl_prediction_optimizer.ml.features import (
    FEATURE_COLUMNS,
    build_fixture_features,
    build_training_frame,
)


def _matches() -> pd.DataFrame:
    teams = ["Arsenal", "Everton", "Chelsea", "Fulham"]
    rows = []
    for week in range(1, 13):
        home = teams[week % len(teams)]
        away = teams[(week + 1) % len(teams)]
        rows.append(
            {
                "date": f"2025-09-{week:02d}",
                "home_team": home,
                "away_team": away,
                "home_goals": float(week % 3),
                "away_goals": float((week + 1) % 3),
                "home_elo": 1600.0 + week,
                "away_elo": 1500.0 - week,
                "season": "2526",
                "match_id": f"m-{week}",
                "contest_week": week,
            }
        )
    return pd.DataFrame(rows)


def test_fixture_features_reproduce_training_features() -> None:
    """A held-out match scored as a fixture gets the same features as in training."""
    matches = _matches()
    history = matches.iloc[:-1]
    held_out = matches.iloc[-1:]

    training = build_training_frame(matches)
    expected = training[training["match_id"] == "m-12"].iloc[0]

    fixture = held_out.drop(columns=["home_goals", "away_goals"])
    actual = build_fixture_features(fixture, history=history).iloc[0]

    for column in FEATURE_COLUMNS:
        assert actual[column] == expected[column], column


def test_unplayed_fixtures_do_not_feed_back_into_form() -> None:
    """Two future fixtures both see history as it stood, not each other's results."""
    matches = _matches()
    fixtures = pd.DataFrame(
        [
            {
                "date": "2025-10-01",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_elo": 1620.0,
                "away_elo": 1510.0,
                "match_id": "f-1",
                "contest_week": 13,
            },
            {
                "date": "2025-10-08",
                "home_team": "Arsenal",
                "away_team": "Fulham",
                "home_elo": 1620.0,
                "away_elo": 1490.0,
                "match_id": "f-2",
                "contest_week": 14,
            },
        ]
    )
    features = build_fixture_features(fixtures, history=matches)
    assert features.loc[0, "home_form_points"] == features.loc[1, "home_form_points"]
    assert features.loc[0, "home_season_matches"] == features.loc[1, "home_season_matches"]


def test_history_path_beats_neutral_defaults_on_form_features() -> None:
    """Without history the form features fall back to defaults, not real values."""
    matches = _matches()
    fixtures = matches.iloc[-1:].drop(columns=["home_goals", "away_goals"])

    with_history = build_fixture_features(fixtures, history=matches.iloc[:-1]).iloc[0]
    without_history = build_fixture_features(fixtures).iloc[0]

    assert with_history["home_form_points"] != 0.0
    assert without_history["home_form_points"] == 0.0
    assert without_history["home_draw_tendency"] == 0.25


def test_fixture_metadata_survives_feature_building() -> None:
    matches = _matches()
    fixtures = matches.iloc[-1:].drop(columns=["home_goals", "away_goals"]).copy()
    fixtures["kickoff_utc"] = "2025-09-12T14:00:00Z"
    features = build_fixture_features(fixtures, history=matches.iloc[:-1])
    assert features.loc[0, "kickoff_utc"] == "2025-09-12T14:00:00Z"
    assert features.loc[0, "contest_week"] == 12
    assert features.loc[0, "match_id"] == "m-12"


def test_a_played_fixture_is_not_described_by_its_own_result() -> None:
    """The season plan re-predicts past rounds; those must stay leak-free."""
    matches = _matches()
    fixture = matches.iloc[-1:].drop(columns=["home_goals", "away_goals"])

    # History that already contains the match, and history that does not.
    with_own_result = build_fixture_features(fixture, history=matches).iloc[0]
    without_own_result = build_fixture_features(fixture, history=matches.iloc[:-1]).iloc[0]

    for column in FEATURE_COLUMNS:
        assert with_own_result[column] == without_own_result[column], column
