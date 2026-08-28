"""Local challenge-management helpers for actual submitted picks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from epl_prediction_optimizer.paths import PROCESSED_DIR

if TYPE_CHECKING:
    from epl_prediction_optimizer.storage.database import Database


def load_challenge_rows(runtime_dir: Path, season: str, actual_picks: list[dict]) -> list[dict]:
    """Combine probabilities, known results, and user's stored picks for the UI."""
    probabilities = _read_optional_csv(
        runtime_dir / "data" / "exports" / "fixture_probabilities.csv"
    )
    recommendations = _read_optional_csv(runtime_dir / "data" / "exports" / "optimized_picks.csv")
    results = _read_optional_csv(runtime_dir / "data" / "processed" / "historical_matches.csv")
    if probabilities.empty:
        return []
    rows = probabilities.copy()
    rows["season"] = season
    if not results.empty and "match_id" in results:
        rows["match_id"] = rows["match_id"].astype(str)
        results["match_id"] = results["match_id"].astype(str)
        result_columns = ["match_id", "home_goals", "away_goals"]
        if "season" in results:
            result_columns.append("season")
        rows = rows.merge(
            results[result_columns],
            on="match_id",
            how="left",
            suffixes=("", "_result"),
        )
        if "season_result" in rows:
            rows["season"] = rows["season_result"].fillna(rows["season"])
            rows = rows.drop(columns=["season_result"])
    picks_by_week = {int(pick["contest_week"]): pick for pick in actual_picks}
    recommendations_by_week = {}
    if not recommendations.empty:
        recommendations_by_week = {
            int(pick["contest_week"]): pick for pick in recommendations.to_dict(orient="records")
        }
    used_teams = {pick["team"] for pick in actual_picks}
    used_team_venues = {(pick["team"], pick["venue"]) for pick in actual_picks}
    output = []
    for row in rows.sort_values(["contest_week", "date"]).to_dict(orient="records"):
        pick = picks_by_week.get(int(row["contest_week"]))
        recommendation = recommendations_by_week.get(int(row["contest_week"]))
        row["home_expected_points"] = round(3 * row["p_home_win"] + row["p_draw"], 3)
        row["away_expected_points"] = round(3 * row["p_away_win"] + row["p_draw"], 3)
        row["expected_gap"] = round(
            abs(row["home_expected_points"] - row["away_expected_points"]),
            3,
        )
        row["is_close_call"] = row["expected_gap"] <= 0.18
        row["model_favorite"] = (
            row["home_team"]
            if row["home_expected_points"] >= row["away_expected_points"]
            else row["away_team"]
        )
        row["actual_pick"] = pick
        row["recommended_pick"] = recommendation
        row["is_override"] = bool(
            pick and recommendation and pick["team"] != recommendation["team"]
        )
        row["home_used"] = row["home_team"] in used_teams
        row["away_used"] = row["away_team"] in used_teams
        row["home_venue_used"] = (row["home_team"], "home") in used_team_venues
        row["away_venue_used"] = (row["away_team"], "away") in used_team_venues
        row["result_label"] = _result_label(row)
        output.append(row)
    return output


def summarize_challenge(rows: list[dict], actual_picks: list[dict]) -> dict[str, object]:
    """Return compact season-management stats for the challenge page."""
    points = sum(pick["actual_points"] or 0 for pick in actual_picks)
    pending = sum(1 for pick in actual_picks if pick["actual_points"] is None)
    overrides = 0
    for row in rows:
        if row.get("is_override"):
            overrides += 1
    weeks = sorted({int(row["contest_week"]) for row in rows})
    picked_weeks = {int(pick["contest_week"]) for pick in actual_picks}
    next_open_week = next((week for week in weeks if week not in picked_weeks), None)
    close_calls = sum(1 for row in rows if row.get("is_close_call"))
    return {
        "fixtures": len(rows),
        "weeks": len(weeks),
        "submitted_picks": len(actual_picks),
        "pending_results": pending,
        "points": points,
        "overrides": overrides,
        "next_open_week": next_open_week,
        "close_calls": close_calls,
    }


def score_manual_pick(
    runtime_dir: Path,
    match_id: str,
    team: str,
) -> int | None:
    """Score a manual pick if the selected match already has a known result."""
    results = _read_optional_csv(runtime_dir / "data" / "processed" / "historical_matches.csv")
    if results.empty or "match_id" not in results:
        return None
    match = results[results["match_id"].astype(str) == str(match_id)]
    if match.empty:
        return None
    row = match.iloc[0]
    if pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")):
        return None
    if row["home_goals"] == row["away_goals"]:
        return 1
    if team == row["home_team"] and row["home_goals"] > row["away_goals"]:
        return 3
    if team == row["away_team"] and row["away_goals"] > row["home_goals"]:
        return 3
    return 0


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _result_label(row: dict) -> str:
    home_goals = row.get("home_goals")
    away_goals = row.get("away_goals")
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "Pending"
    if home_goals == away_goals:
        return f"Draw {int(home_goals)}-{int(away_goals)}"
    winner = row["home_team"] if home_goals > away_goals else row["away_team"]
    return f"{winner} won {int(home_goals)}-{int(away_goals)}"


def pick_points(pick_team: str, home_team: str, home_goals: float, away_goals: float) -> int:
    """Contest points a pick earns: 3 for a win, 1 for a draw, 0 for a defeat."""
    if home_goals == away_goals:
        return 1
    picked_home = pick_team == home_team
    home_won = home_goals > away_goals
    return 3 if picked_home == home_won else 0


def score_committed_picks(
    database: Database,
    season: str,
    matches: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Fill in ``actual_points`` for committed picks whose match has a final score.

    Committing a pick records the intent; nothing else was ever writing the
    result back, so a season's points stayed at zero however many rounds had
    been played. This resolves every stored pick against the processed match
    record and is safe to re-run — a pick already carrying the same score is
    left alone, and a corrected result overwrites the old one.
    """
    if matches is None:
        matches_path = PROCESSED_DIR / "historical_matches.csv"
        matches = _read_optional_csv(matches_path)
    if matches.empty or "match_id" not in matches:
        return {"scored": 0, "pending": len(database.list_actual_picks(season))}

    played = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    played["match_id"] = played["match_id"].astype(str)
    results = played.drop_duplicates(subset="match_id", keep="last").set_index("match_id")

    # Picks are stored with the match_id of whichever source produced the
    # fixture list, which is not always the id in the processed match record —
    # the official API numbers a match differently from football-data.co.uk.
    # Team, venue, and round identify the same match under either scheme.
    by_slot = _matches_by_slot(played, season)

    scored = 0
    pending = 0
    for pick in database.list_actual_picks(season):
        match = None
        match_id = str(pick.get("match_id", ""))
        if match_id in results.index:
            match = results.loc[match_id]
        else:
            match = by_slot.get((int(pick["contest_week"]), pick["team"], pick["venue"]))
        if match is None:
            pending += 1
            continue
        points = pick_points(
            pick["team"], match["home_team"], match["home_goals"], match["away_goals"]
        )
        if pick.get("actual_points") == points:
            continue
        database.set_pick_points(pick["season"], int(pick["contest_week"]), points)
        scored += 1
    return {"scored": scored, "pending": pending}


def _matches_by_slot(played: pd.DataFrame, season: str) -> dict[tuple[int, str, str], pd.Series]:
    """Index a season's completed matches by (contest week, team, venue)."""
    if "season" not in played or "contest_week" not in played:
        return {}
    season_matches = played[played["season"].astype(str).str.zfill(4) == str(season).zfill(4)]
    index: dict[tuple[int, str, str], pd.Series] = {}
    for match in season_matches.itertuples(index=False):
        week = int(match.contest_week)
        row = pd.Series(
            {
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_goals": match.home_goals,
                "away_goals": match.away_goals,
            }
        )
        index[(week, match.home_team, "home")] = row
        index[(week, match.away_team, "away")] = row
    return index
