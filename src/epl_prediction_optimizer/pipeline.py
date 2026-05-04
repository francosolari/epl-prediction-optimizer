"""End-to-end commands that connect refresh, training, prediction, and optimization."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, log_loss

from epl_prediction_optimizer.data.sources import (
    DEFAULT_SEASON_CODES,
    OPENFOOTBALL_SEASON_CODES,
    UNDERSTAT_SEASON_CODES,
    apply_official_gameweeks,
    attach_historical_elo,
    attach_understat_xg,
    download_historical_results,
    fetch_clubelo_history,
    fetch_openfootball_gameweeks,
    fetch_understat_xg,
    fetch_upcoming_fixtures,
    get_cached_pl_matches,
    read_historical_results,
)
from epl_prediction_optimizer.ml.features import FEATURE_COLUMNS, build_training_frame
from epl_prediction_optimizer.ml.model import ModelRun, apply_dixon_coles_correction, season_decay_weights, train_model
from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.paths import (
    ARTIFACT_DIR,
    EXPORT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_data_dirs,
)
from epl_prediction_optimizer.storage.database import Database


def refresh_data(
    seasons: list[str] | None = None,
    use_network: bool = False,
    force_current: bool = True,
) -> dict[str, str]:
    """Refresh processed historical match and fixture CSVs."""
    ensure_data_dirs()
    if use_network and seasons is None and (PROCESSED_DIR / "historical_matches.csv").exists():
        return refresh_current_season(force=force_current)
    seasons = seasons or (DEFAULT_SEASON_CODES if use_network else ["2526"])
    historical_frames = []
    if use_network:
        for season in seasons:
            try:
                historical_frames.append(
                    read_historical_results(
                        download_historical_results(
                            season,
                            RAW_DIR / "football-data" / f"{season}_E0.csv",
                            force=force_current and season == "2526",
                        ),
                        season,
                    )
                )
            except requests.RequestException:
                continue
        historical_frames = [
            _overlay_cached_official_gameweeks(frame, str(frame["season"].iloc[0]))
            for frame in historical_frames
        ]
    else:
        historical_frames.append(_sample_historical_results())
    historical = pd.concat(historical_frames, ignore_index=True)
    historical = _attach_elo_cache(historical) if use_network else historical
    fixtures = _fixtures_from_current_season(historical, "2526")
    if fixtures.empty:
        fixtures = fetch_upcoming_fixtures()
    if use_network:
        fixtures = _attach_elo_cache(fixtures)
    historical_path = PROCESSED_DIR / "historical_matches.csv"
    fixtures_path = PROCESSED_DIR / "fixtures.csv"
    historical.to_csv(historical_path, index=False)
    fixtures.to_csv(fixtures_path, index=False)
    seasons_path = PROCESSED_DIR / "available_seasons.csv"
    pd.DataFrame({"season": sorted(historical["season"].unique(), key=str)}).to_csv(
        seasons_path,
        index=False,
    )
    return {
        "historical": str(historical_path),
        "fixtures": str(fixtures_path),
        "seasons": str(seasons_path),
    }


def train_from_processed() -> ModelRun:
    """Train from processed historical CSVs and save model artifacts."""
    ensure_data_dirs()
    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv").dropna(
        subset=["home_goals", "away_goals"]
    )
    training = build_training_frame(matches)
    import datetime
    current_season = str(datetime.date.today().year)[-2:] + str(datetime.date.today().year + 1)[-2:]
    weights = season_decay_weights(training["season"], reference_season=current_season)
    model_run = train_model(training, sample_weight=weights)
    joblib.dump(model_run, ARTIFACT_DIR / "model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(model_run.metrics, indent=2),
        encoding="utf-8",
    )
    return model_run


def predict_from_processed(database: Database | None = None) -> pd.DataFrame:
    """Generate probabilities from saved fixtures and persist CSV/app state."""
    ensure_data_dirs()
    model_run: ModelRun = joblib.load(ARTIFACT_DIR / "model.joblib")
    fixtures = pd.read_csv(PROCESSED_DIR / "fixtures.csv")
    current_stats = _compute_current_season_stats()
    predictions = model_run.predict(fixtures, season_stats=current_stats)
    output_path = EXPORT_DIR / "fixture_probabilities.csv"
    predictions.to_csv(output_path, index=False)
    (database or Database()).set_json("predictions", predictions.to_dict(orient="records"))
    return predictions


def optimize_from_predictions(database: Database | None = None) -> pd.DataFrame:
    """Build candidates from probabilities and persist optimized picks."""
    ensure_data_dirs()
    predictions = pd.read_csv(EXPORT_DIR / "fixture_probabilities.csv")
    candidates = build_pick_candidates(predictions)
    picks = optimize_picks(candidates)
    output_path = EXPORT_DIR / "optimized_picks.csv"
    picks.to_csv(output_path, index=False)
    db = database or Database()
    db.set_json("picks", picks.to_dict(orient="records"))
    db.set_json("status", {"status": "ready", "optimized_picks": len(picks)})
    return picks


def run_all(use_network: bool = False) -> dict[str, int]:
    """Run the full offline or live-data pipeline."""
    refresh_data(use_network=use_network)
    train_from_processed()
    predictions = predict_from_processed()
    picks = optimize_from_predictions()
    return {"predictions": len(predictions), "picks": len(picks)}


def rebuild_processed_history() -> dict[str, int]:
    """Re-parse all cached raw CSVs into SQLite + historical_matches.csv.

    Fully offline — uses only locally cached files (raw CSVs, clubelo cache, understat cache).
    No network calls. Safe to run repeatedly; each run is idempotent.
    """
    ensure_data_dirs()
    raw_dir = RAW_DIR / "football-data"
    frames = []
    for season_code in DEFAULT_SEASON_CODES:
        csv_path = raw_dir / f"{season_code}_E0.csv"
        if not csv_path.exists():
            continue
        try:
            frame = read_historical_results(csv_path, season_code)
            # Prefer football-data.org official cache; fall back to openfootball
            official_cache = RAW_DIR / "football-data-org" / f"{season_code.zfill(4)}_PL_matches.csv"
            openfootball_cache = RAW_DIR / "openfootball" / f"{season_code}_PL_matchdays.csv"
            if official_cache.exists():
                import pandas as _pd
                official = _pd.read_csv(official_cache)
                frame = apply_official_gameweeks(frame, official)
            elif openfootball_cache.exists():
                import pandas as _pd
                official = _pd.read_csv(openfootball_cache)
                frame = apply_official_gameweeks(frame, official)
            frames.append(frame)
        except Exception:
            continue
    if not frames:
        return {"seasons": 0, "matches": 0}
    historical = pd.concat(frames, ignore_index=True)
    historical = _attach_elo_cache(historical)
    # Attach xG from local understat cache only (no network)
    historical = _attach_all_understat_xg(historical, network=False)
    # Write to SQLite
    db = Database()
    db.upsert_historical_matches(historical)
    # Also write CSV for backward compatibility with scripts that read it directly
    historical_path = PROCESSED_DIR / "historical_matches.csv"
    historical.to_csv(historical_path, index=False)
    pd.DataFrame({"season": sorted(historical["season"].unique(), key=str)}).to_csv(
        PROCESSED_DIR / "available_seasons.csv", index=False
    )
    return {"seasons": historical["season"].nunique(), "matches": len(historical)}


def refresh_full_history() -> dict[str, str]:
    """Download all known football-data.co.uk EPL seasons, attach ClubElo and Understat xG."""
    result = refresh_data(seasons=DEFAULT_SEASON_CODES, use_network=True)
    historical_path = PROCESSED_DIR / "historical_matches.csv"
    if historical_path.exists():
        historical = pd.read_csv(historical_path)
        historical = _attach_all_understat_xg(historical, network=True)
        historical.to_csv(historical_path, index=False)
        db = Database()
        db.upsert_historical_matches(historical)
    return result


def refresh_current_season(target_season: str = "2526", force: bool = True) -> dict[str, str]:
    """Refresh only the current season and merge it into cached full history."""
    ensure_data_dirs()
    historical_path = PROCESSED_DIR / "historical_matches.csv"
    existing = pd.read_csv(historical_path) if historical_path.exists() else pd.DataFrame()
    current = read_historical_results(
        download_historical_results(
            target_season,
            RAW_DIR / "football-data" / f"{target_season}_E0.csv",
            force=force,
        ),
        target_season,
    )
    official_current = _get_cached_official_season(target_season, force=force)
    current = apply_official_gameweeks(current, official_current)
    if not existing.empty and "season" in existing:
        prior_2526 = existing[existing["season"].astype(str).str.zfill(4) == target_season]
        if not prior_2526.empty:
            elo_cols = ["match_id", "home_elo", "away_elo"]
            current = current.drop(columns=["home_elo", "away_elo"], errors="ignore").merge(
                prior_2526[elo_cols], on="match_id", how="left"
            )
            current["home_elo"] = current["home_elo"].fillna(1500.0)
            current["away_elo"] = current["away_elo"].fillna(1500.0)
        existing = existing[existing["season"].astype(str).str.zfill(4) != target_season]
        historical = pd.concat([existing, current], ignore_index=True)
    else:
        historical = current
    fixtures = _upcoming_fixtures_from_official(official_current)
    if fixtures.empty:
        fixtures = _fixtures_from_current_season(historical, target_season)
    historical = _attach_all_understat_xg(historical)
    historical.to_csv(historical_path, index=False)
    fixtures_path = PROCESSED_DIR / "fixtures.csv"
    fixtures.to_csv(fixtures_path, index=False)
    seasons_path = PROCESSED_DIR / "available_seasons.csv"
    pd.DataFrame({"season": sorted(historical["season"].unique(), key=str)}).to_csv(
        seasons_path,
        index=False,
    )
    return {
        "historical": str(historical_path),
        "fixtures": str(fixtures_path),
        "seasons": str(seasons_path),
        "mode": "current-season",
        "gameweek_source": "football-data.org" if not official_current.empty else "date-bucket",
    }


WINNER_BENCHMARKS: dict[str, int] = {
    "2425": 86,
    "2324": 83,
    "2223": 78,
    # Older seasons — add when contest winner data is confirmed
}


def backtest_season(
    matches: pd.DataFrame,
    target_season: str = "2526",
    winner_points: int | None = None,
) -> dict[str, float | int]:
    """Train before the target season, predict completed target matches, and score picks.

    Features are computed over all seasons together so in-season stats accumulate
    correctly through the target season (no train/predict feature mismatch).
    """
    completed = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    completed["season_code"] = completed["season"].astype(str).str.zfill(4)
    train_matches = completed[completed["season_code"] != target_season]
    target_matches = completed[completed["season_code"] == target_season]
    if train_matches.empty or target_matches.empty:
        raise ValueError("Backtest requires completed training matches and target-season matches.")

    # Build features over all completed matches so in-season context accumulates properly,
    # then split by season for train vs evaluate.
    all_features = build_training_frame(completed)
    all_features["season_code"] = all_features["season"].astype(str).str.zfill(4)
    train_features = all_features[all_features["season_code"] != target_season]
    target_features = all_features[all_features["season_code"] == target_season].reset_index(drop=True)

    weights = season_decay_weights(train_features["season"], reference_season=target_season)
    model_run = train_model(train_features, sample_weight=weights)

    # Predict directly from pre-computed features (bypasses build_fixture_features).
    proba = model_run.estimator.predict_proba(target_features[FEATURE_COLUMNS])
    classes = list(model_run.estimator.classes_)
    import numpy as np
    proba_df = pd.DataFrame(proba, columns=classes)
    predictions = target_features[["match_id", "contest_week", "date", "home_team", "away_team"]].copy()
    predictions["p_home_win"] = proba_df.get("HOME_WIN", 0.0)
    predictions["p_draw"] = proba_df.get("DRAW", 0.0)
    predictions["p_away_win"] = proba_df.get("AWAY_WIN", 0.0)
    totals = predictions[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    predictions[["p_home_win", "p_draw", "p_away_win"]] = predictions[
        ["p_home_win", "p_draw", "p_away_win"]
    ].div(totals, axis=0)
    # DC correction improves draw calibration but can hurt Stage B pick selection.
    # Disabled in backtest until we validate it improves combined season points.
    # predictions = apply_dixon_coles_correction(predictions)

    actual = target_matches[
        ["match_id", "home_team", "away_team", "home_goals", "away_goals"]
    ].copy()
    evaluated = predictions.merge(actual, on=["match_id", "home_team", "away_team"])
    evaluated["actual"] = evaluated.apply(_actual_outcome, axis=1)
    probability_columns = ["p_home_win", "p_draw", "p_away_win"]
    log_loss_columns = ["p_away_win", "p_draw", "p_home_win"]
    class_order = ["AWAY_WIN", "DRAW", "HOME_WIN"]
    y_true = evaluated["actual"]
    y_pred = evaluated[probability_columns].idxmax(axis=1).map(
        {
            "p_home_win": "HOME_WIN",
            "p_draw": "DRAW",
            "p_away_win": "AWAY_WIN",
        }
    )
    candidates = build_pick_candidates(predictions)
    picks = optimize_picks(candidates)
    scored_picks = _score_optimized_picks(picks, evaluated)

    ensure_data_dirs()
    predictions.to_csv(EXPORT_DIR / f"{target_season}_backtest_probabilities.csv", index=False)
    scored_picks.to_csv(EXPORT_DIR / f"{target_season}_backtest_optimized_picks.csv", index=False)
    resolved_winner = winner_points if winner_points is not None else WINNER_BENCHMARKS.get(target_season)
    metrics: dict[str, float | int | str | None] = {
        "target_season": target_season,
        "training_matches": int(len(train_matches)),
        "target_matches": int(len(target_matches)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, evaluated[log_loss_columns], labels=class_order)),
        "optimized_picks": int(len(scored_picks)),
        "optimized_points": int(scored_picks["actual_points"].sum()),
        "optimized_expected_points": float(scored_picks["expected_points"].sum()),
        "winner_points": resolved_winner,
    }
    (ARTIFACT_DIR / f"{target_season}_backtest_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    db = Database()
    db.set_json("backtest", metrics)
    # Extract hyperparameters from the trained estimator for experiment tracking.
    try:
        base_params = model_run.estimator.estimator.get_params()
    except AttributeError:
        base_params = {}
    db.log_experiment_run(
        season=target_season,
        metrics=metrics,  # type: ignore[arg-type]
        model_params=base_params,
        feature_columns=FEATURE_COLUMNS,
        winner_points=resolved_winner,
    )
    return metrics


def backtest_from_processed(target_season: str = "2526") -> dict[str, float | int]:
    """Run a target-season backtest from processed historical data."""
    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")
    return backtest_season(matches, target_season)


def _sample_historical_results() -> pd.DataFrame:
    rows = []
    teams = ["Arsenal", "Everton", "Chelsea", "Fulham", "Liverpool", "Tottenham Hotspur"]
    for day in range(1, 31):
        rows.append(
            {
                "date": f"2025-08-{day:02d}",
                "home_team": teams[day % len(teams)],
                "away_team": teams[(day + 1) % len(teams)],
                "home_goals": 2 + (day % 3),
                "away_goals": day % 2,
                "home_elo": 1580 + day,
                "away_elo": 1500 - day,
                "season": "2425",
                "match_id": f"sample-2425-home-{day}",
                "contest_week": day,
            }
        )
        rows.append(
            {
                "date": f"2025-09-{day:02d}",
                "home_team": teams[(day + 2) % len(teams)],
                "away_team": teams[(day + 3) % len(teams)],
                "home_goals": day % 2,
                "away_goals": 2 + (day % 3),
                "home_elo": 1460 - day,
                "away_elo": 1580 + day,
                "season": "2425",
                "match_id": f"sample-2425-away-{day}",
                "contest_week": day,
            }
        )
        rows.append(
            {
                "date": f"2025-10-{day:02d}",
                "home_team": teams[(day + 4) % len(teams)],
                "away_team": teams[(day + 5) % len(teams)],
                "home_goals": 1,
                "away_goals": 1,
                "home_elo": 1500,
                "away_elo": 1500,
                "season": "2425",
                "match_id": f"sample-2425-draw-{day}",
                "contest_week": day,
            }
        )
    return pd.DataFrame(rows)


def _upcoming_fixtures_from_official(official: pd.DataFrame) -> pd.DataFrame:
    """Extract all matches from the official API data as fixtures (full season)."""
    if official.empty:
        return pd.DataFrame()
    frame = official.copy()
    frame = _attach_elo_cache(frame)
    return frame[["match_id", "contest_week", "date", "home_team", "away_team", "home_elo", "away_elo"]]


def _fixtures_from_current_season(historical: pd.DataFrame, target_season: str) -> pd.DataFrame:
    target = historical[historical["season"] == target_season].copy()
    if target.empty:
        return pd.DataFrame()
    return target[
        [
            "match_id",
            "contest_week",
            "date",
            "home_team",
            "away_team",
            "home_elo",
            "away_elo",
        ]
    ]


def _attach_elo_cache(matches: pd.DataFrame) -> pd.DataFrame:
    elo_path = PROCESSED_DIR / "clubelo_history.csv"
    if elo_path.exists():
        return attach_historical_elo(matches, pd.read_csv(elo_path))
    teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
    frames = []
    for team in teams:
        cache_path = RAW_DIR / "clubelo" / f"{team.replace('/', '-')}.csv"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            cached = pd.read_csv(cache_path)
            if not cached.empty:
                frames.append(cached)
            continue
        try:
            history = fetch_clubelo_history(team)
        except requests.RequestException:
            history = pd.DataFrame(columns=["date", "team", "elo"])
        history.to_csv(cache_path, index=False)
        if not history.empty:
            frames.append(history)
    if not frames:
        return matches
    elo_history = pd.concat(frames, ignore_index=True)
    elo_path.parent.mkdir(parents=True, exist_ok=True)
    elo_history.to_csv(elo_path, index=False)
    return attach_historical_elo(matches, elo_history)


def _overlay_cached_official_gameweeks(frame: pd.DataFrame, season_code: str) -> pd.DataFrame:
    official = _get_cached_official_season(season_code, force=False)
    return apply_official_gameweeks(frame, official)


def _get_cached_official_season(season_code: str, force: bool = False) -> pd.DataFrame:
    season_code = str(season_code).zfill(4)
    season_start_year = 2000 + int(season_code[:2])
    cache_path = RAW_DIR / "football-data-org" / f"{season_code}_PL_matches.csv"
    try:
        return get_cached_pl_matches(
            cache_path,
            season_start_year=season_start_year,
            force=force,
        )
    except requests.RequestException:
        return pd.DataFrame()


def _compute_current_season_stats(
    current_season: str = "2526",
) -> dict[str, list[int]]:
    """Compute cumulative in-season points/GD/matches for each team from processed history."""
    path = PROCESSED_DIR / "historical_matches.csv"
    if not path.exists():
        return {}
    matches = pd.read_csv(path).dropna(subset=["home_goals", "away_goals"])
    matches = matches[matches["season"].astype(str).str.zfill(4) == current_season]
    if matches.empty:
        return {}
    stats: dict[str, list[int]] = {}
    for row in matches.sort_values("date").itertuples(index=False):
        hg, ag = int(row.home_goals), int(row.away_goals)
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if hg == ag else 0)
        gd = hg - ag
        for team, pts, team_gd in [(row.home_team, h_pts, gd), (row.away_team, a_pts, -gd)]:
            s = stats.get(team, [0, 0, 0])
            stats[team] = [s[0] + pts, s[1] + team_gd, s[2] + 1]
    return stats


def _attach_all_understat_xg(historical: pd.DataFrame, network: bool = True) -> pd.DataFrame:
    """Attach Understat xG for all available seasons.

    network=True fetches missing seasons from understat.com and stores them in SQLite.
    network=False only uses locally cached CSV files (for fast offline rebuilds).
    """
    cache_dir = RAW_DIR / "understat"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = historical.drop(columns=["home_xg", "away_xg"], errors="ignore")

    all_xg_frames = []
    for season_code in UNDERSTAT_SEASON_CODES:
        cache_path = cache_dir / f"{season_code}_understat_xg.csv"
        if network:
            xg = fetch_understat_xg(season_code, cache_dir=cache_dir)
        elif cache_path.exists():
            try:
                xg = pd.read_csv(cache_path)
            except Exception:
                xg = pd.DataFrame()
        else:
            xg = pd.DataFrame()
        if not xg.empty:
            all_xg_frames.append(xg)

    if not all_xg_frames:
        frame["home_xg"] = float("nan")
        frame["away_xg"] = float("nan")
        return frame

    combined_xg = pd.concat(all_xg_frames, ignore_index=True)
    return attach_understat_xg(frame, combined_xg)


def _actual_outcome(row: pd.Series) -> str:
    if row["home_goals"] > row["away_goals"]:
        return "HOME_WIN"
    if row["home_goals"] < row["away_goals"]:
        return "AWAY_WIN"
    return "DRAW"


def _score_optimized_picks(picks: pd.DataFrame, evaluated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    actual_by_match = evaluated.set_index("match_id")
    for pick in picks.itertuples(index=False):
        match = actual_by_match.loc[pick.match_id]
        if match.home_goals == match.away_goals:
            points = 1
        elif pick.team == match.home_team and match.home_goals > match.away_goals:
            points = 3
        elif pick.team == match.away_team and match.away_goals > match.home_goals:
            points = 3
        else:
            points = 0
        row = pick._asdict()
        row["actual_points"] = points
        row["home_goals"] = int(match.home_goals)
        row["away_goals"] = int(match.away_goals)
        rows.append(row)
    return pd.DataFrame(rows)
