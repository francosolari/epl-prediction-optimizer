"""SQLite-backed application state for predictions, picks, and results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Database:
    """Small JSON state store wrapped around SQLite."""

    def __init__(self, path: Path | str = "data/state.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    match_id TEXT PRIMARY KEY,
                    picked_team TEXT NOT NULL,
                    winner TEXT,
                    loser TEXT,
                    points INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS actual_picks (
                    season TEXT NOT NULL,
                    contest_week INTEGER NOT NULL,
                    match_id TEXT NOT NULL,
                    team TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    actual_points INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (season, contest_week)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    season TEXT NOT NULL,
                    training_matches INTEGER,
                    accuracy REAL,
                    log_loss REAL,
                    optimized_points INTEGER,
                    expected_points REAL,
                    winner_points INTEGER,
                    gap_to_winner INTEGER,
                    picks INTEGER,
                    model_params TEXT NOT NULL DEFAULT '{}',
                    feature_columns TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_matches (
                    match_id     TEXT PRIMARY KEY,
                    season       TEXT NOT NULL,
                    date         TEXT NOT NULL,
                    contest_week INTEGER,
                    home_team    TEXT NOT NULL,
                    away_team    TEXT NOT NULL,
                    home_goals   REAL,
                    away_goals   REAL,
                    home_elo     REAL,
                    away_elo     REAL,
                    home_shots_ot REAL,
                    away_shots_ot REAL,
                    home_odds    REAL,
                    draw_odds    REAL,
                    away_odds    REAL,
                    home_xg      REAL,
                    away_xg      REAL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_hm_season ON historical_matches (season)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS match_xg (
                    season    TEXT NOT NULL,
                    date      TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    home_xg   REAL NOT NULL,
                    away_xg   REAL NOT NULL,
                    PRIMARY KEY (season, home_team, away_team)
                )
                """
            )

    def upsert_historical_matches(self, matches: "pd.DataFrame") -> int:
        """Bulk upsert historical matches. Returns number of rows written."""
        import pandas as _pd
        cols = [
            "match_id", "season", "date", "contest_week",
            "home_team", "away_team", "home_goals", "away_goals",
            "home_elo", "away_elo", "home_shots_ot", "away_shots_ot",
            "home_odds", "draw_odds", "away_odds", "home_xg", "away_xg",
        ]
        frame = matches.reindex(columns=cols)
        records = [
            tuple(None if (isinstance(v, float) and v != v) else v for v in row)
            for row in frame.itertuples(index=False)
        ]
        with self._connect() as connection:
            connection.executemany(
                f"""INSERT OR REPLACE INTO historical_matches ({", ".join(cols)})
                    VALUES ({", ".join("?" * len(cols))})""",
                records,
            )
        return len(records)

    def load_historical_matches(self, season: str | None = None) -> "pd.DataFrame":
        """Load historical matches as a DataFrame, optionally filtered by season."""
        import pandas as _pd
        query = "SELECT * FROM historical_matches"
        params: tuple[Any, ...] = ()
        if season:
            query += " WHERE season = ?"
            params = (season,)
        query += " ORDER BY date"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return _pd.DataFrame()
        return _pd.DataFrame([dict(r) for r in rows])

    def upsert_match_xg(self, xg_frame: "pd.DataFrame") -> int:
        """Bulk upsert Understat xG rows (season, home_team, away_team are the key)."""
        records = [
            (str(r.season), str(r.date), r.home_team, r.away_team, float(r.home_xg), float(r.away_xg))
            for r in xg_frame.itertuples(index=False)
            if hasattr(r, "season")
        ]
        with self._connect() as connection:
            connection.executemany(
                """INSERT OR REPLACE INTO match_xg (season, date, home_team, away_team, home_xg, away_xg)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                records,
            )
        return len(records)

    def load_match_xg(self, season: str | None = None) -> "pd.DataFrame":
        """Load Understat xG data as a DataFrame."""
        import pandas as _pd
        query = "SELECT * FROM match_xg"
        params: tuple[Any, ...] = ()
        if season:
            query += " WHERE season = ?"
            params = (season,)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return _pd.DataFrame([dict(r) for r in rows]) if rows else _pd.DataFrame()

    def set_json(self, key: str, value: dict[str, Any] | list[dict[str, Any]]) -> None:
        """Persist JSON-serializable application state by key."""
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO run_state(key, value) VALUES (?, ?)",
                (key, json.dumps(value, default=str)),
            )

    def get_json(self, key: str, default: Any) -> Any:
        """Read JSON application state by key, returning the default when absent."""
        with self._connect() as connection:
            row = connection.execute("SELECT value FROM run_state WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def upsert_actual_pick(self, pick: dict[str, Any]) -> dict[str, Any]:
        """Store the user's actual submitted pick for a season/week."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO actual_picks (
                    season, contest_week, match_id, team, venue, notes, actual_points
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(season, contest_week) DO UPDATE SET
                    match_id = excluded.match_id,
                    team = excluded.team,
                    venue = excluded.venue,
                    notes = excluded.notes,
                    actual_points = excluded.actual_points,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    pick["season"],
                    int(pick["contest_week"]),
                    pick["match_id"],
                    pick["team"],
                    pick["venue"],
                    pick.get("notes", ""),
                    pick.get("actual_points"),
                ),
            )
        return pick

    def list_actual_picks(self, season: str | None = None) -> list[dict[str, Any]]:
        """List stored user picks, optionally scoped to a season."""
        query = "SELECT * FROM actual_picks"
        params: tuple[Any, ...] = ()
        if season:
            query += " WHERE season = ?"
            params = (season,)
        query += " ORDER BY season, contest_week"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def log_experiment_run(
        self,
        season: str,
        metrics: dict[str, Any],
        model_params: dict[str, Any],
        feature_columns: list[str],
        winner_points: int | None = None,
    ) -> int:
        """Record a backtest run with its model config and results. Returns the new row id."""
        pts = metrics.get("optimized_points")
        gap = (winner_points - pts) if (winner_points is not None and pts is not None) else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO experiment_runs (
                    season, training_matches, accuracy, log_loss,
                    optimized_points, expected_points, winner_points, gap_to_winner,
                    picks, model_params, feature_columns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    season,
                    metrics.get("training_matches"),
                    metrics.get("accuracy"),
                    metrics.get("log_loss"),
                    pts,
                    metrics.get("optimized_expected_points"),
                    winner_points,
                    gap,
                    metrics.get("optimized_picks"),
                    json.dumps(model_params, default=str),
                    json.dumps(feature_columns),
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def list_experiment_runs(self, season: str | None = None) -> list[dict[str, Any]]:
        """Return experiment history, newest first, optionally filtered by season."""
        query = "SELECT * FROM experiment_runs"
        params: tuple[Any, ...] = ()
        if season:
            query += " WHERE season = ?"
            params = (season,)
        query += " ORDER BY id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["model_params"] = json.loads(d["model_params"])
            d["feature_columns"] = json.loads(d["feature_columns"])
            result.append(d)
        return result
