"""Feature engineering for EPL three-outcome probability models.

Stage A feature pipeline. All features are pre-match (no future leakage).
build_training_frame processes matches in date order, accumulating state.
build_fixture_features is used at prediction time with optional current-season context.
"""

from __future__ import annotations

import math

import pandas as pd

OUTCOME_HOME = "HOME_WIN"
OUTCOME_DRAW = "DRAW"
OUTCOME_AWAY = "AWAY_WIN"

FEATURE_COLUMNS = [
    # Elo (dominant signal)
    "elo_diff",
    "home_elo",
    "away_elo",
    # Rolling mixed-venue form (last 5)
    "home_form_points",
    "away_form_points",
    "home_goal_diff_form",
    "away_goal_diff_form",
    # Venue-specific form (last 5)
    "home_form_home_only",
    "away_form_away_only",
    "home_goal_diff_home_only",
    "away_goal_diff_away_only",
    # Momentum
    "home_win_streak",
    "away_win_streak",
    "home_form_decayed",
    "away_form_decayed",
    # In-season table position
    "home_season_points",
    "away_season_points",
    "home_season_gd",
    "away_season_gd",
    "home_season_matches",
    "away_season_matches",
    # Draw tendency (rolling 10-game draw rate)
    "home_draw_tendency",
    "away_draw_tendency",
    # Head-to-head record (last 5 meetings between this pair)
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_meetings",
    # --- Candidates for future inclusion ---
    # "home_shots_ot_form", "away_shots_ot_form",
    # "home_xg_form", "away_xg_form",
    # Judge these on log loss / RPS over several held-out seasons, not on
    # combined contest points — 38 picks a season cannot separate signal from
    # luck. See docs/model-evaluation.md.
    #
    # Market prices deliberately stay out of this list. Odds exist for the next
    # round or two, never for the whole remaining season, so a model fitted on
    # them would describe near and far fixtures on different terms and make the
    # season plan incoherent. They are pooled into the output instead, per
    # fixture and only where a price exists — see ml/market.py.
]


# Fallback values for features that cannot be computed from history. These match
# the cold-start values the accumulator itself emits, so a fixture with no prior
# record looks the same to the model in training and in production.
NEUTRAL_FEATURE_DEFAULTS: dict[str, float] = {
    "home_draw_tendency": 0.25,
    "away_draw_tendency": 0.25,
    # Optional signals stay NaN when unavailable; the gradient booster treats
    # missing natively rather than reading a zero as a real observation.
    "home_shots_ot_form": float("nan"),
    "away_shots_ot_form": float("nan"),
    "home_xg_form": float("nan"),
    "away_xg_form": float("nan"),
    "market_prob_home": float("nan"),
    "market_prob_draw": float("nan"),
    "market_prob_away": float("nan"),
}


def normalize_match_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Return historical matches with canonical dates, goals, and Elo columns."""
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    for column in ["home_goals", "away_goals"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["home_elo"] = pd.to_numeric(frame.get("home_elo", 1500), errors="coerce").fillna(1500)
    frame["away_elo"] = pd.to_numeric(frame.get("away_elo", 1500), errors="coerce").fillna(1500)
    return frame.sort_values("date").reset_index(drop=True)


def build_training_frame(matches: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Build model-ready rows with pre-match features and target outcomes.

    Processes matches in chronological order, accumulating:
    - Cross-venue and venue-specific rolling form
    - In-season table position (resets each season)
    - Draw tendency (rolling 10-game)
    - Head-to-head history for each fixture pair
    - Rolling shots on target (5-game) per team
    - Rolling xG (5-game) per team — NaN when unavailable
    - Betting market implied probabilities from B365 odds
    """
    frame = normalize_match_frame(matches)
    has_season = "season" in frame.columns
    has_match_id = "match_id" in frame.columns
    has_contest_week = "contest_week" in frame.columns

    # All-venue form: [(points, goal_diff), ...]
    team_history: dict[str, list[tuple[int, int]]] = {}
    # Venue-specific form
    team_home_history: dict[str, list[tuple[int, int]]] = {}
    team_away_history: dict[str, list[tuple[int, int]]] = {}
    # In-season: [cumulative_pts, cumulative_gd, matches_played]
    season_stats: dict[tuple[str, str], list[int]] = {}
    # H2H: canonical key (min, max) -> outcomes from min-team perspective ('W','D','L')
    h2h_history: dict[tuple[str, str], list[str]] = {}
    # Rolling shots on target: team -> list of (home_sot, away_sot) from team's perspective
    team_shots_ot: dict[str, list[float]] = {}
    # Rolling xG: team -> list of xg values
    team_xg: dict[str, list[float]] = {}

    rows: list[dict[str, object]] = []

    for match in frame.itertuples(index=False):
        home = match.home_team
        away = match.away_team
        season = str(getattr(match, "season", "unknown"))

        all_home = team_history.get(home, [])
        all_away = team_history.get(away, [])
        home_only = team_home_history.get(home, [])
        away_only = team_away_history.get(away, [])
        h_stats = season_stats.get((season, home), [0, 0, 0])
        a_stats = season_stats.get((season, away), [0, 0, 0])

        # Head-to-head
        h2h_key = (min(home, away), max(home, away))
        h2h = h2h_history.get(h2h_key, [])
        recent_h2h = h2h[-5:]
        n_h2h = len(recent_h2h)
        if n_h2h == 0:
            h2h_home_win_rate = 0.0
            h2h_draw_rate = 0.0
        else:
            if home == h2h_key[0]:
                h2h_home_wins = sum(1 for o in recent_h2h if o == "W")
                h2h_draws = sum(1 for o in recent_h2h if o == "D")
            else:
                h2h_home_wins = sum(1 for o in recent_h2h if o == "L")
                h2h_draws = sum(1 for o in recent_h2h if o == "D")
            h2h_home_win_rate = h2h_home_wins / n_h2h
            h2h_draw_rate = h2h_draws / n_h2h

        # Rolling shots on target (NaN when no history yet — HGBC handles natively)
        home_sot_hist = team_shots_ot.get(home, [])
        away_sot_hist = team_shots_ot.get(away, [])
        home_shots_ot_form = _rolling_mean_or_nan(home_sot_hist, rolling_window)
        away_shots_ot_form = _rolling_mean_or_nan(away_sot_hist, rolling_window)

        # Rolling xG (NaN when no xG data available — HGBC handles natively)
        home_xg_hist = team_xg.get(home, [])
        away_xg_hist = team_xg.get(away, [])
        home_xg_form = _rolling_mean_or_nan(home_xg_hist, rolling_window)
        away_xg_form = _rolling_mean_or_nan(away_xg_hist, rolling_window)

        # Betting market implied probabilities from B365 opening odds
        h_odds = getattr(match, "home_odds", None)
        d_odds = getattr(match, "draw_odds", None)
        a_odds = getattr(match, "away_odds", None)
        market_prob_home, market_prob_draw, market_prob_away = _implied_probs(
            h_odds, d_odds, a_odds
        )

        row: dict[str, object] = {
            "date": match.date,
            "home_team": home,
            "away_team": away,
            "home_elo": float(match.home_elo),
            "away_elo": float(match.away_elo),
            "elo_diff": float(match.home_elo) - float(match.away_elo),
            "home_form_points": _rolling_points(all_home, rolling_window),
            "away_form_points": _rolling_points(all_away, rolling_window),
            "home_goal_diff_form": _rolling_goal_diff(all_home, rolling_window),
            "away_goal_diff_form": _rolling_goal_diff(all_away, rolling_window),
            "home_form_home_only": _rolling_points(home_only, rolling_window),
            "away_form_away_only": _rolling_points(away_only, rolling_window),
            "home_goal_diff_home_only": _rolling_goal_diff(home_only, rolling_window),
            "away_goal_diff_away_only": _rolling_goal_diff(away_only, rolling_window),
            "home_win_streak": _current_streak(all_home),
            "away_win_streak": _current_streak(all_away),
            "home_form_decayed": _decayed_form(all_home, rolling_window),
            "away_form_decayed": _decayed_form(all_away, rolling_window),
            "home_season_points": float(h_stats[0]),
            "away_season_points": float(a_stats[0]),
            "home_season_gd": float(h_stats[1]),
            "away_season_gd": float(a_stats[1]),
            "home_season_matches": float(h_stats[2]),
            "away_season_matches": float(a_stats[2]),
            "home_draw_tendency": _draw_tendency(all_home),
            "away_draw_tendency": _draw_tendency(all_away),
            "h2h_home_win_rate": h2h_home_win_rate,
            "h2h_draw_rate": h2h_draw_rate,
            "h2h_meetings": float(n_h2h),
            "home_shots_ot_form": home_shots_ot_form,
            "away_shots_ot_form": away_shots_ot_form,
            "home_xg_form": home_xg_form,
            "away_xg_form": away_xg_form,
            "market_prob_home": market_prob_home,
            "market_prob_draw": market_prob_draw,
            "market_prob_away": market_prob_away,
            "outcome": _outcome(match.home_goals, match.away_goals),
        }
        if has_season:
            row["season"] = season
        if has_match_id:
            row["match_id"] = match.match_id
        if has_contest_week:
            row["contest_week"] = match.contest_week

        rows.append(row)

        if row["outcome"] is None:
            # Unplayed fixture: its features are emitted from state as it stands,
            # but it contributes nothing back to form, table, or head-to-head.
            continue

        home_pts = _points_for(match.home_goals, match.away_goals)
        away_pts = _points_for(match.away_goals, match.home_goals)
        home_gd = int(match.home_goals) - int(match.away_goals)

        team_history.setdefault(home, []).append((home_pts, home_gd))
        team_history.setdefault(away, []).append((away_pts, -home_gd))
        team_home_history.setdefault(home, []).append((home_pts, home_gd))
        team_away_history.setdefault(away, []).append((away_pts, -home_gd))

        h = season_stats.get((season, home), [0, 0, 0])
        season_stats[(season, home)] = [h[0] + home_pts, h[1] + home_gd, h[2] + 1]
        a = season_stats.get((season, away), [0, 0, 0])
        season_stats[(season, away)] = [a[0] + away_pts, a[1] - home_gd, a[2] + 1]

        # Update H2H
        if home_gd > 0:
            outcome = "W" if home == h2h_key[0] else "L"
        elif home_gd < 0:
            outcome = "L" if home == h2h_key[0] else "W"
        else:
            outcome = "D"
        h2h_history.setdefault(h2h_key, []).append(outcome)

        # Update rolling shots on target history (store team's own shots as attacker)
        for team, value in (
            (home, getattr(match, "home_shots_ot", None)),
            (away, getattr(match, "away_shots_ot", None)),
        ):
            if _is_observed(value):
                team_shots_ot.setdefault(team, []).append(float(value))

        # Update rolling xG history
        for team, value in (
            (home, getattr(match, "home_xg", None)),
            (away, getattr(match, "away_xg", None)),
        ):
            if _is_observed(value):
                team_xg.setdefault(team, []).append(float(value))

    return pd.DataFrame(rows)


def build_fixture_features(
    fixtures: pd.DataFrame,
    history: pd.DataFrame | None = None,
    season_stats: dict[str, list[int]] | None = None,
    fixture_season: str | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Build prediction-time features for upcoming fixtures.

    When ``history`` is supplied the fixtures are appended to the completed
    match record and run through the same accumulator that produces the
    training rows, so a fixture is described by exactly the features the model
    was fitted on. Without history only Elo and the supplied season table are
    populated and every form feature falls back to a neutral default, which is
    a materially weaker input — pass history whenever it exists.
    """
    if history is not None and not history.empty:
        return _fixture_features_from_history(
            fixtures,
            history,
            fixture_season=fixture_season,
            rolling_window=rolling_window,
        )

    frame = fixtures.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame["home_elo"] = pd.to_numeric(frame.get("home_elo", 1500), errors="coerce").fillna(1500)
    frame["away_elo"] = pd.to_numeric(frame.get("away_elo", 1500), errors="coerce").fillna(1500)
    frame["elo_diff"] = frame["home_elo"] - frame["away_elo"]

    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = NEUTRAL_FEATURE_DEFAULTS.get(column, 0.0)

    if season_stats:
        for idx, row in frame.iterrows():
            h = season_stats.get(row["home_team"], [0, 0, 0])
            a = season_stats.get(row["away_team"], [0, 0, 0])
            frame.at[idx, "home_season_points"] = float(h[0])
            frame.at[idx, "away_season_points"] = float(a[0])
            frame.at[idx, "home_season_gd"] = float(h[1])
            frame.at[idx, "away_season_gd"] = float(a[1])
            frame.at[idx, "home_season_matches"] = float(h[2])
            frame.at[idx, "away_season_matches"] = float(a[2])

    return frame


def _fixture_features_from_history(
    fixtures: pd.DataFrame,
    history: pd.DataFrame,
    fixture_season: str | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Run fixtures through the training accumulator seeded with completed matches."""
    completed = history.copy()
    if "home_goals" in completed and "away_goals" in completed:
        completed = completed.dropna(subset=["home_goals", "away_goals"])

    upcoming = fixtures.copy()
    upcoming["date"] = pd.to_datetime(upcoming["date"]).dt.date
    upcoming["home_goals"] = pd.NA
    upcoming["away_goals"] = pd.NA
    if "season" not in upcoming or upcoming["season"].isna().all():
        upcoming["season"] = fixture_season or _latest_season(completed)
    if "match_id" not in upcoming:
        upcoming["match_id"] = [f"fixture-{index}" for index in range(len(upcoming))]
    upcoming["match_id"] = upcoming["match_id"].astype(str)

    # Re-predicting a match that has already been played is normal here: the
    # season plan covers every round, past ones included. Drop the completed
    # copy so the fixture is never described by its own result.
    if "match_id" in completed.columns:
        completed = completed[~completed["match_id"].astype(str).isin(set(upcoming["match_id"]))]

    shared = [column for column in completed.columns if column in upcoming.columns]
    combined = pd.concat(
        [completed[shared], upcoming[shared]],
        ignore_index=True,
    )
    accumulated = build_training_frame(combined, rolling_window=rolling_window)
    accumulated["match_id"] = accumulated["match_id"].astype(str)

    fixture_ids = set(upcoming["match_id"])
    features = accumulated[accumulated["match_id"].isin(fixture_ids)]
    features = features.drop_duplicates(subset="match_id", keep="last")

    # Keep the caller's fixture metadata (kickoff time, official gameweek, Elo)
    # and overwrite only the modelled feature columns with accumulated values.
    by_match = features.set_index("match_id")
    output = upcoming.copy()
    for column in FEATURE_COLUMNS:
        default = NEUTRAL_FEATURE_DEFAULTS.get(column, 0.0)
        if column in by_match.columns:
            output[column] = output["match_id"].map(by_match[column]).fillna(default)
        else:
            output[column] = default
    return output.reset_index(drop=True)


def _latest_season(matches: pd.DataFrame) -> str:
    """Return the season code of the most recent completed match."""
    if "season" not in matches or matches.empty:
        return "unknown"
    ordered = matches.sort_values("date")
    return str(ordered["season"].iloc[-1])


# ---------------------------------------------------------------------------
# Outcome helpers
# ---------------------------------------------------------------------------

def _outcome(home_goals: object, away_goals: object) -> str | None:
    """Return the match outcome, or None for a fixture that has not been played."""
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return OUTCOME_HOME
    if home_goals < away_goals:
        return OUTCOME_AWAY
    return OUTCOME_DRAW


def _points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Form aggregators
# ---------------------------------------------------------------------------

def _rolling_points(history: list[tuple[int, int]], window: int) -> float:
    recent = history[-window:]
    return float(sum(pts for pts, _ in recent) / max(len(recent), 1))


def _rolling_goal_diff(history: list[tuple[int, int]], window: int) -> float:
    recent = history[-window:]
    return float(sum(gd for _, gd in recent) / max(len(recent), 1))


def _decayed_form(history: list[tuple[int, int]], window: int, decay: float = 0.7) -> float:
    """Exponentially weighted average of recent points (most recent = highest weight)."""
    recent = history[-window:]
    if not recent:
        return 0.0
    n = len(recent)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_w = sum(weights)
    return float(
        sum(w * pts for w, (pts, _) in zip(weights, recent, strict=True)) / total_w
    )


def _draw_tendency(history: list[tuple[int, int]], window: int = 10) -> float:
    """Rolling draw rate — proportion of recent games ending in a draw."""
    recent = history[-window:]
    if not recent:
        return 0.25  # league-average prior
    return float(sum(1 for pts, _ in recent if pts == 1) / len(recent))


def _current_streak(history: list[tuple[int, int]]) -> int:
    """Consecutive wins (positive) or losses (negative); draws break the streak."""
    if not history:
        return 0
    streak = 0
    for pts, _ in reversed(history):
        if pts == 3:
            if streak < 0:
                break
            streak += 1
        elif pts == 0:
            if streak > 0:
                break
            streak -= 1
        else:
            break
    return streak


def _is_observed(value: object) -> bool:
    """True when an optional per-match statistic was actually recorded."""
    if value is None:
        return False
    return not (isinstance(value, float) and math.isnan(value))


def _rolling_mean_or_nan(history: list[float], window: int) -> float:
    """Rolling mean of last `window` values, or NaN when history is empty."""

    recent = history[-window:]
    if not recent:
        return float("nan")
    return float(sum(recent) / len(recent))


def _implied_probs(
    home_odds: object,
    draw_odds: object,
    away_odds: object,
) -> tuple[float, float, float]:
    """Convert decimal B365 odds to normalised implied probabilities.

    Returns (0.0, 0.0, 0.0) when any odds value is missing or invalid.
    """
    _nan = float("nan")
    try:
        h = float(home_odds)  # type: ignore[arg-type]
        d = float(draw_odds)  # type: ignore[arg-type]
        a = float(away_odds)  # type: ignore[arg-type]
        if math.isnan(h) or math.isnan(d) or math.isnan(a):
            return _nan, _nan, _nan
        if h <= 0 or d <= 0 or a <= 0:
            return _nan, _nan, _nan
        raw_h, raw_d, raw_a = 1.0 / h, 1.0 / d, 1.0 / a
        total = raw_h + raw_d + raw_a
        return raw_h / total, raw_d / total, raw_a / total
    except (TypeError, ValueError, ZeroDivisionError):
        return _nan, _nan, _nan
