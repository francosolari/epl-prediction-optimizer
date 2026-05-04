"""Candidate pick generation from match probability rows."""

from __future__ import annotations

import pandas as pd


def build_pick_candidates(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Create one selectable team row per weekend fixture side."""
    frame = probabilities.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame[pd.to_datetime(frame["date"]).dt.dayofweek.isin([5, 6])]
    rows: list[dict[str, object]] = []
    for match in frame.itertuples(index=False):
        rows.append(
            {
                "contest_week": int(match.contest_week),
                "match_id": match.match_id,
                "date": match.date,
                "team": match.home_team,
                "opponent": match.away_team,
                "venue": "home",
                "p_win": float(match.p_home_win),
                "p_draw": float(match.p_draw),
                "p_loss": float(match.p_away_win),
                "expected_points": round(3 * float(match.p_home_win) + float(match.p_draw), 6),
            }
        )
        rows.append(
            {
                "contest_week": int(match.contest_week),
                "match_id": match.match_id,
                "date": match.date,
                "team": match.away_team,
                "opponent": match.home_team,
                "venue": "away",
                "p_win": float(match.p_away_win),
                "p_draw": float(match.p_draw),
                "p_loss": float(match.p_home_win),
                "expected_points": round(3 * float(match.p_away_win) + float(match.p_draw), 6),
            }
        )
    return pd.DataFrame(rows)
