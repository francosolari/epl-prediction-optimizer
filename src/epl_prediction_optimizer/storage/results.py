"""Result scoring helpers for EPL pick outcomes."""

from __future__ import annotations


def score_pick(picked_team: str, winner: str | None, loser: str | None) -> int:
    """Score a pick as 3 for a win, 1 for a draw, and 0 for a loss."""
    if winner is None:
        return 1
    if picked_team == winner:
        return 3
    if loser is None or picked_team == loser:
        return 0
    raise ValueError("Picked team must be winner or loser unless the match was a draw.")
