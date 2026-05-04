"""Integer-programming optimizer for season-long contest picks."""

from __future__ import annotations

import pandas as pd
import pulp


def optimize_picks(candidates: pd.DataFrame) -> pd.DataFrame:
    """Maximize expected points while enforcing contest usage constraints."""
    if candidates.empty:
        return candidates.copy()
    frame = candidates.reset_index(drop=True).copy()
    weeks = sorted(frame["contest_week"].unique())
    teams = sorted(frame["team"].unique())
    problem = pulp.LpProblem("epl_pick_optimizer", pulp.LpMaximize)
    variables = {
        index: pulp.LpVariable(f"pick_{index}", lowBound=0, upBound=1, cat="Binary")
        for index in frame.index
    }
    problem += pulp.lpSum(
        frame.loc[index, "expected_points"] * variables[index] for index in frame.index
    )
    for week in weeks:
        indexes = frame.index[frame["contest_week"] == week]
        problem += pulp.lpSum(variables[index] for index in indexes) == 1
    for team in teams:
        team_indexes = frame.index[frame["team"] == team]
        problem += pulp.lpSum(variables[index] for index in team_indexes) >= 1
        problem += pulp.lpSum(variables[index] for index in team_indexes) <= 2
        for venue in ["home", "away"]:
            venue_indexes = frame.index[(frame["team"] == team) & (frame["venue"] == venue)]
            problem += pulp.lpSum(variables[index] for index in venue_indexes) <= 1
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"No optimal pick plan found: {pulp.LpStatus[status]}")
    selected = [index for index, variable in variables.items() if variable.value() == 1]
    output = frame.loc[selected].sort_values("contest_week").reset_index(drop=True)
    return output
