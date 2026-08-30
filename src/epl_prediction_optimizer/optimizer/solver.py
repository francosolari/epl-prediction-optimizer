"""Integer-programming optimizer for season-long contest picks."""

from __future__ import annotations

import pandas as pd
import pulp

# Only reachable when a caller explicitly allows incomplete coverage. Leaving a
# team out must still cost more than any plan can gain — a pick is worth at most
# 3 points across fewer than 40 rounds — so the solver drops a team only when
# there is genuinely no way to place it.
UNCOVERED_TEAM_PENALTY = 1000.0


def optimize_picks(
    candidates: pd.DataFrame,
    locked_picks: list[dict[str, object]] | None = None,
    objective_column: str = "expected_points",
    allow_incomplete_coverage: bool = False,
) -> pd.DataFrame:
    """Maximize a per-candidate value while enforcing contest rules and locks.

    ``objective_column`` names the value being maximised. It defaults to
    expected points; the tournament objective substitutes a risk-adjusted value
    while leaving every contest constraint untouched.

    Every contest rule is hard, including selecting each team at least once:
    a plan that breaks a rule is not a plan, so an unsatisfiable request raises
    rather than quietly returning something ineligible.

    ``allow_incomplete_coverage`` exists only for the state where the rule can
    no longer be met by any plan — a round that passed without a pick has left
    the candidate set, so some team has nowhere left to go. Callers use it to
    show a best-effort plan *after* a hard solve has failed, and must present
    the result as rule-breaking. The teams that could not be placed are listed
    in ``result.attrs["uncovered_teams"]``, which is empty on every hard solve.
    """
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
    uncovered = {
        team: pulp.LpVariable(f"uncovered_{position}", lowBound=0, upBound=1, cat="Binary")
        for position, team in enumerate(teams)
    }
    if not allow_incomplete_coverage:
        for slack in uncovered.values():
            slack.upBound = 0
    problem += pulp.lpSum(
        frame.loc[index, objective_column] * variables[index] for index in frame.index
    ) - pulp.lpSum(UNCOVERED_TEAM_PENALTY * slack for slack in uncovered.values())
    for week in weeks:
        indexes = frame.index[frame["contest_week"] == week]
        problem += pulp.lpSum(variables[index] for index in indexes) == 1
    for team in teams:
        team_indexes = frame.index[frame["team"] == team]
        problem += pulp.lpSum(variables[index] for index in team_indexes) + uncovered[team] >= 1
        problem += pulp.lpSum(variables[index] for index in team_indexes) <= 2
        for venue in ["home", "away"]:
            venue_indexes = frame.index[(frame["team"] == team) & (frame["venue"] == venue)]
            problem += pulp.lpSum(variables[index] for index in venue_indexes) <= 1
    for locked in locked_picks or []:
        indexes = frame.index[
            (frame["contest_week"] == int(locked["contest_week"]))
            & (frame["match_id"].astype(str) == str(locked["match_id"]))
            & (frame["team"] == str(locked["team"]))
            & (frame["venue"] == str(locked["venue"]))
        ]
        if len(indexes) != 1:
            raise ValueError(
                f"Locked pick is not an eligible candidate: week {locked['contest_week']} "
                f"{locked['team']}"
            )
        problem += variables[int(indexes[0])] == 1
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"No optimal pick plan found: {pulp.LpStatus[status]}")
    selected = [index for index, variable in variables.items() if variable.value() == 1]
    output = frame.loc[selected].sort_values("contest_week").reset_index(drop=True)
    output.attrs["uncovered_teams"] = sorted(
        team for team, slack in uncovered.items() if slack.value() == 1
    )
    return output
