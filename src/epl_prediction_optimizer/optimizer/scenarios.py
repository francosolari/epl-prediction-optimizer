"""Exact counterfactual scenarios for a selected contest week."""

from __future__ import annotations

from typing import Any

import pandas as pd

from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.optimizer.tournament import win_probability_against_field


def build_scenarios(
    candidates: pd.DataFrame,
    contest_week: int,
    committed_picks: list[dict[str, Any]],
    field_totals: list[int] | None = None,
    points_banked: int = 0,
) -> list[dict[str, Any]]:
    """Re-solve the season once for every eligible choice in ``contest_week``.

    When ``field_totals`` are supplied — the finishing scores of a real
    completed field — each resulting plan is also scored on its chance of
    finishing first. That is the objective the contest actually pays for, and
    it folds this round's expected points, the rest of the season, and the
    spread of outcomes into one comparable number.
    """
    if candidates.empty:
        return []
    prior = [p for p in committed_picks if int(p["contest_week"]) < contest_week]
    # Selecting every team is a contest rule, so solve it hard. It only stops
    # being satisfiable once a round has passed with no pick recorded: that
    # round is gone from the candidate set and some team has nowhere left to
    # go. In that state the desk still has to show the best remaining plan, so
    # fall back once, for every solve, and mark the output as rule-breaking.
    coverage_unsatisfiable = False
    try:
        baseline = optimize_picks(candidates, prior)
    except ValueError:
        baseline = optimize_picks(candidates, prior, allow_incomplete_coverage=True)
        coverage_unsatisfiable = True
    baseline_ev = float(baseline["expected_points"].sum())
    baseline_uncovered = list(baseline.attrs.get("uncovered_teams", []))
    baseline_by_week = {int(row.contest_week): row for row in baseline.itertuples(index=False)}
    committed_weeks = {int(p["contest_week"]) for p in prior}
    decided_weeks = {int(p["contest_week"]) for p in committed_picks}
    lookahead_assumptions = [
        {
            "contest_week": int(row.contest_week),
            "match_id": str(row.match_id),
            "team": row.team,
            "venue": row.venue,
        }
        for row in baseline.itertuples(index=False)
        if int(row.contest_week) < contest_week and int(row.contest_week) not in committed_weeks
    ]
    scenarios: list[dict[str, Any]] = []
    current = candidates[candidates["contest_week"] == contest_week]
    for row in current.itertuples(index=False):
        lock = {
            "contest_week": contest_week,
            "match_id": str(row.match_id),
            "team": row.team,
            "venue": row.venue,
        }
        try:
            plan = optimize_picks(
                candidates,
                [*prior, *lookahead_assumptions, lock],
                allow_incomplete_coverage=coverage_unsatisfiable,
            )
        except ValueError as exc:
            scenarios.append(
                {
                    **lock,
                    "opponent": row.opponent,
                    "date": str(row.date),
                    "kickoff_utc": getattr(row, "kickoff_utc", None),
                    "p_win": float(row.p_win),
                    "expected_points": float(row.expected_points),
                    "feasible": False,
                    "reason": str(exc),
                    "win_chance": None,
                    "uncovered_teams": [],
                    "new_uncovered_teams": [],
                    "coverage_rule_unsatisfiable": coverage_unsatisfiable,
                }
            )
            continue
        total_ev = float(plan["expected_points"].sum())
        win_chance = None
        if field_totals:
            future = plan[~plan["contest_week"].isin(decided_weeks)]
            win_chance = win_probability_against_field(
                future, field_totals, points_so_far=points_banked
            )["win_probability"]
        # Empty on every hard solve; only populated in the rule-breaking
        # fallback, where the desk needs to name what could not be placed.
        uncovered = list(plan.attrs.get("uncovered_teams", []))
        changes = []
        cumulative = 0.0
        horizon = []
        for pick in plan.itertuples(index=False):
            cumulative += float(pick.expected_points)
            horizon.append({"week": int(pick.contest_week), "value": round(cumulative, 4)})
            old = baseline_by_week.get(int(pick.contest_week))
            if old is not None and old.team != pick.team:
                changes.append(
                    {
                        "week": int(pick.contest_week),
                        "from": old.team,
                        "to": pick.team,
                    }
                )
        scenarios.append(
            {
                **lock,
                "opponent": row.opponent,
                "date": str(row.date),
                "kickoff_utc": getattr(row, "kickoff_utc", None),
                "p_win": float(row.p_win),
                "expected_points": float(row.expected_points),
                "season_ev": round(total_ev, 6),
                "season_cost": round(max(0.0, baseline_ev - total_ev), 6),
                "first_affected_week": changes[0]["week"] if changes else None,
                "changes": changes,
                "horizon": horizon,
                "feasible": True,
                "reason": None,
                "win_chance": win_chance,
                "uncovered_teams": uncovered,
                "new_uncovered_teams": [
                    team for team in uncovered if team not in baseline_uncovered
                ],
                "coverage_rule_unsatisfiable": coverage_unsatisfiable,
                "lookahead_assumptions": lookahead_assumptions,
            }
        )
    feasible = [item for item in scenarios if item["feasible"]]
    immediate = sorted(feasible, key=lambda x: (-x["expected_points"], x["team"]))
    season = sorted(feasible, key=lambda x: (-x["season_ev"], -x["expected_points"], x["team"]))
    for rank, item in enumerate(immediate, 1):
        item["immediate_rank"] = rank
    for rank, item in enumerate(season, 1):
        item["season_rank"] = rank
    _score_recommendations(feasible)
    return sorted(
        scenarios,
        key=lambda x: (
            not x["feasible"],
            x.get("season_rank", 999),
            x["team"],
        ),
    )


def _score_recommendations(feasible: list[dict[str, Any]]) -> None:
    """Attach a 0-100 recommendation score and an equivalence tier.

    The score is driven by season expected value, not by the chance of
    winning. Those rank differently: a heavy underdog adds variance, and
    variance nudges win probability up even as expected points fall, so
    scoring on win chance alone recommends 22% shots over 51% ones. Backtests
    put the value of that extra spread at between nothing and five percent
    relative (see docs/model-evaluation.md), far less than the expected points
    it costs. Expected value leads; win chance is reported alongside.

    A quarter-point of season expected value costs about six points of score,
    so options within roughly 0.2 points of the best land at 95 or above and
    read as equivalent.
    """
    if not feasible:
        return
    best_ev = max(item["season_ev"] for item in feasible)
    for item in feasible:
        cost = best_ev - item["season_ev"]
        item["recommendation"] = max(0, round(100 - 25 * cost))
        item["score_basis"] = "season expected value"

    for item in feasible:
        score = item["recommendation"]
        if score >= 95:
            item["tier"] = "equivalent"
        elif score >= 85:
            item["tier"] = "slightly behind"
        else:
            item["tier"] = "behind"
