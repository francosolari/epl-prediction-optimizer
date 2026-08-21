"""Exact counterfactual scenarios for a selected contest week."""

from __future__ import annotations

from typing import Any

import pandas as pd

from epl_prediction_optimizer.optimizer.solver import optimize_picks


def build_scenarios(
    candidates: pd.DataFrame,
    contest_week: int,
    committed_picks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-solve the season once for every eligible choice in ``contest_week``."""
    if candidates.empty:
        return []
    prior = [p for p in committed_picks if int(p["contest_week"]) < contest_week]
    baseline = optimize_picks(candidates, prior)
    baseline_ev = float(baseline["expected_points"].sum())
    baseline_by_week = {int(row.contest_week): row for row in baseline.itertuples(index=False)}
    committed_weeks = {int(p["contest_week"]) for p in prior}
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
            plan = optimize_picks(candidates, [*prior, *lookahead_assumptions, lock])
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
                }
            )
            continue
        total_ev = float(plan["expected_points"].sum())
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
    return sorted(
        scenarios,
        key=lambda x: (
            not x["feasible"],
            x.get("season_rank", 999),
            x["team"],
        ),
    )
