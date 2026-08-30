"""Plan for winning the contest rather than for the best average season.

Maximising expected points produces the plan with the best *typical* outcome.
That is not the same as the plan most likely to finish first. A contest is won
by clearing whatever the best rival scores, and when that threshold sits well
above your expected total, the plan that most often reaches it is not the one
with the highest mean — it is the one whose distribution reaches furthest.

Everything here works on the exact distribution of a plan's season total.
A pick pays 3, 1, or 0, so the total over N picks is an integer under 3N and
its distribution is a convolution of N three-point distributions: cheap to
compute exactly, with no normal approximation and no simulation error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epl_prediction_optimizer.optimizer.solver import optimize_picks

WIN_POINTS = 3
DRAW_POINTS = 1
RISK_COLUMN = "risk_adjusted_points"
# Weight on a pick's point variance, in units of points. Zero reproduces the
# expected-points plan; larger values trade mean for spread.
DEFAULT_RISK_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)


def pick_point_variance(candidates: pd.DataFrame) -> pd.Series:
    """Variance of the points a single pick returns.

    With outcomes worth 3, 1, and 0, variance peaks on genuinely uncertain
    matches and is lowest on heavy favourites — which is why a plan that needs
    an unlikely total prefers the uncertain ones.
    """
    p_win = candidates["p_win"].astype(float)
    p_draw = candidates["p_draw"].astype(float)
    mean = WIN_POINTS * p_win + DRAW_POINTS * p_draw
    second_moment = (WIN_POINTS**2) * p_win + (DRAW_POINTS**2) * p_draw
    return second_moment - mean**2


def total_points_pmf(picks: pd.DataFrame) -> np.ndarray:
    """Exact probability mass function of a plan's season total.

    Picks are treated as independent. They are not perfectly so — the same
    weekend's results share conditions — but a plan takes one match per round,
    which removes the dominant source of dependence.
    """
    distribution = np.array([1.0])
    for pick in picks.itertuples(index=False):
        p_win = float(pick.p_win)
        p_draw = float(pick.p_draw)
        outcome = np.zeros(WIN_POINTS + 1)
        outcome[0] = max(0.0, 1.0 - p_win - p_draw)
        outcome[DRAW_POINTS] += p_draw
        outcome[WIN_POINTS] += p_win
        distribution = np.convolve(distribution, outcome)
    return distribution


def probability_at_least(picks: pd.DataFrame, target: float) -> float:
    """Probability the plan's season total reaches ``target`` points."""
    if picks.empty:
        return 0.0
    distribution = total_points_pmf(picks)
    threshold = int(np.ceil(target))
    if threshold <= 0:
        return 1.0
    if threshold >= len(distribution):
        return 0.0
    return float(distribution[threshold:].sum())


def plan_summary(picks: pd.DataFrame, target: float) -> dict[str, float]:
    """Mean, spread, and win probability for a plan."""
    if picks.empty:
        return {"expected_points": 0.0, "std_dev": 0.0, "win_probability": 0.0}
    distribution = total_points_pmf(picks)
    totals = np.arange(len(distribution))
    mean = float((distribution * totals).sum())
    variance = float((distribution * (totals - mean) ** 2).sum())
    return {
        "expected_points": mean,
        "std_dev": float(np.sqrt(variance)),
        "win_probability": probability_at_least(picks, target),
    }


def optimize_for_target(
    candidates: pd.DataFrame,
    target: float,
    locked_picks: list[dict[str, object]] | None = None,
    risk_grid: tuple[float, ...] = DEFAULT_RISK_GRID,
) -> pd.DataFrame:
    """Return the feasible plan most likely to reach ``target`` points.

    A plan's total variance is the sum of its picks' variances, so a linear
    objective of ``expected_points + risk * variance`` stays inside the same
    integer program. The risk weight that actually maximises the chance of
    clearing the target is not known in advance, so each weight on the grid is
    solved and the resulting plans are compared on their exact probability of
    reaching the target. The winning plan carries its risk weight, expected
    points, spread, and win probability in ``attrs``.
    """
    if candidates.empty:
        return candidates.copy()

    frame = candidates.reset_index(drop=True).copy()
    frame["point_variance"] = pick_point_variance(frame)

    best_plan: pd.DataFrame | None = None
    best_summary: dict[str, float] = {}
    best_risk = 0.0
    for risk in risk_grid:
        frame[RISK_COLUMN] = frame["expected_points"] + risk * frame["point_variance"]
        plan = optimize_picks(frame, locked_picks, objective_column=RISK_COLUMN)
        summary = plan_summary(plan, target)
        better = best_plan is None or summary["win_probability"] > best_summary["win_probability"]
        # Ties go to the higher mean: when the target is out of reach the win
        # probability can round to zero for every weight, and there is no
        # reason to give up points for spread that buys nothing.
        if not better and summary["win_probability"] == best_summary["win_probability"]:
            better = summary["expected_points"] > best_summary["expected_points"]
        if better:
            best_plan, best_summary, best_risk = plan, summary, risk

    assert best_plan is not None
    result = best_plan.drop(columns=[RISK_COLUMN, "point_variance"], errors="ignore")
    result.attrs = dict(best_plan.attrs)
    result.attrs.update({"risk_weight": best_risk, "target_points": float(target), **best_summary})
    return result


def win_probability_against_field(
    picks: pd.DataFrame,
    field_totals: list[int] | np.ndarray,
    rivals: int | None = None,
    points_so_far: int = 0,
) -> dict[str, float]:
    """Chance of finishing top given how a comparable field actually scored.

    Planning against a single target score assumes you know what it will take.
    You do not — the winning score is the best of however many rivals enter, and
    the only honest description of that is the spread a real field produced.
    ``field_totals`` are the final totals of a completed season of the same
    contest; each rival is treated as an independent draw from that empirical
    distribution.

    Ties are counted as losses, so this is the probability of finishing alone
    at the top rather than sharing it.
    """
    totals = np.asarray([total for total in field_totals if total is not None], dtype=float)
    if picks.empty or totals.size == 0:
        return {"win_probability": 0.0, "rivals": 0.0, "field_median": 0.0}

    rival_count = float(rivals if rivals is not None else totals.size - 1)
    distribution = total_points_pmf(picks)
    scores = np.arange(len(distribution)) + points_so_far
    # Share of the field a given final score beats outright.
    beaten = np.array([(totals < score).mean() for score in scores])
    return {
        "win_probability": float((distribution * beaten**rival_count).sum()),
        "rivals": rival_count,
        "field_median": float(np.median(totals)),
    }
