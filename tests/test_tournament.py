"""Tests for planning against a target score rather than for the best average."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from epl_prediction_optimizer.optimizer.tournament import (
    optimize_for_target,
    pick_point_variance,
    plan_summary,
    probability_at_least,
    total_points_pmf,
    win_probability_against_field,
)


def _picks(rows: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contest_week": index + 1,
                "match_id": f"m{index}",
                "team": f"Team {index}",
                "opponent": "Opponent",
                "venue": "home",
                "p_win": p_win,
                "p_draw": p_draw,
                "p_loss": 1.0 - p_win - p_draw,
                "expected_points": 3 * p_win + p_draw,
            }
            for index, (p_win, p_draw) in enumerate(rows)
        ]
    )


def test_single_pick_distribution_matches_its_outcomes() -> None:
    pmf = total_points_pmf(_picks([(0.5, 0.2)]))
    assert pmf[0] == pytest.approx(0.3)
    assert pmf[1] == pytest.approx(0.2)
    assert pmf[3] == pytest.approx(0.5)


def test_distribution_is_a_valid_pmf_over_reachable_totals() -> None:
    picks = _picks([(0.5, 0.2), (0.4, 0.3), (0.6, 0.2)])
    pmf = total_points_pmf(picks)
    assert pmf.sum() == pytest.approx(1.0)
    assert len(pmf) == 3 * 3 + 1
    assert (pmf >= 0).all()


def test_mean_of_the_distribution_equals_summed_expected_points() -> None:
    picks = _picks([(0.5, 0.2), (0.4, 0.3), (0.6, 0.2)])
    pmf = total_points_pmf(picks)
    mean = float((pmf * np.arange(len(pmf))).sum())
    assert mean == pytest.approx(picks["expected_points"].sum())


def test_probability_at_least_covers_the_boundaries() -> None:
    picks = _picks([(0.5, 0.2), (0.4, 0.3)])
    assert probability_at_least(picks, 0) == pytest.approx(1.0)
    assert probability_at_least(picks, 7) == 0.0
    assert probability_at_least(picks, 6) == pytest.approx(0.5 * 0.4)


def test_variance_peaks_on_uncertain_matches() -> None:
    variance = pick_point_variance(_picks([(0.9, 0.05), (0.45, 0.25)]))
    assert variance.iloc[1] > variance.iloc[0]


def _weekly_candidates() -> pd.DataFrame:
    """Four rounds, four teams, one clear favourite and one coin flip per round."""
    rows = []
    profiles = [(0.70, 0.18), (0.45, 0.25), (0.40, 0.28), (0.35, 0.30)]
    for week in range(1, 5):
        for index, (p_win, p_draw) in enumerate(profiles):
            rows.append(
                {
                    "contest_week": week,
                    "match_id": f"{week}-{index}",
                    "team": f"Team {index}",
                    "opponent": "Opponent",
                    "venue": "home" if week % 2 == index % 2 else "away",
                    "p_win": p_win,
                    "p_draw": p_draw,
                    "p_loss": 1.0 - p_win - p_draw,
                    "expected_points": 3 * p_win + p_draw,
                }
            )
    return pd.DataFrame(rows)


def test_a_reachable_target_keeps_the_expected_points_plan() -> None:
    """With the target below the mean there is nothing to gain from spread."""
    candidates = _weekly_candidates()
    plan = optimize_for_target(candidates, target=1.0)
    assert plan.attrs["risk_weight"] == 0.0
    assert plan.attrs["win_probability"] == pytest.approx(1.0, abs=0.01)


def test_planning_for_a_target_never_lowers_its_own_win_probability() -> None:
    """Zero is on the risk grid, so the result is at least the plain plan."""
    from epl_prediction_optimizer.optimizer.solver import optimize_picks

    candidates = _weekly_candidates()
    target = 9.0
    baseline = optimize_picks(candidates)
    tournament = optimize_for_target(candidates, target=target)
    assert tournament.attrs["win_probability"] >= plan_summary(baseline, target)[
        "win_probability"
    ] - 1e-12


def test_the_plan_still_obeys_every_contest_rule() -> None:
    candidates = _weekly_candidates()
    plan = optimize_for_target(candidates, target=9.0)
    assert plan["contest_week"].nunique() == 4
    assert plan.groupby("team").size().max() <= 2
    assert plan.groupby(["team", "venue"]).size().max() <= 1
    assert plan.attrs["uncovered_teams"] == []


def test_summary_reports_mean_spread_and_win_probability() -> None:
    summary = plan_summary(_picks([(0.5, 0.2), (0.4, 0.3)]), target=4)
    assert summary["expected_points"] == pytest.approx(1.7 + 1.5)
    assert summary["std_dev"] > 0
    assert 0.0 <= summary["win_probability"] <= 1.0


def test_win_probability_uses_the_field_that_actually_entered() -> None:
    """A plan that beats most of a weak field wins more often than against a strong one."""
    picks = _picks([(0.6, 0.2)] * 6)
    weak = [2, 4, 5, 6, 7, 8]
    strong = [16, 17, 17, 18, 18, 18]
    against_weak = win_probability_against_field(picks, weak)
    against_strong = win_probability_against_field(picks, strong)
    assert against_weak["win_probability"] > against_strong["win_probability"]
    assert against_weak["rivals"] == 5


def test_more_rivals_lower_the_chance_of_finishing_top() -> None:
    picks = _picks([(0.6, 0.2)] * 6)
    field = [8, 10, 12, 14]
    few = win_probability_against_field(picks, field, rivals=3)
    many = win_probability_against_field(picks, field, rivals=60)
    assert few["win_probability"] > many["win_probability"]


def test_points_already_banked_count_toward_the_final_total() -> None:
    picks = _picks([(0.5, 0.2)] * 3)
    field = [10, 11, 12]
    behind = win_probability_against_field(picks, field, points_so_far=0)
    ahead = win_probability_against_field(picks, field, points_so_far=9)
    assert ahead["win_probability"] > behind["win_probability"]


def test_an_empty_field_gives_no_estimate() -> None:
    assert win_probability_against_field(_picks([(0.5, 0.2)]), [])["win_probability"] == 0.0
