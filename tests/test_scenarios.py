import pandas as pd

from epl_prediction_optimizer.optimizer.scenarios import build_scenarios


def test_scenarios_reoptimize_and_rank_current_choices():
    rows = []
    for week, home, away, home_ev, away_ev in [
        (1, "A", "B", 2.0, 1.8),
        (2, "A", "C", 2.4, 1.2),
        (3, "B", "C", 2.3, 1.5),
    ]:
        for team, opponent, venue, ev in [
            (home, away, "home", home_ev),
            (away, home, "away", away_ev),
        ]:
            rows.append(
                {
                    "contest_week": week,
                    "match_id": f"m{week}",
                    "date": f"2026-08-{20 + week}",
                    "team": team,
                    "opponent": opponent,
                    "venue": venue,
                    "p_win": ev / 3,
                    "p_draw": 0.0,
                    "p_loss": 1 - ev / 3,
                    "expected_points": ev,
                }
            )
    scenarios = build_scenarios(pd.DataFrame(rows), 1, [])
    assert len(scenarios) == 2
    assert {item["immediate_rank"] for item in scenarios} == {1, 2}
    assert {item["season_rank"] for item in scenarios} == {1, 2}
    assert all(item["season_cost"] >= 0 for item in scenarios)
    assert all(item["horizon"] for item in scenarios)


def _round_candidates() -> "pd.DataFrame":
    import pandas as pd

    rows = []
    profiles = [(0.70, 0.18), (0.55, 0.22), (0.40, 0.28), (0.30, 0.30)]
    for week in range(1, 5):
        for index, (p_win, p_draw) in enumerate(profiles):
            rows.append(
                {
                    "contest_week": week,
                    "match_id": f"{week}-{index}",
                    "date": f"2026-09-{week:02d}",
                    "team": f"Team {index}",
                    "opponent": "Opponent",
                    "venue": "home" if week % 2 == index % 2 else "away",
                    "p_win": p_win,
                    "p_draw": p_draw,
                    "p_loss": 1 - p_win - p_draw,
                    "expected_points": 3 * p_win + p_draw,
                }
            )
    return pd.DataFrame(rows)


def test_scenarios_score_every_feasible_option():
    scenarios = build_scenarios(_round_candidates(), 1, [])
    feasible = [item for item in scenarios if item["feasible"]]
    assert feasible
    assert all(item["recommendation"] is not None for item in feasible)
    assert max(item["recommendation"] for item in feasible) == 100
    assert all(item["tier"] in {"equivalent", "slightly behind", "behind"} for item in feasible)


def test_win_chance_is_reported_when_a_field_is_supplied():
    field = [3, 5, 7, 9, 11]
    scenarios = build_scenarios(_round_candidates(), 1, [], field_totals=field)
    feasible = [item for item in scenarios if item["feasible"]]
    assert all(item["win_chance"] is not None for item in feasible)


def test_scores_always_come_from_season_expected_value():
    """Variance nudges win chance up as expected points fall; EV must lead."""
    scenarios = build_scenarios(_round_candidates(), 1, [], field_totals=[3, 5, 7, 9, 11])
    feasible = [item for item in scenarios if item["feasible"]]
    assert all(item["score_basis"] == "season expected value" for item in feasible)
    best = max(feasible, key=lambda item: item["recommendation"])
    assert best["season_rank"] == 1


def test_a_longshot_never_outranks_a_clearly_better_plan():
    """The failure this guards: a 22% pick scoring above a 51% one on variance."""
    scenarios = build_scenarios(_round_candidates(), 1, [], field_totals=[40, 45, 50])
    feasible = [item for item in scenarios if item["feasible"]]
    ordered = sorted(feasible, key=lambda item: -item["recommendation"])
    assert [item["season_rank"] for item in ordered] == sorted(
        item["season_rank"] for item in feasible
    )


def test_the_top_scored_option_is_the_top_season_ranked_option():
    scenarios = build_scenarios(_round_candidates(), 1, [], field_totals=[3, 5, 7, 9, 11])
    feasible = [item for item in scenarios if item["feasible"]]
    best_score = max(feasible, key=lambda item: item["recommendation"])
    assert best_score["season_rank"] == 1


def test_an_unreachable_field_still_scores_the_options():
    scenarios = build_scenarios(_round_candidates(), 1, [], field_totals=[400, 500])
    feasible = [item for item in scenarios if item["feasible"]]
    assert max(item["recommendation"] for item in feasible) == 100
