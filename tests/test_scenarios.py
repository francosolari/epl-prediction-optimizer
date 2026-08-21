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
