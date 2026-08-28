"""The weekly choice must resolve to an instruction, not a table to interpret."""

from __future__ import annotations

from epl_prediction_optimizer.optimizer.verdict import decision_verdict


def _option(team: str, score: int, **extra) -> dict:
    return {
        "team": team,
        "feasible": True,
        "recommendation": score,
        "score_basis": "season expected value",
        "tier": "equivalent" if score >= 95 else "slightly behind" if score >= 85 else "behind",
        **extra,
    }


def test_no_feasible_options_gives_no_verdict() -> None:
    assert decision_verdict([]) is None
    assert decision_verdict([{"team": "Arsenal", "feasible": False}]) is None


def test_a_single_option_is_stated_as_such() -> None:
    verdict = decision_verdict([_option("Arsenal", 100)])
    assert verdict["band"] == "only"
    assert verdict["headline"] == "Take Arsenal"


def test_close_options_are_called_a_toss_up_and_both_named() -> None:
    verdict = decision_verdict([_option("Coventry City", 100), _option("Manchester United", 95)])
    assert verdict["band"] == "toss-up"
    assert verdict["team"] == "Coventry City"
    assert verdict["equivalent"] == ["Coventry City", "Manchester United"]
    assert "Manchester United" in verdict["guidance"]


def test_a_small_but_real_gap_is_a_slight_edge() -> None:
    verdict = decision_verdict([_option("Liverpool", 100), _option("Chelsea", 88)])
    assert verdict["band"] == "slight edge"
    assert "know something concrete" in verdict["guidance"]


def test_a_large_gap_is_stated_as_decisive() -> None:
    verdict = decision_verdict([_option("Arsenal", 100), _option("Burnley", 60)])
    assert verdict["band"] == "clear"
    assert "40 points clear" in verdict["guidance"]


def test_the_top_scoring_option_wins_regardless_of_list_order() -> None:
    verdict = decision_verdict([_option("Burnley", 60), _option("Arsenal", 100)])
    assert verdict["team"] == "Arsenal"


def test_options_without_a_score_are_ignored() -> None:
    verdict = decision_verdict(
        [_option("Arsenal", 100), {"team": "Hull City", "feasible": True, "recommendation": None}]
    )
    assert verdict["team"] == "Arsenal"


def test_a_committed_pick_that_matches_the_model_says_nothing() -> None:
    """The round is decided; repeating the recommendation is just noise."""
    scenarios = [_option("Coventry City", 100), _option("Manchester United", 95)]
    assert decision_verdict(scenarios, committed={"team": "Coventry City"}) is None


def test_a_committed_pick_inside_the_margin_is_reported_as_fine() -> None:
    scenarios = [_option("Coventry City", 100), _option("Manchester United", 96)]
    verdict = decision_verdict(scenarios, committed={"team": "Manchester United"})
    assert verdict["band"] == "committed-equal"
    assert verdict["headline"] == "Committed: Manchester United"
    assert "no reason to change" in verdict["guidance"]


def test_a_committed_pick_the_model_dislikes_is_flagged() -> None:
    scenarios = [_option("Coventry City", 100), _option("Hull City", 70)]
    verdict = decision_verdict(scenarios, committed={"team": "Hull City"})
    assert verdict["band"] == "committed-differs"
    assert "prefers Coventry City" in verdict["guidance"]
    assert "changeable until kickoff" in verdict["guidance"]


def test_a_committed_pick_not_among_the_options_is_ignored() -> None:
    scenarios = [_option("Coventry City", 100)]
    assert decision_verdict(scenarios, committed={"team": "Everton"}) is None
