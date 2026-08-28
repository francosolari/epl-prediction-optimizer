"""Tests for market-implied probabilities and model/market blending."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from epl_prediction_optimizer.ml.market import (
    DEVIG_METHODS,
    blend_probabilities,
    implied_probabilities,
    sharpen_probabilities,
)


@pytest.mark.parametrize("method", DEVIG_METHODS)
def test_devig_returns_a_normalised_distribution(method: str) -> None:
    probabilities = implied_probabilities(1.36, 5.0, 8.5, method=method)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
    assert all(0.0 < value < 1.0 for value in probabilities)


@pytest.mark.parametrize("method", DEVIG_METHODS)
def test_devig_preserves_favourite_ordering(method: str) -> None:
    home, draw, away = implied_probabilities(1.36, 5.0, 8.5, method=method)
    assert home > draw > away


def test_bias_corrections_shrink_longshots_relative_to_proportional() -> None:
    """Power and Shin should take more off the longshot than flat scaling does."""
    flat = implied_probabilities(1.36, 5.0, 8.5, method="proportional")
    power = implied_probabilities(1.36, 5.0, 8.5, method="power")
    shin = implied_probabilities(1.36, 5.0, 8.5, method="shin")
    assert power[2] < flat[2]
    assert shin[2] < flat[2]
    assert power[0] > flat[0]


@pytest.mark.parametrize("method", DEVIG_METHODS)
def test_fair_odds_are_returned_unchanged(method: str) -> None:
    probabilities = implied_probabilities(3.0, 3.0, 3.0, method=method)
    assert all(math.isclose(value, 1 / 3, abs_tol=1e-6) for value in probabilities)


@pytest.mark.parametrize("odds", [(0.0, 3.0, 3.0), (None, 3.0, 3.0), (float("nan"), 3.0, 3.0)])
def test_invalid_prices_produce_nan(odds: tuple[object, object, object]) -> None:
    assert all(math.isnan(value) for value in implied_probabilities(*odds))


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["1", "2"],
            "date": ["2026-08-22", "2026-08-29"],
            "home_team": ["Hull City", "Everton"],
            "away_team": ["Manchester United", "Chelsea"],
            "p_home_win": [0.30, 0.40],
            "p_draw": [0.20, 0.25],
            "p_away_win": [0.50, 0.35],
        }
    )


def _odds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-08-22"],
            "home_team": ["Hull City"],
            "away_team": ["Manchester United"],
            "home_odds": [8.5],
            "draw_odds": [5.0],
            "away_odds": [1.36],
        }
    )


def test_blend_only_moves_priced_fixtures() -> None:
    blended = blend_probabilities(_predictions(), _odds(), weight=0.5)
    assert blended.loc[0, "market_weight"] == 0.5
    assert blended.loc[1, "market_weight"] == 0.0
    # The unpriced fixture keeps pure model probabilities.
    assert blended.loc[1, "p_home_win"] == pytest.approx(0.40)
    # The priced fixture moves toward the heavily favoured away side.
    assert blended.loc[0, "p_away_win"] > 0.50


def test_blend_rows_stay_normalised() -> None:
    blended = blend_probabilities(_predictions(), _odds(), weight=0.7)
    totals = blended[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
    assert totals.round(9).eq(1.0).all()


def test_zero_weight_is_a_no_op() -> None:
    blended = blend_probabilities(_predictions(), _odds(), weight=0.0)
    pd.testing.assert_frame_equal(
        blended[["p_home_win", "p_draw", "p_away_win"]],
        _predictions()[["p_home_win", "p_draw", "p_away_win"]],
    )


def test_missing_market_data_leaves_predictions_untouched() -> None:
    blended = blend_probabilities(_predictions(), pd.DataFrame(), weight=0.6)
    assert blended["market_weight"].eq(0.0).all()


def test_blend_tolerates_a_one_day_feed_offset() -> None:
    """The odds feed publishes local kickoff dates; predictions carry UTC dates."""
    odds = _odds()
    odds.loc[0, "date"] = "2026-08-23"
    blended = blend_probabilities(_predictions(), odds, weight=0.5)
    assert blended.loc[0, "market_weight"] == 0.5


def test_blend_ignores_the_same_pairing_in_another_season() -> None:
    odds = _odds()
    odds.loc[0, "date"] = "2025-08-22"
    blended = blend_probabilities(_predictions(), odds, weight=0.5)
    assert blended["market_weight"].eq(0.0).all()


def test_blend_preserves_row_order_and_count() -> None:
    predictions = _predictions()
    blended = blend_probabilities(predictions, _odds(), weight=0.5)
    assert list(blended["match_id"]) == list(predictions["match_id"])


def test_blend_survives_predictions_that_already_carry_market_columns() -> None:
    """Re-blending a frame from an earlier join must not collide on columns."""
    predictions = _predictions()
    once = blend_probabilities(predictions, _odds(), weight=0.5)
    once["market_prob_home"] = 0.9  # a stale column left over from analysis code
    twice = blend_probabilities(once, _odds(), weight=0.5)
    assert twice.loc[0, "market_weight"] == 0.5
    assert "market_prob_home" not in twice.columns


def test_sharpening_concentrates_probability_without_reordering() -> None:
    frame = _predictions()
    sharper = sharpen_probabilities(frame, 2.0)
    flatter = sharpen_probabilities(frame, 0.5)
    assert sharper.loc[0, "p_away_win"] > frame.loc[0, "p_away_win"]
    assert flatter.loc[0, "p_away_win"] < frame.loc[0, "p_away_win"]
    for out in (sharper, flatter):
        row = out.loc[0, ["p_home_win", "p_draw", "p_away_win"]]
        assert row.idxmax() == "p_away_win"
        assert row.sum() == pytest.approx(1.0)


def test_sharpening_by_one_is_a_no_op() -> None:
    frame = _predictions()
    pd.testing.assert_frame_equal(sharpen_probabilities(frame, 1.0), frame)
