"""Market-implied probabilities and model/market probability blending.

Bookmaker prices are treated purely as a data source: decimal prices are
converted to a de-vigged probability distribution over HOME_WIN / DRAW /
AWAY_WIN and consumed like any other feature or prior. Nothing here places,
prices, or recommends a wager.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

PROBABILITY_COLUMNS = ["p_home_win", "p_draw", "p_away_win"]
MARKET_COLUMNS = ["market_prob_home", "market_prob_draw", "market_prob_away"]

DEVIG_METHODS = ("proportional", "power", "shin")


def implied_probabilities(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    method: str = "power",
) -> tuple[float, float, float]:
    """De-vig decimal odds into a normalised (home, draw, away) distribution.

    ``proportional`` divides raw inverse odds by the overround, which is known
    to leave favourite-longshot bias in place. ``power`` solves for the
    exponent k with sum(p_i ** k) == 1, and ``shin`` solves Shin's
    insider-trading model; both shrink longshots more than favourites.
    Returns NaNs when any price is missing or invalid.
    """
    nan = float("nan")
    try:
        prices = [float(home_odds), float(draw_odds), float(away_odds)]
    except (TypeError, ValueError):
        return nan, nan, nan
    if any(math.isnan(price) or price <= 1.0 for price in prices):
        return nan, nan, nan

    raw = [1.0 / price for price in prices]
    total = sum(raw)
    if total <= 0:
        return nan, nan, nan

    if method == "proportional":
        return tuple(value / total for value in raw)  # type: ignore[return-value]
    if method == "power":
        return _power_devig(raw)
    if method == "shin":
        return _shin_devig(raw)
    raise ValueError(f"Unknown de-vig method: {method}")


def _power_devig(raw: list[float]) -> tuple[float, float, float]:
    """Find k such that sum(raw_i ** k) == 1 by bisection on k >= 1.

    Each raw inverse-odds value is below 1, so sum(raw_i ** k) falls
    monotonically as k rises, and it shrinks longshots harder than favourites
    — which is the favourite-longshot bias that proportional scaling ignores.
    """
    if abs(sum(raw) - 1.0) < 1e-9:
        return tuple(raw)  # type: ignore[return-value]
    low, high = 1.0, 8.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if sum(value**mid for value in raw) > 1.0:
            low = mid
        else:
            high = mid
    k = (low + high) / 2.0
    adjusted = [value**k for value in raw]
    total = sum(adjusted)
    return tuple(value / total for value in adjusted)  # type: ignore[return-value]


def _shin_devig(raw: list[float]) -> tuple[float, float, float]:
    """Solve Shin's model for the insider fraction z by bisection on z in [0, 0.2)."""
    overround = sum(raw)
    if abs(overround - 1.0) < 1e-9:
        return tuple(raw)  # type: ignore[return-value]

    def probabilities(z: float) -> list[float]:
        return [
            (math.sqrt(z**2 + 4.0 * (1.0 - z) * value**2 / overround) - z) / (2.0 * (1.0 - z))
            for value in raw
        ]

    low, high = 0.0, 0.2
    for _ in range(60):
        mid = (low + high) / 2.0
        if sum(probabilities(mid)) > 1.0:
            low = mid
        else:
            high = mid
    result = probabilities((low + high) / 2.0)
    total = sum(result)
    return tuple(value / total for value in result)  # type: ignore[return-value]


def market_probability_frame(
    odds: pd.DataFrame,
    method: str = "power",
    home_column: str = "home_odds",
    draw_column: str = "draw_odds",
    away_column: str = "away_odds",
) -> pd.DataFrame:
    """Add market_prob_home/draw/away columns to a frame carrying decimal odds."""
    frame = odds.copy()
    if frame.empty:
        for column in ("market_prob_home", "market_prob_draw", "market_prob_away"):
            frame[column] = pd.Series(dtype=float)
        return frame
    probabilities = [
        implied_probabilities(row_home, row_draw, row_away, method=method)
        for row_home, row_draw, row_away in zip(
            frame.get(home_column, pd.Series(np.nan, index=frame.index)),
            frame.get(draw_column, pd.Series(np.nan, index=frame.index)),
            frame.get(away_column, pd.Series(np.nan, index=frame.index)),
            strict=False,
        )
    ]
    frame["market_prob_home"] = [value[0] for value in probabilities]
    frame["market_prob_draw"] = [value[1] for value in probabilities]
    frame["market_prob_away"] = [value[2] for value in probabilities]
    return frame


def sharpen_probabilities(predictions: pd.DataFrame, exponent: float) -> pd.DataFrame:
    """Raise each outcome probability to ``exponent`` and renormalise.

    An exponent above 1 makes the distribution more confident, below 1 flatter.
    One parameter shared across all fixtures preserves the ordering of outcomes
    within a match while moving the whole model onto a different scale — which
    is what is needed when only part of the season is priced and the two halves
    must remain comparable to the optimizer.
    """
    frame = predictions.copy()
    if exponent == 1.0:
        return frame
    values = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    scaled = np.clip(values, 1e-12, 1.0) ** exponent
    frame[PROBABILITY_COLUMNS] = scaled / scaled.sum(axis=1, keepdims=True)
    return frame


def blend_probabilities(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    weight: float,
    method: str = "power",
) -> pd.DataFrame:
    """Linearly pool model probabilities with market-implied probabilities.

    ``weight`` is the share given to the market (0.0 keeps the model
    untouched, 1.0 replaces it entirely). Rows with no matching priced fixture
    keep their model probabilities, so a season-long plan stays coherent when
    only the near rounds are priced. Adds a ``market_weight`` column recording
    how much market signal each row actually received.
    """
    frame = predictions.copy()
    frame["market_weight"] = 0.0
    if market.empty or weight <= 0.0:
        return frame.drop(columns=MARKET_COLUMNS, errors="ignore")

    priced = market_probability_frame(market, method=method)
    priced = priced.dropna(subset=MARKET_COLUMNS)
    if priced.empty:
        return frame.drop(columns=MARKET_COLUMNS, errors="ignore")

    # A caller may hand us a frame that already carries market columns from an
    # earlier join; they would collide with the ones added below and silently
    # get suffixed out of the way.
    left = frame.drop(columns=MARKET_COLUMNS, errors="ignore").copy()
    left["_row"] = range(len(left))
    left["_date"] = pd.to_datetime(left["date"], errors="coerce")
    right = priced.copy()
    right["_date"] = pd.to_datetime(right["date"], errors="coerce")
    right = right[["home_team", "away_team", "_date", *MARKET_COLUMNS]]

    # The two feeds disagree by up to a day: football-data.co.uk publishes the
    # local kickoff date, the fixtures API a UTC timestamp. A pairing recurs
    # every season and twice within one, so candidates outside that window are
    # dropped before de-duplicating rather than blanked — otherwise a meeting
    # from a decade ago can win the row.
    candidates = left[["_row", "home_team", "away_team", "_date"]].merge(
        right, on=["home_team", "away_team"], how="inner", suffixes=("", "_odds")
    )
    offset = (candidates["_date_odds"] - candidates["_date"]).abs()
    candidates = candidates[offset <= pd.Timedelta(days=1)]
    candidates = candidates.sort_values("_date_odds").drop_duplicates(subset="_row", keep="last")

    merged = left.merge(candidates[["_row", *MARKET_COLUMNS]], on="_row", how="left")

    matched = merged["market_prob_home"].notna()
    for model_column, market_column in zip(PROBABILITY_COLUMNS, MARKET_COLUMNS, strict=True):
        merged.loc[matched, model_column] = (1.0 - weight) * merged.loc[
            matched, model_column
        ] + weight * merged.loc[matched, market_column]
    merged.loc[matched, "market_weight"] = weight

    totals = merged[PROBABILITY_COLUMNS].sum(axis=1)
    merged[PROBABILITY_COLUMNS] = merged[PROBABILITY_COLUMNS].div(totals, axis=0)
    return merged.drop(columns=["_row", "_date", *MARKET_COLUMNS])
