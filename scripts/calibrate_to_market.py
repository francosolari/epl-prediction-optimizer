#!/usr/bin/env python3
"""Measure and correct the sharpness gap between this model and the market.

Only the next round or two is ever priced, so a season plan mixes blended
probabilities for near fixtures with raw model probabilities for far ones. If
the model is systematically flatter than the market, the two halves are not on
the same scale: expected points for priced rounds sit higher than for unpriced
rounds for no footballing reason, and the optimizer front-loads its best teams
into whichever half is inflated.

This fits one exponent k, applied as p_i ** k renormalised, that brings model
probabilities onto the market's scale. It is fitted against market
probabilities rather than outcomes: a settled match provides one bit of
evidence, whereas its price provides a full distribution, so the target is far
lower variance per fixture. The exponent is fitted on earlier seasons and
scored on later ones against real results.

Usage:
    uv run scripts/calibrate_to_market.py
    uv run scripts/calibrate_to_market.py --fit 1819 1920 2021 --test 2223 2324 2425 2526
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from backtest_production_path import replay_season  # noqa: E402

from epl_prediction_optimizer.data.sources import market_odds_from_matches
from epl_prediction_optimizer.ml.market import (
    PROBABILITY_COLUMNS,
    blend_probabilities,
    market_probability_frame,
    sharpen_probabilities,
)
from epl_prediction_optimizer.paths import ARTIFACT_DIR, PROCESSED_DIR
from epl_prediction_optimizer.scoring import probability_metrics

DEFAULT_FIT = ["1819", "1920", "2021", "2122"]
DEFAULT_TEST = ["2223", "2324", "2425", "2526"]
MARKET_COLUMNS = ["market_prob_home", "market_prob_draw", "market_prob_away"]


def with_market(evaluated: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Attach de-vigged market probabilities to replayed predictions."""
    priced = market_probability_frame(odds, method="power").dropna(subset=MARKET_COLUMNS)
    frame = evaluated.copy()
    frame["_date"] = pd.to_datetime(frame["date"])
    priced = priced.copy()
    priced["_date"] = pd.to_datetime(priced["date"])
    merged = frame.merge(
        priced[["home_team", "away_team", "_date", *MARKET_COLUMNS]],
        on=["home_team", "away_team"],
        how="left",
        suffixes=("", "_odds"),
    )
    offset = (merged["_date_odds"] - merged["_date"]).abs()
    merged = merged[offset <= pd.Timedelta(days=1)]
    return merged.dropna(subset=MARKET_COLUMNS).reset_index(drop=True)


def sharpness(frame: pd.DataFrame, columns: list[str]) -> float:
    """Mean probability assigned to the most likely outcome."""
    return float(frame[columns].to_numpy(dtype=float).max(axis=1).mean())


def cross_entropy_to_market(frame: pd.DataFrame, k: float) -> float:
    """Cross-entropy of the market distribution under the k-scaled model."""
    scaled = sharpen_probabilities(frame, k)[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    target = frame[MARKET_COLUMNS].to_numpy(dtype=float)
    return float(-(target * np.log(np.clip(scaled, 1e-12, 1.0))).sum(axis=1).mean())


def fit_exponent(frame: pd.DataFrame) -> float:
    """Golden-section search for the exponent that best matches market prices."""
    low, high = 0.5, 3.0
    for _ in range(80):
        left = low + (high - low) / 3.0
        right = high - (high - low) / 3.0
        if cross_entropy_to_market(frame, left) < cross_entropy_to_market(frame, right):
            high = right
        else:
            low = left
    return (low + high) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", nargs="+", default=DEFAULT_FIT)
    parser.add_argument("--test", nargs="+", default=DEFAULT_TEST)
    args = parser.parse_args()

    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")
    odds = market_odds_from_matches(matches)

    print("\nreplaying seasons (train on earlier seasons only)...")
    replays = {
        season: with_market(replay_season(matches, season, use_history=True), odds)
        for season in [*args.fit, *args.test]
    }

    print(f"\n{'season':<9}{'model':>9}{'market':>9}{'gap':>9}")
    print("-" * 36)
    for season, frame in replays.items():
        model_sharpness = sharpness(frame, PROBABILITY_COLUMNS)
        market_sharpness = sharpness(frame, MARKET_COLUMNS)
        print(
            f"{season:<9}{model_sharpness:>9.4f}{market_sharpness:>9.4f}"
            f"{market_sharpness - model_sharpness:>9.4f}"
        )

    fit_frame = pd.concat([replays[season] for season in args.fit], ignore_index=True)
    exponent = fit_exponent(fit_frame)
    print(f"\nfitted exponent on {', '.join(args.fit)}: k = {exponent:.4f}")
    print("(k > 1 sharpens the model, k < 1 flattens it)")

    header = f"{'season':<9}{'variant':<12}{'LogLoss':>10}{'RPS':>9}{'Acc':>7}{'ECE':>8}"
    print("\nheld-out seasons, scored against real results\n")
    print(header)
    print("-" * len(header))
    totals: dict[str, list[float]] = {"raw": [], "scaled": []}
    for season in args.test:
        frame = replays[season]
        for label, variant in (("raw", frame), ("scaled", sharpen_probabilities(frame, exponent))):
            metrics = probability_metrics(variant)
            totals[label].append(metrics["log_loss"])
            print(
                f"{season if label == 'raw' else '':<9}{label:<12}{metrics['log_loss']:>10.4f}"
                f"{metrics['rps']:>9.4f}{metrics['accuracy']:>7.3f}"
                f"{metrics['expected_calibration_error']:>8.4f}"
            )
    print("-" * len(header))
    for label in ("raw", "scaled"):
        mean_loss = sum(totals[label]) / len(totals[label])
        print(f"{'mean':<9}{label:<12}{mean_loss:>10.4f}")

    # The scaling only matters for unpriced fixtures; priced ones are dominated
    # by the blend. Confirm it does not degrade the blended output either.
    print("\nblended at the production weight\n")
    print(header)
    print("-" * len(header))
    for season in args.test:
        frame = replays[season]
        for label, variant in (("raw", frame), ("scaled", sharpen_probabilities(frame, exponent))):
            blended = blend_probabilities(variant, odds, weight=0.9)
            metrics = probability_metrics(blended)
            print(
                f"{season if label == 'raw' else '':<9}{label:<12}{metrics['log_loss']:>10.4f}"
                f"{metrics['rps']:>9.4f}{metrics['accuracy']:>7.3f}"
                f"{metrics['expected_calibration_error']:>8.4f}"
            )

    output = ARTIFACT_DIR / "market_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "exponent": exponent,
                "fit_seasons": args.fit,
                "test_seasons": args.test,
                "mean_log_loss_raw": sum(totals["raw"]) / len(totals["raw"]),
                "mean_log_loss_scaled": sum(totals["scaled"]) / len(totals["scaled"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()
