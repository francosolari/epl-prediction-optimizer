#!/usr/bin/env python3
"""Sweep the model/market blend weight across seasons of real closing prices.

football-data.co.uk ships B365 1X2 closing odds with every season CSV from
2002-03 onwards, so the market side of this backtest uses the same schema the
live fixtures feed provides. For each target season the model is trained on
earlier seasons only, predicted through the production fixture-feature path,
then pooled with de-vigged market probabilities at a range of weights.

Log loss / RPS / Brier decide the weight. Contest points are printed because
they are the thing the product optimises, but 38 picks a season cannot
distinguish a real edge from luck and must not drive the choice.

Usage:
    uv run scripts/backtest_market_blend.py
    uv run scripts/backtest_market_blend.py --devig shin --seasons 2324 2425
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.data.sources import market_odds_from_matches
from epl_prediction_optimizer.ml.market import blend_probabilities
from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.solver import optimize_picks
from epl_prediction_optimizer.paths import ARTIFACT_DIR, PROCESSED_DIR
from epl_prediction_optimizer.scoring import probability_metrics, score_picks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_production_path import replay_season  # noqa: E402

DEFAULT_SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
DEFAULT_WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--weights", nargs="+", type=float, default=DEFAULT_WEIGHTS)
    parser.add_argument("--devig", default="power", choices=["proportional", "power", "shin"])
    args = parser.parse_args()

    matches = pd.read_csv(PROCESSED_DIR / "historical_matches.csv")
    odds = market_odds_from_matches(matches)
    if odds.empty:
        raise SystemExit("No historical odds columns found in historical_matches.csv")

    # Replaying a season is the expensive part; do it once and reuse the
    # model probabilities for every blend weight.
    replays = {season: replay_season(matches, season, use_history=True) for season in args.seasons}

    results: list[dict[str, float | str]] = []
    print(f"\nde-vig method: {args.devig}")
    header = (
        f"{'weight':>7}{'LogLoss':>10}{'RPS':>9}{'Brier':>9}"
        f"{'Acc':>7}{'ECE':>8}{'Points':>8}{'Priced':>8}"
    )
    print(header)
    print("-" * len(header))

    for weight in args.weights:
        season_metrics: list[dict[str, float]] = []
        total_points = 0
        priced_share: list[float] = []
        for season, evaluated in replays.items():
            blended = blend_probabilities(evaluated, odds, weight=weight, method=args.devig)
            metrics = probability_metrics(blended)
            picks = optimize_picks(build_pick_candidates(blended))
            scored = score_picks(picks, blended)
            total_points += int(scored["actual_points"].sum())
            priced_share.append(float((blended["market_weight"] > 0).mean()))
            season_metrics.append(metrics)
            results.append({"weight": weight, "season": season, **metrics})

        mean = {
            key: sum(entry[key] for entry in season_metrics) / len(season_metrics)
            for key in ("log_loss", "rps", "brier", "accuracy", "expected_calibration_error")
        }
        print(
            f"{weight:>7.2f}{mean['log_loss']:>10.4f}{mean['rps']:>9.4f}{mean['brier']:>9.4f}"
            f"{mean['accuracy']:>7.3f}{mean['expected_calibration_error']:>8.4f}"
            f"{total_points:>8}{sum(priced_share) / len(priced_share):>8.2f}"
        )

    frame = pd.DataFrame(results)
    output = ARTIFACT_DIR / f"market_blend_sweep_{args.devig}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8")

    by_weight = frame.groupby("weight")["log_loss"].mean()
    best = float(by_weight.idxmin())
    print(f"\nbest weight by mean log loss: {best:.2f}  ({by_weight.min():.4f})")
    print(f"per-season detail written to {output}")


if __name__ == "__main__":
    main()
