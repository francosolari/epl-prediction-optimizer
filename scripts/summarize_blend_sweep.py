#!/usr/bin/env python3
"""Print the per-season detail behind a blend-weight sweep.

The sweep prints cross-season means; this shows whether a weight helps in
every season or only on average, which is the difference between a real edge
and a lucky one.

Usage:
    uv run scripts/summarize_blend_sweep.py --weights 0.0 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from epl_prediction_optimizer.paths import ARTIFACT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devig", default="power")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.85])
    parser.add_argument("--metric", default="log_loss")
    args = parser.parse_args()

    path = ARTIFACT_DIR / f"market_blend_sweep_{args.devig}.json"
    if not path.exists():
        raise SystemExit(f"No sweep found at {path}; run scripts/backtest_market_blend.py first")
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))

    available = sorted(frame["weight"].unique())
    weights = [weight for weight in args.weights if weight in available]
    if len(weights) != 2:
        raise SystemExit(f"Need two weights present in the sweep; available: {available}")

    pivot = frame[frame["weight"].isin(weights)].pivot(
        index="season", columns="weight", values=args.metric
    )
    pivot["delta"] = pivot[weights[1]] - pivot[weights[0]]

    print(f"\n{args.metric} by season ({args.devig} de-vig)\n")
    print(f"{'season':<9}{weights[0]:>10.2f}{weights[1]:>10.2f}{'delta':>10}")
    print("-" * 39)
    for season, row in pivot.iterrows():
        print(f"{season:<9}{row[weights[0]]:>10.4f}{row[weights[1]]:>10.4f}{row['delta']:>10.4f}")
    print("-" * 39)
    # Lower is better for loss-style metrics, higher for accuracy.
    lower_is_better = args.metric not in {"accuracy"}
    improved = int(
        (pivot["delta"] < 0).sum() if lower_is_better else (pivot["delta"] > 0).sum()
    )
    print(f"{'mean':<9}{pivot[weights[0]].mean():>10.4f}{pivot[weights[1]].mean():>10.4f}"
          f"{pivot['delta'].mean():>10.4f}")
    print(f"\nimproved in {improved} of {len(pivot)} seasons")


if __name__ == "__main__":
    main()
