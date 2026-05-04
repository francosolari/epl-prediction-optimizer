#!/usr/bin/env python3
"""
Analyze the trained Stage A probability model.

Stage A = pure prediction (P(home win), P(draw), P(away win)).
Stage B = optimizer (constrained pick selection, separate concern).

Usage:
    uv run scripts/analyze_model.py
    uv run scripts/analyze_model.py --season 2425   # backtest-specific analysis
    uv run scripts/analyze_model.py --no-importance  # skip slow permutation step
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd

from epl_prediction_optimizer.ml.analysis import (
    feature_importance_report,
    pick_accuracy_report,
    print_backtest_summary,
    print_feature_importance,
    print_pick_accuracy,
)
from epl_prediction_optimizer.ml.features import build_training_frame
from epl_prediction_optimizer.paths import ARTIFACT_DIR, EXPORT_DIR, PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage A model")
    parser.add_argument("--season", default=None, help="Season to analyze picks for (e.g. 2425)")
    parser.add_argument("--no-importance", action="store_true", help="Skip feature importance (slow)")
    args = parser.parse_args()

    # --- Training metrics ---
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        print("\n=== Stage A: Training Metrics ===")
        print(f"  Trained at : {metrics.get('trained_at', '—')}")
        print(f"  Rows       : {int(metrics.get('rows', 0)):,}")
        print(f"  Accuracy   : {metrics.get('accuracy', 0):.4f}")
        print(f"  Log loss   : {metrics.get('log_loss', 0):.4f}")
    else:
        print("No trained model found. Run: eplpo train")
        return

    # --- Feature importance ---
    if not args.no_importance:
        print("\n=== Stage A: Feature Importance (permutation) ===")
        model_run = joblib.load(ARTIFACT_DIR / "model.joblib")
        hist_path = PROCESSED_DIR / "historical_matches.csv"
        if hist_path.exists():
            matches = pd.read_csv(hist_path).dropna(subset=["home_goals", "away_goals"])
            training_frame = build_training_frame(matches)
            importance_df = feature_importance_report(model_run, training_frame)
            print_feature_importance(importance_df)
        else:
            print("  No historical data found. Run: eplpo refresh --network")

    # --- Pick accuracy for chosen season ---
    season = args.season
    if season:
        picks_path = EXPORT_DIR / f"{season}_backtest_optimized_picks.csv"
        if picks_path.exists():
            picks = pd.read_csv(picks_path)
            print(f"\n=== Stage B: Pick Accuracy ({season}) ===")
            print_pick_accuracy(pick_accuracy_report(picks))
        else:
            print(f"\nNo backtest picks for {season}. Run: eplpo backtest --season {season}")


if __name__ == "__main__":
    main()
