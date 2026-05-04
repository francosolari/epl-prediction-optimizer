"""Command-line entry points for the EPL prediction optimizer."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import chdir
from pathlib import Path

import uvicorn

from epl_prediction_optimizer.pipeline import (
    backtest_from_processed,
    optimize_from_predictions,
    predict_from_processed,
    refresh_data,
    refresh_full_history,
    run_all,
    train_from_processed,
)


def main() -> None:
    """Parse CLI arguments and dispatch project commands."""
    parser = argparse.ArgumentParser(description="EPL prediction optimizer")
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory for data/artifacts (default: current directory)",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    # --- data commands ---
    refresh = subcommands.add_parser("refresh", help="Refresh current-season data and fixtures")
    refresh.add_argument("--network", action="store_true", help="Download live data")

    subcommands.add_parser("refresh-full-history", help="Download all available EPL history")

    # --- model commands (Stage A) ---
    subcommands.add_parser("train", help="Train the Stage A probability model")
    subcommands.add_parser("predict", help="Generate fixture probability CSV (Stage A output)")

    analyze = subcommands.add_parser("analyze", help="Analyze Stage A model quality")
    analyze.add_argument("--season", default=None, help="Show pick breakdown for this season")
    analyze.add_argument(
        "--no-importance",
        action="store_true",
        help="Skip permutation importance (faster)",
    )

    # --- optimizer commands (Stage B) ---
    subcommands.add_parser("optimize", help="Generate optimized picks from Stage A probabilities")

    backtest = subcommands.add_parser(
        "backtest", help="Backtest Stage A+B pipeline on a completed season"
    )
    backtest.add_argument("--season", default="2526")

    # --- pipeline shortcuts ---
    run = subcommands.add_parser("run-all", help="Refresh → train → predict → optimize")
    run.add_argument("--network", action="store_true")

    serve = subcommands.add_parser("serve", help="Start the web dashboard")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    with chdir(workdir):
        if args.command == "refresh":
            print(json.dumps(refresh_data(use_network=args.network), indent=2))

        elif args.command == "refresh-full-history":
            print(json.dumps(refresh_full_history(), indent=2))

        elif args.command == "train":
            metrics = train_from_processed().metrics
            print("\n=== Stage A: Training complete ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

        elif args.command == "predict":
            count = len(predict_from_processed())
            print(f"Wrote {count} predictions → data/exports/fixture_probabilities.csv")

        elif args.command == "optimize":
            count = len(optimize_from_predictions())
            print(f"Wrote {count} picks → data/exports/optimized_picks.csv")

        elif args.command == "run-all":
            result = run_all(use_network=args.network)
            print(json.dumps(result, indent=2))

        elif args.command == "backtest":
            result = backtest_from_processed(args.season)
            pts = result["optimized_points"]
            winner = result.get("winner_points")
            print(f"\n=== Backtest: {args.season} ===")
            for k, v in result.items():
                print(f"  {k}: {v}")
            if winner:
                print(f"\n  Gap to winner: {pts - winner:+d} pts")

        elif args.command == "analyze":
            _run_analyze(args)

        elif args.command == "serve":
            uvicorn.run(
                "epl_prediction_optimizer.app.main:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
            )


def _run_analyze(args: argparse.Namespace) -> None:
    """Run Stage A model analysis."""
    import json

    import joblib
    import pandas as pd

    from epl_prediction_optimizer.ml.analysis import (
        feature_importance_report,
        pick_accuracy_report,
        print_backtest_summary,
        print_feature_importance,
        print_pick_accuracy,
        backtest_summary_report,
    )
    from epl_prediction_optimizer.ml.features import build_training_frame
    from epl_prediction_optimizer.paths import ARTIFACT_DIR, EXPORT_DIR, PROCESSED_DIR

    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not metrics_path.exists():
        print("No trained model found. Run: eplpo train")
        return

    metrics = json.loads(metrics_path.read_text())
    print("\n=== Stage A: Training Metrics ===")
    print(f"  Trained at  : {metrics.get('trained_at', '—')}")
    print(f"  Training set: {int(metrics.get('rows', 0)):,} matches")
    print(f"  Accuracy    : {metrics.get('accuracy', 0):.4f}")
    print(f"  Log loss    : {metrics.get('log_loss', 0):.4f}")

    if not args.no_importance:
        print("\n=== Stage A: Feature Importance (permutation, ~30s) ===")
        model_run = joblib.load(ARTIFACT_DIR / "model.joblib")
        hist_path = PROCESSED_DIR / "historical_matches.csv"
        if hist_path.exists():
            matches = pd.read_csv(hist_path).dropna(subset=["home_goals", "away_goals"])
            training_frame = build_training_frame(matches)
            importance_df = feature_importance_report(model_run, training_frame)
            print_feature_importance(importance_df)
        else:
            print("  No historical data. Run: eplpo refresh --network")

    # Backtest summary
    from epl_prediction_optimizer.pipeline import WINNER_BENCHMARKS
    known_seasons = list(WINNER_BENCHMARKS.keys())
    summary = backtest_summary_report(ARTIFACT_DIR, known_seasons)
    if not summary.empty:
        print("\n=== Stage A+B: Backtest Summary ===")
        print_backtest_summary(summary)

    if args.season:
        picks_path = EXPORT_DIR / f"{args.season}_backtest_optimized_picks.csv"
        if picks_path.exists():
            picks = pd.read_csv(picks_path)
            print(f"\n=== Stage B: Pick Accuracy ({args.season}) ===")
            print_pick_accuracy(pick_accuracy_report(picks))
        else:
            print(f"\nNo backtest picks for {args.season}. Run: eplpo backtest --season {args.season}")
