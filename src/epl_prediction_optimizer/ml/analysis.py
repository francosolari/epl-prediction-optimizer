"""Stage A model analysis: feature importance, backtest comparison, pick diagnostics.

Stage A = the probability model (predicts P(home win), P(draw), P(away win)).
Stage B = the optimizer (selects picks given Stage A probabilities + contest constraints).
This module analyzes Stage A in isolation so its quality can be improved independently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from epl_prediction_optimizer.ml.features import FEATURE_COLUMNS
from epl_prediction_optimizer.ml.model import ModelRun


def feature_importance_report(
    model_run: ModelRun,
    training_frame: pd.DataFrame,
    n_repeats: int = 8,
) -> pd.DataFrame:
    """Permutation feature importances for the Stage A estimator.

    Uses permutation importance (not in-bag) so the result reflects actual
    out-of-training-distribution impact rather than fitting artifact.
    """
    x = training_frame[FEATURE_COLUMNS]
    y = training_frame["outcome"]
    result = permutation_importance(
        model_run.estimator,
        x,
        y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def backtest_summary_report(
    artifact_dir: Path,
    seasons: list[str],
) -> pd.DataFrame:
    """Load and compare Stage A+B backtest metrics across multiple completed seasons."""
    rows = []
    for season in seasons:
        path = artifact_dir / f"{season}_backtest_metrics.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        winner = data.get("winner_points")
        pts = data.get("optimized_points", 0)
        rows.append({
            "season": season,
            "training_matches": data.get("training_matches"),
            "accuracy": round(data.get("accuracy", 0), 4),
            "log_loss": round(data.get("log_loss", 0), 4),
            "picks": data.get("optimized_picks"),
            "model_points": pts,
            "expected_points": round(data.get("optimized_expected_points", 0), 1),
            "winner_points": winner,
            "gap_to_winner": (winner - pts) if winner else None,
        })
    return pd.DataFrame(rows)


def pick_accuracy_report(picks: pd.DataFrame) -> dict[str, object]:
    """Win/draw/loss breakdown and averages for a scored picks frame."""
    total = len(picks)
    if total == 0:
        return {"total_picks": 0}
    wins = int((picks["actual_points"] == 3).sum())
    draws = int((picks["actual_points"] == 1).sum())
    losses = int((picks["actual_points"] == 0).sum())
    return {
        "total_picks": total,
        "total_points": int(picks["actual_points"].sum()),
        "avg_points_per_pick": round(float(picks["actual_points"].mean()), 3),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / total, 3),
        "draw_rate": round(draws / total, 3),
        "loss_rate": round(losses / total, 3),
        "expected_vs_actual": round(
            float(picks["actual_points"].sum() - picks["expected_points"].sum()), 2,
        ) if "expected_points" in picks.columns else None,
    }


def print_feature_importance(df: pd.DataFrame) -> None:
    """Print feature importance table to stdout."""
    print(f"\n{'Feature':<38} {'Importance':>12} {'±Std':>8}")
    print("-" * 60)
    for _, row in df.iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"{row['feature']:<38} {row['importance']:>12.4f} {row['std']:>8.4f}  {bar}")


def print_backtest_summary(df: pd.DataFrame) -> None:
    """Print backtest comparison table to stdout."""
    print(f"\n{'Season':<8} {'Acc':>6} {'LogLoss':>8} {'Picks':>6} {'Pts':>5} {'xPts':>7} {'Winner':>7} {'Gap':>5}")
    print("-" * 62)
    for _, row in df.iterrows():
        gap = f"{row['gap_to_winner']:+.0f}" if row["gap_to_winner"] is not None else " —"
        wp = row["winner_points"]
        winner = str(int(wp)) if wp and not (isinstance(wp, float) and wp != wp) else "—"
        print(
            f"{row['season']:<8} {row['accuracy']:>6.3f} {row['log_loss']:>8.4f} "
            f"{int(row['picks'] or 0):>6} {int(row['model_points'] or 0):>5} "
            f"{row['expected_points']:>7.1f} {winner:>7} {gap:>5}"
        )


def print_pick_accuracy(report: dict) -> None:
    """Print pick accuracy breakdown to stdout."""
    if not report.get("total_picks"):
        print("No scored picks found.")
        return
    total = report["total_picks"]
    print(f"\n  Picks : {total}")
    print(f"  Points: {report['total_points']}  ({report['avg_points_per_pick']:.2f}/pick)")
    print(f"  Wins  : {report['wins']}  ({report['win_rate']:.1%})")
    print(f"  Draws : {report['draws']}  ({report['draw_rate']:.1%})")
    print(f"  Losses: {report['losses']}  ({report['loss_rate']:.1%})")
    if report.get("expected_vs_actual") is not None:
        delta = report["expected_vs_actual"]
        sign = "+" if delta >= 0 else ""
        print(f"  Luck  : {sign}{delta:.1f} pts vs expected")
