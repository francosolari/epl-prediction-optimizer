"""Training and inference for calibrated multinomial match probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from epl_prediction_optimizer.ml.features import (
    FEATURE_COLUMNS,
    OUTCOME_AWAY,
    OUTCOME_DRAW,
    OUTCOME_HOME,
    build_fixture_features,
)


@dataclass
class ModelRun:
    """A trained estimator plus serializable training metrics."""

    estimator: CalibratedClassifierCV
    metrics: dict[str, float | str]

    def predict(
        self,
        fixtures: pd.DataFrame,
        season_stats: dict[str, list[int]] | None = None,
        apply_dc: bool = True,
    ) -> pd.DataFrame:
        """Generate normalized HOME/DRAW/AWAY probabilities for fixtures."""
        frame = build_fixture_features(fixtures, season_stats=season_stats).reset_index(drop=True)
        probabilities = self.estimator.predict_proba(frame[FEATURE_COLUMNS])
        classes = list(self.estimator.classes_)
        prob_frame = pd.DataFrame(probabilities, columns=classes)
        output = frame[["match_id", "contest_week", "date", "home_team", "away_team"]].copy()
        output["p_home_win"] = prob_frame.get(OUTCOME_HOME, 0.0)
        output["p_draw"] = prob_frame.get(OUTCOME_DRAW, 0.0)
        output["p_away_win"] = prob_frame.get(OUTCOME_AWAY, 0.0)
        totals = output[["p_home_win", "p_draw", "p_away_win"]].sum(axis=1)
        output[["p_home_win", "p_draw", "p_away_win"]] = output[
            ["p_home_win", "p_draw", "p_away_win"]
        ].div(totals, axis=0)
        if apply_dc:
            output = apply_dixon_coles_correction(output)
        return output


def train_model(
    training_frame: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    feature_columns: list[str] | None = None,
) -> ModelRun:
    """Train a calibrated gradient boosting model from engineered features."""
    if training_frame["outcome"].nunique() < 3:
        raise ValueError("Training data must include home wins, draws, and away wins.")
    cols = feature_columns if feature_columns is not None else FEATURE_COLUMNS
    x = training_frame[cols]
    y = training_frame["outcome"]
    sw = sample_weight.values if sample_weight is not None else None
    base = HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=15,
        l2_regularization=0.5,
        random_state=42,
    )
    cv = min(3, y.value_counts().min())
    estimator = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    estimator.fit(x, y, sample_weight=sw)
    probabilities = estimator.predict_proba(x)
    predictions = estimator.predict(x)
    metrics = {
        "trained_at": datetime.now(UTC).isoformat(),
        "rows": float(len(training_frame)),
        "log_loss": float(log_loss(y, probabilities, labels=list(estimator.classes_))),
        "accuracy": float(accuracy_score(y, predictions)),
    }
    return ModelRun(estimator=estimator, metrics=metrics)


def season_decay_weights(
    season_series: pd.Series,
    reference_season: str,
    decay: float = 0.9,
) -> pd.Series:
    """Return per-row sample weights with exponential decay by season age.

    Decay of 0.9/season means data 10 seasons back gets 0.9^10 ≈ 35% weight.
    reference_season is the target season being backtested (not included in training).
    """
    ref_year = _season_start_year(reference_season)
    weights = season_series.map(lambda s: decay ** max(0, ref_year - _season_start_year(str(s))))
    return weights.astype(float)


def apply_dixon_coles_correction(
    predictions: pd.DataFrame,
    rho: float = -0.07,
) -> pd.DataFrame:
    """Apply a simplified Dixon-Coles draw correction to predicted probabilities.

    For evenly-matched games (|p_home - p_away| < 0.3), boosts p_draw by a
    factor proportional to rho and how close the teams are, then renormalises.
    rho is negative (the canonical DC value) so we take abs(rho) as the boost.
    """
    frame = predictions.copy()
    p_h = frame["p_home_win"].values.copy()
    p_d = frame["p_draw"].values.copy()
    p_a = frame["p_away_win"].values.copy()

    diff = abs(p_h - p_a)
    closeness = (1.0 - diff / 0.3).clip(0.0, 1.0)  # 1 when perfectly even, 0 at 0.3 gap
    boost = abs(rho) * closeness
    p_d_new = p_d + boost

    total = p_h + p_d_new + p_a
    frame["p_home_win"] = p_h / total
    frame["p_draw"] = p_d_new / total
    frame["p_away_win"] = p_a / total
    return frame


def _season_start_year(season_code: str) -> int:
    """Parse a 2- or 4-char season code (e.g. '2324', '9596', '1011') to start year."""
    code = str(season_code).zfill(4)
    yy = int(code[:2])
    return (1900 + yy) if yy >= 50 else (2000 + yy)
