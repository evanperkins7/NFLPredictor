"""Chronological logistic-regression training and evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2_000)),
        ]
    )


def time_split(data: pd.DataFrame, test_seasons: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(data["season"].unique())
    if len(seasons) <= test_seasons:
        raise ValueError("Need more seasons than the requested test window.")
    cutoff = seasons[-test_seasons]
    return data[data["season"] < cutoff].copy(), data[data["season"] >= cutoff].copy()


def fit_and_evaluate(data: pd.DataFrame, test_seasons: int = 2) -> tuple[Pipeline, dict[str, float]]:
    train, test = time_split(data, test_seasons=test_seasons)
    model = make_model()
    model.fit(train[FEATURE_COLUMNS], train["home_win"])
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    train_home_win_rate = float(train["home_win"].mean())
    metrics = {
        "log_loss": float(log_loss(test["home_win"], probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(test["home_win"], probabilities)),
        "accuracy": float(accuracy_score(test["home_win"], predictions)),
        "always_home_accuracy": float(accuracy_score(test["home_win"], [1] * len(test))),
        "constant_rate_log_loss": float(
            log_loss(test["home_win"], [train_home_win_rate] * len(test), labels=[0, 1])
        ),
        "constant_rate_brier_score": float(
            brier_score_loss(test["home_win"], [train_home_win_rate] * len(test))
        ),
        "train_home_win_rate": train_home_win_rate,
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
    }
    return model, metrics


def add_prediction_labels(games: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Add interpretable probability, winner, and confidence outputs to game rows."""
    output = games.copy()
    home_probability = pd.Series(
        model.predict_proba(output[FEATURE_COLUMNS])[:, 1], index=output.index
    )
    confidence = home_probability.where(home_probability >= 0.5, 1 - home_probability)
    output["home_win_probability"] = home_probability
    output["predicted_winner"] = output.apply(
        lambda row: row["home_team"] if row["home_win_probability"] >= 0.5 else row["away_team"],
        axis=1,
    )
    output["confidence_tier"] = pd.cut(
        confidence,
        bins=[0.0, 0.55, 0.70, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return output
