"""Chronological logistic-regression training and evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, NEUTRAL_PACE_FEATURE

EPA_FEATURES = ["offensive_epa_diff", "defensive_epa_diff"]
FEATURE_GROUPS = {
    "epa": EPA_FEATURES,
    "epa_rest": [*EPA_FEATURES, "rest_days_diff"],
    "epa_pace": [*EPA_FEATURES, "pace_diff"],
    "epa_neutral_pace": [*EPA_FEATURES, NEUTRAL_PACE_FEATURE],
    "epa_rest_neutral_pace": [*EPA_FEATURES, "rest_days_diff", NEUTRAL_PACE_FEATURE],
    "all": FEATURE_COLUMNS,
    "all_neutral_pace": [*FEATURE_COLUMNS, NEUTRAL_PACE_FEATURE],
    "rest": ["rest_days_diff"],
    "pace": ["pace_diff"],
}
DEFAULT_FEATURE_GROUPS = ("all", "epa", "epa_pace", "epa_rest", "pace", "rest")


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2_000)),
        ]
    )


def fit_model(data: pd.DataFrame, feature_columns: list[str] | None = None) -> Pipeline:
    """Fit a model on completed feature rows using the requested feature columns."""
    selected_features = feature_columns or FEATURE_COLUMNS
    model = make_model()
    model.fit(data[selected_features], data["home_win"])
    return model


def time_split(data: pd.DataFrame, test_seasons: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(data["season"].unique())
    if len(seasons) <= test_seasons:
        raise ValueError("Need more seasons than the requested test window.")
    cutoff = seasons[-test_seasons]
    return data[data["season"] < cutoff].copy(), data[data["season"] >= cutoff].copy()


def fit_and_evaluate(
    data: pd.DataFrame,
    test_seasons: int = 2,
    feature_columns: list[str] | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    selected_features = feature_columns or FEATURE_COLUMNS
    train, test = time_split(data, test_seasons=test_seasons)
    model = fit_model(train, selected_features)
    probabilities = model.predict_proba(test[selected_features])[:, 1]
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


def add_prediction_labels(
    games: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str] | None = None,
    probabilities: pd.Series | None = None,
) -> pd.DataFrame:
    """Add interpretable probability, winner, and confidence outputs to game rows."""
    selected_features = feature_columns or FEATURE_COLUMNS
    output = games.copy()
    home_probability = probabilities
    if home_probability is None:
        home_probability = pd.Series(
            model.predict_proba(output[selected_features])[:, 1], index=output.index
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
