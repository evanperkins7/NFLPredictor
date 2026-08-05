"""Generate the weekly prediction artifact from historical and target schedules."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from .calibration import calibrated_probabilities, fit_calibrated_model
from .features import build_game_features, build_upcoming_features
from .model import FEATURE_GROUPS, add_prediction_labels


def generate_weekly_predictions(
    team_stats: pd.DataFrame,
    historical_schedule: pd.DataFrame,
    target_schedule: pd.DataFrame,
    rolling_window: int = 8,
    feature_group: str = "epa",
    calibration_method: str = "sigmoid",
) -> tuple[pd.DataFrame, Pipeline]:
    """Fit on completed history and label unplayed games in the target schedule."""
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {feature_group}")
    feature_columns = FEATURE_GROUPS[feature_group]
    training = build_game_features(
        team_stats,
        historical_schedule,
        rolling_window=rolling_window,
    )
    schedule = pd.concat([historical_schedule, target_schedule], ignore_index=True)
    upcoming = build_upcoming_features(team_stats, schedule, rolling_window=rolling_window)
    if upcoming.empty:
        raise ValueError("Target schedule contains no unplayed regular-season games with features.")
    model, calibrator = fit_calibrated_model(
        training, feature_columns, method=calibration_method
    )
    probabilities = calibrated_probabilities(model, calibrator, upcoming, feature_columns)
    return add_prediction_labels(upcoming, model, feature_columns, probabilities), model
