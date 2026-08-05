"""Time-aware probability calibration and reliability evaluation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .ablations import walk_forward_splits
from .model import fit_model

CALIBRATION_METHODS = ("raw", "sigmoid", "isotonic")


def calibration_split(
    train: pd.DataFrame, calibration_seasons: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the latest training seasons for calibration without using test data."""
    seasons = sorted(train["season"].unique())
    if calibration_seasons < 1 or len(seasons) <= calibration_seasons:
        raise ValueError("Need more training seasons than calibration seasons.")
    cutoff = seasons[-calibration_seasons]
    return (
        train[train["season"] < cutoff].copy(),
        train[train["season"] >= cutoff].copy(),
    )


def _clip(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


def reliability_table(
    outcomes: pd.Series,
    probabilities: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize observed outcomes and predictions in equal-width probability bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    frame = pd.DataFrame({"outcome": outcomes.astype(int), "probability": _clip(probabilities)})
    frame["bin"] = np.minimum((frame["probability"] * n_bins).astype(int), n_bins - 1)
    table = (
        frame.groupby("bin", as_index=False)
        .agg(count=("outcome", "size"), mean_probability=("probability", "mean"), outcome_rate=("outcome", "mean"))
    )
    table["bin_lower"] = table["bin"] / n_bins
    table["bin_upper"] = (table["bin"] + 1) / n_bins
    table["absolute_gap"] = (table["mean_probability"] - table["outcome_rate"]).abs()
    return table[["bin", "bin_lower", "bin_upper", "count", "mean_probability", "outcome_rate", "absolute_gap"]]


def calibration_metrics(outcomes: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    """Return probability quality, calibration error, and calibration line estimates."""
    clipped = _clip(probabilities)
    table = reliability_table(outcomes, clipped)
    ece = float((table["count"] * table["absolute_gap"]).sum() / len(outcomes))
    if outcomes.nunique() < 2:
        intercept = float("nan")
        slope = float("nan")
    else:
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibration_line = LogisticRegression(C=1_000_000, max_iter=2_000).fit(logits, outcomes)
        intercept = float(calibration_line.intercept_[0])
        slope = float(calibration_line.coef_[0][0])
    return {
        "accuracy": float(accuracy_score(outcomes, clipped >= 0.5)),
        "log_loss": float(log_loss(outcomes, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(outcomes, clipped)),
        "expected_calibration_error": ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def fit_probability_calibrator(
    method: str,
    probabilities: pd.Series,
    outcomes: pd.Series,
) -> Callable[[pd.Series], pd.Series]:
    """Fit a probability mapping on a chronologically reserved calibration set."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method: {method}")
    if method == "raw":
        return lambda values: pd.Series(_clip(values), index=values.index)
    if outcomes.nunique() < 2:
        raise ValueError("Calibration outcomes must contain both classes.")
    if method == "sigmoid":
        calibrator = LogisticRegression(C=1_000_000, max_iter=2_000).fit(
            np.log(_clip(probabilities) / (1 - _clip(probabilities))).reshape(-1, 1), outcomes
        )
        return lambda values: pd.Series(
            calibrator.predict_proba(
                np.log(_clip(values) / (1 - _clip(values))).reshape(-1, 1)
            )[:, 1],
            index=values.index,
        )
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(_clip(probabilities), outcomes)
    return lambda values: pd.Series(calibrator.predict(_clip(values)), index=values.index)


def fit_calibrated_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    method: str = "sigmoid",
    calibration_seasons: int = 1,
) -> tuple[object, Callable[[pd.Series], pd.Series]]:
    """Fit a production model and an optional calibration mapping from prior seasons."""
    if method == "raw":
        return fit_model(data, feature_columns), fit_probability_calibrator(
            "raw", pd.Series(dtype=float), pd.Series(dtype=int)
        )
    model_data, calibration_data = calibration_split(data, calibration_seasons)
    model = fit_model(model_data, feature_columns)
    calibration_probabilities = pd.Series(
        model.predict_proba(calibration_data[feature_columns])[:, 1], index=calibration_data.index
    )
    calibrator = fit_probability_calibrator(
        method, calibration_probabilities, calibration_data["home_win"].astype(int)
    )
    return model, calibrator


def calibrated_probabilities(
    model: object,
    calibrator: Callable[[pd.Series], pd.Series],
    games: pd.DataFrame,
    feature_columns: list[str],
) -> pd.Series:
    """Apply a fitted model and its calibration mapping to game features."""
    raw = pd.Series(model.predict_proba(games[feature_columns])[:, 1], index=games.index)
    return calibrator(raw)


def _predict_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    method: str,
    calibration_seasons: int,
) -> tuple[pd.Series, pd.Series]:
    model, calibrator = fit_calibrated_model(
        train, feature_columns, method, calibration_seasons
    )
    return test["home_win"].astype(int), calibrated_probabilities(
        model, calibrator, test, feature_columns
    )


def run_calibration_experiment(
    data: pd.DataFrame,
    feature_columns: list[str],
    min_train_seasons: int = 4,
    test_seasons: int = 2,
    calibration_seasons: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate raw and calibrated probabilities with chronological outer splits."""
    seasons = sorted(data["season"].unique())
    if len(seasons) <= test_seasons:
        raise ValueError("Need more seasons than the fixed test window.")
    fixed_train = data[data["season"] < seasons[-test_seasons]].copy()
    fixed_test = data[data["season"] >= seasons[-test_seasons]].copy()
    evaluations = {
        "walk_forward": list(walk_forward_splits(data, min_train_seasons)),
        "fixed_holdout": [(-1, fixed_train, fixed_test)],
    }
    season_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    reliability_rows: list[pd.DataFrame] = []
    for method in CALIBRATION_METHODS:
        for evaluation, splits in evaluations.items():
            outcomes_all: list[pd.Series] = []
            probabilities_all: list[pd.Series] = []
            for test_season, train, test in splits:
                outcomes, probabilities = _predict_split(
                    train, test, feature_columns, method, calibration_seasons
                )
                season_rows.append(
                    {
                        "method": method,
                        "evaluation": evaluation,
                        "test_season": int(test_season) if test_season != -1 else "all",
                        "rows": len(test),
                        **calibration_metrics(outcomes, probabilities),
                    }
                )
                outcomes_all.append(outcomes)
                probabilities_all.append(probabilities)
            outcomes = pd.concat(outcomes_all, ignore_index=True)
            probabilities = pd.concat(probabilities_all, ignore_index=True)
            summary_rows.append(
                {
                    "method": method,
                    "evaluation": evaluation,
                    "test_season": "all",
                    "rows": len(outcomes),
                    **calibration_metrics(outcomes, probabilities),
                }
            )
            reliability = reliability_table(outcomes, probabilities)
            reliability.insert(0, "method", method)
            reliability.insert(1, "evaluation", evaluation)
            reliability_rows.append(reliability)
    return (
        pd.DataFrame(season_rows),
        pd.DataFrame(summary_rows),
        pd.concat(reliability_rows, ignore_index=True),
    )
