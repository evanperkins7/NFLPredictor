import pandas as pd

from nfl_predictor.calibration import (
    calibration_metrics,
    calibration_split,
    reliability_table,
    run_calibration_experiment,
)


def _features():
    rows = []
    for season in range(2016, 2023):
        for game in range(8):
            rows.append(
                {
                    "season": season,
                    "offensive_epa_diff": (game - 3) / 3,
                    "defensive_epa_diff": (3 - game) / 4,
                    "home_win": int((game + season) % 3 != 0),
                }
            )
    return pd.DataFrame(rows)


def test_calibration_split_reserves_only_latest_training_season():
    model_data, calibration_data = calibration_split(_features()[lambda frame: frame.season < 2022])
    assert model_data["season"].max() == 2020
    assert calibration_data["season"].min() == 2021


def test_reliability_metrics_return_expected_columns():
    outcomes = pd.Series([0, 0, 1, 1])
    probabilities = pd.Series([0.1, 0.2, 0.8, 0.9])
    table = reliability_table(outcomes, probabilities, n_bins=2)
    metrics = calibration_metrics(outcomes, probabilities)
    assert table["count"].sum() == 4
    assert set(metrics) >= {"expected_calibration_error", "calibration_slope"}


def test_calibration_experiment_is_chronological():
    season_results, summary_results, reliability = run_calibration_experiment(
        _features(), ["offensive_epa_diff", "defensive_epa_diff"]
    )
    assert set(season_results["method"]) == {"raw", "sigmoid", "isotonic"}
    assert set(summary_results["evaluation"]) == {"walk_forward", "fixed_holdout"}
    assert set(reliability["method"]) == {"raw", "sigmoid", "isotonic"}
