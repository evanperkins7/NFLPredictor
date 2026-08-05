import pandas as pd

from nfl_predictor.analysis import build_analysis_markdown


def test_analysis_report_explains_coefficients_and_season_baseline():
    report = build_analysis_markdown(
        {
            "train_seasons": [2020],
            "test_seasons": [2021],
            "test_rows": 10,
            "accuracy": 0.6,
            "log_loss": 0.65,
            "brier_score": 0.23,
            "always_home_accuracy": 0.5,
        },
        pd.DataFrame(
            [{"season": 2021, "rows": 10, "accuracy": 0.6, "log_loss": 0.65, "brier_score": 0.23, "home_win_rate": 0.5}]
        ),
        pd.DataFrame(
            [{"feature": "offensive_epa_diff", "coefficient": 0.5}]
        ),
    )
    assert "Offensive EPA differential" in report
    assert "Accuracy vs. home baseline" in report
    assert "association" in report

