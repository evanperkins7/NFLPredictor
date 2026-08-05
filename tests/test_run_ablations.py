import pandas as pd

from nfl_predictor.ablations import run_ablations


def test_run_ablations_labels_windows_and_feature_groups():
    data = pd.DataFrame(
        [
            {
                "season": season,
                "offensive_epa_diff": 0.2,
                "defensive_epa_diff": -0.1,
                "pace_diff": 1.0,
                "rest_days_diff": 0.0,
                "home_win": int(season % 2 == 0),
            }
            for season in range(2016, 2022)
            for _ in range(2)
        ]
    )
    season_results, summary_results = run_ablations(
        {"expanding": data, "3": data}, feature_groups=["epa"]
    )
    assert set(season_results["rolling_window"]) == {"expanding", "3"}
    assert set(summary_results["feature_group"]) == {"epa"}
