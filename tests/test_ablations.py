import pandas as pd

from nfl_predictor.ablations import evaluate_feature_table, walk_forward_splits


def _features():
    rows = []
    for season in range(2016, 2022):
        for game in range(2):
            rows.append(
                {
                    "season": season,
                    "offensive_epa_diff": 0.2 if game == 0 else -0.1,
                    "defensive_epa_diff": -0.1,
                    "pace_diff": 1.0,
                    "rest_days_diff": 0.0,
                    "home_win": int(game == 0),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_splits_use_only_prior_seasons():
    data = _features()
    splits = list(walk_forward_splits(data, min_train_seasons=4))
    assert [season for season, _, _ in splits] == [2020, 2021]
    assert all(train["season"].max() < season for season, train, _ in splits)
    assert all(test["season"].min() == season for season, _, test in splits)


def test_feature_group_evaluation_returns_walk_forward_and_fixed_holdout():
    season_results, summary_results = evaluate_feature_table(_features(), "epa")
    assert set(season_results["evaluation"]) == {"walk_forward", "fixed_holdout"}
    assert set(summary_results["evaluation"]) == {"walk_forward", "fixed_holdout"}
    assert set(summary_results.columns) >= {"accuracy", "log_loss", "brier_score"}
