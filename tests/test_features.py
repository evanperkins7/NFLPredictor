import pandas as pd
import pytest

from nfl_predictor.features import FEATURE_COLUMNS, build_game_features


def _fixtures():
    schedule = pd.DataFrame(
        {
            "game_id": ["2024_01_A_B", "2024_02_A_B"],
            "season": [2024, 2024],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "gameday": ["2024-09-01", "2024-09-08"],
            "home_team": ["B", "A"],
            "away_team": ["A", "B"],
            "home_score": [20, 10],
            "away_score": [10, 21],
        }
    )
    team_stats = pd.DataFrame(
        {
            "game_id": ["2024_01_A_B", "2024_01_A_B", "2024_02_A_B", "2024_02_A_B"],
            "season": [2024] * 4,
            "week": [1, 1, 2, 2],
            "team": ["A", "B", "A", "B"],
            "off_epa": [1.0, -1.0, 99.0, -99.0],
            "def_epa": [-0.5, 0.5, 99.0, -99.0],
            "plays": [60, 55, 70, 50],
            "opponent_team": ["B", "A", "B", "A"],
        }
    )
    return team_stats, schedule


def test_game_stats_are_not_used_in_the_same_game_prediction():
    stats, schedule = _fixtures()
    result = build_game_features(stats, schedule)
    week_two = result.loc[result["week"] == 2].iloc[0]
    assert week_two["offensive_epa_diff"] == pytest.approx(2.0)
    assert week_two["pace_diff"] == pytest.approx(5.0)
    assert week_two["home_margin"] == -11
    assert set(FEATURE_COLUMNS).issubset(result.columns)


def test_missing_schema_is_explicit():
    stats, schedule = _fixtures()
    stats = stats.drop(columns=["off_epa", "def_epa", "plays"])
    with pytest.raises(ValueError, match="passing EPA"):
        build_game_features(stats, schedule)
