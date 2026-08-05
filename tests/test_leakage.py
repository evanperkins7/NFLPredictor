import pandas as pd

from nfl_predictor.features import build_game_features
from nfl_predictor.leakage import audit_feature_table, mutation_audit


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
    stats = pd.DataFrame(
        {
            "game_id": ["2024_01_A_B", "2024_01_A_B", "2024_02_A_B", "2024_02_A_B"],
            "season": [2024] * 4,
            "week": [1, 1, 2, 2],
            "team": ["A", "B", "A", "B"],
            "opponent_team": ["B", "A", "B", "A"],
            "off_epa": [1.0, -1.0, 99.0, -99.0],
            "def_epa": [-0.5, 0.5, 99.0, -99.0],
            "plays": [60, 55, 70, 50],
        }
    )
    return stats, schedule


def test_structural_audit_passes_for_valid_features():
    stats, schedule = _fixtures()
    features = build_game_features(stats, schedule)
    audit = audit_feature_table(features)
    assert audit["passed"] is True
    assert audit["duplicate_game_rows"] == 0


def test_mutating_a_game_does_not_change_current_or_prior_rows():
    stats, schedule = _fixtures()
    audit = mutation_audit(stats, schedule, "2024_02_A_B")
    assert audit["protected_rows"] == 2
    assert audit["passed"] is True
