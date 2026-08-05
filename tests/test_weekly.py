import pandas as pd

from nfl_predictor.weekly import generate_weekly_predictions


def _data():
    schedules = []
    stats = []
    for week in range(1, 7):
        game_id = f"2024_{week:02d}_A_B"
        home_score, away_score = ((20, 10) if week % 2 else (10, 20))
        schedules.append(
            {
                "game_id": game_id,
                "season": 2024,
                "week": week,
                "game_type": "REG",
                "gameday": f"2024-09-{week:02d}",
                "home_team": "A" if week % 2 else "B",
                "away_team": "B" if week % 2 else "A",
                "home_score": home_score,
                "away_score": away_score,
            }
        )
        stats.extend(
            [
                {
                    "game_id": game_id,
                    "season": 2024,
                    "week": week,
                    "team": "A",
                    "opponent_team": "B",
                    "off_epa": 1.0,
                    "def_epa": -0.5,
                    "plays": 60,
                },
                {
                    "game_id": game_id,
                    "season": 2024,
                    "week": week,
                    "team": "B",
                    "opponent_team": "A",
                    "off_epa": -1.0,
                    "def_epa": 0.5,
                    "plays": 55,
                },
            ]
        )
    target = pd.DataFrame(
        [
            {
                "game_id": "2024_07_A_B",
                "season": 2024,
                "week": 7,
                "game_type": "REG",
                "gameday": "2024-10-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": float("nan"),
                "away_score": float("nan"),
            }
        ]
    )
    return pd.DataFrame(stats), pd.DataFrame(schedules), target


def test_weekly_predictions_produce_probability_and_confidence_outputs():
    stats, historical, target = _data()
    predictions, _ = generate_weekly_predictions(
        stats, historical, target, calibration_method="raw"
    )
    assert len(predictions) == 1
    assert 0 <= predictions.iloc[0]["home_win_probability"] <= 1
    assert predictions.iloc[0]["predicted_winner"] in {"A", "B"}
    assert predictions.iloc[0]["confidence_tier"] in {"low", "medium", "high"}
