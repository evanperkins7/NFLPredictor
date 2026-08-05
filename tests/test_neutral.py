import pandas as pd
import pytest

from nfl_predictor.neutral import build_neutral_pace, neutral_play_mask


def _pbp():
    return pd.DataFrame(
        {
            "season": [2024] * 7,
            "game_id": ["g1"] * 7,
            "posteam": ["A", "A", "A", "A", "A", None, "A"],
            "play_type": ["pass", "run", "pass", "punt", "pass", "pass", "pass"],
            "qtr": [1, 2, 3, 2, 4, 2, 2],
            "score_differential": [0, 8, -8, 0, 0, 0, 9],
            "qb_kneel": [False] * 7,
            "qb_spike": [False, False, False, False, False, False, True],
            "penalty": [False, False, False, False, False, False, False],
            "play_deleted": [False] * 7,
            "aborted_play": [False] * 7,
        }
    )


def test_neutral_play_mask_excludes_non_neutral_plays():
    assert neutral_play_mask(_pbp()).tolist() == [True, True, True, False, False, False, False]


def test_build_neutral_pace_counts_team_games():
    pace = build_neutral_pace(_pbp())
    assert pace.to_dict("records") == [{"season": 2024, "game_id": "g1", "team": "A", "neutral_pace": 3}]


def test_neutral_play_mask_requires_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        neutral_play_mask(pd.DataFrame({"game_id": ["g1"]}))
