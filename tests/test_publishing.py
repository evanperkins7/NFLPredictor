from datetime import date

import pandas as pd
import pytest

from nfl_predictor.publishing import next_unplayed_week, season_for_date


def test_season_for_date_uses_previous_year_during_january_and_february():
    assert season_for_date(date(2027, 1, 12)) == 2026
    assert season_for_date(date(2026, 9, 1)) == 2026


def test_next_unplayed_week_selects_earliest_regular_season_game():
    schedule = pd.DataFrame(
        {
            "game_type": ["REG", "REG", "POST"],
            "week": [1, 2, 19],
            "home_score": [24, None, None],
            "away_score": [17, None, None],
        }
    )
    assert next_unplayed_week(schedule) == 2


def test_next_unplayed_week_returns_none_after_regular_season():
    schedule = pd.DataFrame(
        {
            "game_type": ["REG", "POST"],
            "week": [18, 19],
            "home_score": [24, None],
            "away_score": [17, None],
        }
    )
    assert next_unplayed_week(schedule) is None


def test_next_unplayed_week_rejects_incomplete_schedule_schema():
    with pytest.raises(ValueError, match="home_score"):
        next_unplayed_week(pd.DataFrame({"game_type": ["REG"], "week": [1]}))
