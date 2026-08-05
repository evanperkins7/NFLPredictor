"""Helpers for selecting the next weekly prediction artifact to publish."""

from __future__ import annotations

from datetime import date

import pandas as pd


def season_for_date(reference_date: date) -> int:
    """Return the NFL season label active around a calendar date."""
    return reference_date.year - 1 if reference_date.month <= 2 else reference_date.year


def next_unplayed_week(schedule: pd.DataFrame) -> int | None:
    """Return the earliest regular-season week with an unplayed game, if any."""
    required = {"game_type", "week", "home_score", "away_score"}
    missing = sorted(required - set(schedule.columns))
    if missing:
        raise ValueError(f"Schedule is missing required columns: {missing}")

    regular_season = schedule[schedule["game_type"] == "REG"]
    unplayed = regular_season[
        regular_season["home_score"].isna() | regular_season["away_score"].isna()
    ]
    if unplayed.empty:
        return None
    return int(unplayed["week"].min())
