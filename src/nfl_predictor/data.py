"""Data acquisition boundary for nflverse."""

from __future__ import annotations

import pandas as pd


def last_completed_seasons(end_season: int, count: int = 10) -> list[int]:
    """Return a contiguous, inclusive list of completed NFL season labels."""
    if count < 1:
        raise ValueError("count must be positive")
    if end_season < 1999:
        raise ValueError("nflverse team statistics start in 1999")
    return list(range(end_season - count + 1, end_season + 1))


def load_nflverse_data(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download weekly team stats and schedules and convert them to pandas."""
    import nflreadpy as nfl

    team_stats = nfl.load_team_stats(seasons=seasons, summary_level="week").to_pandas()
    schedule = nfl.load_schedules(seasons=seasons).to_pandas()
    return team_stats, schedule


def load_nflverse_schedules(seasons: list[int]) -> pd.DataFrame:
    """Download schedules without requiring team statistics for those seasons."""
    import nflreadpy as nfl

    return nfl.load_schedules(seasons=seasons).to_pandas()
