"""Neutral-situation play filtering and team pace aggregation."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "game_id",
    "posteam",
    "play_type",
    "qtr",
    "score_differential",
    "qb_kneel",
    "qb_spike",
    "penalty",
    "play_deleted",
    "aborted_play",
}
OFFENSIVE_PLAY_TYPES = {"pass", "run", "sack"}


def neutral_play_mask(pbp: pd.DataFrame) -> pd.Series:
    """Return plays in quarters 1–3 within one possession of the opponent."""
    missing = sorted(REQUIRED_COLUMNS - set(pbp.columns))
    if missing:
        raise ValueError(f"Play-by-play data is missing required columns: {missing}")

    return (
        pbp["posteam"].notna()
        & pbp["play_type"].isin(OFFENSIVE_PLAY_TYPES)
        & pbp["qtr"].between(1, 3)
        & pbp["score_differential"].abs().le(8)
        & ~pbp["qb_kneel"].fillna(False).astype(bool)
        & ~pbp["qb_spike"].fillna(False).astype(bool)
        & ~pbp["penalty"].fillna(False).astype(bool)
        & ~pbp["play_deleted"].fillna(False).astype(bool)
        & ~pbp["aborted_play"].fillna(False).astype(bool)
    )


def build_neutral_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Count neutral offensive plays for each team in each game."""
    neutral = pbp.loc[neutral_play_mask(pbp), ["season", "game_id", "posteam"]].copy()
    neutral = neutral.rename(columns={"posteam": "team"})
    return (
        neutral.groupby(["season", "game_id", "team"], as_index=False)
        .size()
        .rename(columns={"size": "neutral_pace"})
    )
