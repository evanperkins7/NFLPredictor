"""Audits for temporal leakage in the game feature table."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .features import FEATURE_COLUMNS, build_game_features


AUDIT_COLUMNS = ["game_id", "season", "week", "game_date", *FEATURE_COLUMNS, "home_win"]


def audit_feature_table(features: pd.DataFrame) -> dict[str, Any]:
    """Return deterministic structural checks for a materialized feature table."""
    required = set(AUDIT_COLUMNS)
    missing_columns = sorted(required - set(features.columns))
    duplicate_game_rows = int(features["game_id"].duplicated().sum()) if "game_id" in features else -1
    missing_feature_values = int(features[FEATURE_COLUMNS].isna().sum().sum()) if not missing_columns else -1
    invalid_team_matchups = int((features["home_team"] == features["away_team"]).sum()) if {"home_team", "away_team"} <= set(features.columns) else -1
    invalid_week_values = int((features["week"] < 1).sum()) if "week" in features else -1
    return {
        "rows": len(features),
        "missing_columns": missing_columns,
        "duplicate_game_rows": duplicate_game_rows,
        "missing_feature_values": missing_feature_values,
        "invalid_team_matchups": invalid_team_matchups,
        "invalid_week_values": invalid_week_values,
        "passed": not missing_columns
        and duplicate_game_rows == 0
        and missing_feature_values == 0
        and invalid_team_matchups == 0
        and invalid_week_values == 0,
    }


def mutation_audit(
    team_stats: pd.DataFrame, schedule: pd.DataFrame, mutation_game_id: str
) -> dict[str, Any]:
    """Verify that mutating a game's stats cannot change current or earlier rows."""
    baseline = build_game_features(team_stats, schedule)
    if mutation_game_id not in set(baseline["game_id"]):
        raise ValueError(f"Mutation game {mutation_game_id!r} is not represented in the feature table.")
    mutated_stats = team_stats.copy()
    mask = mutated_stats["game_id"] == mutation_game_id
    if not mask.any():
        raise ValueError(f"Mutation game {mutation_game_id!r} was not found in team stats.")
    numeric_columns = mutated_stats.loc[mask].select_dtypes(include="number").columns
    for column in numeric_columns:
        if column not in {"season", "week"}:
            mutated_stats.loc[mask, column] = mutated_stats.loc[mask, column].fillna(0) + 1_000_000

    mutated = build_game_features(mutated_stats, schedule)
    baseline_by_game = baseline.set_index("game_id").sort_index()
    mutated_by_game = mutated.set_index("game_id").sort_index()
    common_games = baseline_by_game.index.intersection(mutated_by_game.index)
    target_date = baseline_by_game.loc[mutation_game_id, "game_date"]
    protected_games = baseline_by_game.loc[common_games, "game_date"] <= target_date
    protected_ids = common_games[protected_games]
    compared_columns = [*FEATURE_COLUMNS, "home_win", "home_margin"]
    unchanged = baseline_by_game.loc[protected_ids, compared_columns].equals(
        mutated_by_game.loc[protected_ids, compared_columns]
    )
    return {
        "mutation_game_id": mutation_game_id,
        "protected_rows": len(protected_ids),
        "protected_rows_unchanged": bool(unchanged),
        "passed": bool(unchanged),
    }
