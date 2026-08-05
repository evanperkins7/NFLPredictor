"""Build pre-game features from nflverse weekly team statistics and schedules."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

FEATURE_COLUMNS = [
    "offensive_epa_diff",
    "defensive_epa_diff",
    "pace_diff",
    "rest_days_diff",
]
NEUTRAL_PACE_FEATURE = "neutral_pace_diff"


def _find_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    options = ", ".join(candidates)
    raise ValueError(f"Could not find {label} in nflverse data. Expected one of: {options}")


def _date_column(schedule: pd.DataFrame) -> str:
    return _find_column(schedule, ("gameday", "game_date", "date"), "game date")


def _prepare_team_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Normalize the small schema surface needed by the feature builder."""
    required = {
        "season": _find_column(team_stats, ("season",), "season"),
        "week": _find_column(team_stats, ("week",), "week"),
        "team": _find_column(team_stats, ("team",), "team"),
        "game_id": _find_column(team_stats, ("game_id", "gsis_id"), "game id"),
        "opponent_team": _find_column(team_stats, ("opponent_team",), "opponent team"),
    }
    out = team_stats.rename(columns={value: key for key, value in required.items()}).copy()
    direct_offense = next(
        (column for column in ("offensive_epa", "off_epa", "epa") if column in out.columns),
        None,
    )
    if direct_offense:
        out["offensive_epa"] = out[direct_offense]
    else:
        passing_epa = _find_column(out, ("passing_epa",), "passing EPA")
        rushing_epa = _find_column(out, ("rushing_epa",), "rushing EPA")
        out["offensive_epa"] = out[passing_epa].fillna(0) + out[rushing_epa].fillna(0)

    direct_defense = next(
        (column for column in ("defensive_epa", "def_epa") if column in out.columns), None
    )
    if direct_defense:
        out["defensive_epa"] = out[direct_defense]
    else:
        offense_by_team = out.set_index(["game_id", "team"])["offensive_epa"]
        opponent_keys = pd.MultiIndex.from_frame(out[["game_id", "opponent_team"]])
        out["defensive_epa"] = offense_by_team.reindex(opponent_keys).to_numpy()

    direct_pace = next(
        (column for column in ("plays", "offensive_plays", "total_plays") if column in out.columns),
        None,
    )
    if direct_pace:
        out["pace"] = out[direct_pace]
    else:
        attempts = _find_column(out, ("attempts",), "pass attempts")
        carries = _find_column(out, ("carries",), "rushing attempts")
        sacks = _find_column(out, ("sacks_suffered",), "sacks suffered")
        out["pace"] = out[attempts].fillna(0) + out[carries].fillna(0) + out[sacks].fillna(0)

    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out[
        ["season", "week", "team", "opponent_team", "game_id", "offensive_epa", "defensive_epa", "pace"]
    ]


def build_game_features(
    team_stats: pd.DataFrame,
    schedule: pd.DataFrame,
    rolling_window: int | None = None,
    include_unplayed: bool = False,
    neutral_pace: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per game using only information available before kickoff.

    The weekly team statistics are shifted by one game within each team before rolling
    aggregation. This is the core leakage guard: a game's own stats cannot become an
    input to its prediction row.
    """
    if rolling_window is not None and rolling_window < 1:
        raise ValueError("rolling_window must be positive or None")

    stats = _prepare_team_stats(team_stats)
    metric_names = ["offensive_epa", "defensive_epa", "pace"]
    if neutral_pace is not None:
        required_pace_columns = {"game_id", "team", "neutral_pace"}
        missing_pace_columns = sorted(required_pace_columns - set(neutral_pace.columns))
        if missing_pace_columns:
            raise ValueError(
                f"Neutral pace data is missing required columns: {missing_pace_columns}"
            )
        stats = stats.merge(
            neutral_pace[["game_id", "team", "neutral_pace"]],
            on=["game_id", "team"],
            how="left",
        )
        stats["neutral_pace"] = stats["neutral_pace"].fillna(0)
        metric_names.append("neutral_pace")
    schedule = schedule.copy()
    required = {"season", "week", "game_id", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule is missing required columns: {sorted(missing)}")

    date_col = _date_column(schedule)
    schedule["game_date"] = pd.to_datetime(schedule[date_col], errors="coerce")
    schedule = schedule[schedule["game_type"] == "REG"].copy()
    if not include_unplayed:
        schedule = schedule[schedule["home_score"].notna() & schedule["away_score"].notna()].copy()
    schedule = schedule.sort_values(["season", "game_date", "game_id"])

    stats = stats.merge(schedule[["game_id", "game_date"]], on="game_id", how="inner")
    stats = stats.sort_values(["team", "game_date", "game_id"])
    for name in metric_names:
        stats[f"prior_{name}"] = stats.groupby("team", sort=False)[name].shift(1)
        prior_values = stats.groupby("team", sort=False)[f"prior_{name}"]
        if rolling_window is None:
            rolling_values = prior_values.transform(
                lambda series: series.expanding(min_periods=1).mean()
            )
        else:
            rolling_values = prior_values.transform(
                lambda series: series.rolling(rolling_window, min_periods=1).mean()
            )
        stats[f"rolling_{name}"] = rolling_values

    game_dates = schedule[["game_id", "home_team", "away_team", "game_date"]].copy()
    latest_team_stats = stats.sort_values(["team", "game_date", "game_id"]).drop_duplicates(
        "team", keep="last"
    )
    for side in ("home", "away"):
        side_stats = stats.rename(columns={"team": f"{side}_team"})
        side_stats = side_stats[
            [
                "game_id",
                f"{side}_team",
                *[f"rolling_{name}" for name in metric_names],
            ]
        ].rename(
            columns={
                f"rolling_{name}": f"{side}_{name}"
                for name in metric_names
            }
        )
        game_dates = game_dates.merge(side_stats, on=["game_id", f"{side}_team"], how="left")
        latest_side_stats = latest_team_stats.rename(columns={"team": f"{side}_team"})[
            [f"{side}_team", *[f"rolling_{name}" for name in metric_names]]
        ].rename(
            columns={
                f"rolling_{name}": f"latest_{side}_{name}"
                for name in metric_names
            }
        )
        game_dates = game_dates.merge(latest_side_stats, on=f"{side}_team", how="left")
        for metric in metric_names:
            current_column = f"{side}_{metric}"
            latest_column = f"latest_{side}_{metric}"
            game_dates[current_column] = game_dates[current_column].fillna(game_dates[latest_column])
            game_dates = game_dates.drop(columns=latest_column)

    appearances = pd.concat(
        [
            schedule[["game_id", "home_team", "game_date"]].rename(columns={"home_team": "team"}),
            schedule[["game_id", "away_team", "game_date"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).sort_values(["team", "game_date", "game_id"])
    appearances["rest_days"] = appearances.groupby("team", sort=False)["game_date"].diff().dt.days
    appearances["rest_days"] = appearances["rest_days"].fillna(7)
    schedule = schedule.merge(
        appearances[["game_id", "team", "rest_days"]].rename(
            columns={"team": "home_team", "rest_days": "home_rest_days"}
        ),
        on=["game_id", "home_team"],
        how="left",
    ).merge(
        appearances[["game_id", "team", "rest_days"]].rename(
            columns={"team": "away_team", "rest_days": "away_rest_days"}
        ),
        on=["game_id", "away_team"],
        how="left",
    )

    result = game_dates.merge(
        schedule[
            [
                "game_id", "season", "week", "home_score", "away_score",
                "home_rest_days", "away_rest_days",
            ]
        ],
        on="game_id",
        how="left",
    )
    result["home_win"] = (result["home_score"] > result["away_score"]).where(
        result["home_score"].notna() & result["away_score"].notna()
    )
    result["home_margin"] = result["home_score"] - result["away_score"]
    result["offensive_epa_diff"] = result["home_offensive_epa"] - result["away_offensive_epa"]
    result["defensive_epa_diff"] = result["home_defensive_epa"] - result["away_defensive_epa"]
    result["pace_diff"] = result["home_pace"] - result["away_pace"]
    result["rest_days_diff"] = result["home_rest_days"] - result["away_rest_days"]
    if neutral_pace is not None:
        result[NEUTRAL_PACE_FEATURE] = (
            result["home_neutral_pace"] - result["away_neutral_pace"]
        )
    return result.dropna(subset=FEATURE_COLUMNS).sort_values(["season", "week", "game_id"])


def build_upcoming_features(
    team_stats: pd.DataFrame,
    schedule: pd.DataFrame,
    rolling_window: int | None = None,
    neutral_pace: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build pre-game features for schedule rows without completed scores."""
    features = build_game_features(
        team_stats,
        schedule,
        rolling_window=rolling_window,
        include_unplayed=True,
        neutral_pace=neutral_pace,
    )
    upcoming = features[features["home_win"].isna()].copy()
    return upcoming.drop(columns=["home_win", "home_margin"])
