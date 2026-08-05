"""Download seasons and materialize the leakage-aware game feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from nfl_predictor.data import load_nflverse_data
from nfl_predictor.features import build_game_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2024)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/features"))
    args = parser.parse_args()
    end_season = args.end_season or args.start_season
    seasons = list(range(args.start_season, end_season + 1))

    team_stats, schedule = load_nflverse_data(seasons)
    features = build_game_features(team_stats, schedule)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"game_features_{args.start_season}_{end_season}.csv"
    features.to_csv(destination, index=False)
    print(f"Built {len(features):,} rows across seasons {args.start_season}-{end_season}.")
    print(f"Saved feature table to {destination}")
    print(features[["season", "week", "home_team", "away_team", *features.columns[-6:]]].head())


if __name__ == "__main__":
    main()

