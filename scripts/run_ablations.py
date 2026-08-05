"""Run the Step 4 feature-group and rolling-window experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from nfl_predictor.ablations import run_ablations
from nfl_predictor.data import load_nflverse_data
from nfl_predictor.features import build_game_features
from nfl_predictor.model import FEATURE_GROUPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--min-train-seasons", type=int, default=4)
    parser.add_argument("--test-seasons", type=int, default=2)
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=sorted(FEATURE_GROUPS),
        default=sorted(FEATURE_GROUPS),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/model/ablations"))
    args = parser.parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))
    team_stats, schedule = load_nflverse_data(seasons)

    feature_tables = {
        "expanding": build_game_features(team_stats, schedule),
        "3": build_game_features(team_stats, schedule, rolling_window=3),
        "5": build_game_features(team_stats, schedule, rolling_window=5),
        "8": build_game_features(team_stats, schedule, rolling_window=8),
    }
    season_results, summary_results = run_ablations(
        feature_tables,
        feature_groups=args.feature_groups,
        min_train_seasons=args.min_train_seasons,
        test_seasons=args.test_seasons,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    season_path = args.output / "season_results.csv"
    summary_path = args.output / "summary_results.csv"
    season_results.to_csv(season_path, index=False)
    summary_results.to_csv(summary_path, index=False)
    print(f"Saved per-season results to {season_path}")
    print(f"Saved pooled results to {summary_path}")
    print(
        summary_results[summary_results["evaluation"] == "walk_forward"]
        .sort_values("log_loss")
        [["rolling_window", "feature_group", "accuracy", "log_loss", "brier_score"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
