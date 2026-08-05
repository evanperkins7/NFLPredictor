"""Evaluate time-aware probability calibration for the selected model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from nfl_predictor.calibration import run_calibration_experiment
from nfl_predictor.data import load_nflverse_data
from nfl_predictor.features import build_game_features
from nfl_predictor.model import FEATURE_GROUPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-season", type=int, default=2016)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--min-train-seasons", type=int, default=4)
    parser.add_argument("--test-seasons", type=int, default=2)
    parser.add_argument("--calibration-seasons", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model/calibration"))
    args = parser.parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))
    team_stats, schedule = load_nflverse_data(seasons)
    features = build_game_features(team_stats, schedule, rolling_window=8)
    season_results, summary_results, reliability = run_calibration_experiment(
        features,
        FEATURE_GROUPS["epa"],
        min_train_seasons=args.min_train_seasons,
        test_seasons=args.test_seasons,
        calibration_seasons=args.calibration_seasons,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    season_path = args.output / "season_results.csv"
    summary_path = args.output / "summary_results.csv"
    reliability_path = args.output / "reliability.csv"
    chart_path = args.output / "reliability.png"
    season_results.to_csv(season_path, index=False)
    summary_results.to_csv(summary_path, index=False)
    reliability.to_csv(reliability_path, index=False)

    chart_data = reliability[reliability["evaluation"] == "walk_forward"]
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.lineplot(
        data=chart_data,
        x="mean_probability",
        y="outcome_rate",
        hue="method",
        marker="o",
        ax=axis,
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        title="Walk-forward calibration comparison",
        xlabel="Mean predicted probability",
        ylabel="Observed home-win rate",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    print(f"Saved per-season results to {season_path}")
    print(f"Saved pooled results to {summary_path}")
    print(f"Saved reliability data to {reliability_path}")
    print(f"Saved reliability chart to {chart_path}")
    print(
        summary_results[summary_results["evaluation"] == "walk_forward"]
        .sort_values(["log_loss", "brier_score"])
        [["method", "accuracy", "log_loss", "brier_score", "expected_calibration_error"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
