"""Refresh data and export weekly NFL win probabilities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from nfl_predictor.data import last_completed_seasons, load_nflverse_data, load_nflverse_schedules
from nfl_predictor.weekly import generate_weekly_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--history-seasons", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions"))
    args = parser.parse_args()
    history_seasons = last_completed_seasons(args.season - 1, args.history_seasons)
    team_stats, historical_schedule = load_nflverse_data(history_seasons)
    target_schedule = load_nflverse_schedules([args.season])
    predictions, _ = generate_weekly_predictions(
        team_stats,
        historical_schedule,
        target_schedule,
        rolling_window=8,
        feature_group="epa",
    )
    if args.week is not None:
        predictions = predictions[predictions["week"] == args.week].copy()
    if predictions.empty:
        raise SystemExit("No upcoming games matched the requested season/week.")

    args.output.mkdir(parents=True, exist_ok=True)
    week_label = f"week_{args.week}" if args.week is not None else "upcoming"
    csv_path = args.output / f"predictions_{args.season}_{week_label}.csv"
    chart_path = args.output / f"predictions_{args.season}_{week_label}.png"
    report_path = args.output / f"predictions_{args.season}_{week_label}.md"
    export_columns = [
        "game_id", "season", "week", "game_date", "home_team", "away_team",
        "home_win_probability", "predicted_winner", "confidence_tier",
    ]
    predictions[export_columns].to_csv(csv_path, index=False)

    chart_data = predictions.assign(
        matchup=predictions["away_team"] + " at " + predictions["home_team"]
    ).sort_values("home_win_probability")
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(9, max(4, len(chart_data) * 0.35)))
    sns.barplot(
        data=chart_data,
        x="home_win_probability",
        y="matchup",
        hue="confidence_tier",
        dodge=False,
        palette={"low": "#9ca3af", "medium": "#60a5fa", "high": "#2563eb"},
        legend=False,
        ax=axis,
    )
    axis.set(xlim=(0, 1), title=f"NFL home-win probabilities: {args.season} {week_label}", xlabel="Home-team win probability", ylabel="")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    report_path.write_text(
        "\n".join(
            [
                f"# NFL predictions: {args.season} {week_label}",
                "",
                "- Model: logistic regression with standardized 8-game EPA history.",
                f"- Games: {len(predictions)}",
                f"- CSV: `{csv_path}`",
                f"- Chart: `{chart_path}`",
                "",
                "Predictions are probabilities, not guarantees; confidence tiers are a communication aid.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved predictions to {csv_path}")
    print(f"Saved chart to {chart_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
