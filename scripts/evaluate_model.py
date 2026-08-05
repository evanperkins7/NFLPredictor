"""Create an explainable evaluation report for the chronological model holdout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)

from nfl_predictor.features import FEATURE_COLUMNS
from nfl_predictor.model import make_model, time_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-seasons", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model/evaluation"))
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    train, test = time_split(data, test_seasons=args.test_seasons)
    model = make_model()
    model.fit(train[FEATURE_COLUMNS], train["home_win"])
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    args.output.mkdir(parents=True, exist_ok=True)
    season_rows = []
    for season, group in test.assign(_probability=probabilities).groupby("season"):
        season_probability = group["_probability"]
        season_rows.append(
            {
                "season": int(season),
                "rows": len(group),
                "accuracy": accuracy_score(group["home_win"], season_probability >= 0.5),
                "log_loss": log_loss(group["home_win"], season_probability, labels=[0, 1]),
                "brier_score": brier_score_loss(group["home_win"], season_probability),
                "home_win_rate": group["home_win"].mean(),
            }
        )
    season_metrics = pd.DataFrame(season_rows)
    season_metrics.to_csv(args.output / "season_metrics.csv", index=False)

    classifier = model.named_steps["classifier"]
    coefficients = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "coefficient": classifier.coef_[0]}
    ).sort_values("coefficient", ascending=False)
    coefficients.to_csv(args.output / "coefficients.csv", index=False)

    metrics = {
        "train_seasons": sorted(train["season"].unique().tolist()),
        "test_seasons": sorted(test["season"].unique().tolist()),
        "train_rows": len(train),
        "test_rows": len(test),
        "accuracy": accuracy_score(test["home_win"], predictions),
        "log_loss": log_loss(test["home_win"], probabilities, labels=[0, 1]),
        "brier_score": brier_score_loss(test["home_win"], probabilities),
        "always_home_accuracy": accuracy_score(test["home_win"], [1] * len(test)),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    observed, predicted = calibration_curve(test["home_win"], probabilities, n_bins=10, strategy="quantile")
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(predicted, observed, marker="o", label="NFL predictor")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set(title="Home-win probability calibration", xlabel="Mean predicted probability", ylabel="Observed home-win rate")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "calibration.png", dpi=160)
    plt.close(fig)

    matrix = confusion_matrix(test["home_win"], predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(title="Home-win prediction confusion matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(args.output / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation report to {args.output}")


if __name__ == "__main__":
    main()
