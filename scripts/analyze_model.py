"""Create the interview-ready Markdown interpretation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_predictor.analysis import build_analysis_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, default=Path("artifacts/model/evaluation"))
    args = parser.parse_args()

    metrics = json.loads((args.evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
    season_metrics = pd.read_csv(args.evaluation_dir / "season_metrics.csv")
    coefficients = pd.read_csv(args.evaluation_dir / "coefficients.csv")
    report = build_analysis_markdown(metrics, season_metrics, coefficients)
    destination = args.evaluation_dir / "analysis.md"
    destination.write_text(report, encoding="utf-8")
    print(f"Saved model analysis to {destination}")


if __name__ == "__main__":
    main()

