"""Train and evaluate the first chronological logistic-regression baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_predictor.model import fit_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--test-seasons", type=int, default=2)
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/model/metrics.json"))
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    _, metrics = fit_and_evaluate(data, test_seasons=args.test_seasons)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()

