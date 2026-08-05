"""Run structural and adversarial leakage audits on a feature table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nfl_predictor.data import load_nflverse_data
from nfl_predictor.leakage import audit_feature_table, mutation_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model/leakage_audit.json"))
    parser.add_argument("--skip-raw-download", action="store_true")
    args = parser.parse_args()

    features = pd.read_csv(args.input, parse_dates=["game_date"])
    structural = audit_feature_table(features)
    if args.skip_raw_download:
        adversarial = {"status": "not_run", "reason": "Raw nflverse download was skipped."}
    else:
        seasons = sorted(features["season"].unique().tolist())
        team_stats, schedule = load_nflverse_data([int(season) for season in seasons])
        mutation_game_id = str(features.iloc[len(features) // 2]["game_id"])
        adversarial = mutation_audit(team_stats, schedule, mutation_game_id)
    report = {
        "structural": structural,
        "adversarial": adversarial,
        "complete": not args.skip_raw_download,
        "passed": structural["passed"]
        and (adversarial.get("passed", False) if not args.skip_raw_download else True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved leakage audit to {args.output}")
    if not report["passed"]:
        raise SystemExit("Leakage audit failed.")


if __name__ == "__main__":
    main()
