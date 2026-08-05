"""Inspect the nflverse schemas used by the first modeling milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=Path("artifacts/schema"))
    args = parser.parse_args()

    team_stats = nfl.load_team_stats(seasons=[args.season], summary_level="week").to_pandas()
    schedules = nfl.load_schedules(seasons=[args.season]).to_pandas()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "season": args.season,
        "team_stats": {
            "rows": len(team_stats),
            "columns": list(team_stats.columns),
            "epa_candidates": [
                column
                for column in team_stats.columns
                if "epa" in column.lower()
            ],
            "pace_candidates": [
                column
                for column in team_stats.columns
                if any(term in column.lower() for term in ("play", "pace", "snap"))
            ],
        },
        "schedules": {
            "rows": len(schedules),
            "columns": list(schedules.columns),
            "date_candidates": [
                column
                for column in schedules.columns
                if any(term in column.lower() for term in ("date", "day"))
            ],
        },
    }
    destination = args.output / f"schema_{args.season}.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved schema report to {destination}")


if __name__ == "__main__":
    main()

