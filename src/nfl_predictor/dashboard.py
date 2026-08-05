"""Artifact loading and refresh helpers for the local Streamlit demo."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PREDICTION_PATTERN = re.compile(r"predictions_(?P<season>\d{4})_week_(?P<week>\d+)\.csv")


def prediction_runs(project_root: Path) -> list[tuple[int, int]]:
    """Return available prediction artifacts as sorted (season, week) pairs."""
    runs = []
    for path in (project_root / "artifacts" / "predictions").glob("predictions_*_week_*.csv"):
        match = PREDICTION_PATTERN.fullmatch(path.name)
        if match:
            runs.append((int(match["season"]), int(match["week"])))
    return sorted(set(runs))


def prediction_paths(project_root: Path, season: int, week: int) -> dict[str, Path]:
    """Return the artifact paths associated with one prediction run."""
    base = project_root / "artifacts" / "predictions" / f"predictions_{season}_week_{week}"
    return {"csv": base.with_suffix(".csv"), "chart": base.with_suffix(".png"), "report": base.with_suffix(".md")}


def load_predictions(project_root: Path, season: int, week: int) -> pd.DataFrame:
    """Load a generated prediction CSV or fail with an actionable error."""
    path = prediction_paths(project_root, season, week)["csv"]
    if not path.exists():
        raise FileNotFoundError(f"No saved predictions for {season} Week {week}. Run a refresh first.")
    return pd.read_csv(path, parse_dates=["game_date"])


def load_json(path: Path) -> dict:
    """Load a JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_predictions(project_root: Path, season: int, week: int) -> subprocess.CompletedProcess[str]:
    """Run the existing prediction CLI without duplicating pipeline behavior in the UI."""
    return subprocess.run(
        [
            sys.executable,
            "scripts/predict_week.py",
            "--season",
            str(season),
            "--week",
            str(week),
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
