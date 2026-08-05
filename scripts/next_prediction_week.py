"""Print the next unplayed regular-season week for the publishing workflow."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from nfl_predictor.data import load_nflverse_schedules
from nfl_predictor.publishing import next_unplayed_week, season_for_date


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    args = parser.parse_args()

    season = args.season or season_for_date(datetime.now(UTC).date())
    week = next_unplayed_week(load_nflverse_schedules([season]))
    if week is not None:
        print(season, week)


if __name__ == "__main__":
    main()
