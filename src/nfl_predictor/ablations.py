"""Run chronological feature and rolling-window ablation experiments."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .model import FEATURE_GROUPS, make_model


def walk_forward_splits(
    data: pd.DataFrame, min_train_seasons: int = 4
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield one future season at a time after an expanding training history."""
    seasons = sorted(data["season"].unique())
    if min_train_seasons < 1 or len(seasons) <= min_train_seasons:
        raise ValueError("Need more seasons than the minimum training window.")
    for test_season in seasons[min_train_seasons:]:
        yield (
            int(test_season),
            data[data["season"] < test_season].copy(),
            data[data["season"] == test_season].copy(),
        )


def _evaluate_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.Series, pd.Series]:
    model = make_model()
    model.fit(train[feature_columns], train["home_win"])
    probabilities = pd.Series(model.predict_proba(test[feature_columns])[:, 1], index=test.index)
    outcomes = test["home_win"].astype(int)
    return outcomes, probabilities


def _metrics(outcomes: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(outcomes, probabilities >= 0.5)),
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(outcomes, probabilities)),
    }


def evaluate_feature_table(
    data: pd.DataFrame,
    feature_group: str,
    min_train_seasons: int = 4,
    test_seasons: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-season and pooled results for one feature table and group."""
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {feature_group}")
    feature_columns = FEATURE_GROUPS[feature_group]
    seasons = sorted(data["season"].unique())
    fixed_train = data[data["season"] < seasons[-test_seasons]].copy()
    fixed_test = data[data["season"] >= seasons[-test_seasons]].copy()
    evaluations = {
        "walk_forward": list(walk_forward_splits(data, min_train_seasons)),
        "fixed_holdout": [(-1, fixed_train, fixed_test)],
    }

    season_rows: list[dict[str, float | int | str]] = []
    pooled: list[tuple[str, pd.Series, pd.Series]] = []
    for evaluation, splits in evaluations.items():
        all_outcomes: list[pd.Series] = []
        all_probabilities: list[pd.Series] = []
        for test_season, train, test in splits:
            outcomes, probabilities = _evaluate_split(train, test, feature_columns)
            row = {
                "evaluation": evaluation,
                "test_season": int(test_season) if test_season != -1 else "all",
                "rows": len(test),
                **_metrics(outcomes, probabilities),
            }
            season_rows.append(row)
            all_outcomes.append(outcomes)
            all_probabilities.append(probabilities)
        pooled.append(
            (
                evaluation,
                pd.concat(all_outcomes),
                pd.concat(all_probabilities),
            )
        )

    summary_rows = [
        {
            "evaluation": evaluation,
            "test_season": "all",
            "rows": len(outcomes),
            **_metrics(outcomes, probabilities),
        }
        for evaluation, outcomes, probabilities in pooled
    ]
    return pd.DataFrame(season_rows), pd.DataFrame(summary_rows)


def run_ablations(
    feature_tables: dict[str, pd.DataFrame],
    feature_groups: Iterable[str] = FEATURE_GROUPS,
    min_train_seasons: int = 4,
    test_seasons: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all requested feature-group experiments across rolling windows."""
    all_season_rows: list[pd.DataFrame] = []
    all_summary_rows: list[pd.DataFrame] = []
    for window, data in feature_tables.items():
        for feature_group in feature_groups:
            season_results, summary_results = evaluate_feature_table(
                data,
                feature_group,
                min_train_seasons=min_train_seasons,
                test_seasons=test_seasons,
            )
            for frame in (season_results, summary_results):
                frame.insert(0, "feature_group", feature_group)
                frame.insert(0, "rolling_window", window)
            all_season_rows.append(season_results)
            all_summary_rows.append(summary_results)
    return pd.concat(all_season_rows, ignore_index=True), pd.concat(all_summary_rows, ignore_index=True)
