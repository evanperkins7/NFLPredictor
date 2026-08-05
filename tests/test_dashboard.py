from pathlib import Path

from nfl_predictor.dashboard import prediction_paths, prediction_runs


def test_prediction_runs_discovers_saved_csvs(tmp_path: Path):
    directory = tmp_path / "artifacts" / "predictions"
    directory.mkdir(parents=True)
    (directory / "predictions_2026_week_1.csv").write_text("game_id\nexample\n", encoding="utf-8")
    (directory / "unexpected.csv").write_text("ignore\n", encoding="utf-8")
    assert prediction_runs(tmp_path) == [(2026, 1)]


def test_prediction_paths_are_deterministic(tmp_path: Path):
    paths = prediction_paths(tmp_path, 2026, 1)
    assert paths["csv"].name == "predictions_2026_week_1.csv"
    assert paths["chart"].name == "predictions_2026_week_1.png"
