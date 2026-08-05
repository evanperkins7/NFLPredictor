import pytest

from nfl_predictor.data import last_completed_seasons


def test_last_completed_seasons_is_contiguous():
    assert last_completed_seasons(2025, 10) == list(range(2016, 2026))


def test_last_completed_seasons_rejects_invalid_count():
    with pytest.raises(ValueError, match="positive"):
        last_completed_seasons(2025, 0)

