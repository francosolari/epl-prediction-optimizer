"""Tests for the active EPL season context."""

from datetime import date

import pytest

from epl_prediction_optimizer.config.season import SeasonContext


def test_from_date_uses_new_season_after_august_boundary() -> None:
    season = SeasonContext.from_date(date(2026, 8, 21))

    assert season.code == "2627"
    assert season.start_year == 2026
    assert season.label == "2026–27"


def test_from_date_uses_previous_season_before_july_boundary() -> None:
    season = SeasonContext.from_date(date(2027, 2, 1))

    assert season.code == "2627"


def test_current_season_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EPLPO_SEASON", "2425")

    assert SeasonContext.current_season(today=date(2026, 8, 21)).code == "2425"


def test_from_code_rejects_non_four_digit_code() -> None:
    with pytest.raises(ValueError, match="four digits"):
        SeasonContext.from_code("26x7")


def test_from_code_rejects_non_consecutive_season_years() -> None:
    with pytest.raises(ValueError, match="consecutive years"):
        SeasonContext.from_code("2026")
