"""Centralized context for the active Premier League season."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


class SeasonMismatchError(ValueError):
    """Raised when a source belongs to a season other than the active season."""

    def __init__(self, source_season: str, expected_season: SeasonContext) -> None:
        self.source_season = source_season
        self.expected_season = expected_season
        super().__init__(
            f"Source season {source_season} does not match active season {expected_season.code}."
        )


@dataclass(frozen=True)
class SeasonContext:
    """Validated identifiers and display values for one EPL season."""

    code: str
    start_year: int
    label: str

    @classmethod
    def from_code(cls, code: str) -> SeasonContext:
        """Build a season context from a compact code such as ``2627``."""
        if len(code) != 4 or not code.isdigit():
            raise ValueError("Season code must contain four digits.")
        start_suffix = int(code[:2])
        end_suffix = int(code[2:])
        if (start_suffix + 1) % 100 != end_suffix:
            raise ValueError("Season code must contain consecutive years.")
        start_year = 2000 + start_suffix
        return cls(code=code, start_year=start_year, label=f"{start_year}–{end_suffix:02d}")

    @classmethod
    def from_date(cls, today: date) -> SeasonContext:
        """Return the EPL season containing ``today`` using the July boundary."""
        start_year = today.year if today.month >= 7 else today.year - 1
        return cls.from_code(f"{start_year % 100:02d}{(start_year + 1) % 100:02d}")

    @classmethod
    def current_season(
        cls,
        today: date | None = None,
        override: str | None = None,
    ) -> SeasonContext:
        """Return the configured active season, respecting the environment override."""
        configured_code = override or os.getenv("EPLPO_SEASON")
        return cls.from_code(configured_code or cls.from_date(today or date.today()).code)

    def require_source_season(self, source_season: str) -> None:
        """Raise a typed error unless ``source_season`` is this active season."""
        normalized_source = str(source_season).zfill(4)
        if normalized_source != self.code:
            raise SeasonMismatchError(normalized_source, self)


def require_source_season(source_season: str, season: SeasonContext) -> None:
    """Require that a source season matches ``season``."""
    season.require_source_season(source_season)
