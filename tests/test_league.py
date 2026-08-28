"""Tests for importing the contest standings spreadsheet."""

from __future__ import annotations

import pytest

from epl_prediction_optimizer.data.league import (
    ContestSheetError,
    parse_contest_sheet,
    parse_slot,
    sheet_export_url,
)

SHEET_URL = "https://docs.google.com/spreadsheets/d/ABC123_x/edit?pli=1&gid=1052054025#gid=1052054025"

SAMPLE = """Entrant / place in standings:,Ann Lee,,1,Bo Ray,,2
,Auto,Points,GD,Auto,Points,GD
,xx,12,4,,9,2
,Week,Result,(+/-),Week,Result,(+/-)
Arsenal FC (H),Week 2,3,5,Week 15,3,1
Arsenal FC (A),Week 10,1,0,,,
Brighton & Hove (A),Week 4,3,2,Week 4,0,-1
Wolverhampton (H),,,,Week 7,3,1
,,,,,,
Player,,,,,,
Ann Lee,,,,,,
"""


def test_export_url_is_derived_from_an_ordinary_edit_link() -> None:
    assert sheet_export_url(SHEET_URL) == (
        "https://docs.google.com/spreadsheets/d/ABC123_x/export?format=csv&gid=1052054025"
    )


def test_export_url_defaults_to_the_first_tab() -> None:
    url = "https://docs.google.com/spreadsheets/d/ABC123_x/edit"
    assert url.replace("/edit", "/export?format=csv&gid=0") == sheet_export_url(url)


def test_a_non_sheets_url_is_rejected() -> None:
    with pytest.raises(ContestSheetError):
        sheet_export_url("https://example.com/standings.csv")


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Arsenal FC (H)", ("Arsenal", "home")),
        ("Brighton & Hove (A)", ("Brighton & Hove Albion", "away")),
        ("Wolverhampton (H)", ("Wolverhampton Wanderers", "home")),
        ("Leeds (A)", ("Leeds United", "away")),
    ],
)
def test_slot_labels_map_to_canonical_team_and_venue(label: str, expected: tuple) -> None:
    assert parse_slot(label) == expected


def test_rows_without_a_venue_marker_are_not_slots() -> None:
    assert parse_slot("Player") is None
    assert parse_slot("") is None


def test_entrants_carry_place_and_totals() -> None:
    entrants, _ = parse_contest_sheet(SAMPLE, "2526")
    assert list(entrants["entrant"]) == ["Ann Lee", "Bo Ray"]
    ann = entrants.set_index("entrant").loc["Ann Lee"]
    assert ann["place"] == "1"
    assert ann["points"] == 12
    assert ann["goal_diff"] == 4


def test_picks_are_one_row_per_used_slot() -> None:
    _, picks = parse_contest_sheet(SAMPLE, "2526")
    ann = picks[picks["entrant"] == "Ann Lee"].set_index("contest_week")
    assert set(ann.index) == {2, 10, 4}
    assert ann.loc[2, "team"] == "Arsenal"
    assert ann.loc[2, "venue"] == "home"
    assert ann.loc[2, "points"] == 3
    assert ann.loc[10, "points"] == 1


def test_unused_slots_produce_no_pick() -> None:
    _, picks = parse_contest_sheet(SAMPLE, "2526")
    bo = picks[picks["entrant"] == "Bo Ray"]
    # Bo never used Arsenal away, so that slot contributes nothing.
    assert not ((bo["team"] == "Arsenal") & (bo["venue"] == "away")).any()
    assert set(bo["contest_week"]) == {15, 4, 7}


def test_the_standings_block_below_the_grid_is_not_read_as_picks() -> None:
    _, picks = parse_contest_sheet(SAMPLE, "2526")
    assert "Player" not in set(picks["team"])
    assert "Ann Lee" not in set(picks["team"])


def test_a_sheet_without_entrant_columns_is_rejected() -> None:
    with pytest.raises(ContestSheetError):
        parse_contest_sheet("a,b\nc,d\ne,f\ng,h\ni,j\n", "2526")


def test_too_short_a_sheet_is_rejected() -> None:
    with pytest.raises(ContestSheetError):
        parse_contest_sheet("only,one,row\n", "2526")


def test_sharing_failures_are_explained_in_plain_terms(monkeypatch) -> None:
    """The UI is the only interface, so an HTTP status is not a useful message."""
    import requests

    from epl_prediction_optimizer.data import league

    def deny(_url: str, timeout: int = 30) -> bytes:
        response = requests.Response()
        response.status_code = 403
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(league, "http_get_bytes", deny)
    with pytest.raises(ContestSheetError, match="Anyone with the link"):
        league.fetch_contest_sheet(SHEET_URL)


def test_a_missing_sheet_says_so(monkeypatch) -> None:
    import requests

    from epl_prediction_optimizer.data import league

    def missing(_url: str, timeout: int = 30) -> bytes:
        response = requests.Response()
        response.status_code = 404
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(league, "http_get_bytes", missing)
    with pytest.raises(ContestSheetError, match="No sheet found"):
        league.fetch_contest_sheet(SHEET_URL)


def test_a_login_page_instead_of_data_is_caught(monkeypatch) -> None:
    from epl_prediction_optimizer.data import league

    monkeypatch.setattr(league, "http_get_bytes", lambda *_a, **_k: b"<html>sign in</html>")
    with pytest.raises(ContestSheetError, match="not readable by link"):
        league.fetch_contest_sheet(SHEET_URL)
