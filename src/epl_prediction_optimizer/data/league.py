"""Import a contest standings spreadsheet and normalise it into picks and totals.

The contest is run from a Google Sheet laid out entrant-by-entrant across the
columns: three columns each (the round a team was used, the points it returned,
and the goal difference), with one row per team-and-venue slot. That layout is
convenient to fill in by hand and awkward to compute with, so this reads the
published CSV and turns it into two tidy frames — one row per entrant, and one
row per entrant pick.

Only the sheet's own published export is used. Nothing here signs in, and a
sheet that is not link-readable simply fails.
"""

from __future__ import annotations

import csv
import re
from io import StringIO

import pandas as pd
import requests

from epl_prediction_optimizer.data.sources import http_get_bytes, normalize_team_name

SHEET_ID_PATTERN = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")
GID_PATTERN = re.compile(r"[#&?]gid=([0-9]+)")
WEEK_PATTERN = re.compile(r"(\d+)")

# Column offsets within each entrant's three-column block.
SLOT_COLUMN = 0
POINTS_COLUMN = 1
GOAL_DIFF_COLUMN = 2

NAME_ROW = 0
TOTALS_ROW = 2
FIRST_PICK_ROW = 4

# Names the sheet uses that differ from the canonical ones.
SHEET_TEAM_ALIASES = {
    "Arsenal FC": "Arsenal",
    "Brighton & Hove": "Brighton & Hove Albion",
    "Chelsea FC": "Chelsea",
    "Everton FC": "Everton",
    "Liverpool FC": "Liverpool",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Leeds": "Leeds United",
    "Newcastle United": "Newcastle United",
}


class ContestSheetError(RuntimeError):
    """Raised when a sheet cannot be read or does not have the expected shape."""


def sheet_export_url(url: str, gid: str | None = None) -> str:
    """Turn any Google Sheets link into its CSV export URL.

    Accepts the ordinary ``/edit#gid=...`` address people copy from the
    browser, so the user never has to construct an export link by hand.
    """
    match = SHEET_ID_PATTERN.search(url)
    if not match:
        raise ContestSheetError(f"Not a Google Sheets URL: {url}")
    sheet_id = match.group(1)
    tab = gid
    if tab is None:
        gid_match = GID_PATTERN.search(url)
        tab = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={tab}"


def fetch_contest_sheet(url: str, gid: str | None = None) -> str:
    """Download the sheet's published CSV.

    Failures here are nearly always sharing settings or a mistyped link, so
    they are reported in those terms rather than as an HTTP status.
    """
    export_url = sheet_export_url(url, gid=gid)
    try:
        payload = http_get_bytes(export_url)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in (401, 403):
            raise ContestSheetError(
                "Google refused to share that sheet. Open it, choose Share, and set "
                "General access to 'Anyone with the link' as Viewer."
            ) from exc
        if status == 404:
            raise ContestSheetError(
                "No sheet found at that link. Copy the URL again from the browser "
                "address bar with the standings tab open."
            ) from exc
        raise ContestSheetError(f"Could not download the sheet ({status or exc}).") from exc
    except requests.RequestException as exc:
        raise ContestSheetError(f"Could not reach Google Sheets: {exc}") from exc

    text = payload.decode("utf-8-sig", errors="replace")
    if text.lstrip().startswith("<"):
        raise ContestSheetError(
            "That link returned a web page rather than data, which means the sheet is "
            "not readable by link. Set General access to 'Anyone with the link'."
        )
    return text


def parse_contest_sheet(text: str, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the sheet into an entrants frame and a picks frame."""
    rows = list(csv.reader(StringIO(text)))
    if len(rows) <= FIRST_PICK_ROW:
        raise ContestSheetError("Sheet has too few rows to be a contest standings tab")

    header = rows[NAME_ROW]
    totals = rows[TOTALS_ROW] if len(rows) > TOTALS_ROW else []

    entrants: list[dict[str, object]] = []
    picks: list[dict[str, object]] = []

    for start in range(1, len(header), 3):
        name = header[start].strip() if start < len(header) else ""
        if not name:
            continue
        entrants.append(
            {
                "season": season,
                "entrant": name,
                "place": _cell(header, start + GOAL_DIFF_COLUMN).strip() or None,
                "points": _to_int(_cell(totals, start + POINTS_COLUMN)),
                "goal_diff": _to_int(_cell(totals, start + GOAL_DIFF_COLUMN)),
            }
        )
        picks.extend(_entrant_picks(rows, start, name, season))

    if not entrants:
        raise ContestSheetError("No entrant columns found; is this the standings tab?")
    if not _has_team_slots(rows):
        # Guards against pointing at the wrong tab: without team-and-venue row
        # labels the column blocks are not picks, whatever else they contain.
        raise ContestSheetError(
            "No team rows such as 'Arsenal FC (H)' were found; this looks like a "
            "different tab of the workbook."
        )
    return pd.DataFrame(entrants), pd.DataFrame(picks)


def _has_team_slots(rows: list[list[str]]) -> bool:
    """True when the grid has at least one recognisable team-and-venue row."""
    return any(parse_slot(_cell(row, 0)) is not None for row in rows[FIRST_PICK_ROW:])


def _entrant_picks(
    rows: list[list[str]],
    start: int,
    name: str,
    season: str,
) -> list[dict[str, object]]:
    """Read one entrant's column block down the team-and-venue rows."""
    picks: list[dict[str, object]] = []
    for row in rows[FIRST_PICK_ROW:]:
        label = _cell(row, 0).strip()
        if not label:
            # A blank slot label ends the pick grid; later blocks in the sheet
            # repeat the standings in a different shape and are not picks.
            break
        parsed = parse_slot(label)
        if parsed is None:
            continue
        team, venue = parsed
        week = _to_week(_cell(row, start + SLOT_COLUMN))
        if week is None:
            continue
        picks.append(
            {
                "season": season,
                "entrant": name,
                "team": team,
                "venue": venue,
                "contest_week": week,
                "points": _to_int(_cell(row, start + POINTS_COLUMN)),
                "goal_diff": _to_int(_cell(row, start + GOAL_DIFF_COLUMN)),
            }
        )
    return picks


def parse_slot(label: str) -> tuple[str, str] | None:
    """Turn a row label such as ``Brighton & Hove (A)`` into (team, venue)."""
    text = label.strip()
    if text.endswith("(H)"):
        venue = "home"
    elif text.endswith("(A)"):
        venue = "away"
    else:
        return None
    raw = text[:-3].strip()
    return normalize_team_name(SHEET_TEAM_ALIASES.get(raw, raw)), venue


def _cell(row: list[str], index: int) -> str:
    return row[index] if 0 <= index < len(row) else ""


def _to_int(value: str) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _to_week(value: str) -> int | None:
    match = WEEK_PATTERN.search(str(value))
    return int(match.group(1)) if match else None
