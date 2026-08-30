"""External and sample data source helpers for EPL results, fixtures, and Elo."""

from __future__ import annotations

import os
import time
from collections import deque
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_FOOTBALL_DATA_ORG_TIMES: deque[float] = deque()
_RATE_LIMIT_CALLS = 10
_RATE_LIMIT_WINDOW = 60.0


def _throttle_football_data_org() -> None:
    """Block until a football-data.org request is within the 10 req/min free-plan limit."""
    now = time.monotonic()
    while _FOOTBALL_DATA_ORG_TIMES and now - _FOOTBALL_DATA_ORG_TIMES[0] >= _RATE_LIMIT_WINDOW:
        _FOOTBALL_DATA_ORG_TIMES.popleft()
    if len(_FOOTBALL_DATA_ORG_TIMES) >= _RATE_LIMIT_CALLS:
        sleep_for = _RATE_LIMIT_WINDOW - (now - _FOOTBALL_DATA_ORG_TIMES[0]) + 0.1
        print(f"[rate-limit] football-data.org: sleeping {sleep_for:.1f}s to stay under 10 req/min")
        time.sleep(sleep_for)
    _FOOTBALL_DATA_ORG_TIMES.append(time.monotonic())


FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"
EPL_DIVISION = "E0"
FOOTBALL_DATA_ORG_FIXTURES = "https://api.football-data.org/v4/competitions/PL/matches"
DEFAULT_SEASON_CODES = [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(1995, 2026)]

TEAM_NAME_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Sunderland AFC": "Sunderland",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Brighton and Hove Albion": "Brighton & Hove Albion",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "West Ham": "West Ham United",
    "Birmingham": "Birmingham City",
    "Bradford": "Bradford City",
    "Cardiff": "Cardiff City",
    "Coventry": "Coventry City",
    "Derby": "Derby County",
    "Hull": "Hull City",
    "Hull City AFC": "Hull City",
    "Ipswich": "Ipswich Town",
    "Norwich": "Norwich City",
    "Oldham": "Oldham Athletic",
    "QPR": "Queens Park Rangers",
    "Sheffield United": "Sheffield United",
    "Sheffield Weds": "Sheffield Wednesday",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "West Brom": "West Bromwich Albion",
    "Wimbledon": "Wimbledon",
}

CLUBELO_NAME_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "AstonVilla",
    "Barnsley": "Barnsley",
    "Birmingham City": "Birmingham",
    "Blackburn": "Blackburn",
    "Blackpool": "Blackpool",
    "Bolton": "Bolton",
    "Bournemouth": "Bournemouth",
    "Bradford City": "Bradford",
    "Brentford": "Brentford",
    "Brighton & Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Cardiff City": "Cardiff",
    "Charlton": "Charlton",
    "Chelsea": "Chelsea",
    "Coventry City": "Coventry",
    "Crystal Palace": "CrystalPalace",
    "Derby County": "Derby",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Manchester City": "ManCity",
    "Manchester United": "ManUnited",
    "Middlesbrough": "Middlesbrough",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Forest",
    "Portsmouth": "Portsmouth",
    "Queens Park Rangers": "QPR",
    "Reading": "Reading",
    "Sheffield United": "SheffieldUnited",
    "Sheffield Wednesday": "SheffieldWeds",
    "Southampton": "Southampton",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "West Bromwich Albion": "WestBrom",
    "West Ham United": "WestHam",
    "Wigan": "Wigan",
    "Wimbledon": "Wimbledon",
    "Wolverhampton Wanderers": "Wolves",
}


def read_historical_results(path: Path, season_code: str | None = None) -> pd.DataFrame:
    """Read football-data.co.uk EPL CSVs into the canonical match schema."""
    import numpy as np

    raw = pd.read_csv(path, encoding="latin1", on_bad_lines="skip")
    frame = raw.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        }
    )
    columns = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    frame = frame[columns].dropna()
    frame["date"] = _parse_football_data_dates(frame["date"])
    frame = frame.dropna(subset=["date"])
    frame["home_team"] = frame["home_team"].map(normalize_team_name)
    frame["away_team"] = frame["away_team"].map(normalize_team_name)
    frame["season"] = season_code or path.parent.name
    frame["match_id"] = [f"{frame.loc[index, 'season']}-{index:04d}" for index in frame.index]
    frame["contest_week"] = build_contest_weeks(frame["date"])
    frame["home_elo"] = 1500.0
    frame["away_elo"] = 1500.0

    # Optional columns from raw CSV (present from 2000-01 onwards for shots,
    # 2002-03 onwards for B365 odds; fill with NaN when absent).
    frame["home_shots_ot"] = pd.to_numeric(raw.get("HST", np.nan), errors="coerce")
    frame["away_shots_ot"] = pd.to_numeric(raw.get("AST", np.nan), errors="coerce")
    frame["home_odds"] = pd.to_numeric(raw.get("B365H", np.nan), errors="coerce")
    frame["draw_odds"] = pd.to_numeric(raw.get("B365D", np.nan), errors="coerce")
    frame["away_odds"] = pd.to_numeric(raw.get("B365A", np.nan), errors="coerce")

    return frame


def download_historical_results(
    season_code: str,
    destination: Path,
    force: bool = False,
) -> Path:
    """Download a football-data.co.uk EPL CSV for a season code such as ``2526``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    url = f"{FOOTBALL_DATA_BASE}/{season_code}/{EPL_DIVISION}.csv"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def fetch_upcoming_fixtures(api_key: str | None = None) -> pd.DataFrame:
    """Fetch scheduled EPL fixtures or return sample fixtures without an API key."""
    token = api_key or os.getenv("FOOTBALL_DATA_ORG_API_KEY")
    if not token:
        return sample_upcoming_fixtures()
    _throttle_football_data_org()
    response = requests.get(
        FOOTBALL_DATA_ORG_FIXTURES,
        headers={"X-Auth-Token": token},
        params={"status": "SCHEDULED"},
        timeout=30,
    )
    response.raise_for_status()
    matches = response.json()["matches"]
    rows = []
    for index, match in enumerate(matches, start=1):
        rows.append(
            {
                "match_id": str(match["id"]),
                "contest_week": int(match.get("matchday") or index),
                "date": match["utcDate"][:10],
                "kickoff_utc": match["utcDate"],
                "home_team": normalize_team_name(match["homeTeam"]["name"]),
                "away_team": normalize_team_name(match["awayTeam"]["name"]),
                "home_elo": 1500.0,
                "away_elo": 1500.0,
            }
        )
    return pd.DataFrame(rows)


def fetch_pl_matches(
    api_key: str | None = None,
    season_start_year: int = 2025,
) -> pd.DataFrame:
    """Fetch Premier League matches with official matchday/gameweek values."""
    token = api_key or os.getenv("FOOTBALL_DATA_ORG_API_KEY")
    if not token:
        return pd.DataFrame()
    _throttle_football_data_org()
    response = requests.get(
        FOOTBALL_DATA_ORG_FIXTURES,
        headers={"X-Auth-Token": token},
        params={"season": season_start_year},
        timeout=30,
    )
    response.raise_for_status()
    rows = []
    for match in response.json()["matches"]:
        full_time = match.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        rows.append(
            {
                "date": match["utcDate"][:10],
                "kickoff_utc": match["utcDate"],
                "home_team": normalize_team_name(match["homeTeam"]["name"]),
                "away_team": normalize_team_name(match["awayTeam"]["name"]),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "season": f"{season_start_year % 100:02d}{(season_start_year + 1) % 100:02d}",
                "match_id": str(match["id"]),
                "contest_week": int(match["matchday"]),
                "home_elo": 1500.0,
                "away_elo": 1500.0,
                "status": match.get("status", ""),
            }
        )
    return pd.DataFrame(rows)


def get_cached_pl_matches(
    cache_path: Path,
    api_key: str | None = None,
    season_start_year: int = 2025,
    force: bool = False,
) -> pd.DataFrame:
    """Read or fetch official PL matches with matchday values for one season."""
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)
    matches = fetch_pl_matches(api_key=api_key, season_start_year=season_start_year)
    if not matches.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        matches.to_csv(cache_path, index=False)
    return matches


def apply_official_gameweeks(historical: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """Overlay official PL matchdays onto football-data.co.uk rows."""
    if official.empty:
        return historical
    frame = historical.copy()
    frame["_match_date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
    frame["_home_key"] = frame["home_team"].map(normalize_team_name)
    frame["_away_key"] = frame["away_team"].map(normalize_team_name)
    official_frame = official.copy()
    official_frame["_match_date"] = pd.to_datetime(official_frame["date"]).dt.date.astype(str)
    official_frame["_home_key"] = official_frame["home_team"].map(normalize_team_name)
    official_frame["_away_key"] = official_frame["away_team"].map(normalize_team_name)
    official_frame = official_frame[
        ["_match_date", "_home_key", "_away_key", "contest_week"]
    ].rename(columns={"contest_week": "official_contest_week"})
    merged = frame.merge(
        official_frame,
        on=["_match_date", "_home_key", "_away_key"],
        how="left",
    )
    merged["contest_week"] = (
        merged["official_contest_week"].fillna(merged["contest_week"]).astype(int)
    )
    return merged.drop(columns=["_match_date", "_home_key", "_away_key", "official_contest_week"])


def fetch_clubelo_current() -> pd.DataFrame:
    """Fetch current English club Elo ratings from ClubElo."""
    response = requests.get("http://api.clubelo.com/ENG", timeout=30)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text))
    return frame[["Club", "Elo"]].rename(columns={"Club": "team", "Elo": "elo"})


def fetch_clubelo_history(team: str) -> pd.DataFrame:
    """Fetch historical ClubElo rows for a mapped team name."""
    clubelo_name = CLUBELO_NAME_MAP.get(normalize_team_name(team))
    if not clubelo_name:
        return pd.DataFrame(columns=["date", "team", "elo"])
    response = requests.get(f"http://api.clubelo.com/{clubelo_name}", timeout=12)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.empty or "Elo" not in frame:
        return pd.DataFrame(columns=["date", "team", "elo"])
    return pd.DataFrame(
        {
            "date": pd.to_datetime(frame["To"], errors="coerce").dt.date,
            "team": normalize_team_name(team),
            "elo": pd.to_numeric(frame["Elo"], errors="coerce"),
        }
    ).dropna()


def attach_current_elo(fixtures: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Attach current Elo ratings to home and away teams in fixture rows."""
    frame = fixtures.copy()
    ratings = {normalize_team_name(row.team): float(row.elo) for row in elo.itertuples(index=False)}
    frame["home_elo"] = frame["home_team"].map(ratings).fillna(frame.get("home_elo", 1500))
    frame["away_elo"] = frame["away_team"].map(ratings).fillna(frame.get("away_elo", 1500))
    return frame


def attach_historical_elo(matches: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match ClubElo ratings by team and date using a vectorized as-of join."""
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if elo_history.empty:
        frame["home_elo"] = 1500.0
        frame["away_elo"] = 1500.0
        frame["date"] = frame["date"].dt.date
        return frame

    history = elo_history.copy()
    history["date"] = pd.to_datetime(history["date"]).astype("datetime64[us]")
    history = history.sort_values("date")

    def _asof_elo(side: str) -> pd.Series:
        left = frame[["date", f"{side}_team"]].rename(columns={f"{side}_team": "team"}).copy()
        left["date"] = left["date"].astype("datetime64[us]")
        left = left.sort_values("date")
        merged = pd.merge_asof(
            left.reset_index(),
            history[["date", "team", "elo"]],
            on="date",
            by="team",
            direction="backward",
        ).set_index("index")
        return merged["elo"].reindex(frame.index).fillna(1500.0)

    frame["home_elo"] = _asof_elo("home")
    frame["away_elo"] = _asof_elo("away")
    frame["date"] = frame["date"].dt.date
    return frame


def sample_upcoming_fixtures() -> pd.DataFrame:
    """Return deterministic fixtures for offline demos and tests."""
    teams = ["Arsenal", "Everton", "Chelsea", "Fulham"]
    rows = []
    first_saturday = date(2026, 8, 22)
    for week in range(1, 5):
        for match_index in range(0, len(teams), 2):
            home = teams[(match_index + week) % len(teams)]
            away = teams[(match_index + week + 1) % len(teams)]
            rows.append(
                {
                    "match_id": f"sample-{week}-{match_index}",
                    "contest_week": week,
                    "date": first_saturday + timedelta(days=(week - 1) * 7),
                    "kickoff_utc": (first_saturday + timedelta(days=(week - 1) * 7)).isoformat()
                    + "T15:00:00Z",
                    "home_team": home,
                    "away_team": away,
                    "home_elo": 1500 + (10 * match_index),
                    "away_elo": 1490 + (8 * match_index),
                }
            )
    return pd.DataFrame(rows)


OPENFOOTBALL_SEASON_CODES = [
    f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(2000, 2025)
]

# openfootball team names → canonical names used in this project
_OPENFOOTBALL_TEAM_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Barnsley FC": "Barnsley",
    "Birmingham City FC": "Birmingham City",
    "Blackburn Rovers FC": "Blackburn",
    "Blackpool FC": "Blackpool",
    "Bolton Wanderers FC": "Bolton",
    "AFC Bournemouth": "Bournemouth",
    "Bradford City AFC": "Bradford City",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Burnley FC": "Burnley",
    "Cardiff City FC": "Cardiff City",
    "Charlton Athletic FC": "Charlton",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry City",
    "Crystal Palace FC": "Crystal Palace",
    "Derby County FC": "Derby County",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Huddersfield Town AFC": "Huddersfield",
    "Hull City AFC": "Hull City",
    "Ipswich Town FC": "Ipswich Town",
    "Leeds United AFC": "Leeds United",
    "Leicester City FC": "Leicester City",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Middlesbrough FC": "Middlesbrough",
    "Newcastle United FC": "Newcastle United",
    "Norwich City FC": "Norwich City",
    "Nottingham Forest FC": "Nottingham Forest",
    "Oldham Athletic AFC": "Oldham Athletic",
    "Portsmouth FC": "Portsmouth",
    "Queens Park Rangers FC": "Queens Park Rangers",
    "Reading FC": "Reading",
    "Sheffield United FC": "Sheffield United",
    "Sheffield Wednesday FC": "Sheffield Wednesday",
    "Southampton FC": "Southampton",
    "Stoke City FC": "Stoke City",
    "Sunderland AFC": "Sunderland",
    "Swansea City AFC": "Swansea City",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Watford FC": "Watford",
    "West Bromwich Albion FC": "West Bromwich Albion",
    "West Ham United FC": "West Ham United",
    "Wigan Athletic FC": "Wigan",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
}


def _openfootball_season_dir(season_code: str) -> str:
    """Convert '2122' → '2021-22' for the openfootball GitHub directory name."""
    code = str(season_code).zfill(4)
    yy_start = int(code[:2])
    yy_end = int(code[2:])
    year_start = (1900 + yy_start) if yy_start >= 50 else (2000 + yy_start)
    return f"{year_start}-{yy_end:02d}"


def fetch_openfootball_gameweeks(
    season_code: str,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch official PL matchday (gameweek) numbers from openfootball GitHub.

    Returns DataFrame with columns: date, home_team, away_team, contest_week.
    Caches to cache_dir/{season_code}_PL_matchdays.csv.
    Returns empty DataFrame on failure.
    """
    import re

    cache_path = cache_dir / f"{season_code}_PL_matchdays.csv"
    if cache_path.exists() and not force:
        try:
            return pd.read_csv(cache_path, parse_dates=["date"])
        except Exception:
            pass

    season_dir = _openfootball_season_dir(season_code)
    url = f"https://raw.githubusercontent.com/openfootball/eng-england/master/{season_dir}/1-premierleague.txt"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        text = response.text
    except Exception as exc:
        print(f"[openfootball] Failed to fetch {season_code}: {exc}")
        return pd.DataFrame(columns=["date", "home_team", "away_team", "contest_week"])

    rows = []
    current_matchday = None
    current_year = int(_openfootball_season_dir(season_code)[:4])
    current_date = None

    # Month abbreviations used in openfootball format
    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    matchday_re = re.compile(r"^Matchday\s+(\d+)", re.IGNORECASE)
    date_re = re.compile(r"\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w+)/(\d+)\]")
    # Match line: "  HH:MM  Team A  score  Team B" or "         Team A  score  Team B"
    # Score pattern: digits-digits (with optional ET/AET qualifier)
    match_re = re.compile(
        r"^\s{2,}"  # leading indent
        r"(?:\d{1,2}[.:]\d{2}\s+)?"  # optional kick-off time
        r"(.+?)\s{2,}"  # home team (2+ spaces separator)
        r"(\d+)-(\d+)"  # score
        r"(?:\s*\(\d+-\d+\))?"  # optional half-time score
        r"(?:\s*(?:aet|AET))?"  # optional AET
        r"\s{2,}(.+?)\s*$"  # away team
    )

    for line in text.splitlines():
        md_match = matchday_re.match(line)
        if md_match:
            current_matchday = int(md_match.group(1))
            continue

        date_match = date_re.search(line)
        if date_match:
            month_str, day_str = date_match.group(1), date_match.group(2)
            month = month_map.get(month_str)
            if month:
                # Seasons span two calendar years; Jan-Jul belong to end year
                year = current_year + 1 if month <= 7 else current_year
                try:
                    from datetime import date as _date

                    current_date = _date(year, month, int(day_str))
                except ValueError:
                    current_date = None
            continue

        if current_matchday is None or current_date is None:
            continue

        m = match_re.match(line)
        if m:
            raw_home = m.group(1).strip()
            raw_away = m.group(4).strip()
            home = _OPENFOOTBALL_TEAM_MAP.get(raw_home, normalize_team_name(raw_home))
            away = _OPENFOOTBALL_TEAM_MAP.get(raw_away, normalize_team_name(raw_away))
            rows.append(
                {
                    "date": current_date,
                    "home_team": home,
                    "away_team": away,
                    "contest_week": current_matchday,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "contest_week"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(
        f"[openfootball] {season_code}: {len(df)} matches across {df['contest_week'].nunique()} matchdays → {cache_path.name}"
    )
    return df


UNDERSTAT_SEASON_CODES = [f"{year % 100:02d}{(year + 1) % 100:02d}" for year in range(2014, 2025)]

_UNDERSTAT_TEAM_MAP = {
    "Manchester United": "Manchester United",
    "Manchester City": "Manchester City",
    "Arsenal": "Arsenal",
    "Chelsea": "Chelsea",
    "Liverpool": "Liverpool",
    "Tottenham": "Tottenham Hotspur",
    "Everton": "Everton",
    "Leicester": "Leicester City",
    "West Brom": "West Bromwich Albion",
    "Swansea": "Swansea City",
    "Stoke": "Stoke City",
    "Crystal Palace": "Crystal Palace",
    "Southampton": "Southampton",
    "Burnley": "Burnley",
    "Watford": "Watford",
    "Bournemouth": "Bournemouth",
    "West Ham": "West Ham United",
    "Hull": "Hull City",
    "Middlesbrough": "Middlesbrough",
    "Sunderland": "Sunderland",
    "Newcastle United": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Huddersfield": "Huddersfield",
    "Cardiff": "Cardiff City",
    "Fulham": "Fulham",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",
    "Sheffield United": "Sheffield United",
    "Aston Villa": "Aston Villa",
    "Leeds United": "Leeds United",
    "Brentford": "Brentford",
    "Nottingham Forest": "Nottingham Forest",
    "Luton": "Luton",
    "Ipswich": "Ipswich Town",
}


def fetch_understat_xg(
    season_code: str,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch Understat xG data for one EPL season, caching to CSV.

    season_code is like "2324" (= 2023-24, start year 2023).
    Returns DataFrame with columns: date, home_team, away_team, home_xg, away_xg.
    Returns empty DataFrame on any failure.
    """

    cache_path = cache_dir / f"{season_code}_understat_xg.csv"
    if cache_path.exists() and not force:
        try:
            return pd.read_csv(cache_path, parse_dates=["date"])
        except Exception:
            pass

    # Convert season_code "2324" → start year 2023
    yy = int(str(season_code).zfill(4)[:2])
    start_year = (1900 + yy) if yy >= 50 else (2000 + yy)
    url = f"https://understat.com/getLeagueData/EPL/{start_year}"
    try:
        response = requests.get(
            url,
            headers={
                "Referer": f"https://understat.com/league/EPL/{start_year}",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_xg", "away_xg"])

    rows = []
    for match in data.get("dates", []):
        if not match.get("isResult"):
            continue
        try:
            match_date = pd.to_datetime(match["datetime"]).date()
            home = normalize_team_name(
                _UNDERSTAT_TEAM_MAP.get(match["h"]["title"], match["h"]["title"])
            )
            away = normalize_team_name(
                _UNDERSTAT_TEAM_MAP.get(match["a"]["title"], match["a"]["title"])
            )
            home_xg = float(match["xG"]["h"])
            away_xg = float(match["xG"]["a"])
            rows.append(
                {
                    "date": match_date,
                    "home_team": home,
                    "away_team": away,
                    "home_xg": home_xg,
                    "away_xg": away_xg,
                }
            )
        except (KeyError, ValueError, TypeError):
            continue

    frame = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_xg", "away_xg"])
    if not frame.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    return frame


def attach_understat_xg(matches: pd.DataFrame, xg_data: pd.DataFrame) -> pd.DataFrame:
    """Left-join Understat xG onto matches by date + home_team + away_team.

    Adds home_xg and away_xg columns (NaN where no match found).
    """
    if xg_data.empty:
        frame = matches.copy()
        frame["home_xg"] = float("nan")
        frame["away_xg"] = float("nan")
        return frame

    xg = xg_data.copy()
    xg["date"] = pd.to_datetime(xg["date"]).dt.date
    left = matches.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.date

    merged = left.merge(
        xg[["date", "home_team", "away_team", "home_xg", "away_xg"]],
        on=["date", "home_team", "away_team"],
        how="left",
    )
    return merged


def normalize_team_name(name: str) -> str:
    """Normalize common team-name variants across data providers."""
    clean = str(name).replace(" FC", "").strip()
    return TEAM_NAME_MAP.get(clean, clean)


def build_contest_weeks(dates: pd.Series) -> list[int]:
    """Convert match dates into stable contest week numbers by sorted weekend blocks."""
    parsed = pd.to_datetime(dates)
    week_starts = parsed.dt.to_period("W-MON").dt.start_time.dt.date
    ordered = {week: index + 1 for index, week in enumerate(sorted(week_starts.unique()))}
    return [ordered[week] for week in week_starts]


def _parse_football_data_dates(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            dates.loc[missing],
            format="%d/%m/%y",
            dayfirst=True,
            errors="coerce",
        )
    return parsed.dt.date


# ---------------------------------------------------------------------------
# Market odds
# ---------------------------------------------------------------------------

FOOTBALL_DATA_FIXTURES_FEED = "https://www.football-data.co.uk/fixtures.csv"
MARKET_ODDS_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "avg_home_odds",
    "avg_draw_odds",
    "avg_away_odds",
    "captured_at",
]


def http_get_bytes(url: str, timeout: int = 30) -> bytes:
    """GET a URL, using curl_cffi browser impersonation when it is installed.

    football-data.co.uk sits behind bot protection that intermittently rejects
    the default python TLS fingerprint. curl_cffi impersonates a real Chrome
    handshake; plain requests remains the fallback so the package still works
    without the optional dependency.
    """
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    try:
        response = curl_requests.get(url, impersonate="chrome", timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception:  # noqa: BLE001 - any transport failure falls back to requests
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content


def fetch_market_odds(division: str = EPL_DIVISION) -> pd.DataFrame:
    """Fetch pre-match 1X2 prices for upcoming fixtures from football-data.co.uk.

    The public fixtures feed carries the same B365H/B365D/B365A columns as the
    per-season historical CSVs, so live prices and backtest prices come from
    one schema. Avg* columns are the cross-bookmaker consensus, which is the
    more stable signal when a single book moves early.
    """
    payload = http_get_bytes(FOOTBALL_DATA_FIXTURES_FEED)
    raw = pd.read_csv(StringIO(payload.decode("utf-8-sig")), on_bad_lines="skip")
    if "Div" not in raw.columns:
        return pd.DataFrame(columns=MARKET_ODDS_COLUMNS)
    raw = raw[raw["Div"] == division]
    if raw.empty:
        return pd.DataFrame(columns=MARKET_ODDS_COLUMNS)

    frame = pd.DataFrame(
        {
            "date": _parse_football_data_dates(raw["Date"]),
            "home_team": raw["HomeTeam"].map(normalize_team_name),
            "away_team": raw["AwayTeam"].map(normalize_team_name),
            "home_odds": pd.to_numeric(raw.get("B365H"), errors="coerce"),
            "draw_odds": pd.to_numeric(raw.get("B365D"), errors="coerce"),
            "away_odds": pd.to_numeric(raw.get("B365A"), errors="coerce"),
            "avg_home_odds": pd.to_numeric(raw.get("AvgH"), errors="coerce"),
            "avg_draw_odds": pd.to_numeric(raw.get("AvgD"), errors="coerce"),
            "avg_away_odds": pd.to_numeric(raw.get("AvgA"), errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["date", "home_team", "away_team"])
    frame["captured_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return frame[MARKET_ODDS_COLUMNS].reset_index(drop=True)


def update_market_odds_ledger(ledger_path: Path, fetched: pd.DataFrame) -> pd.DataFrame:
    """Merge newly fetched prices into the durable odds ledger and return it.

    The upcoming-fixtures feed drops a match as soon as it kicks off, so prices
    must be persisted to stay available for scoring and for retraining once the
    match becomes history. The most recent capture for a fixture wins.
    """
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    combined = pd.concat([existing, fetched], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=MARKET_ODDS_COLUMNS)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date.astype(str)
    combined = combined.dropna(subset=["date"])
    combined = combined.sort_values("captured_at").drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
    )
    combined = combined.sort_values(["date", "home_team"]).reset_index(drop=True)
    combined.to_csv(ledger_path, index=False)
    return combined


def market_odds_from_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Extract the canonical odds frame from historical match rows.

    football-data.co.uk carries B365 closing prices from 2002-03 onwards, which
    is what makes an honest market backtest possible without any scraping.
    """
    required = {"date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"}
    if not required.issubset(matches.columns):
        return pd.DataFrame(columns=MARKET_ODDS_COLUMNS)
    frame = matches[sorted(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame = frame.dropna(subset=["home_odds", "draw_odds", "away_odds"])
    return frame.reset_index(drop=True)
