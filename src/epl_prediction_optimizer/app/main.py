"""FastAPI dashboard and JSON endpoints for operating the optimizer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager, chdir, suppress
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import pandas as pd
import requests
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from epl_prediction_optimizer.challenge import (
    load_challenge_rows,
    score_manual_pick,
    summarize_challenge,
)
from epl_prediction_optimizer.config.contest import compose_pick_email, entrant_name
from epl_prediction_optimizer.config.season import SeasonContext
from epl_prediction_optimizer.data.league import (
    ContestSheetError,
    fetch_contest_sheet,
    parse_contest_sheet,
)
from epl_prediction_optimizer.optimizer.candidates import build_pick_candidates
from epl_prediction_optimizer.optimizer.scenarios import build_scenarios
from epl_prediction_optimizer.optimizer.tournament import win_probability_against_field
from epl_prediction_optimizer.optimizer.verdict import decision_verdict
from epl_prediction_optimizer.pipeline import (
    MARKET_BLEND_WEIGHT,
    WINNER_BENCHMARKS,
    backtest_from_processed,
    optimize_from_predictions,
    predict_from_processed,
    refresh_data,
    refresh_full_history,
    train_from_processed,
)
from epl_prediction_optimizer.storage.database import Database

PACKAGE_ROOT = Path(__file__).resolve().parent


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _asset_version() -> str:
    """Fingerprint the static bundle by its most recently modified file."""
    static_dir = PACKAGE_ROOT / "static"
    if not static_dir.exists():
        return "0"
    newest = max(
        (path.stat().st_mtime for path in static_dir.rglob("*") if path.is_file()),
        default=0.0,
    )
    return str(int(newest))


def create_app(
    database: Database | None = None,
    workdir: Path | str | None = None,
    use_live_data: bool = True,
    auto_refresh: bool = False,
) -> FastAPI:
    """Create a FastAPI app using the provided or default SQLite database.

    ``auto_refresh`` re-runs the whole pipeline every 30 minutes in the
    background. It is off by default: a long-running server would otherwise
    rewrite the published forecasts and picks on its own schedule, including
    over a plan the user had just generated, and a server left running across a
    code change would keep republishing from the old code. Refresh is a
    deliberate action taken from the UI or the CLI.
    """
    db = database or Database()
    runtime_dir = Path(workdir or ".").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    active_season = SeasonContext.current_season()
    refresh_mode = "live" if use_live_data else "offline"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task: asyncio.Task[None] | None = None

        async def refresh_loop() -> None:
            while True:
                db.set_json(
                    "status",
                    {"status": "refreshing", "last_action": "automatic-refresh"},
                )
                try:
                    result = await asyncio.to_thread(
                        _run_background_pipeline,
                        runtime_dir,
                        db,
                        active_season,
                    )
                    db.set_json(
                        "status",
                        {
                            "status": "ready",
                            "last_action": "automatic-refresh",
                            "updated_at": _utc_now(),
                            **result,
                        },
                    )
                except Exception as exc:
                    db.set_json(
                        "status",
                        {
                            "status": "refresh_failed",
                            "last_action": "automatic-refresh",
                            "updated_at": _utc_now(),
                            "error": str(exc),
                        },
                    )
                await asyncio.sleep(30 * 60)

        if use_live_data and auto_refresh:
            task = asyncio.create_task(refresh_loop())
        yield
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="EPL Prediction Optimizer", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))
    # Browsers cache /static aggressively, so a CSS change can leave a running
    # page styled by the previous version with no sign anything is wrong.
    # Stamping the newest asset mtime onto every link makes that impossible.
    templates.env.globals["asset_version"] = _asset_version()
    app.mount(
        "/static",
        StaticFiles(directory=str(PACKAGE_ROOT / "static")),
        name="static",
    )

    def decision_payload(selected_week: int | None = None) -> dict[str, object]:
        predictions = _active_season_predictions(db.get_json("predictions", []), active_season)
        actual_picks = db.list_actual_picks(active_season.code)
        candidates = (
            build_pick_candidates(pd.DataFrame(predictions)) if predictions else pd.DataFrame()
        )
        if not candidates.empty:
            committed_matches = {str(pick["match_id"]) for pick in actual_picks}
            eligible = candidates.apply(
                lambda row: (
                    _candidate_kickoff(row.to_dict()) > datetime.now(UTC)
                    or str(row["match_id"]) in committed_matches
                ),
                axis=1,
            )
            candidates = candidates[eligible].reset_index(drop=True)
        weeks = sorted(
            int(value) for value in candidates.get("contest_week", pd.Series(dtype=int)).unique()
        )
        picked_weeks = {int(pick["contest_week"]) for pick in actual_picks}
        now = datetime.now(UTC)
        future_weeks = {
            int(candidate["contest_week"])
            for candidate in candidates.to_dict(orient="records")
            if _candidate_kickoff(candidate) > now
        }
        next_open = next(
            (week for week in weeks if week not in picked_weeks and week in future_weeks),
            None,
        )
        week = (
            selected_week if selected_week in weeks else next_open or (weeks[-1] if weeks else None)
        )
        banked = sum(
            int(pick["actual_points"])
            for pick in actual_picks
            if pick.get("actual_points") is not None
        )
        scenarios = (
            build_scenarios(
                candidates,
                week,
                actual_picks,
                field_totals=_reference_field(db, active_season.code),
                points_banked=banked,
            )
            if week is not None
            else []
        )
        committed = next((p for p in actual_picks if int(p["contest_week"]) == week), None)
        for item in scenarios:
            item["selected"] = bool(
                committed
                and committed["match_id"] == item["match_id"]
                and committed["team"] == item["team"]
            )
        return {
            "predictions": predictions,
            "actual_picks": actual_picks,
            "candidates": candidates.to_dict(orient="records") if not candidates.empty else [],
            "weeks": weeks,
            "selected_week": week,
            "next_open_week": next_open,
            "scenarios": scenarios,
            "verdict": decision_verdict(scenarios, committed),
            "committed": committed,
            "submission": _submission_email(committed, scenarios),
            "entrant": entrant_name(),
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, week: int | None = None) -> HTMLResponse:
        picks = db.get_json("picks", [])
        decision = decision_payload(week)
        actual_picks = decision["actual_picks"]
        season_summary = _season_summary(actual_picks)
        current_week = decision["selected_week"]
        week_recommendation = next(
            (p for p in picks if current_week and p["contest_week"] == current_week), None
        )
        recent_picks = _recent_picks_with_results(runtime_dir, actual_picks)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "readiness": build_readiness(
                    db, runtime_dir, active_season, MARKET_BLEND_WEIGHT
                ),
                "status": db.get_json("status", {"status": "needs_refresh"}),
                "predictions": decision["predictions"],
                "picks": picks,
                "season_summary": season_summary,
                "current_week": current_week,
                "week_recommendation": week_recommendation,
                "recent_picks": recent_picks,
                "season": active_season,
                "decision": decision,
            },
        )

    @app.get("/api/weeks/{contest_week}")
    def week_decision(contest_week: int) -> dict[str, object]:
        payload = decision_payload(contest_week)
        if payload["selected_week"] != contest_week:
            raise HTTPException(status_code=404, detail="Contest week not found")
        return payload

    def commit_pick(contest_week: int, payload: dict[str, object]) -> dict[str, object]:
        decision = decision_payload(contest_week)
        scenario = next(
            (
                item
                for item in decision["scenarios"]
                if item["match_id"] == str(payload.get("match_id"))
                and item["team"] == str(payload.get("team"))
                and item["venue"] == str(payload.get("venue"))
            ),
            None,
        )
        if scenario is None or not scenario.get("feasible"):
            raise HTTPException(status_code=409, detail="Choice is no longer feasible; refresh")
        existing = decision["committed"]
        now = datetime.now(UTC)
        if existing and existing.get("kickoff_at"):
            existing_kickoff = datetime.fromisoformat(str(existing["kickoff_at"]))
            if now >= existing_kickoff:
                raise HTTPException(
                    status_code=409, detail="Pick is locked because its match started"
                )
        kickoff = _candidate_kickoff(scenario)
        if now >= kickoff:
            raise HTTPException(status_code=409, detail="This match has already started")
        stored = db.upsert_actual_pick(
            {
                "season": active_season.code,
                "contest_week": contest_week,
                "match_id": scenario["match_id"],
                "team": scenario["team"],
                "venue": scenario["venue"],
                "kickoff_at": kickoff.isoformat(),
                "notes": str(payload.get("notes", "")),
                "news_risk": str(payload.get("news_risk", "none")),
                "actual_points": None,
                "state": "committed",
            }
        )
        return {"status": "saved", "pick": stored, "decision": decision_payload(contest_week)}

    @app.put("/api/picks/{contest_week}")
    def put_pick(contest_week: int, payload: dict[str, object]) -> dict[str, object]:
        return commit_pick(contest_week, payload)

    @app.post("/picks/{contest_week}", response_class=RedirectResponse)
    def post_pick(
        contest_week: int,
        match_id: str = Form(...),
        team: str = Form(...),
        venue: str = Form(...),
        notes: str = Form(""),
        news_risk: str = Form("none"),
    ) -> RedirectResponse:
        commit_pick(
            contest_week,
            {
                "match_id": match_id,
                "team": team,
                "venue": venue,
                "notes": notes,
                "news_risk": news_risk,
            },
        )
        return RedirectResponse(f"/?week={contest_week}", status_code=303)

    @app.get("/scorecard", response_class=HTMLResponse)
    def scorecard(
        request: Request,
        season: str | None = None,
        entrant: str | None = None,
        error: str | None = None,
        imported: str | None = None,
    ) -> HTMLResponse:
        selected_season = season or active_season.code
        context = build_scorecard(db, runtime_dir, selected_season, entrant)
        context["import_error"] = error
        context["imported"] = _imported_banner(imported)
        return templates.TemplateResponse(request, "scorecard.html", context)

    @app.post("/actions/import-league", response_class=RedirectResponse)
    def import_league(url: str = Form(...), season: str = Form(...)) -> RedirectResponse:
        try:
            text = fetch_contest_sheet(url)
            entrants, picks = parse_contest_sheet(text, season)
            db.replace_league_data(
                season, entrants.to_dict(orient="records"), picks.to_dict(orient="records")
            )
        except (ContestSheetError, requests.RequestException) as exc:
            return RedirectResponse(
                f"/scorecard?season={season}&error={quote(str(exc))}", status_code=303
            )
        db.set_json("league_sheet_url", {"season": season, "url": url})
        banner = f"{season}:{len(entrants)}:{len(picks)}"
        return RedirectResponse(
            f"/scorecard?season={season}&imported={quote(banner)}", status_code=303
        )

    @app.get("/model", response_class=HTMLResponse)
    def model_view(request: Request) -> HTMLResponse:
        import json as _json

        metrics_path = runtime_dir / "data" / "artifacts" / "metrics.json"
        metrics = _json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        from epl_prediction_optimizer.ml.analysis import backtest_summary_report

        _all_backtest_seasons = ["2021", "2122", "2223", "2324", "2425"]
        summary = backtest_summary_report(
            runtime_dir / "data" / "artifacts",
            _all_backtest_seasons,
        )
        experiments = db.list_experiment_runs()
        return templates.TemplateResponse(
            request,
            "model.html",
            {
                "metrics": metrics,
                "backtest_summary": summary.to_dict(orient="records") if not summary.empty else [],
                "experiments": experiments,
                "season": active_season,
            },
        )

    @app.get("/api/experiments")
    def list_experiments(season: str | None = None) -> list[dict]:
        return db.list_experiment_runs(season=season)

    @app.get("/backtest", response_class=HTMLResponse)
    def backtest_view(request: Request, season: str = "2425") -> HTMLResponse:
        metrics = _load_backtest_metrics(runtime_dir, season)
        picks = _load_backtest_picks(runtime_dir, season)
        chart = _build_chart(picks, metrics.get("winner_points") if metrics else None)
        return templates.TemplateResponse(
            request,
            "backtest.html",
            {
                "season": season,
                "metrics": metrics,
                "picks": picks,
                "chart": chart,
                "active_season": active_season,
            },
        )

    @app.get("/data", response_class=HTMLResponse)
    def data_explorer(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "data.html",
            {
                "datasets": _list_datasets(runtime_dir),
                "season": active_season,
            },
        )

    @app.get("/challenge", response_class=HTMLResponse)
    def challenge_manager(
        request: Request,
        season: str | None = None,
        week: str = "next",
        view: str = "all",
        team: str = "",
    ) -> HTMLResponse:
        selected_season = season or active_season.code
        actual_picks = db.list_actual_picks(selected_season)
        rows = load_challenge_rows(runtime_dir, selected_season, actual_picks)
        summary = summarize_challenge(rows, actual_picks)
        filtered_rows = _filter_challenge_rows(rows, week, view, team, summary)
        return templates.TemplateResponse(
            request,
            "challenge.html",
            {
                "season": selected_season,
                "season_context": active_season,
                "week": week,
                "view": view,
                "team": team,
                "rows": filtered_rows,
                "all_rows": rows,
                "actual_picks": actual_picks,
                "summary": summary,
                "weeks": sorted({int(row["contest_week"]) for row in rows}),
                "teams": sorted(
                    {row["home_team"] for row in rows}.union({row["away_team"] for row in rows})
                ),
            },
        )

    @app.get("/api/status")
    def status() -> dict[str, object]:
        return {"status": "needs_refresh", **db.get_json("status", {})}

    @app.get("/api/readiness")
    def readiness() -> dict[str, object]:
        state = db.get_json("status", {})
        return {
            "season": active_season.code,
            "status": state.get("status", "needs_refresh"),
            "last_action": state.get("last_action"),
            "updated_at": state.get("updated_at"),
            "predictions": len(db.get_json("predictions", [])),
            "optimized_picks": len(db.get_json("picks", [])),
            "source_mode": refresh_mode,
        }

    @app.get("/api/predictions")
    def predictions() -> list[dict[str, object]]:
        return db.get_json("predictions", [])

    @app.get("/api/picks")
    def picks() -> list[dict[str, object]]:
        return db.get_json("picks", [])

    @app.get("/api/data/{dataset_path:path}")
    def data_preview(dataset_path: str, limit: int = 100) -> dict[str, object]:
        path = _dataset_path(runtime_dir, dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        frame = _read_csv_preview(path, limit)
        return {
            "dataset": dataset_path,
            "path": str(path),
            "columns": list(frame.columns),
            "rows": frame.to_dict(orient="records"),
        }

    @app.post("/api/manual-picks")
    def manual_pick(
        season: str = Form(...),
        contest_week: int = Form(...),
        match_id: str = Form(...),
        team: str = Form(...),
        venue: str = Form(...),
        notes: str = Form(""),
    ) -> dict[str, object]:
        pick = {
            "season": season,
            "contest_week": contest_week,
            "match_id": match_id,
            "team": team,
            "venue": venue,
            "notes": notes,
            "actual_points": score_manual_pick(runtime_dir, match_id, team),
        }
        return {"status": "saved", "pick": db.upsert_actual_pick(pick)}

    @app.post("/manual-picks", response_class=RedirectResponse)
    def manual_pick_form(
        season: str = Form(...),
        contest_week: int = Form(...),
        match_id: str = Form(...),
        team: str = Form(...),
        venue: str = Form(...),
        notes: str = Form(""),
    ) -> RedirectResponse:
        manual_pick(season, contest_week, match_id, team, venue, notes)
        return RedirectResponse(f"/challenge?season={season}", status_code=303)

    @app.post("/api/refresh")
    def refresh() -> dict[str, object]:
        outputs = _run_action(
            runtime_dir,
            lambda: refresh_data(mode=refresh_mode, season=active_season),
        )
        db.set_json(
            "status",
            {
                "status": "complete",
                "last_action": "refresh",
                "outputs": outputs,
                "updated_at": _utc_now(),
            },
        )
        return {"status": "complete", "action": "refresh", "outputs": outputs}

    @app.post("/api/refresh-full-history")
    def refresh_history() -> dict[str, object]:
        outputs = _run_action(runtime_dir, lambda: refresh_full_history(season=active_season))
        db.set_json(
            "status",
            {"status": "complete", "last_action": "refresh-full-history", "outputs": outputs},
        )
        return {"status": "complete", "action": "refresh-full-history", "outputs": outputs}

    @app.post("/api/train")
    def train() -> dict[str, object]:
        model_run = _run_action(runtime_dir, lambda: train_from_processed(season=active_season))
        db.set_json(
            "status",
            {
                "status": "complete",
                "last_action": "train",
                "updated_at": _utc_now(),
                **model_run.metrics,
            },
        )
        return {"status": "complete", "action": "train", "metrics": model_run.metrics}

    @app.post("/api/predict")
    def predict() -> dict[str, object]:
        prediction_frame = _run_action(
            runtime_dir,
            lambda: predict_from_processed(db, season=active_season),
        )
        db.set_json(
            "status",
            {
                "status": "complete",
                "last_action": "predict",
                "predictions": len(prediction_frame),
                "updated_at": _utc_now(),
            },
        )
        return {"status": "complete", "action": "predict", "predictions": len(prediction_frame)}

    @app.post("/api/optimize")
    def optimize() -> dict[str, object]:
        pick_frame = _run_action(
            runtime_dir,
            lambda: optimize_from_predictions(db, season=active_season),
        )
        db.set_json(
            "status",
            {
                "status": "complete",
                "last_action": "optimize",
                "picks": len(pick_frame),
                "updated_at": _utc_now(),
            },
        )
        return {"status": "complete", "action": "optimize", "picks": len(pick_frame)}

    @app.post("/api/run-all")
    def run_full_pipeline() -> dict[str, object]:
        result = _run_action(
            runtime_dir,
            lambda: run_all_with_database(db, refresh_mode, active_season),
        )
        db.set_json(
            "status",
            {"status": "complete", "last_action": "run-all", "updated_at": _utc_now(), **result},
        )
        return {"status": "complete", "action": "run-all", **result}

    @app.post("/api/backtest/{target_season}")
    def backtest(target_season: str) -> dict[str, object]:
        result = _run_action(runtime_dir, lambda: backtest_from_processed(target_season))
        db.set_json("backtest", result)
        db.set_json("status", {"status": "complete", "last_action": "backtest", **result})
        return {"status": "complete", "action": "backtest", **result}

    @app.post("/api/backtest-season/{target_season}", response_class=RedirectResponse)
    def backtest_season_redirect(target_season: str) -> RedirectResponse:
        _run_action(runtime_dir, lambda: backtest_from_processed(target_season))
        return RedirectResponse(f"/backtest?season={target_season}", status_code=303)

    @app.post("/actions/{action_name}", response_class=RedirectResponse)
    def run_action_from_form(action_name: str) -> RedirectResponse:
        actions = {
            "refresh": refresh,
            "refresh-full-history": refresh_history,
            "train": train,
            "predict": predict,
            "optimize": optimize,
            "run-all": run_full_pipeline,
            "backtest-current": lambda: backtest(active_season.code),
            "backtest-2425": lambda: backtest_season_redirect("2425"),
            "backtest-2324": lambda: backtest_season_redirect("2324"),
        }
        if action_name not in actions:
            raise HTTPException(status_code=404, detail="Unknown action")
        actions[action_name]()
        return RedirectResponse("/", status_code=303)

    return app


def run_all_with_database(
    database: Database,
    mode: Literal["live", "offline"] = "live",
    season: SeasonContext | None = None,
) -> dict[str, int]:
    """Run the full pipeline while writing prediction and pick state to the UI database."""
    active_season = season or SeasonContext.current_season()
    refresh_data(mode=mode, season=active_season)
    train_from_processed(season=active_season)
    predictions = predict_from_processed(database, season=active_season)
    picks = optimize_from_predictions(database, season=active_season)
    return {"predictions": len(predictions), "picks": len(picks)}


def _run_background_pipeline(
    runtime_dir: Path,
    database: Database,
    season: SeasonContext,
) -> dict[str, int]:
    """Refresh live inputs and republish decisions from the application lifespan."""
    with chdir(runtime_dir):
        return run_all_with_database(database, "live", season)


def _run_action[T](runtime_dir: Path, action: Callable[[], T]) -> T:
    """Execute a filesystem-writing pipeline action from the app runtime directory."""
    try:
        with chdir(runtime_dir):
            return action()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _list_datasets(runtime_dir: Path) -> list[dict[str, object]]:
    datasets = []
    for group in ["processed", "exports", "raw/football-data", "raw/clubelo"]:
        directory = runtime_dir / "data" / group
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            try:
                row_count = sum(1 for _ in path.open(encoding="utf-8", errors="ignore")) - 1
            except OSError:
                row_count = 0
            datasets.append(
                {
                    "group": group,
                    "name": path.stem,
                    "file": path.name,
                    "rows": max(row_count, 0),
                    "size": path.stat().st_size,
                }
            )
    return datasets


def _dataset_path(runtime_dir: Path, dataset_path: str) -> Path:
    allowed_groups = {"processed", "exports", "raw/football-data", "raw/clubelo"}
    parts = dataset_path.split("/")
    if len(parts) < 2:
        raise HTTPException(status_code=404, detail="Dataset group not found")
    group = "/".join(parts[:-1])
    name = parts[-1]
    if group not in allowed_groups:
        raise HTTPException(status_code=404, detail="Dataset group not found")
    safe_name = Path(name).stem
    return runtime_dir / "data" / group / f"{safe_name}.csv"


def _read_csv_preview(path: Path, limit: int) -> pd.DataFrame:
    bounded_limit = max(1, min(limit, 500))
    return pd.read_csv(path, nrows=bounded_limit).fillna("")


def _season_summary(actual_picks: list[dict]) -> dict:
    points = sum(p["actual_points"] or 0 for p in actual_picks)
    wins = sum(1 for p in actual_picks if p.get("actual_points") == 3)
    draws = sum(1 for p in actual_picks if p.get("actual_points") == 1)
    losses = sum(1 for p in actual_picks if p.get("actual_points") == 0)
    pending = sum(1 for p in actual_picks if p.get("actual_points") is None)
    submitted = len(actual_picks)
    avg_pts = points / submitted if submitted else 0.0
    picked_weeks = {int(p["contest_week"]) for p in actual_picks}
    # Find next open week from picks stored in DB (won't know all weeks without predictions).
    next_open = None
    for w in range(1, 40):
        if w not in picked_weeks:
            next_open = w
            break
    return {
        "points": points,
        "submitted": submitted,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "pending": pending,
        "avg_pts": avg_pts,
        "next_open_week": next_open,
    }


def _candidate_kickoff(candidate: dict[str, object]) -> datetime:
    """Return an aware kickoff, using the legacy date-only 15:00 UTC fallback."""
    raw = candidate.get("kickoff_utc")
    if raw and not pd.isna(raw):
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    date_value = datetime.fromisoformat(str(candidate["date"])).date()
    return datetime.combine(date_value, time(15, 0), tzinfo=UTC)


def _active_season_predictions(
    predictions: list[dict[str, object]],
    season: SeasonContext,
) -> list[dict[str, object]]:
    """Reject cached predictions outside the active season's date window."""
    if not predictions:
        return []
    frame = pd.DataFrame(predictions)
    if "date" not in frame:
        return []
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    start = pd.Timestamp(year=season.start_year, month=7, day=1, tz="UTC")
    end = pd.Timestamp(year=season.start_year + 1, month=7, day=1, tz="UTC")
    current = frame[dates.ge(start) & dates.lt(end)]
    return current.to_dict(orient="records")


def _recent_picks_with_results(
    runtime_dir: Path,
    actual_picks: list[dict],
    n: int = 6,
) -> list[dict]:
    import pandas as pd

    results_path = runtime_dir / "data" / "processed" / "historical_matches.csv"
    results_by_id: dict[str, dict] = {}
    if results_path.exists():
        try:
            hist = pd.read_csv(results_path)
            for row in hist.itertuples(index=False):
                hg = row.home_goals if not pd.isna(row.home_goals) else None
                ag = row.away_goals if not pd.isna(row.away_goals) else None
                if hg is not None and ag is not None:
                    if hg == ag:
                        label = f"Draw {int(hg)}-{int(ag)}"
                    elif hg > ag:
                        label = f"{row.home_team} {int(hg)}-{int(ag)}"
                    else:
                        label = f"{row.away_team} {int(ag)}-{int(hg)}"
                    results_by_id[str(row.match_id)] = {"result_label": label}
        except Exception:
            pass
    recent = sorted(actual_picks, key=lambda p: int(p["contest_week"]), reverse=True)[:n]
    for pick in recent:
        result = results_by_id.get(str(pick.get("match_id", "")), {})
        pick["result_label"] = result.get("result_label")
    return list(reversed(recent))


def _load_backtest_metrics(runtime_dir: Path, season: str) -> dict | None:
    import json

    path = runtime_dir / "data" / "artifacts" / f"{season}_backtest_metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "winner_points" not in data:
        data["winner_points"] = WINNER_BENCHMARKS.get(season)
    return data


def _load_backtest_picks(runtime_dir: Path, season: str) -> list[dict]:
    import pandas as pd

    path = runtime_dir / "data" / "exports" / f"{season}_backtest_optimized_picks.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path).fillna("")
    cumulative = 0
    rows = []
    for pick in frame.to_dict(orient="records"):
        cumulative += int(pick.get("actual_points") or 0)
        pick["cumulative"] = cumulative
        if pick["venue"] == "home":
            pick["home_team_label"] = pick["team"]
            pick["away_team_label"] = pick["opponent"]
        else:
            pick["home_team_label"] = pick["opponent"]
            pick["away_team_label"] = pick["team"]
        rows.append(pick)
    return rows


def _build_chart(picks: list[dict], winner_points: int | None) -> dict | None:
    if not picks:
        return None
    width, height = 600, 160
    max_pts = (
        max(
            (winner_points or 0),
            picks[-1]["cumulative"] if picks else 0,
            10,
        )
        + 8
    )
    n = len(picks)

    dots = []
    for i, pick in enumerate(picks):
        x = round((i + 1) / n * width, 1)
        y = round(height - pick["cumulative"] / max_pts * height, 1)
        dots.append({"x": x, "y": y, "week": pick["contest_week"]})

    model_points = " ".join(f"{d['x']},{d['y']}" for d in dots)

    winner_y = None
    if winner_points:
        winner_y = round(height - winner_points / max_pts * height, 1)

    y_ticks = []
    step = 10
    for val in range(0, int(max_pts) + 1, step):
        y = round(height - val / max_pts * height, 1)
        y_ticks.append({"y": y, "label": str(val)})

    return {
        "width": width,
        "height": height,
        "model_points": model_points,
        "winner_y": winner_y,
        "dots": dots,
        "y_ticks": y_ticks,
    }


def _filter_challenge_rows(
    rows: list[dict[str, object]],
    week: str,
    view: str,
    team: str,
    summary: dict[str, object],
) -> list[dict[str, object]]:
    filtered = rows
    selected_week = summary["next_open_week"] if week == "next" else week
    if selected_week != "all" and selected_week is not None:
        filtered = [row for row in filtered if str(row["contest_week"]) == str(selected_week)]
    if team:
        filtered = [row for row in filtered if row["home_team"] == team or row["away_team"] == team]
    if view == "open":
        filtered = [row for row in filtered if not row.get("actual_pick")]
    elif view == "picked":
        filtered = [row for row in filtered if row.get("actual_pick")]
    elif view == "close":
        filtered = [row for row in filtered if row.get("is_close_call")]
    elif view == "overrides":
        filtered = [row for row in filtered if row.get("is_override")]
    return filtered


app = create_app()


def build_scorecard(
    database: Database,
    runtime_dir: Path,
    season: str,
    entrant: str | None = None,
) -> dict[str, object]:
    """Assemble the scorecard: my scored rounds, the field, and where I sit in it."""
    picks = database.list_actual_picks(season)
    results = _match_results(runtime_dir)
    probabilities = _pick_probabilities(runtime_dir)

    my_picks: list[dict[str, object]] = []
    running = 0
    for pick in sorted(picks, key=lambda row: int(row["contest_week"])):
        points = pick.get("actual_points")
        if points is not None:
            running += int(points)
        result = results.get(str(pick.get("match_id", "")), {})
        my_picks.append(
            {
                "contest_week": int(pick["contest_week"]),
                "team": pick["team"],
                "venue": pick["venue"],
                "opponent": result.get("opponent_for", {}).get(pick["team"]),
                "result_label": result.get("label"),
                "p_win": probabilities.get((int(pick["contest_week"]), pick["team"])),
                "points": points,
                "running_total": running,
                "state_class": _pick_state_class(points),
                "email": compose_pick_email(
                    int(pick["contest_week"]),
                    pick["team"],
                    result.get("opponent_for", {}).get(pick["team"]) or "opponent",
                ),
            }
        )

    scored = [pick for pick in picks if pick.get("actual_points") is not None]
    summary = {
        "points": sum(int(pick["actual_points"]) for pick in scored),
        "scored": len(scored),
        "submitted": len(picks),
        "pending": len(picks) - len(scored),
        "wins": sum(1 for pick in scored if pick["actual_points"] == 3),
        "draws": sum(1 for pick in scored if pick["actual_points"] == 1),
        "losses": sum(1 for pick in scored if pick["actual_points"] == 0),
    }
    summary["avg_points"] = summary["points"] / len(scored) if scored else 0.0

    league_url = (database.get_json("league_sheet_url", {}) or {}).get("url")
    entrants, standing = _league_standings(database, season, summary["points"])
    league_picks = database.list_league_picks(season, entrant) if entrant else []

    return {
        "season": season,
        "summary": summary,
        "my_picks": my_picks,
        "entrants": entrants,
        "standing": standing,
        "selected_entrant": entrant,
        "entrant_picks": league_picks,
        "popular": _field_consensus(database, season, my_picks),
        "chances": _win_chances(database, season, runtime_dir, summary["points"]),
        "league_url": league_url,
    }


def _pick_state_class(points: object) -> str:
    if points is None:
        return ""
    return {3: "is-win", 1: "is-draw", 0: "is-loss"}.get(int(points), "")


def _match_results(runtime_dir: Path) -> dict[str, dict]:
    """Final scores by match id, with a readable label and each side's opponent."""
    path = runtime_dir / "data" / "processed" / "historical_matches.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path).dropna(subset=["home_goals", "away_goals"])
    results: dict[str, dict] = {}
    for match in frame.itertuples(index=False):
        home, away = int(match.home_goals), int(match.away_goals)
        if home == away:
            label = f"{home}-{away} draw"
        elif home > away:
            label = f"{match.home_team} {home}-{away}"
        else:
            label = f"{match.away_team} {away}-{home}"
        results[str(match.match_id)] = {
            "label": label,
            "opponent_for": {match.home_team: match.away_team, match.away_team: match.home_team},
        }
    return results


def _pick_probabilities(runtime_dir: Path) -> dict[tuple[int, str], float]:
    """Modelled win probability per (round, team) from the published forecasts."""
    path = runtime_dir / "data" / "exports" / "fixture_probabilities.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    mapping: dict[tuple[int, str], float] = {}
    for row in frame.itertuples(index=False):
        week = int(row.contest_week)
        mapping[(week, row.home_team)] = float(row.p_home_win)
        mapping[(week, row.away_team)] = float(row.p_away_win)
    return mapping


def _league_standings(
    database: Database,
    season: str,
    my_points: int,
) -> tuple[list[dict], dict | None]:
    """Imported entrants with round counts, plus where the user sits."""
    entrants = database.list_league_entrants(season)
    if not entrants:
        return [], None
    picks = database.list_league_picks(season)
    rounds: dict[str, int] = {}
    for pick in picks:
        rounds[pick["entrant"]] = rounds.get(pick["entrant"], 0) + 1

    rows = []
    for item in entrants:
        rows.append({**item, "rounds": rounds.get(item["entrant"], 0), "is_me": False})
    ahead = sum(1 for row in rows if (row["points"] or 0) > my_points)
    return rows, {"place": ahead + 1, "entrants": len(rows), "points": my_points}


def _field_consensus(database: Database, season: str, my_picks: list[dict]) -> list[dict]:
    """The most-taken team per round, and what the user took instead."""
    picks = database.list_league_picks(season)
    if not picks:
        return []
    frame = pd.DataFrame(picks)
    mine = {row["contest_week"]: row["team"] for row in my_picks}
    rows = []
    for week, group in frame.groupby("contest_week"):
        counts = group["team"].value_counts()
        top = counts.index[0]
        taken = group[group["team"] == top]
        rows.append(
            {
                "contest_week": int(week),
                "team": top,
                "share": float(counts.iloc[0] / len(group)),
                "avg_points": (
                    float(taken["points"].mean()) if taken["points"].notna().any() else 0.0
                ),
                "mine": mine.get(int(week)),
            }
        )
    return sorted(rows, key=lambda row: row["contest_week"])


def _win_chances(
    database: Database,
    season: str,
    runtime_dir: Path,
    points_so_far: int,
) -> dict | None:
    """Chance of finishing top, using a completed season's field as the comparison.

    The current season has no final totals yet, so the field is taken from the
    most recent completed import. It is an estimate of the standard this
    contest is won at, not a reading of this season's rivals.
    """
    plan_path = runtime_dir / "data" / "exports" / "optimized_picks.csv"
    if not plan_path.exists():
        return None
    field_season = next(
        (
            code
            for code in sorted(_imported_seasons(database), reverse=True)
            if code != season
        ),
        None,
    )
    if field_season is None:
        return None
    totals = [
        entrant["points"]
        for entrant in database.list_league_entrants(field_season)
        if entrant["points"] is not None
    ]
    if not totals:
        return None
    plan = pd.read_csv(plan_path)
    remaining = plan[~plan["contest_week"].isin({row for row in _scored_weeks(database, season)})]
    chances = win_probability_against_field(remaining, totals, points_so_far=points_so_far)
    return {**chances, "field_season": field_season}


def _imported_seasons(database: Database) -> list[str]:
    seasons: set[str] = set()
    for code in ("2223", "2324", "2425", "2526", "2627"):
        if database.list_league_entrants(code):
            seasons.add(code)
    return sorted(seasons)


def _scored_weeks(database: Database, season: str) -> set[int]:
    return {
        int(pick["contest_week"])
        for pick in database.list_actual_picks(season)
        if pick.get("actual_points") is not None
    }


def _imported_banner(value: str | None) -> dict[str, object] | None:
    """Decode the post-import confirmation carried through the redirect."""
    if not value:
        return None
    parts = value.split(":")
    if len(parts) != 3:
        return None
    season, entrants, picks = parts
    if not (entrants.isdigit() and picks.isdigit()):
        return None
    return {"season": season, "entrants": int(entrants), "picks": int(picks)}


def build_readiness(
    database: Database,
    runtime_dir: Path,
    season: SeasonContext,
    market_weight: float,
) -> dict[str, object]:
    """Provenance for the pick about to be made: what fed it, and when.

    A refresh that silently used stale prices or an un-retrained model looks
    exactly like one that did not, so every claim here is read back off the
    artifact that would actually be used rather than from a status flag.
    """
    exports = runtime_dir / "data" / "exports"
    processed = runtime_dir / "data" / "processed"
    artifacts = runtime_dir / "data" / "artifacts"

    metrics = _read_json(artifacts / "metrics.json")
    forecasts = _read_csv(exports / "fixture_probabilities.csv")
    fixtures = _read_csv(processed / "fixtures.csv")
    results = _read_csv(processed / "historical_matches.csv")

    played = pd.DataFrame()
    if not results.empty and "season" in results:
        played = results[results["season"].astype(str).str.zfill(4) == season.code]
        played = played.dropna(subset=["home_goals", "away_goals"])

    round_number, priced, total = _next_round_pricing(fixtures, forecasts)
    entrants = database.list_league_entrants(season.code)

    checks = [
        {
            "label": "Live data pulled",
            "value": _file_age(processed / "fixtures.csv"),
            "ok": _is_fresh(processed / "fixtures.csv"),
            "detail": f"{len(fixtures)} fixtures, {len(played)} results recorded",
        },
        {
            "label": "Model retrained",
            "value": _timestamp_age(metrics.get("trained_at")),
            "ok": _trained_after(metrics.get("trained_at"), processed / "fixtures.csv"),
            "detail": f"{int(metrics.get('rows', 0)):,} matches",
        },
        {
            "label": "Forecasts published",
            "value": _file_age(exports / "fixture_probabilities.csv"),
            "ok": _is_fresh(exports / "fixture_probabilities.csv"),
            "detail": f"{len(forecasts)} fixtures priced by the model",
        },
        {
            "label": f"Round {round_number} market prices" if round_number else "Market prices",
            "value": f"{priced}/{total} fixtures" if total else "no round pending",
            "ok": bool(total) and priced == total,
            "detail": (
                f"blended at weight {market_weight:g}"
                if priced
                else "the market has not opened this round yet"
            ),
        },
        {
            "label": "Contest sheet",
            "value": f"{len(entrants)} entrants" if entrants else "not imported",
            "ok": bool(entrants),
            "detail": f"season {season.code}",
        },
    ]
    return {"checks": checks, "all_ok": all(check["ok"] for check in checks)}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (ValueError, OSError):
        return pd.DataFrame()


def _next_round_pricing(
    fixtures: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> tuple[int | None, int, int]:
    """How many of the next weekend round's fixtures carry a market price."""
    if fixtures.empty or forecasts.empty or "market_weight" not in forecasts:
        return None, 0, 0
    frame = fixtures.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    today = pd.Timestamp(datetime.now(UTC)).tz_localize(None).normalize()
    weekend = frame[(frame["date"] >= today) & (frame["date"].dt.dayofweek.isin([5, 6]))]
    if weekend.empty:
        return None, 0, 0
    round_number = int(weekend["contest_week"].min())
    ids = set(weekend[weekend["contest_week"] == round_number]["match_id"].astype(str))
    scored = forecasts[forecasts["match_id"].astype(str).isin(ids)]
    priced = int((pd.to_numeric(scored["market_weight"], errors="coerce") > 0).sum())
    return round_number, priced, len(ids)


def _file_age(path: Path) -> str:
    if not path.exists():
        return "missing"
    return _describe_age(datetime.fromtimestamp(path.stat().st_mtime, tz=UTC))


def _timestamp_age(value: object) -> str:
    if not value:
        return "never"
    try:
        return _describe_age(datetime.fromisoformat(str(value)))
    except ValueError:
        return str(value)


def _describe_age(moment: datetime) -> str:
    minutes = (datetime.now(UTC) - moment).total_seconds() / 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)} min ago"
    if minutes < 60 * 24:
        return f"{int(minutes // 60)}h ago"
    return f"{int(minutes // (60 * 24))}d ago"


def _is_fresh(path: Path, hours: int = 24) -> bool:
    """Artifacts older than a day predate this weekend's team news and prices."""
    if not path.exists():
        return False
    age = datetime.now(UTC) - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return age.total_seconds() < hours * 3600


def _trained_after(trained_at: object, data_path: Path) -> bool:
    """The model must be newer than the data, or it never saw the latest results."""
    if not trained_at or not data_path.exists():
        return False
    try:
        trained = datetime.fromisoformat(str(trained_at))
    except ValueError:
        return False
    refreshed = datetime.fromtimestamp(data_path.stat().st_mtime, tz=UTC)
    return trained >= refreshed and _is_fresh(data_path)


def _reference_field(database: Database, season: str) -> list[int]:
    """Finishing totals from the most recent completed field, for scoring plans.

    The season in progress has no final totals, so the closest completed import
    stands in as the standard this contest is won at.
    """
    for code in sorted(_imported_seasons(database), reverse=True):
        if code == season:
            continue
        totals = [
            entrant["points"]
            for entrant in database.list_league_entrants(code)
            if entrant["points"] is not None
        ]
        if totals:
            return totals
    return []


def _submission_email(
    committed: dict | None,
    scenarios: list[dict],
) -> dict[str, str] | None:
    """Compose the organiser email once a pick for the round is committed.

    The opponent is not stored on the pick, so it is read back from the
    scenario the pick corresponds to.
    """
    if not committed:
        return None
    match = next(
        (
            item
            for item in scenarios
            if str(item.get("match_id")) == str(committed.get("match_id"))
            and item.get("team") == committed.get("team")
        ),
        None,
    )
    if match is None:
        return None
    return compose_pick_email(
        int(committed["contest_week"]), committed["team"], match["opponent"]
    )
