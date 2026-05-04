"""FastAPI dashboard and JSON endpoints for operating the optimizer."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import chdir
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from epl_prediction_optimizer.challenge import (
    load_challenge_rows,
    score_manual_pick,
    summarize_challenge,
)
from epl_prediction_optimizer.pipeline import (
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


def create_app(
    database: Database | None = None,
    workdir: Path | str | None = None,
    use_live_data: bool = True,
) -> FastAPI:
    """Create a FastAPI app using the provided or default SQLite database."""
    db = database or Database()
    runtime_dir = Path(workdir or ".").resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="EPL Prediction Optimizer")
    templates = Jinja2Templates(directory=str(PACKAGE_ROOT / "templates"))
    app.mount(
        "/static",
        StaticFiles(directory=str(PACKAGE_ROOT / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        picks = db.get_json("picks", [])
        actual_picks = db.list_actual_picks("2526")
        season_summary = _season_summary(actual_picks)
        current_week = season_summary["next_open_week"]
        week_recommendation = next(
            (p for p in picks if current_week and p["contest_week"] == current_week), None
        )
        recent_picks = _recent_picks_with_results(runtime_dir, actual_picks)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "status": db.get_json("status", {"status": "ready"}),
                "predictions": db.get_json("predictions", []),
                "picks": picks,
                "season_summary": season_summary,
                "current_week": current_week,
                "week_recommendation": week_recommendation,
                "recent_picks": recent_picks,
            },
        )

    @app.get("/model", response_class=HTMLResponse)
    def model_view(request: Request) -> HTMLResponse:
        import json as _json
        metrics_path = runtime_dir / "data" / "artifacts" / "metrics.json"
        metrics = _json.loads(metrics_path.read_text()) if metrics_path.exists() else None
        from epl_prediction_optimizer.pipeline import WINNER_BENCHMARKS
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
            },
        )

    @app.get("/data", response_class=HTMLResponse)
    def data_explorer(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "data.html",
            {
                "datasets": _list_datasets(runtime_dir),
            },
        )

    @app.get("/challenge", response_class=HTMLResponse)
    def challenge_manager(
        request: Request,
        season: str = "2526",
        week: str = "next",
        view: str = "all",
        team: str = "",
    ) -> HTMLResponse:
        actual_picks = db.list_actual_picks(season)
        rows = load_challenge_rows(runtime_dir, season, actual_picks)
        summary = summarize_challenge(rows, actual_picks)
        filtered_rows = _filter_challenge_rows(rows, week, view, team, summary)
        return templates.TemplateResponse(
            request,
            "challenge.html",
            {
                "season": season,
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
        return {"status": "ready", **db.get_json("status", {})}

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
        outputs = _run_action(runtime_dir, lambda: refresh_data(use_network=use_live_data))
        db.set_json("status", {"status": "complete", "last_action": "refresh", "outputs": outputs})
        return {"status": "complete", "action": "refresh", "outputs": outputs}

    @app.post("/api/refresh-full-history")
    def refresh_history() -> dict[str, object]:
        outputs = _run_action(runtime_dir, refresh_full_history)
        db.set_json(
            "status",
            {"status": "complete", "last_action": "refresh-full-history", "outputs": outputs},
        )
        return {"status": "complete", "action": "refresh-full-history", "outputs": outputs}

    @app.post("/api/train")
    def train() -> dict[str, object]:
        model_run = _run_action(runtime_dir, train_from_processed)
        db.set_json("status", {"status": "complete", "last_action": "train", **model_run.metrics})
        return {"status": "complete", "action": "train", "metrics": model_run.metrics}

    @app.post("/api/predict")
    def predict() -> dict[str, object]:
        prediction_frame = _run_action(runtime_dir, lambda: predict_from_processed(db))
        db.set_json(
            "status",
            {
                "status": "complete",
                "last_action": "predict",
                "predictions": len(prediction_frame),
            },
        )
        return {"status": "complete", "action": "predict", "predictions": len(prediction_frame)}

    @app.post("/api/optimize")
    def optimize() -> dict[str, object]:
        pick_frame = _run_action(runtime_dir, lambda: optimize_from_predictions(db))
        db.set_json(
            "status",
            {"status": "complete", "last_action": "optimize", "picks": len(pick_frame)},
        )
        return {"status": "complete", "action": "optimize", "picks": len(pick_frame)}

    @app.post("/api/run-all")
    def run_full_pipeline() -> dict[str, object]:
        result = _run_action(runtime_dir, lambda: run_all_with_database(db, use_live_data))
        db.set_json("status", {"status": "complete", "last_action": "run-all", **result})
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
            "backtest-2526": lambda: backtest("2526"),
            "backtest-2425": lambda: backtest_season_redirect("2425"),
            "backtest-2324": lambda: backtest_season_redirect("2324"),
        }
        if action_name not in actions:
            raise HTTPException(status_code=404, detail="Unknown action")
        actions[action_name]()
        return RedirectResponse("/", status_code=303)

    return app


def run_all_with_database(database: Database, use_live_data: bool = True) -> dict[str, int]:
    """Run the full pipeline while writing prediction and pick state to the UI database."""
    refresh_data(use_network=use_live_data)
    train_from_processed()
    predictions = predict_from_processed(database)
    picks = optimize_from_predictions(database)
    return {"predictions": len(predictions), "picks": len(picks)}


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
    # find next open week from picks stored in DB (won't know all weeks without predictions)
    all_weeks = sorted(picked_weeks | {int(p["contest_week"]) for p in actual_picks})
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


def _recent_picks_with_results(runtime_dir: Path, actual_picks: list[dict], n: int = 6) -> list[dict]:
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
    max_pts = max(
        (winner_points or 0),
        picks[-1]["cumulative"] if picks else 0,
        10,
    ) + 8
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
        filtered = [
            row
            for row in filtered
            if row["home_team"] == team or row["away_team"] == team
        ]
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
