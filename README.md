# EPL Prediction Optimizer

Python/FastAPI implementation of the EPL prediction optimizer plan:

- Stage A: ingest EPL results/fixtures, engineer features, train calibrated
  HOME_WIN / DRAW / AWAY_WIN probabilities.
- Stage B: convert probabilities into candidate team picks and solve the
  season-long integer program with the contest constraints.
- Stage C: serve a lightweight dashboard backed by SQLite and CSV exports.

## Environment

This repository uses `uv` for dependency and environment management.

```bash
uv sync
uv run pytest
```

## Commands

Run the full live-data pipeline:

```bash
uv run eplpo run-all
```

Run each stage separately:

```bash
uv run eplpo refresh
uv run eplpo train
uv run eplpo predict
uv run eplpo optimize
```

Start the dashboard:

```bash
uv run eplpo serve --port 8000
```

Then open `http://127.0.0.1:8000`.

The decision desk automatically refreshes live data at startup and every 30
minutes while it is running. From the UI you can also manually:

- `Pull Data`
- `Train Model`
- `Predict Fixtures`
- `Optimize Picks`
- `Prepare season`
- `Explore Stored Data`
- `Challenge Manager`

The UI actions write the same CSV/model/SQLite artifacts as the CLI commands.
The Challenge Manager stores your actual submitted pick separately from the
optimizer recommendation, including notes for injury/news overrides and scored
points when the result is known.

For weekly use after the first full-history pull, use:

```bash
uv run eplpo refresh
uv run eplpo train
uv run eplpo predict
uv run eplpo optimize
```

`refresh` refreshes the active 2026-27 season and reuses cached
historical seasons and ClubElo files. Use `refresh-full-history` only when you
want to rebuild the long-term historical cache.

Backtest a completed season using only earlier seasons for training:

```bash
uv run eplpo backtest --season 2425
```

## Live Data

Live fixture refresh requires a football-data.org API key:

```bash
FOOTBALL_DATA_ORG_API_KEY=... uv run eplpo run-all
```

The pipeline downloads public historical results and current ClubElo ratings.
If live fixtures cannot be verified, it reports the failure instead of silently
presenting sample data as current.

## Outputs

- `data/processed/historical_matches.csv`
- `data/processed/fixtures.csv`
- `data/artifacts/model.joblib`
- `data/artifacts/metrics.json`
- `data/exports/fixture_probabilities.csv`
- `data/exports/optimized_picks.csv`
- `data/state.sqlite`
# epl-prediction-optimizer
