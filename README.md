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

Run the full sample-data pipeline:

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

From the HTML UI you can run:

- `Pull Data`
- `Train Model`
- `Predict Fixtures`
- `Optimize Picks`
- `Run Full Pipeline`
- `Explore Stored Data`
- `Challenge Manager`

The UI actions write the same CSV/model/SQLite artifacts as the CLI commands.
The Challenge Manager stores your actual submitted pick separately from the
optimizer recommendation, including notes for injury/news overrides and scored
points when the result is known.

For weekly use after the first full-history pull, use:

```bash
uv run eplpo refresh --network
uv run eplpo train
uv run eplpo predict
uv run eplpo optimize
```

`refresh --network` refreshes the current 2025-26 CSV and reuses cached
historical seasons and ClubElo files. Use `refresh-full-history` only when you
want to rebuild the long-term historical cache.

Backtest the in-progress 2025-26 season against completed matches:

```bash
uv run eplpo backtest --season 2526
```

## Live Data

By default, `refresh` uses deterministic sample fixtures so the project works
offline. To download public data where possible:

```bash
FOOTBALL_DATA_ORG_API_KEY=... uv run eplpo run-all --network
```

The `--network` mode downloads football-data.co.uk historical CSVs and attempts
to attach current ClubElo ratings. If no football-data.org API key is present,
fixture refresh falls back to sample fixtures.

## Outputs

- `data/processed/historical_matches.csv`
- `data/processed/fixtures.csv`
- `data/artifacts/model.joblib`
- `data/artifacts/metrics.json`
- `data/exports/fixture_probabilities.csv`
- `data/exports/optimized_picks.csv`
- `data/state.sqlite`
# epl-prediction-optimizer
