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

## Weekly run

Start the dashboard and do everything from there:

```bash
uv run eplpo serve --port 8000     # then open http://127.0.0.1:8000
```

Click **Refresh everything** in the decision header. It pulls current fixtures
and results, captures market prices, retrains, re-forecasts, and re-optimizes —
about a minute or two. Then pick the week and commit.

Run it **Friday or later**. Bookmakers open a round a few days out, and a pick
made before its round is priced loses the strongest input the pipeline has.

The server no longer refreshes on a timer. Nothing overwrites your forecasts
unless you ask it to, so leaving it running is safe.

### Same thing from the CLI

```bash
uv run eplpo run-all               # refresh → train → predict → optimize
```

Or one step at a time:

```bash
uv run eplpo refresh               # fixtures, results, Elo, market prices
uv run eplpo train
uv run eplpo predict
uv run eplpo optimize
uv run eplpo odds                  # prices only
```

### Submitting the pick

Committing a pick shows the submission email with **Send in Gmail** and a
plain `mailto:` fallback.

With more than one Google account signed in, Gmail composes from whichever it
treats as default. It selects an account from the **number** in the URL —
`mail.google.com/mail/u/1/` is account 1 — and ignores an email address there,
falling back to the default without saying so.

So set **Gmail account** under the buttons to that digit: open Gmail on the
account you want to send from and copy the number out of its address bar. The
panel warns while it is unset or holds an address rather than a number. Confirm
the From line the first time either way.

`CONTEST_EMAIL`, `CONTEST_ENTRANT`, and `CONTEST_GMAIL_ACCOUNT` override the
recipient, the name in the subject, and the account index.

### Scorecard and the field

**Scorecard** (nav 03) is the grid view: every committed round with the result,
points, and running total, plus the leaderboard once a contest sheet is
imported.

To import the contest sheet, from the dashboard:

1. Click **03 · Scorecard**.
2. Scroll to **Leaderboard**.
3. Paste the sheet link from your browser's address bar into **Contest sheet
   link**.
4. Set **Season the sheet covers** to the season that sheet is for — last
   year's is `2526`, this year's `2627`. It defaults to the season you are
   viewing, so change it when importing an older sheet.
5. Click **Import sheet**. A confirmation names how many entrants and picks
   were read.

The sheet must be shared as *Anyone with the link · Viewer*. Re-importing
replaces that season's data, so it is safe to repeat whenever the sheet is
updated.

Results settle automatically as part of a refresh, so committed picks pick up
their points without any extra step.

CLI equivalents, if ever needed:

```bash
uv run eplpo import-league "<google sheets url>" --season 2526
uv run eplpo score --season 2627
```

### Check before you commit

```bash
uv run scripts/check_active_season.py
```

Confirms the schedule and results are intact, Elo is attached, and — the one
that matters on Friday — whether the round you are about to pick is priced yet.
If it reports `0/8 priced`, the market has not opened that round; refresh again
closer to kickoff.

## Other commands

```bash
uv run eplpo serve --port 8000
uv run eplpo refresh-full-history  # rebuild the long-term historical cache
uv run eplpo backtest --season 2425
```

`refresh` updates the active 2026-27 season and reuses cached historical
seasons and ClubElo files.

`backtest` predicts straight from the training frame, which does not exercise
the fixture-feature path the live `predict` step uses. To measure what
production actually does, replay seasons week by week:

```bash
uv run scripts/backtest_production_path.py --seasons 2324 2425 2526
```

Measure where the edge is:

```bash
uv run scripts/backtest_optimizer_edge.py       # optimizer vs greedy weekly picks
uv run scripts/hindsight_ceiling.py             # best score the rules allow
uv run scripts/backtest_tournament_objective.py # planning to win vs to average well
uv run scripts/calibrate_to_market.py           # model confidence vs market confidence
```

See `docs/model-evaluation.md` for which metrics decide a change, why contest
points are not one of them, and which levers were measured and rejected.

## Market Prices

Probabilities are pooled with de-vigged bookmaker prices for fixtures that are
priced, at a weight tuned by backtest (`MARKET_BLEND_WEIGHT` in
`pipeline.py`). Prices come from the public football-data.co.uk fixtures feed —
the same B365 columns the historical season CSVs carry, so live and backtest
prices share one schema and need no API key. `curl_cffi` supplies a browser TLS
fingerprint when the plain request is refused.

Only the priced rounds move. Later fixtures keep pure model probabilities, so
the season-long plan stays internally consistent. Every exported probability row
carries a `market_weight` column recording how much market signal it received.

Prices are inputs to a forecast, not a betting product: nothing here prices,
places, or recommends a wager.

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
