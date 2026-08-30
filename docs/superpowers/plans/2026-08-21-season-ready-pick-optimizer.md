# Season-Ready Pick Optimizer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a trustworthy 2026–27 weekly EPL pick decision desk that stays current, explains exact downstream consequences, records one editable-until-kickoff pick, and works equally well in light and dark modes.

**Architecture:** Introduce a small domain layer for season, fixture, contest-round, scenario, and pick-lifecycle contracts; keep data synchronization, probability publication, optimization, persistence, API orchestration, and presentation behind focused interfaces. SQLite stores source/version/pick truth, generated scenarios remain derived data, and the FastAPI/Jinja application progressively enhances a complete server-rendered decision surface with small vanilla-JavaScript modules.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, Jinja2, SQLite, pandas, scikit-learn, PuLP/CBC, vanilla JavaScript, CSS custom properties, pytest/TestClient, Ruff, and Playwright for browser-level interaction/accessibility screenshots.

**Design specification:** `docs/superpowers/specs/2026-08-21-season-ready-pick-optimizer-design.md`

---

## File Structure

New modules are intentionally narrow. Existing large files remain as compatibility façades while behavior moves behind explicit contracts.

```text
src/epl_prediction_optimizer/
  config/season.py                 Active-season derivation and validated override
  domain/models.py                 Shared immutable fixture/round/pick/version records
  domain/rounds.py                 Europe/London contest-round derivation and state
  data/adapters.py                 Adapter protocols and normalized sync results
  data/coordinator.py              Atomic current-season refresh and source policy
  data/artifacts.py                Atomic file replacement and snapshot hashing
  prediction/service.py            Train/publish orchestration and prediction versions
  optimizer/solver.py              Locked-pick optimization and canonical plan
  optimizer/scenarios.py           Exact counterfactual scenario construction
  optimizer/explanations.py        Displacement and plain-language consequence copy
  storage/migrations.py            Ordered, idempotent SQLite schema migrations
  storage/repositories.py          Fixture, source, version, round, pick repositories
  services/readiness.py            Freshness gate and next-open-round summary
  services/reconciliation.py       Kickoff/status/round/pick lifecycle reconciliation
  services/runtime.py              Startup refresh and deadline-aware background checks
  services/picks.py                Transactional commit/change use case
  app/schemas.py                   API request/response/error models
  app/dependencies.py              Per-app service container and clock injection
  app/routes/decision.py           Readiness, round, scenario, and pick endpoints
  app/routes/operations.py         Idempotent current refresh endpoint
  app/presenters.py                Domain-to-template decision view models
  app/static/decision.css          Balanced Horizon layout and component styling
  app/static/decision.js           Preview, navigation, theme, and accessible chart UI
  app/templates/dashboard.html     Decision-desk shell
  app/templates/components/        Server-rendered partials for all core interactions
tests/
  fixtures/                        Explicitly labeled deterministic 2026–27 sample data
  unit/                            Pure domain/optimizer/version tests
  integration/                     SQLite, coordinator, pipeline, and API tests
  browser/                         Playwright UX, theme, responsive, and lock-flow tests
validation/
  baselines/                       Tracked replay inputs, metrics, and hash manifest
  waivers/                         Explicit reviewed metric-trade-off decisions
  reports/                         Tracked release-validation output
```

Files retained and adapted: `pipeline.py` delegates to the new services; `data/sources.py` becomes adapter implementation support; `storage/database.py` becomes a compatibility entry point over migrations/repositories; `app/main.py` becomes composition and legacy expert-route registration; existing model/backtest/data templates stay available as secondary surfaces.

## Global Execution Rules

- Use `UV_CACHE_DIR=/tmp/eplpo-uv-cache` for every `uv run` command in this environment.
- Freeze time and inject fake adapters in deterministic tests; no test marked offline may reach the network.
- Use timezone-aware UTC at every boundary and `ZoneInfo("Europe/London")` only for eligibility grouping.
- Do not alter model features, labels, weights, estimator, calibration, or corrections unless Task 9's model gate is executed and reviewed.
- After each task, run the focused tests, `ruff check` on touched files, and commit only that task's files.
- Preserve `.DS_Store`, `.idea/`, user data, and unrelated worktree changes.

## Chunk 1: Season, Storage, and Data Readiness

### Task 1: Repair the Baseline and Centralize Season Context

**Files:**
- Create: `src/epl_prediction_optimizer/config/__init__.py`
- Create: `src/epl_prediction_optimizer/config/season.py`
- Create: `tests/unit/test_season_context.py`
- Modify: `src/epl_prediction_optimizer/pipeline.py`
- Modify: `src/epl_prediction_optimizer/cli/main.py`
- Modify: `src/epl_prediction_optimizer/app/main.py`
- Modify: `tests/test_app.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing season-boundary and override tests**

```python
def test_season_context_uses_current_start_year_after_july():
    context = SeasonContext.from_date(date(2026, 8, 21))
    assert context.code == "2627"
    assert context.start_year == 2026
    assert context.label == "2026–27"

def test_season_context_uses_previous_start_year_before_july():
    assert SeasonContext.from_date(date(2027, 2, 1)).code == "2627"

def test_explicit_override_wins_and_is_validated(monkeypatch):
    monkeypatch.setenv("EPLPO_SEASON", "2425")
    assert current_season(today=date(2026, 8, 21)).code == "2425"
    with pytest.raises(ValueError, match="four digits"):
        SeasonContext.from_code("26x7")
    with pytest.raises(ValueError, match="consecutive years"):
        SeasonContext.from_code("2026")
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_season_context.py -q`

Expected: collection/import failure because `config.season` does not exist.

- [ ] **Step 3: Implement the immutable context and remove runtime `2526` defaults**

```python
@dataclass(frozen=True, slots=True)
class SeasonContext:
    code: str
    start_year: int
    label: str

    @classmethod
    def from_code(cls, code: str) -> "SeasonContext":
        if len(code) != 4 or not code.isdigit():
            raise ValueError("season code must contain four digits")
        start_suffix, end_suffix = int(code[:2]), int(code[2:])
        start_year = 2000 + start_suffix
        if end_suffix != (start_year + 1) % 100:
            raise ValueError("season code must describe consecutive years")
        return cls(code=code, start_year=start_year, label=f"{start_year}–{end_suffix:02d}")

    @classmethod
    def from_date(cls, value: date) -> "SeasonContext":
        start_year = value.year if value.month >= 7 else value.year - 1
        return cls.from_code(f"{start_year % 100:02d}{(start_year + 1) % 100:02d}")

def current_season(*, today: date | None = None, override: str | None = None) -> SeasonContext:
    configured = override or os.getenv("EPLPO_SEASON")
    return SeasonContext.from_code(configured) if configured else SeasonContext.from_date(today or date.today())

def require_source_season(context: SeasonContext, source_code: str) -> None:
    if source_code != context.code:
        raise SeasonMismatchError(expected=context.code, received=source_code)
```

Pass `SeasonContext` into refresh/train/predict/optimize functions while keeping no-argument CLI compatibility. Replace current-year string slicing, route defaults, template labels, artifact-current-season selection, and challenge defaults. Historical backtest literals remain only where explicitly historical.

- [ ] **Step 4: Make offline fixtures explicit and prohibit fallback network calls**

Add a required keyword-only `mode: Literal["live", "offline"]` to `refresh_data`; every production entry point passes `mode="live"`, and tests/sample commands pass `mode="offline"` explicitly. Offline mode uses deterministic sample fixtures only, and live mode raises a structured error when live fixtures are unavailable. Add a monkeypatched `requests.get` test that raises `AssertionError` if called in offline mode.

- [ ] **Step 5: Update stale UI assertions and run the repaired baseline**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/test_app.py tests/test_pipeline.py tests/unit/test_season_context.py -q`

Expected: all selected tests pass, including the former three baseline failures.

- [ ] **Step 6: Verify no active-season literal remains, lint, and commit**

Run: `rg -n '2526|2025.?26' src/epl_prediction_optimizer`

Expected: only explicitly historical backtest lists/benchmarks remain, each named as historical context.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/config src/epl_prediction_optimizer/pipeline.py src/epl_prediction_optimizer/cli/main.py src/epl_prediction_optimizer/app/main.py tests/unit/test_season_context.py tests/test_app.py tests/test_pipeline.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/config src/epl_prediction_optimizer/pipeline.py src/epl_prediction_optimizer/cli/main.py src/epl_prediction_optimizer/app/main.py tests/unit/test_season_context.py tests/test_app.py tests/test_pipeline.py
git commit -m "fix: centralize active EPL season context"
```

### Task 2: Define Fixture and Contest-Round Domain Contracts

**Files:**
- Create: `src/epl_prediction_optimizer/domain/__init__.py`
- Create: `src/epl_prediction_optimizer/domain/models.py`
- Create: `src/epl_prediction_optimizer/domain/rounds.py`
- Create: `tests/unit/test_contest_rounds.py`
- Create: `tests/fixtures/season_2627.py`
- Modify: `src/epl_prediction_optimizer/optimizer/candidates.py`

- [ ] **Step 1: Write failing round derivation tests**

Cover aware-UTC enforcement, Friday/midweek exclusion, Saturday/Sunday grouping in London time, BST/GMT transitions, multiple league matchweeks in one weekend, void weekends, and stable Saturday-date IDs.

Implement four explicit fixtures-based tests named `test_sunday_2330_utc_after_bst_change_groups_by_london_date`, `test_friday_and_monday_are_visible_but_not_eligible`, `test_two_provider_matchweeks_share_one_calendar_round`, and `test_naive_kickoff_is_rejected`. Each test constructs complete `FixtureRecord` values and asserts the exact expected round IDs or exception.

- [ ] **Step 2: Verify tests fail**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_contest_rounds.py -q`

Expected: import failure for the new domain modules.

- [ ] **Step 3: Implement typed domain records**

Use string enums for `FixtureStatus`, `ContestRoundState`, `PickState`, `NewsRisk`, and `ScenarioMode`. Implement frozen dataclasses for `FixtureRecord`, `ContestRound`, `VersionVector`, and `PickRecord`; `PickRecord` includes its integer `pick_version`. Validate aware datetimes in `__post_init__`.

- [ ] **Step 4: Implement calendar-window derivation**

Implement `contest_round_id(kickoff_utc)`, `derive_contest_rounds(fixtures, picks, now_utc)`, and `eligible_fixture_ids(round_, fixtures, now_utc)`. `contest_round_id` converts to London time and returns the preceding/current Saturday only for local Saturday/Sunday values. `derive_contest_rounds` sorts by kickoff/match ID and uses the supplied ledger states plus fixture boundaries to derive `future|open|locked|complete|missed|void`; Task 5 reconciliation remains the only mutation mechanism for those states.

Assign display ordinals only after sorting active weekend IDs. Keep provider `league_matchweek` as metadata and never use it as the persistence key.

- [ ] **Step 5: Adapt candidate creation to `contest_round_id`**

Keep DataFrame compatibility but emit `contest_round_id`, `league_matchweek`, `kickoff_utc`, `venue` (`home` or `away`), and stable match/team identity. Prove Friday fixtures produce no candidate rows.

- [ ] **Step 6: Run tests, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_contest_rounds.py tests/test_gameweeks.py tests/test_optimizer.py -q`

Expected: all selected tests pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/domain src/epl_prediction_optimizer/optimizer/candidates.py tests/unit/test_contest_rounds.py tests/fixtures/season_2627.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/domain src/epl_prediction_optimizer/optimizer/candidates.py tests/unit/test_contest_rounds.py tests/fixtures
git commit -m "feat: model calendar-based contest rounds"
```

### Task 3: Add Forward-Only SQLite Migrations and Focused Repositories

**Files:**
- Create: `src/epl_prediction_optimizer/storage/migrations.py`
- Create: `src/epl_prediction_optimizer/storage/repositories.py`
- Create: `tests/integration/test_database_migrations.py`
- Create: `tests/integration/test_repositories.py`
- Modify: `src/epl_prediction_optimizer/storage/database.py`

- [ ] **Step 1: Write migration, repository, and concurrency tests against new and legacy schemas**

Create a legacy database using the exact current columns `season, contest_week, match_id, team, venue, notes, actual_points, created_at, updated_at`. Assert migration is idempotent, preserves rows, records schema version, and maps resolvable/unresolvable rows to `scored|locked|committed|legacy_unverified` without inventing kickoff times. Also write the initially failing repository contract tests, including two SQLite connections replacing with one expected pick version (exactly one success), atomic global-version increments, and publication-pointer reads that always return one complete version/path/hash tuple.

- [ ] **Step 2: Run and verify failure**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_database_migrations.py tests/integration/test_repositories.py -q`

Expected: migration failures because `schema_migrations` and lifecycle columns do not exist, plus repository import/contract failures for publication pointers and versioned replacement.

- [ ] **Step 3: Implement ordered transactional migrations**

Create `schema_migrations`, normalized `fixtures`, `contest_rounds`, `source_sync_runs`, `version_state`, `published_snapshots(kind, season, content_hash, artifact_path, published_at_utc)`, and a new `pick_ledger` table keyed by `(season, contest_round_id)`. Each ledger row has `pick_version INTEGER NOT NULL DEFAULT 1`; the separate season-global `pick_ledger_version` begins at 0 in `version_state`. Copy legacy rows in a transaction, retain the legacy table for rollback inspection, and make every migration safe to rerun.

- [ ] **Step 4: Implement repositories and atomic version increments**

Implement these exact repository operations: `FixtureRepository.publish_snapshot/list_season`, `RoundRepository.replace_derived/next_open/increment_eligibility`, `PickRepository.get/list_season/replace_with_version`, `SourceSyncRepository.start/finish/latest_by_source`, `PublicationRepository.resolve/publish_pointer`, and `VersionRepository.get_vector/increment`. `publish_pointer` updates the content hash, immutable path, published timestamp, and matching version inside the caller's transaction. Use typed parameters and domain return values rather than raw rows.

`replace_with_version` requires `expected_pick_version`: `NULL` only for creation and the current row's integer for replacement. Creation returns `pick_version=1`; replacement uses `UPDATE ... WHERE pick_version=?`, increments the row version, and increments the global `pick_ledger_version` once in the same `BEGIN IMMEDIATE` transaction. Zero updated rows produce `PICK_VERSION_CONFLICT`. Repository tests launch two connections with the same expected value and assert one success, one conflict, and one global increment.

- [ ] **Step 5: Preserve `Database` as a compatibility façade**

Delegate existing JSON/history/experiment methods without breaking secondary pages. New code must consume repositories rather than add more unrelated methods to the already-large façade.

- [ ] **Step 6: Run repository and legacy tests, lint, then commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_database_migrations.py tests/integration/test_repositories.py tests/test_challenge.py tests/test_results.py -q`

Expected: all pass and a second migration run reports no applied migrations.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/storage tests/integration/test_database_migrations.py tests/integration/test_repositories.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/storage tests/integration/test_database_migrations.py tests/integration/test_repositories.py
git commit -m "feat: migrate durable fixture and pick state"
```

### Task 4: Build the Atomic Live-Data Coordinator

**Files:**
- Create: `src/epl_prediction_optimizer/data/adapters.py`
- Create: `src/epl_prediction_optimizer/data/artifacts.py`
- Create: `src/epl_prediction_optimizer/data/coordinator.py`
- Create: `tests/integration/test_data_coordinator.py`
- Modify: `src/epl_prediction_optimizer/data/sources.py`
- Modify: `src/epl_prediction_optimizer/pipeline.py`
- Modify: `src/epl_prediction_optimizer/storage/migrations.py`
- Modify: `src/epl_prediction_optimizer/storage/repositories.py`
- Modify: `tests/integration/test_database_migrations.py`
- Modify: `tests/integration/test_repositories.py`

- [ ] **Step 1: Write fake-adapter coordinator tests**

Test successful normalization/publish, season mismatch, fixture-source failure retaining last-known-good, optional ClubElo/xG failure with warning, no overlapping refresh, source timestamps, correct promoted-club acceptance data, and live mode never receiving sample fixtures. Before implementation, add failure injection at each publication boundary—before file completion, after file completion/before DB commit, during DB commit, and after commit—and assert readers see an entire old or new snapshot with its matching version. Add frozen-clock tests at 23:59:59, 24:00:00, and 24:00:01 after the last successful ClubElo sync to prove it is fetched at most once per 24 hours.

- [ ] **Step 2: Verify failure**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_data_coordinator.py -q`

Expected: import failure for `DataCoordinator`.

- [ ] **Step 3: Extract adapter protocols and implementations**

```python
@dataclass(frozen=True)
class SyncResult:
    source: str
    season: str
    started_at: datetime
    completed_at: datetime
    record_count: int
    state: Literal["success", "warning", "error"]
    error: str | None
```

Wrap existing football-data.org, football-data.co.uk, ClubElo, and cached Understat functions. Normalize team names once at the adapter boundary.

- [ ] **Step 4: Implement recoverable snapshot publication**

Serialize the validated snapshot, hash its sorted normalized content, and write it once to an immutable content-addressed path such as `data/snapshots/fixtures/<sha256>.json`; `fsync` the file and directory. In one SQLite transaction, point `published_snapshots(kind, season)` at that immutable path/hash and increment `fixture_data_version`. Readers resolve only through that committed pointer. A crash before the transaction may leave an unreferenced file that later cleanup can remove, but cannot expose mismatched data/version; a transaction failure leaves the prior pointer/version intact. Compatibility CSV exports are regenerated after publication and are never runtime source-of-truth. Inject failures before file completion, after file completion/before DB commit, during DB commit, and after commit; assert readers see either the entire old or entire new snapshot and matching version.

- [ ] **Step 5: Implement coordinator policy and pipeline delegation**

Use a process lock plus stored in-progress token for idempotency. Fixture data is required; historical results are required only when training; ClubElo/xG may retain prior snapshots with warnings. Consult persisted last-success timestamps and skip ClubElo until 24 hours has elapsed. Record every attempted or policy-skipped source check with its reason.

- [ ] **Step 6: Run tests, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_data_coordinator.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py tests/test_pipeline.py -q`

Expected: all pass; the executable monkeypatch assertions prove offline mode performs no network calls.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/data src/epl_prediction_optimizer/pipeline.py src/epl_prediction_optimizer/storage/migrations.py src/epl_prediction_optimizer/storage/repositories.py tests/integration/test_data_coordinator.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/data src/epl_prediction_optimizer/pipeline.py src/epl_prediction_optimizer/storage/migrations.py src/epl_prediction_optimizer/storage/repositories.py tests/integration/test_data_coordinator.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py
git commit -m "feat: coordinate atomic current-season refresh"
```

### Task 5: Publish Versioned Predictions and Reconcile Eligibility

**Files:**
- Create: `src/epl_prediction_optimizer/prediction/__init__.py`
- Create: `src/epl_prediction_optimizer/prediction/service.py`
- Create: `src/epl_prediction_optimizer/services/__init__.py`
- Create: `src/epl_prediction_optimizer/services/reconciliation.py`
- Create: `src/epl_prediction_optimizer/services/runtime.py`
- Create: `tests/unit/test_versioning.py`
- Create: `tests/integration/test_reconciliation.py`
- Create: `tests/integration/test_runtime_coordinator.py`
- Modify: `src/epl_prediction_optimizer/pipeline.py`
- Modify: `src/epl_prediction_optimizer/app/main.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write all failing version, publication, reconciliation, and runtime tests**

Assert prediction version changes when ClubElo/current-stat/fixture snapshots change without retraining, does not change for byte-identical inputs, local and season eligibility versions increment at kickoff, lookahead dependencies expire at the earliest boundary, and restart/read-time reconciliation catches a crossed kickoff. Before implementation, add failure injection for immutable model and probability publication at every file/transaction boundary. In `test_runtime_coordinator.py`, add red tests for non-blocking stale startup, no overlapping work, the earlier of a 15-minute check/next kickoff, restart recovery, completed-match-triggered train/predict ordering, prior-model retention on train failure, and invocation of an injected scenario invalidator after successful probability publication.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_versioning.py tests/integration/test_reconciliation.py tests/integration/test_runtime_coordinator.py tests/test_pipeline.py tests/test_predictions.py -q`

Expected: import/contract failures for prediction publication, reconciliation, runtime coordination, and newly required pipeline orchestration.

- [ ] **Step 2: Implement deterministic snapshot identity and recoverable probability publication**

Write a trained model to immutable `data/snapshots/models/<sha256>.joblib`, verify it can be loaded, then transactionally publish its pointer and `model_version`; a failure leaves the prior model pointer active. Hash model artifact version, feature-column/config identity, fixture version, supplemental input versions, and current-stat cutoff. Write probabilities to an immutable content-addressed artifact, then atomically update the SQLite publication pointer and monotonic `prediction_version` exactly as in Task 4. The Step 1 boundary-failure tests prove readers always resolve a matching old or new artifact/version.

- [ ] **Step 3: Implement reconciliation as a pure transition plan plus transaction**

Implement pure `plan_reconciliation(fixtures, rounds, picks, now_utc) -> ReconciliationPlan` and transactional `reconcile_season(repositories, season, now_utc) -> VersionVector` functions. The former returns explicit fixture, round, pick, and version mutations; the latter applies that plan atomically.

Cover kickoff locking, missed/void rounds, same-weekend reschedule retention, cross-weekend invalidation, pre/post-lock cancellation, postponed picks, and destination-round candidate exclusion.

- [ ] **Step 4: Integrate deadline-aware runtime orchestration**

Create `RuntimeCoordinator` and attach it through FastAPI's lifespan. On startup, render from last-known-good immediately and enqueue a refresh only when stale. While active, schedule the next check for the earlier of 15 minutes or the next eligible kickoff; use one non-overlapping task guarded by the coordinator lock. After a refresh introduces newly finished matches beyond the training cutoff, run train, publish probabilities, and invoke an injected `ScenarioPublicationPort.invalidate_and_rebuild(season)` in order; retain the prior model/publication if any stage fails. Chunk 1 supplies a no-op implementation that records `scenarios_pending_rebuild=true` in version state without importing future optimizer modules. Task 7 replaces it with the real scenario publisher and clears the flag only after successful rebuild. The Step 1 fake scheduler/clock tests verify every lifecycle behavior.

- [ ] **Step 5: Run focused and cumulative Chunk 1 tests**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest -q`

Expected: the complete suite passes; no naive datetime warnings and no network access from offline tests.

- [ ] **Step 6: Lint and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/config src/epl_prediction_optimizer/domain src/epl_prediction_optimizer/data src/epl_prediction_optimizer/prediction src/epl_prediction_optimizer/services src/epl_prediction_optimizer/storage src/epl_prediction_optimizer/pipeline.py src/epl_prediction_optimizer/app/main.py tests/unit tests/integration tests/test_app.py tests/test_pipeline.py tests/test_predictions.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/prediction src/epl_prediction_optimizer/services src/epl_prediction_optimizer/pipeline.py src/epl_prediction_optimizer/app/main.py tests/unit/test_versioning.py tests/integration/test_reconciliation.py tests/integration/test_runtime_coordinator.py tests/test_pipeline.py
git commit -m "feat: version predictions and reconcile eligibility"
```

## Chunk 2: Committed Optimization and Validation

### Task 6: Extend the Solver with Locked Picks and Canonical Tie-Breaking

**Files:**
- Create: `scripts/build_validation_baseline.py`
- Create: `validation/baselines/manifest.json`
- Create: `validation/baselines/2223_probabilities.csv`
- Create: `validation/baselines/2324_probabilities.csv`
- Create: `validation/baselines/2425_probabilities.csv`
- Create: `validation/baselines/2223_metrics.json`
- Create: `validation/baselines/2324_metrics.json`
- Create: `validation/baselines/2425_metrics.json`
- Modify: `src/epl_prediction_optimizer/optimizer/solver.py`
- Create: `tests/unit/test_validation_baseline_builder.py`
- Create: `tests/unit/test_locked_optimizer.py`
- Modify: `tests/test_optimizer.py`

- [ ] **Step 1: Write and run failing baseline-builder tests**

Test a tiny set with Saturday, Sunday, and midweek rows; duplicate match IDs; a missing result; shuffled input; and a tampered output hash. Assert Saturday/Sunday map to the same Saturday-date `contest_round_id`, midweek has a null round, missing/duplicate outcomes fail closed, every probability triplet sums to one within `1e-6`, the manifest verifies every byte/row count, and two shuffled builds are byte-identical apart from an injected fixed `generated_at_utc`.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_validation_baseline_builder.py -q`

Expected: import failure because the baseline builder and schema validator do not exist.

- [ ] **Step 2: Freeze the pre-change replay baseline on the canonical horizon**

Before editing optimizer code, implement `build_validation_baseline.py` to join each ignored `data/exports/<season>_backtest_probabilities.csv` row to final `home_goals/away_goals` from `data/processed/historical_matches.csv`. Treat historical `date` as the provider's local fixture calendar date—never as a kickoff. Map local Saturday to itself, Sunday to the previous Saturday, and every other weekday to a null `contest_round_id`. For baseline optimization, filter eligible rows, sort distinct Saturday IDs, assign temporary 1-based ordinals, and pass those ordinals as `contest_week` to the untouched solver. This makes pre-change and candidate solvers use the identical active-round set. Record and later require equality of the exact active-round ID list/cardinality. If runtime inputs are absent, run the existing backtest pipeline with the untouched solver first.

```text
probability CSV: schema_version, season, fixture_local_date, contest_round_id nullable,
  source_contest_week nullable, match_id, home_team, away_team, p_home_win, p_draw, p_away_win,
  home_goals, away_goals
metrics JSON: schema_version, season, accuracy, multiclass_log_loss,
  expected_calibration_error, optimized_realized_points,
  optimized_expected_points, expected_vs_realized_mae, feasible, pick_count,
  active_contest_round_ids
manifest JSON: schema_version, generated_at_utc, generator_sha256,
  baseline_optimizer_source_sha256, model_source_identity{digest,files},
  feature_model_identity, files[{path, sha256, rows}]
```

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/build_validation_baseline.py --seasons 2223 2324 2425 --output validation/baselines`

Expected: three 380-row probability/outcome files, three complete metrics files, a manifest whose hashes verify, and no changes to runtime optimizer source.

- [ ] **Step 3: Re-run builder tests green**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_validation_baseline_builder.py -q`

Expected: all baseline join, schema, deterministic-output, and tamper tests pass.

- [ ] **Step 4: Write locked-plan and rule-feasibility tests**

Cover zero/one/many locked picks, unknown/ineligible lock rejection, exactly one choice per active round, team at least once/at most twice, home/away at most once, missed choices consuming no usage, and infeasible conflict diagnostics.

- [ ] **Step 5: Write an equal-objective determinism test**

Shuffle identical candidates ten times and assert the same plan is returned. The expected plan follows chronological lexicographic keys `(team.casefold(), venue, match_id)`.

- [ ] **Step 6: Verify focused failures**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_locked_optimizer.py tests/test_optimizer.py -q`

Expected: typed locked-pick/canonicalization failures plus the legacy DataFrame wrapper contract failure.

- [ ] **Step 7: Implement the `SeasonPlan` interface**

Add a frozen `SeasonPlan` with `picks: tuple[CandidatePick, ...]`, `remaining_ev`, `feasible`, and nullable `conflict_reason`, then implement `optimize(candidates, rules, locked_picks=()) -> SeasonPlan`.

Maximize EV first. Constrain the optimum within `1e-6`, then fix the first globally feasible sorted candidate for each chronological round. Never use epsilon-weighted or rank-sum objectives.

- [ ] **Step 8: Retain the old DataFrame wrapper and run tests**

`optimize_picks()` converts DataFrames into the typed solver and back so scripts remain functional.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_locked_optimizer.py tests/test_optimizer.py -q`

Expected: all pass.

- [ ] **Step 9: Lint and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check scripts/build_validation_baseline.py src/epl_prediction_optimizer/optimizer/solver.py tests/unit/test_validation_baseline_builder.py tests/unit/test_locked_optimizer.py tests/test_optimizer.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add scripts/build_validation_baseline.py validation/baselines src/epl_prediction_optimizer/optimizer/solver.py tests/unit/test_validation_baseline_builder.py tests/unit/test_locked_optimizer.py tests/test_optimizer.py
git commit -m "feat: optimize around committed canonical picks"
```

### Task 7: Build Exact Counterfactual Scenarios and Explanations

**Files:**
- Create: `src/epl_prediction_optimizer/optimizer/scenarios.py`
- Create: `src/epl_prediction_optimizer/optimizer/explanations.py`
- Create: `tests/unit/test_scenarios.py`
- Create: `tests/unit/test_scenario_explanations.py`
- Modify: `src/epl_prediction_optimizer/services/runtime.py`
- Modify: `src/epl_prediction_optimizer/app/main.py`
- Modify: `tests/test_app.py`

- [ ] **Step 1: Write scenario arithmetic and rank tests**

Use a small hand-solvable three-round fixture. Assert `3*p_win+p_draw`, remaining EV, projected season points, non-negative season cost, immediate rank, season-aware rank, independent tie groups, unique ordinals, null and non-null first displacement, complete displaced/replacement lists, infeasible candidates without numeric rank/cost, and every cumulative expected-points value in each horizon path. Assert `scenario_remaining_ev <= baseline_remaining_ev + 1e-6` before deriving/clamping season cost. Include one scored prior pick and one pending committed pick to prove scored points enter only realized/projected totals while the pending pick contributes current EV.

- [ ] **Step 2: Write decision/lookahead tests**

Assert decision mode applies only to the next open round. For a later round, fix every prior open round to the canonical baseline, return those assumptions, and prove that changing an earlier eligibility version changes the scenario token and `valid_until_utc`. Parameterize each token component—season, round ID, fixture data, prediction, model, rules, pick-ledger, local eligibility, season eligibility, and mode—and assert changing exactly one changes the hash. Assert `valid_until_utc` is the earliest dependency kickoff, cache reads fail strictly at/after it, publication/version changes invalidate cache, and candidate ordering is stable across shuffled input. Assert the exact season comparator `(scenario_remaining_ev desc, immediate_expected_points desc, team.casefold asc, match_id asc)` and immediate comparator `(immediate_expected_points desc, team.casefold asc, match_id asc)` with fixtures that distinguish every key.

- [ ] **Step 3: Verify failures**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_scenarios.py tests/unit/test_scenario_explanations.py -q`

Expected: import failures for scenario modules.

- [ ] **Step 4: Implement scenario generation and stable identities**

Implement `build_scenarios(target_round_id, round_topology, candidates, prior_picks, contest_rules, next_open_round_id, version_vector, now_utc) -> ScenarioSet`, a SHA-256 `scenario_version(season, round_id, versions, mode)`, and a stable `scenario_id(round_id, match_id, team, venue)`. The function derives `decision` only when target equals next-open; otherwise it fixes every earlier open topology node to the rules-aware canonical baseline and returns those assumptions. Serialize version inputs as sorted, delimiter-safe JSON before hashing. Store derived scenarios behind a bounded `(scenario_version, valid_until_utc)` cache and refuse reads when the version vector differs or server time reaches expiry.

Include every spec version component and cache only until `valid_until_utc`. Return full changed-pick lists and cumulative horizon series per feasible candidate.

- [ ] **Step 5: Implement human explanations from structured deltas**

Generate deterministic text such as `0.08 expected season points lower; preserves Manchester City for 2026-09-26`. Use `No downstream plan change` exactly for empty displacement. Do not use confidence, odds, betting, or negative-opportunity-cost language.

- [ ] **Step 6: Wire scenario publication, run tests, and commit**

Implement the real `ScenarioPublicationPort` adapter, inject it into `RuntimeCoordinator`, and clear `scenarios_pending_rebuild` only after all version-matched scenario sets are available. Add a runtime integration assertion that a completed-match refresh now rebuilds scenarios rather than only marking them pending.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_scenarios.py tests/unit/test_scenario_explanations.py tests/integration/test_runtime_coordinator.py tests/test_app.py -q`

Expected: scenario units, refresh-triggered scenario publication, and app lifespan wiring all pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/optimizer/scenarios.py src/epl_prediction_optimizer/optimizer/explanations.py src/epl_prediction_optimizer/services/runtime.py src/epl_prediction_optimizer/app/main.py tests/unit/test_scenarios.py tests/unit/test_scenario_explanations.py tests/integration/test_runtime_coordinator.py tests/test_app.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/optimizer/scenarios.py src/epl_prediction_optimizer/optimizer/explanations.py src/epl_prediction_optimizer/services/runtime.py src/epl_prediction_optimizer/app/main.py tests/unit/test_scenarios.py tests/unit/test_scenario_explanations.py tests/integration/test_runtime_coordinator.py tests/test_app.py
git commit -m "feat: calculate exact pick consequence scenarios"
```

### Task 8: Implement the Transactional Pick Lifecycle

**Files:**
- Create: `src/epl_prediction_optimizer/services/picks.py`
- Create: `tests/integration/test_pick_service.py`
- Modify: `src/epl_prediction_optimizer/services/reconciliation.py`

- [ ] **Step 1: Write lifecycle transition tests**

Cover every allowed transition in spec Section 4.6, replacement only when both old and new kickoffs are future, locked overwrite rejection, same-window reschedule retention, cross-window invalidation, pre/post-lock cancellation, postponed/replayed identity, scoring, missed/void behavior, and news note storage without probability mutation.

- [ ] **Step 2: Write simultaneous version-conflict and precedence tests**

Issue two writes with the same `expected_pick_version`; assert exactly one succeeds. Assert precedence: pick conflict, reconciliation/freshness, existing lock, candidate availability, scenario expiry, noncommittable round, scenario identity, then feasibility.

- [ ] **Step 3: Verify failures**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_pick_service.py -q`

Expected: import failure for `PickService`.

- [ ] **Step 4: Implement one transactional use case**

Implement `PickService.commit(command: CommitPick, now_utc: datetime) -> CommitResult` inside `unit_of_work.begin_immediate()`. Code the eight validation stages as separately named private methods invoked in the specification's required order, then perform one ledger write with the row/global version updates defined below.

Return typed domain errors with codes and diagnostic version vectors. A successful creation sets row `pick_version=1`; a successful replacement increments that row version; either write increments season-global `pick_ledger_version` exactly once in the same transaction. Rebuild/invalidate scenarios only after both version mutations commit.

- [ ] **Step 5: Run tests, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_pick_service.py tests/integration/test_reconciliation.py -q`

Expected: all pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/services/picks.py src/epl_prediction_optimizer/services/reconciliation.py tests/integration/test_pick_service.py tests/integration/test_reconciliation.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/services/picks.py src/epl_prediction_optimizer/services/reconciliation.py tests/integration/test_pick_service.py
git commit -m "feat: enforce kickoff-safe pick lifecycle"
```

### Task 9: Add Optimizer Replay and Model-Change Gates

**Files:**
- Create: `scripts/validate_release.py`
- Create: `tests/integration/test_release_validation.py`
- Create: `src/epl_prediction_optimizer/storage/experiments.py`
- Create: `validation/waivers/.gitkeep`
- Create: `validation/reports/2627_optimizer_validation.json`
- Modify: `src/epl_prediction_optimizer/ml/analysis.py`
- Modify: `src/epl_prediction_optimizer/storage/migrations.py`
- Modify: `tests/integration/test_database_migrations.py`
- Modify: `tests/integration/test_repositories.py`
- Modify: `README.md`

- [ ] **Step 1: Write report-schema, baseline-identity, and threshold tests**

Require candidate optimizer/model source hashes, feature list, parameters, training cutoff, seasons, accuracy, multiclass log loss, calibration diagnostics, realized optimized points, expected-versus-realized points, feasibility count, and baseline deltas. `validation/baselines/manifest.json` stores the SHA-256 and schema version of each tracked probability/outcome and metric input plus the model/feature identity used to create it; `baseline_identity` is the hash of that canonical manifest and those verified contents, never a moving Git commit. Missing fields, a manifest hash mismatch, or fewer than three completed seasons fail validation.

Define `candidate_source_identity` as SHA-256 over canonical sorted JSON mapping these relative paths to their content hashes: `pipeline.py`, `optimizer/candidates.py`, `optimizer/solver.py`, `optimizer/scenarios.py`, `optimizer/explanations.py`, `domain/models.py`, `domain/rounds.py`, `services/reconciliation.py`, `ml/features.py`, `ml/model.py`, `prediction/service.py`, `scripts/validate_release.py`, `pyproject.toml`, and `uv.lock`. Resolve and record the full repository-relative paths in the report. The report stores this combined digest and the individual map; waiver filenames use this exact combined digest.

The baseline builder also computes `model_source_identity` as SHA-256 over canonical sorted JSON for exactly `pipeline.py`, `ml/features.py`, `ml/model.py`, `prediction/service.py`, `pyproject.toml`, and `uv.lock`. Release-validation tests assert `--mode auto` selects `optimizer-only` when this digest matches and `model-change` when any one mapped file differs, while reporting both the stored/current path maps and digests.

Default optimizer-only pass criteria are identical active contest-round IDs/cardinality, zero infeasible seasons, and no decline in aggregate realized points versus the identified canonical-horizon baseline. Default model-change criteria are: accuracy decline no greater than 0.01 absolute, multiclass log-loss increase no greater than 0.01, expected calibration error increase no greater than 0.01, aggregate realized optimizer points no lower than baseline, and expected-versus-realized pick-point MAE increase no greater than 0.05. Any breached threshold returns nonzero; an optional matching file under `validation/waivers/<candidate-source-hash>.json` must contain metric, measured delta, rationale, reviewer, and timestamp, and is itself surfaced in the report. There is no CLI skip flag.

Define top-label multiclass ECE with ten fixed equal-width confidence bins `[0.0,0.1), …, [0.9,1.0]`: for every match take the maximum predicted class probability and whether that predicted class was correct; sum `bin_count / total_matches * abs(mean_confidence - bin_accuracy)`, treating empty bins as zero contribution. Define pick-point MAE per season as the arithmetic mean of `abs((3*p_win + p_draw) - actual_points)` over that solver's selected picks; aggregate across seasons by weighting every pick equally, not by averaging season means. Store per-season and aggregate values.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_release_validation.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py -q`

Expected: import/schema failures for the validator, focused experiment repository, baseline identity, and deterministic thresholds.

- [ ] **Step 2: Implement a reproducible release validator**

Add `--mode auto|optimizer-only|model-change`. `auto` compares the current canonical model-affecting source map (`pipeline.py`, `ml/features.py`, `ml/model.py`, `prediction/service.py`, `pyproject.toml`, `uv.lock`) with the baseline manifest's stored `model_source_identity`; any difference selects `model-change`, otherwise it selects `optimizer-only`. Optimizer-only replays checked-in probabilities across at least `2223`, `2324`, and `2425`; model-change reruns training/backtests and compares all probability-quality metrics. Write JSON atomically. Add forward migration fields for baseline identity, code/model version, training cutoff, seasons JSON, calibration and expected-vs-realized metrics; persist through a focused `ExperimentRepository`, not the compatibility `Database` façade.

- [ ] **Step 3: Run deterministic sample validation**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/test_predictions.py tests/integration/test_release_validation.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py -q`

Expected: all pass.

- [ ] **Step 4: Run the required optimizer-only multi-season replay**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/validate_release.py --mode optimizer-only --baseline-dir validation/baselines --seasons 2223 2324 2425 --output validation/reports/2627_optimizer_validation.json`

Expected: `PASS`, zero infeasible historical seasons, and a complete baseline-delta report. If feature/model code changed unexpectedly, stop and use `--mode model-change`; do not waive the gate.

- [ ] **Step 5: Run the cumulative Chunk 2 gate, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest -q`

Expected: the complete suite passes.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/optimizer src/epl_prediction_optimizer/services src/epl_prediction_optimizer/ml/analysis.py src/epl_prediction_optimizer/storage scripts/validate_release.py tests/unit tests/integration tests/test_optimizer.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add scripts/validate_release.py tests/integration/test_release_validation.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py src/epl_prediction_optimizer/ml/analysis.py src/epl_prediction_optimizer/storage/experiments.py src/epl_prediction_optimizer/storage/migrations.py README.md validation
git commit -m "test: gate optimizer changes with season replays"
```

## Chunk 3: Decision APIs

### Task 10: Define API Schemas, Errors, and Dependency Composition

**Files:**
- Create: `src/epl_prediction_optimizer/app/schemas.py`
- Create: `src/epl_prediction_optimizer/app/dependencies.py`
- Create: `tests/unit/test_api_schemas.py`
- Modify: `src/epl_prediction_optimizer/app/main.py`

- [ ] **Step 1: Write schema tests for aware timestamps and error envelopes**

Assert naive datetimes fail, enum/code values serialize exactly, null tie/displacement fields survive, and `SCENARIO_EXPIRED`/`PICK_VERSION_CONFLICT` include latest token, refresh URL, and candidate identity.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_api_schemas.py tests/test_app.py -q`

Expected: schema import failures and app-composition failures before the new contracts exist.

- [ ] **Step 2: Implement Pydantic request/response models**

Create strict, aware-datetime Pydantic models with these required fields:

```text
VersionResponse: fixture_data_version, prediction_version, model_version,
  rules_version, pick_ledger_version, eligibility_version, season_eligibility_version
SourceFreshnessResponse: source, state, started_at_utc, completed_at_utc,
  age_seconds, record_count, error, retry_eligible_at_utc
ReadinessResponse: season{code,label,start_year}, state, next_open_round_id,
  sources[], versions, model{state,training_cutoff_utc}, scenario_state,
  commit_eligible, commit_disabled_code, commit_disabled_reason, checked_at_utc,
  setup_steps[{id,label,state,detail}], primary_action{label,method,url}|null
FixtureResponse: match_id, league_matchweek|null, kickoff_utc, home_team,
  away_team, status, home_goals|null, away_goals|null, eligible
PlanPickResponse: contest_round_id, match_id, team, opponent, venue,
  kickoff_utc, expected_points, state
HorizonPointResponse: contest_round_id, cumulative_expected_points, event_labels[]
MoveTreeNodeResponse: contest_round_id, contest_round_number, lifecycle,
  baseline_pick|null, scenario_pick|null, changed, constraint_labels[]
ScenarioSummaryResponse: every ScenarioSummary field in spec Section 4.4 plus
  explanation and horizon_points[HorizonPointResponse]; immediate_rank,
  scenario_remaining_ev, projected_season_points, season_cost, and
  season_aware_rank are null exactly when feasible=false
RoundDecisionResponse: season, contest_round{id,number,state}, fixtures[],
  league_matchweeks[], committed_pick, scenarios[], highlighted_scenario_ids,
  move_tree[], mode, assumed_prior_picks[], versions, scenario_version,
  committable, as_of_utc, valid_until_utc; scenario_version is null only for
  void/missed rounds with no scenarios
ScenarioDetailResponse: scenario summary plus complete_plan[PlanPickResponse],
  displaced[{round_id,prior_pick,replacement_pick}], replacements[PlanPickResponse]
CommitPickRequest: scenario_id, scenario_version, expected_pick_version,
  match_id, team, venue, notes, news_risk
PickRecordResponse: season, contest_round_id, league_matchweek|null, match_id,
  team, venue, kickoff_at, committed_at, updated_at, locked_at|null, state,
  notes, news_risk, actual_points|null, pick_version
PickResponse: pick[PickRecordResponse], versions, recalculated_round, readiness
RefreshProgressResponse: source, state(queued|running|success|warning|error),
  started_at_utc|null, completed_at_utc|null, record_count|null, error|null
RefreshResponse: state(accepted|in_progress|complete|error), refresh_id,
  started_at_utc, completed_at_utc|null, sources[RefreshProgressResponse], poll_url
ConflictResponse: code, message, latest_scenario_version, refresh_url,
  candidate_identity, versions
```

Nullability is strict: `SourceFreshnessResponse.started_at_utc/completed_at_utc/age_seconds/record_count` are null only when that source has never completed or never started as applicable; `error` is non-null only for warning/error and `retry_eligible_at_utc` only when throttled/backing off. `next_open_round_id` is null only for first-run/no schedule or a completed season. `model.training_cutoff_utc` is null only when no model exists. `commit_disabled_code` and `commit_disabled_reason` are both null exactly when `commit_eligible=true` and both required when false. Empty setup is `[]`; `primary_action` is non-null only when a user action can advance readiness.

- [ ] **Step 3: Add an injected service container and clock**

`AppServices` requires `season_context`, `clock`, `unit_of_work`, fixture/round/pick/source/version/publication repositories, `data_coordinator`, `prediction_service`, `reconciliation_service`, `scenario_service`, `pick_service`, `readiness_service`, and `runtime_coordinator`. `create_app()` accepts an optional complete `AppServices`. The compatibility `database/workdir` path calls one `build_default_services(database, workdir, mode)` factory that constructs every required port; it never creates route-local services. Tests inject fake services and a frozen clock.

- [ ] **Step 4: Run tests, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_api_schemas.py tests/test_app.py -q`

Expected: all pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/app/schemas.py src/epl_prediction_optimizer/app/dependencies.py src/epl_prediction_optimizer/app/main.py tests/unit/test_api_schemas.py tests/test_app.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/app/schemas.py src/epl_prediction_optimizer/app/dependencies.py src/epl_prediction_optimizer/app/main.py tests/unit/test_api_schemas.py
git commit -m "feat: define typed decision API contracts"
```

### Task 11: Expose Readiness and Idempotent Refresh

**Files:**
- Create: `src/epl_prediction_optimizer/services/readiness.py`
- Create: `src/epl_prediction_optimizer/app/routes/__init__.py`
- Create: `src/epl_prediction_optimizer/app/routes/operations.py`
- Create: `tests/integration/test_readiness_api.py`
- Modify: `src/epl_prediction_optimizer/app/main.py`
- Modify: `src/epl_prediction_optimizer/data/coordinator.py`
- Modify: `src/epl_prediction_optimizer/storage/migrations.py`
- Modify: `src/epl_prediction_optimizer/storage/repositories.py`
- Modify: `tests/integration/test_data_coordinator.py`
- Modify: `tests/integration/test_database_migrations.py`
- Modify: `tests/integration/test_repositories.py`

- [ ] **Step 1: Write readiness-state tests**

Cover ready, updating with readable cache, stale outside/inside 24-hour matchday window, required fixture error, optional-source warning, no data/model checklist, exact source ages, next open round, model cutoff/version, prediction/version vector, and commit eligibility. Start from a stored pre-kickoff version, advance the fake clock beyond kickoff without running the background scheduler, and assert the request lazily reconciles before calculating the response.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_readiness_api.py tests/integration/test_data_coordinator.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py -q`

Expected: route/import failures, the sleep/restart eligibility assertion fails, and active-refresh repository/coordinator contracts are absent before implementation.

- [ ] **Step 2: Implement freshness policy**

Call `reconcile_season()` first on every readiness request, then use its returned version vector and round states. Apply 15-minute normal fixture checks, 30-minute matchday freshness, 24-hour ClubElo refresh, completed-match-vs-training-cutoff staleness, and explicit no-sample live state. Return the earliest reason commit is disabled.

- [ ] **Step 3: Implement routes**

Add `GET /api/readiness?season=2627`, `POST /api/refresh/current`, and `GET /api/refresh/{refresh_id}`. Extend `source_sync_runs` with `refresh_id`, active-run state, `heartbeat_at_utc`, and `lease_expires_at_utc`; cover this forward migration from the existing schema. Implement `RefreshRunRepository.begin_or_get_active/update_source/heartbeat/complete/get_and_expire_stale`. `DataCoordinator.start_or_get_active()` calls that repository under `BEGIN IMMEDIATE`, so simultaneous callers receive the same unexpired active record and only the creator schedules work. Workers heartbeat after every source transition and at least every minute during long work; the lease is five minutes. On startup or a new POST, an expired active lease is atomically closed as `error` with `Interrupted before completion`, then a new refresh ID may be claimed. Frozen-clock tests cover crash, restart before expiry, and restart after expiry. A new refresh returns HTTP 202 with `state="accepted"`, a stable UUID `refresh_id`, start timestamp, one queued progress record per configured source, and `poll_url="/api/refresh/<refresh_id>"`. A duplicate while active returns HTTP 202 with `state="in_progress"`, the same ID/start time, and current per-source state/timestamps/count/errors; it starts no work. The GET status endpoint calls `get_and_expire_stale()` so polling after lease expiry atomically closes and returns the run as `error` rather than remaining `in_progress`; a frozen-clock test advances an active run past five minutes and asserts that response. GET otherwise returns HTTP 200 with `in_progress|complete|error`; unknown IDs return HTTP 404 `REFRESH_NOT_FOUND`. Invalid season input returns HTTP 422; completion transactionally closes the active record so the next request receives a new ID.

- [ ] **Step 4: Test, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_readiness_api.py tests/integration/test_data_coordinator.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py -q`

Expected: all pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/services/readiness.py src/epl_prediction_optimizer/app/routes src/epl_prediction_optimizer/app/main.py src/epl_prediction_optimizer/data/coordinator.py src/epl_prediction_optimizer/storage/migrations.py src/epl_prediction_optimizer/storage/repositories.py tests/integration/test_readiness_api.py tests/integration/test_data_coordinator.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/services/readiness.py src/epl_prediction_optimizer/app/routes src/epl_prediction_optimizer/app/main.py src/epl_prediction_optimizer/data/coordinator.py src/epl_prediction_optimizer/storage/migrations.py src/epl_prediction_optimizer/storage/repositories.py tests/integration/test_readiness_api.py tests/integration/test_data_coordinator.py tests/integration/test_database_migrations.py tests/integration/test_repositories.py
git commit -m "feat: expose auditable season readiness"
```

### Task 12: Expose Round and Scenario Read APIs

**Files:**
- Create: `src/epl_prediction_optimizer/app/routes/decision.py`
- Create: `tests/integration/test_decision_api.py`
- Modify: `src/epl_prediction_optimizer/app/main.py`

- [ ] **Step 1: Write decision and lookahead response tests**

Assert fixtures, league metadata, lifecycle, committed state, rankings, highlighted IDs, horizon series, move tree, mode/assumptions, all versions, `as_of_utc`, and dependency-aware `valid_until_utc`. Exact behavior: unknown round returns HTTP 404 `ROUND_NOT_FOUND`; void/missed rounds return HTTP 200 with lifecycle metadata, `committable=false`, empty scenarios, and null scenario version; a round read lazily reconciles and rebuilds an expired cache before returning 200. Detail reads require query parameter `scenario_version`: unknown/wrong-round IDs return 404 `SCENARIO_NOT_FOUND`, while a known scenario ID paired with a noncurrent token returns 409 `SCENARIO_EXPIRED` and the current version vector.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_decision_api.py -q`

Expected: route failures for round/detail reads and every specified error payload.

- [ ] **Step 2: Implement round endpoint**

Add `GET /api/seasons/{season}/rounds/{contest_round_id}`. Lazy-reconcile before reading, derive scenarios through the scenario service, and never expose a lookahead response as committable.

- [ ] **Step 3: Implement detail endpoint**

Add `GET /api/seasons/{season}/rounds/{contest_round_id}/scenarios/{scenario_id}?scenario_version=<token>` returning the complete plan/deltas. Require the scenario to belong to the round and the supplied token to equal the current version; missing query input returns HTTP 422.

- [ ] **Step 4: Test, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_decision_api.py -q`

Expected: all pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/app/routes/decision.py src/epl_prediction_optimizer/app/main.py tests/integration/test_decision_api.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/app/routes/decision.py src/epl_prediction_optimizer/app/main.py tests/integration/test_decision_api.py
git commit -m "feat: serve round consequence scenarios"
```

### Task 13: Expose Commit/Change with Structured Conflict Recovery

**Files:**
- Modify: `src/epl_prediction_optimizer/app/routes/decision.py`
- Create: `tests/integration/test_pick_api.py`
- Modify: `src/epl_prediction_optimizer/app/main.py`
- Modify: `src/epl_prediction_optimizer/app/templates/challenge.html`
- Modify: `tests/test_challenge.py`

- [ ] **Step 1: Write endpoint tests for every structured failure**

Cover 200 creation/change, 409 scenario expiry/pick conflict/lock, candidate unavailable, noncommittable lookahead/non-next round, identity mismatch, constraint conflict, stale matchday data, and a kickoff race using a clock advanced between read and PUT. Every error assertion checks the complete diagnostic `VersionResponse`, latest scenario token, refresh URL, and candidate identity when applicable.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_pick_api.py tests/test_challenge.py -q`

Expected: missing PUT route/error contracts plus failing legacy mutation-path assertions.

- [ ] **Step 2: Implement `PUT /api/seasons/{season}/rounds/{contest_round_id}/pick`**

Map `CommitPickRequest` to `PickService`. Return the recalculated plan/readiness on success and `ConflictResponse` with every fixture/prediction/model/rules/ledger/local/season eligibility component version on error. Never automatically retry a stale commit.

- [ ] **Step 3: Retire duplicate manual-pick mutation paths**

Make the legacy challenge page read-only: remove both manual-pick POST routes and all mutation forms/buttons from `challenge.html`, retain historical rows/results/notes, and link its `Choose this week` action to the decision desk. Update `tests/test_challenge.py` to assert GET history remains and both legacy POST paths return 404/405. The versioned PUT endpoint is the only authoritative write path.

- [ ] **Step 4: Run the cumulative Chunk 3 gate, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest -q`

Expected: the complete suite passes.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/app src/epl_prediction_optimizer/services tests/unit/test_api_schemas.py tests/integration/test_readiness_api.py tests/integration/test_decision_api.py tests/integration/test_pick_api.py tests/test_app.py tests/test_challenge.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/app/routes/decision.py src/epl_prediction_optimizer/app/main.py src/epl_prediction_optimizer/app/templates/challenge.html tests/integration/test_pick_api.py tests/test_challenge.py
git commit -m "feat: commit versioned picks through one API"
```

## Chunk 4: Balanced Horizon Decision Desk

### Task 14: Build the Server-Rendered Decision Shell and Design Tokens

**Files:**
- Create: `src/epl_prediction_optimizer/app/presenters.py`
- Create: `src/epl_prediction_optimizer/app/static/decision.css`
- Create: `src/epl_prediction_optimizer/app/static/analysis-paper-texture.webp`
- Create: `src/epl_prediction_optimizer/app/static/graphite-texture.webp`
- Create: `src/epl_prediction_optimizer/app/templates/components/readiness.html`
- Create: `src/epl_prediction_optimizer/app/templates/components/round_nav.html`
- Create: `src/epl_prediction_optimizer/app/templates/components/candidate_table.html`
- Create: `src/epl_prediction_optimizer/app/templates/components/scenario_horizon.html`
- Create: `src/epl_prediction_optimizer/app/templates/components/move_tree.html`
- Modify: `src/epl_prediction_optimizer/app/templates/dashboard.html`
- Modify: `src/epl_prediction_optimizer/app/main.py`
- Modify: `src/epl_prediction_optimizer/app/routes/decision.py`
- Create: `tests/integration/test_dashboard_rendering.py`

- [ ] **Step 1: Write server-rendered state tests**

Assert useful HTML without JavaScript for ready/updating/stale/error/first-run, decision/lookahead/committed/locked/missed/void, source timestamps, candidate metrics, chart text summaries, navigation links, and a disabled commit reason. Assert one text/SVG path summary per feasible scenario, best-now and best-season labels, one dual label when both winners match, displacement/constraint marker text, and absence of `ensemble`, `confidence path`, percentile, betting, stake, and odds language.

Add no-JavaScript form tests for `POST /seasons/{season}/rounds/{contest_round_id}/pick`: success delegates to the same `PickService` then redirects 303 to the round; stale and locked submissions render the current dashboard with HTTP 409 and an inline error while retaining the submitted candidate/notes.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_dashboard_rendering.py -q`

Expected: template/component imports, forecast semantics, texture checks, and POST fallback behavior fail before implementation.

- [ ] **Step 2: Implement presenter-only display logic**

Convert domain responses into copy, formatted local kickoffs, badges, comparison rows, plain-language effects, and chart/move-tree data. Templates must not calculate ranks, freshness, or eligibility.

Before the first UI edit in this implementation session, run `node /Users/francosolari/.agents/skills/impeccable/scripts/context.mjs --target src/epl_prediction_optimizer/app/templates/dashboard.html` exactly once, read the returned directives plus `/Users/francosolari/.agents/skills/impeccable/reference/craft-floor.md`, and retain the approved surface brief/comp as the visual contract. Do not rerun context during later polish.

- [ ] **Step 3: Implement the approved page structure**

Use the Preparation Room + Balanced Horizon composition: narrow context rail, ruled candidate comparison, scenario horizon, and full-width move tree. Keep model/backtest/data links under an `Analysis` secondary disclosure.

Own the full component grammar in tokens and rendering assertions: primary UI uses `"Arial Narrow", "Aptos Narrow", "Inter Tight", system-ui, sans-serif`; numeric cells use `ui-monospace, "SFMono-Regular", Consolas, monospace` with `font-variant-numeric: tabular-nums`; only the primary decision heading uses `Georgia, "Times New Roman", serif`. Sparse analyst marginalia may use `"Bradley Hand", "Segoe Print", cursive`, is marked nonessential, and repeats no unique state. Define only square, 2px, 4px, and 6px radii; use 1px hairline region rules; define one shallow `--shadow-selected` elevation used only on the selected scenario. Rendering tests assert font-role classes, tabular figures, nonessential marginalia marking, token maximum radius, hairline rule, and absence of any second shadow token.

Implement `POST /seasons/{season}/rounds/{contest_round_id}/pick` as the progressive-enhancement form target. Parse the same fields as `CommitPickRequest`, call the same `PickService`, redirect only on success, and on any typed conflict re-present the refreshed round with the submitted candidate/notes and matching HTTP status. The form includes all scenario/version identity fields as hidden inputs; no separate mutation logic is allowed.

- [ ] **Step 4: Implement theme tokens before component colors**

Define semantic custom properties for warm-paper and graphite themes, cobalt/amber/muted-red accents, borders, focus, chart grid, and elevation. Add `color-scheme`, system defaults, and no sportsbook/currency/odds semantics.

Use `@imagegen` to create two seamless, near-monochrome 512px WebP materials: subtle warm analysis-paper fiber and low-glare graphite drafting grain, with no text, logos, objects, or high-contrast marks. Optimize each below 80 KB, apply them at low opacity behind solid semantic background colors, and disable texture under forced-colors. Rendering tests assert both assets exist, stay below the limit, and are referenced by the correct theme selector. This asset generation is a texture step only; it does not alter the approved composition.

- [ ] **Step 5: Run rendering tests, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/integration/test_dashboard_rendering.py tests/test_app.py -q`

Expected: all pass with core forms/links present without scripts.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/app/presenters.py src/epl_prediction_optimizer/app/routes/decision.py src/epl_prediction_optimizer/app/main.py tests/integration/test_dashboard_rendering.py tests/test_app.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/app/presenters.py src/epl_prediction_optimizer/app/routes/decision.py src/epl_prediction_optimizer/app/static/decision.css src/epl_prediction_optimizer/app/static/analysis-paper-texture.webp src/epl_prediction_optimizer/app/static/graphite-texture.webp src/epl_prediction_optimizer/app/templates src/epl_prediction_optimizer/app/main.py tests/integration/test_dashboard_rendering.py
git commit -m "feat: render Balanced Horizon decision desk"
```

### Task 15: Add Candidate Preview, Scenario Horizon, and Safe Commit Interaction

**Files:**
- Create: `src/epl_prediction_optimizer/app/static/decision.js`
- Create: `tests/browser/test_decision_interactions.py`
- Modify: `src/epl_prediction_optimizer/app/templates/dashboard.html`
- Modify: `src/epl_prediction_optimizer/app/templates/components/candidate_table.html`
- Modify: `src/epl_prediction_optimizer/app/templates/components/scenario_horizon.html`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add browser-test tooling**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv add --dev pytest-playwright && UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run playwright install chromium`

Expected: dependencies lock successfully and Chromium is available. If installation is blocked, request network approval; do not replace browser verification with unit-only claims.

- [ ] **Step 2: Write failing interaction tests**

Selecting a row must update the consequence sentence, selected/highlighted path, move-tree changes, and inline commit summary without committing. Assert the horizon renders exactly one path per feasible candidate; best-now/best-season/dual-winner labels and displacement/constraint markers match the API; prohibited confidence/ensemble terminology is absent. A successful commit refreshes the plan. A 409 refetches, preserves selection only if eligible, explains the change, and never automatically retries. Rapid double-click/Enter sends one PUT while pending. Network/5xx failure announces the error, retains candidate/notes, clears pending state, and re-enables exactly one actionable commit control.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/browser/test_decision_interactions.py -q`

Expected: preview linkage, exact path semantics, pending guard, and error-recovery tests fail before JavaScript implementation.

- [ ] **Step 3: Implement small state and rendering functions**

```javascript
const state = { round: null, selectedScenarioId: null, pendingCommit: false };
function selectScenario(id) { /* render all linked views from one state */ }
function renderHorizon(series, selectedId) { /* accessible SVG paths + summaries */ }
async function commitSelected() { /* one PUT; explicit conflict recovery */ }
```

Use event delegation, `AbortController` for superseded reads, an in-flight commit promise/disabled control as the duplicate guard, `try/catch/finally` to restore action state, and an `aria-live="polite"` region for preview/commit feedback. Do not put decision math in JavaScript. On 409, refetch once but never resubmit; on network/5xx, retain local form state and offer explicit retry.

- [ ] **Step 4: Test, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/browser/test_decision_interactions.py -q`

Expected: all pass in Chromium.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check tests/browser/test_decision_interactions.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add pyproject.toml uv.lock src/epl_prediction_optimizer/app/static/decision.js src/epl_prediction_optimizer/app/templates tests/browser/test_decision_interactions.py
git commit -m "feat: preview and commit connected pick scenarios"
```

### Task 16: Add Complete Round Browsing and Season Move Tree

**Files:**
- Modify: `src/epl_prediction_optimizer/app/static/decision.js`
- Modify: `src/epl_prediction_optimizer/app/templates/components/round_nav.html`
- Modify: `src/epl_prediction_optimizer/app/templates/components/move_tree.html`
- Create: `tests/browser/test_round_navigation.py`

- [ ] **Step 1: Write navigation tests**

Cover previous/next, direct picker, move-tree click, left/right shortcuts outside form fields, preserved per-round browser-session selection, no accidental out-of-order commit, lifecycle labels, and complete-schedule midweek visibility.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/browser/test_round_navigation.py -q`

Expected: navigation, session restoration, keyboard, and lookahead restrictions fail before implementation.

- [ ] **Step 2: Implement URL-addressable navigation**

Use history state and round URLs so refresh/back/forward remain correct. Store only preview scenario IDs per round in `sessionStorage`; never store committed truth there. Add optional pointer swipe while keeping visible buttons.

- [ ] **Step 3: Render conditional lookahead assumptions**

Show `Look ahead · assumes current plan`, list assumed prior picks, and omit/disable commit controls for every future round.

- [ ] **Step 4: Test, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/browser/test_round_navigation.py -q`

Expected: all pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check tests/browser/test_round_navigation.py`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/app/static/decision.js src/epl_prediction_optimizer/app/templates/components tests/browser/test_round_navigation.py
git commit -m "feat: browse future rounds and plan movement"
```

### Task 17: Complete Theme, Responsive, Accessibility, and Reduced-Motion Behavior

**Files:**
- Modify: `src/epl_prediction_optimizer/app/static/decision.css`
- Modify: `src/epl_prediction_optimizer/app/static/decision.js`
- Modify: `src/epl_prediction_optimizer/app/templates/dashboard.html`
- Create: `tests/browser/test_dashboard_accessibility.py`
- Create: `tests/browser/test_dashboard_responsive.py`

- [ ] **Step 1: Write theme and preference tests**

Cover system/light/dark, persisted local preference, correct `color-scheme`, readable readiness/error states, and equivalent information hierarchy in both themes.

- [ ] **Step 2: Write keyboard/accessibility tests**

Verify focus order/visibility, labelled controls and SVG, text chart summaries, state not conveyed by color alone, normal text contrast at least 4.5:1, large text and UI/non-text boundaries at least 3:1, every interactive touch target at least 44×44 CSS pixels on mobile, and exact reduced-motion behavior. Under emulated `prefers-reduced-motion: reduce`, CSS sets animation/transition durations to zero, JavaScript never morphs paths or uses smooth scrolling, and swipe navigation changes rounds without animation; selecting a candidate still updates the summary, horizon, and move tree synchronously in the same interaction. Browser assertions inspect computed durations/scroll behavior and the immediate linked-view state.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/browser/test_dashboard_accessibility.py tests/browser/test_dashboard_responsive.py -q`

Expected: theme persistence, contrast, target-size, responsive ordering, and reduced-motion tests fail before implementation.

- [ ] **Step 3: Implement responsive compositions**

Desktop retains the split; tablet stacks horizon under comparison and puts the move tree in a labelled keyboard-focusable `overflow-x:auto` region with visible scroll affordance; mobile orders comparison, selected summary, scrollable horizon, and equally navigable move tree. Do not hide immediate EV, season cost, kickoff, or lifecycle state.

- [ ] **Step 4: Run browser tests at all target sizes/themes**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/browser/test_dashboard_accessibility.py tests/browser/test_dashboard_responsive.py -q`

Expected: all pass at desktop `1440x1000`, tablet `900x1100`, and mobile `390x844`.

- [ ] **Step 5: Run the cumulative Chunk 4 gate, lint, and commit**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest -q`

Expected: the complete suite passes in both unit/integration and browser groups.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/app tests/integration/test_dashboard_rendering.py tests/browser`

Expected: exit 0 with `All checks passed!`.

```bash
git add src/epl_prediction_optimizer/app/static src/epl_prediction_optimizer/app/templates/dashboard.html tests/browser
git commit -m "feat: finish accessible responsive themes"
```

## Chunk 5: Verification and Season Release

### Task 18: Run the Complete Automated Quality Gate

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/unit/test_fail_on_skip.py`
- Create: `scripts/check_release_literals.py`
- Create: `scripts/check_release_secrets.py`
- Create: `validation/release_literal_allowlist.json`
- Create: `tests/unit/test_release_literal_check.py`
- Create: `tests/unit/test_release_secret_check.py`
- Create: `docs/release/test-results.xml`
- Modify only when a failing test proves ownership: the exact source/test files named by that failure

- [ ] **Step 1: Make skipped release tests fail the session**

First write pytester tests proving a normal test skip, fixture/setup skip, and module-level `pytest.skip(..., allow_module_level=True)` collection skip all exit 0 without the flag and exit nonzero with `--fail-on-skip`, printing node ID/module and reason.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_fail_on_skip.py -q`

Expected: failures because the option/hook does not exist.

Then add a `--fail-on-skip` pytest option in `tests/conftest.py`. Capture skips from both `pytest_collectreport` and runtest setup/call/teardown reports; at session finish set nonzero exit status and print each node ID/module and reason. Enable pytester's fixture plugin only for the hook unit test.

Run: `mkdir -p docs/release`

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest --fail-on-skip --junitxml=docs/release/test-results.xml -q`

Expected: zero failed/skipped tests and a tracked JUnit report containing exact counts and durations.

- [ ] **Step 2: Run static quality checks**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check .`

Expected: `All checks passed!`

- [ ] **Step 3: Implement and run the scoped season/copy gate**

Write failing tests for allowed historical literals, unallowlisted defaults, stale allowlist entries, word boundaries (`between` must not match `bet`), and prohibited presentation copy.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_release_literal_check.py -q`

Expected: import/contract failures before the checker exists.

Then implement `check_release_literals.py` to scan runtime Python plus templates/static assets. It rejects `2526`/2025–26 defaults outside an explicit JSON allowlist of `{path,line_pattern,reason}` historical cases. It separately scans only presentation paths (`app/templates`, `app/static`, `app/presenters.py`) for case-insensitive word-boundary `bet|betting|stake|odds`, plus `ensemble member|confidence path|percentile`. Stale allowlist entries also fail.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_release_literal_check.py -q && UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/check_release_literals.py --allowlist validation/release_literal_allowlist.json`

Expected: tests pass and checker prints `PASS` with zero unallowlisted season literals/prohibited presentation terms.

- [ ] **Step 4: Implement and run the release-evidence secret gate**

Write red tests with safe fixtures and detections for `.env` assignments, `Authorization`/`Cookie` headers, private-key blocks, password/secret/token/API-key names followed by values, JWT-shaped values, and known provider-key prefixes.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_release_secret_check.py -q`

Expected: import/contract failures before the scanner exists.

Implement `check_release_secrets.py` to accept explicit files/directories, fail closed with exit 2 when any requested path is missing, scan UTF-8 text plus ASCII strings embedded in binary/image metadata, report only path/pattern category with the matched value redacted, and return exit 1 on a finding. Add missing-file behavior to its tests. Hashes, version IDs, and the literal words `token`/`cookie` without assigned values do not trigger.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_release_secret_check.py -q`

Expected: all safe/detection/redaction/exit-status tests pass.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/check_release_secrets.py docs/release validation/reports`

Expected: `PASS` with zero findings and no sensitive value printed.

- [ ] **Step 5: Repair failures test-first and commit**

For each failure, add or tighten a reproducing assertion before changing implementation. Re-run the focused test, then Steps 1–3. Build an explicit list from `git diff --name-only` and stage only the reviewed files owned by each repair; never use directory-wide `git add src tests`.

```bash
git add tests/conftest.py tests/unit/test_fail_on_skip.py scripts/check_release_literals.py scripts/check_release_secrets.py validation/release_literal_allowlist.json tests/unit/test_release_literal_check.py tests/unit/test_release_secret_check.py docs/release/test-results.xml
git commit -m "test: close season readiness regressions"
```

### Task 19: Perform Live 2026–27 Data and Model Readiness Validation

**Files:**
- Create: `docs/release/2026-27-readiness-report.md`
- Create: `tests/test_cli.py`
- Modify: `src/epl_prediction_optimizer/cli/main.py`
- Update generated artifacts: `data/processed/**`, `data/exports/**`, `data/artifacts/**`

- [ ] **Step 1: Define and test the season-aware release CLI**

Write failing CLI tests for `eplpo refresh --season 2627 --mode live|offline`, `eplpo train --season 2627`, and `eplpo predict --season 2627`.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/test_cli.py -q`

Expected: parser failures because season/mode flags and versioned JSON output are absent.

Implement each command to delegate to the already-tested season/data/prediction services and print structured JSON containing season and publication versions. Invalid seasons/modes exit 2; live failures exit nonzero and never fall back to sample data.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/test_cli.py -q`

Expected: all CLI parsing, delegation, output, and failure tests pass.

- [ ] **Step 2: Back up version identities in the report, then run live refresh**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run eplpo refresh --season 2627 --mode live`

Expected: successful football-data.org 2026–27 snapshot, per-source timestamps, no sample fallback, and a new atomic fixture version. Normalize and compare the source team set exactly with:

```text
AFC Bournemouth, Arsenal, Aston Villa, Brentford, Brighton & Hove Albion,
Chelsea, Coventry City, Crystal Palace, Everton, Fulham, Hull City,
Ipswich Town, Leeds United, Liverpool, Manchester City, Manchester United,
Newcastle United, Nottingham Forest, Sunderland, Tottenham Hotspur
```

Require exactly 20 teams, 380 unique fixtures, and promoted subset `Coventry City, Hull City, Ipswich Town`; record a set diff and the authoritative Premier League membership/fixture URLs in the report. Any mismatch blocks release. If credentials/network are unavailable, stop and request them; this gate cannot be simulated.

Authoritative comparison URLs: `https://www.premierleague.com/en/news/4673099/the-202627-premier-league-season-officially-starts/` for promoted membership and `https://www.premierleague.com/en/news/4675097/all-380-fixtures-for-202627-premier-league-season/` for the complete fixture list.

- [ ] **Step 3: Train only on completed matches and publish probabilities**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run eplpo train --season 2627`

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run eplpo predict --season 2627`

Expected: visible training cutoff/model version, a new prediction version, normalized probabilities summing to 1 per match, and no partial artifact replacement.

- [ ] **Step 4: Select and run the model/optimizer gate automatically**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/validate_release.py --mode auto --baseline-dir validation/baselines --seasons 2223 2324 2425 --output validation/reports/2627_optimizer_validation.json`

Expected: `PASS`; the report states the source-map comparison and chosen mode. Any model-affecting change since the frozen baseline—including earlier tasks or Task 18 repairs—automatically runs `model-change` and reports accuracy, log loss, calibration, realized points, expected-vs-realized points, thresholds, and waiver/review state.

- [ ] **Step 5: Verify the next open decision round and complete the report**

Fetch readiness and its returned round through TestClient/default services. Confirm Friday-only Arsenal–Coventry is not a candidate and at least one exact feasible scenario exists for the August 22–23 weekend. The report has fixed sections/fields: generation time/timezone; source URLs/states/timestamps/ages/counts/errors; fixture/model/prediction/rules/ledger/local/season-eligibility versions; exact 20-team set, promoted subset, fixture count and diffs; training cutoff/model identity; probability normalization; next-open round/eligible fixtures/scenario count/feasibility/valid-until; test report counts/path; validator mode/identity/metrics/path; known blockers/limitations.

The report and screenshots must never contain API keys, tokens, `.env` values, authorization/request headers, credentials, cookies, or raw provider responses. Add a secret-pattern scan before staging.

- [ ] **Step 6: Test, lint, scan, and commit release evidence**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/test_cli.py tests/integration/test_release_validation.py -q`

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check src/epl_prediction_optimizer/cli/main.py tests/test_cli.py`

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/check_release_secrets.py docs/release/2026-27-readiness-report.md validation/reports/2627_optimizer_validation.json`

Expected: tests/Ruff pass and the secret checker prints `PASS`. Inspect the staged report diff before commit. Runtime `data/` remains ignored and is not staged.

```bash
git add src/epl_prediction_optimizer/cli/main.py tests/test_cli.py docs/release/2026-27-readiness-report.md validation/reports/2627_optimizer_validation.json
git commit -m "chore: prepare verified 2026-27 season data"
```

### Task 20: Complete Visual Review and Release Smoke Test

**Files:**
- Create: `scripts/release_smoke_server.py`
- Create: `tests/unit/test_release_smoke_server.py`
- Create: `docs/release/screenshots/2627-light-desktop.png`
- Create: `docs/release/screenshots/2627-dark-desktop.png`
- Create: `docs/release/screenshots/2627-light-mobile.png`
- Create: `docs/release/screenshots/2627-dark-mobile.png`
- Modify: `docs/release/2026-27-readiness-report.md`
- Create: `docs/release/impeccable-detector.json`
- Create: `docs/release/impeccable-finish-review.md`
- Create: `.impeccable/review/desktop.png`
- Create: `.impeccable/review/mobile.png`
- Create: `.impeccable/review/hero-repro.png`
- Modify only from material review findings: exact reviewed files under `src/epl_prediction_optimizer/app/`

- [ ] **Step 1: Build and test an isolated smoke server**

First write failing tests for loopback/path refusal, copied-ledger isolation, mutable clock advancement, SIGINT cleanup, and source/live database hashes remaining unchanged.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_release_smoke_server.py -q`

Expected: import/contract failures before the isolated server exists.

Then implement `release_smoke_server.py`: it creates a `TemporaryDirectory`, copies the verified state database and every immutable artifact referenced by its publication rows, rewrites all copied publication pointers to paths under the temporary artifact root in one copied-database transaction, and verifies no absolute/source-workdir pointer remains. Build `AppServices` using only the rewritten copied database/root and a mutable frozen clock. Inject a no-network coordinator whose every fetch method raises the test if invoked, disable the background runtime refresh scheduler, and exercise eligibility changes only through lazy reconciliation/read requests. Add QA-only routes to advance the clock and update source-freshness timestamps in the copied database; refuse non-loopback hosts and any database/artifact path outside the temporary directory. Isolation tests build their entire source database/artifact tree under pytest `tmp_path`; only that disposable synthetic source may be deleted. The real-workdir test uses an access-denying filesystem mock after cloning and never chmods, deletes, moves, or rewrites real source data. SIGINT/normal exit stops Uvicorn and deletes the isolated copy, never touching the user's live ledger.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest tests/unit/test_release_smoke_server.py -q`

Expected: isolation/refusal, clock advance, and cleanup tests pass.

- [ ] **Step 2: Start, probe, and own the isolated server session**

Run the server with the execution tool as a persistent PTY/session: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/release_smoke_server.py --source-workdir . --host 127.0.0.1 --port 8010`. Record the returned session ID, wait for `QA server ready`, then probe `http://127.0.0.1:8010/api/readiness?season=2627` and require HTTP 200. All later browser work uses port 8010. Treat shutdown as `finally`: after any failed command, blocked reviewer disposition, interruption, or successful completion, immediately send Ctrl-C through that same session, wait for exit, verify the port no longer answers, and verify the script reports temporary-directory removal before returning/asking for input.

- [ ] **Step 3: Capture the complete first evidence set**

Run: `mkdir -p .impeccable/review docs/release/screenshots`

Use the browser-control workflow to capture `.impeccable/review/desktop.png` at 1440×1000, `.impeccable/review/mobile.png` at 390×844, and `.impeccable/review/hero-repro.png` at the approved comp's dimensions, plus the four light/dark release screenshots. Compare structure, density, hierarchy, material, and interaction placement with `.impeccable/mocks/pick-optimizer/ensemble-split-light.webp`; never assert its synthetic fixture copy or non-normative numbers.

- [ ] **Step 4: Run one mechanical detector and one independent finish review**

Run once: `node /Users/francosolari/.agents/skills/impeccable/scripts/detect.mjs --json src/epl_prediction_optimizer/app/templates src/epl_prediction_optimizer/app/static src/epl_prediction_optimizer/app/presenters.py`. Save the exact JSON output to `docs/release/impeccable-detector.json` with `apply_patch`; do not run a second detector pass. Every blocking finding is added to the independent review's material-fix packet and resolved in the single batch; acceptance is based on documented resolution plus the post-fix finish-reviewer `ship` verdict, not a false claim that the pre-fix detector JSON became clean.

Dispatch a fresh `impeccable_finish_reviewer` subagent with only: original request/answers; PRODUCT.md; surface brief and direction contract; approved comp/sidecar; craft-floor path; detector JSON; the three `.impeccable/review` captures; and representative final template/CSS/JS paths. Save its exact five-section verdict in `docs/release/impeccable-finish-review.md`. One build agent applies all material fixes in a single batch.

- [ ] **Step 5: Perform the isolated lifecycle smoke test**

Against port 8010 only: preview two candidates and verify table/chart/tree update together; commit and change a copied-database pick before kickoff. Immediately before advancing across kickoff, use the QA-only copied-state route to set fixture freshness to the frozen current time, then cross kickoff by one second and assert `PICK_LOCKED` rather than a freshness error. Call a QA-only reset route that deletes only the temporary cloned state and atomically reclones/repoints it from the unchanged verified source snapshot, resets the mutable clock, and returns the initial publication hashes; assert those hashes match the first clone. In that fresh copied-state case, leave freshness unchanged, advance beyond the 30-minute matchday threshold while still pre-kickoff, and assert stale-data commit disabling. Browse backward/forward and inspect lookahead assumptions; confirm theme persistence and no-JavaScript POST submission. Record outcomes without copying the isolated ledger into live state. Then stop the current PTY session in the unconditional cleanup path and verify the port/temp directory are gone before applying review fixes.

- [ ] **Step 6: Recapture and run the single finish-verdict pass**

After the fix batch, start `release_smoke_server.py` again as a new owned PTY session. This creates a fresh, eligible, matchday-ready cloned database/artifact root and reloads the changed code; probe readiness and require the open decision surface before capture. Overwrite the same `.impeccable/review` and four release screenshot paths with post-fix captures. Return those exact recaptures to the same finish reviewer for its one bounded verdict pass; append the exact two-section verdict/disposition. `ship` is required. A `rebuild`, `fix`, or `recapture` disposition blocks release and immediately enters unconditional server cleanup rather than an open-ended polish loop.

- [ ] **Step 7: Re-run every final gate**

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run pytest --fail-on-skip --junitxml=docs/release/test-results.xml -q`

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run ruff check .`

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/check_release_literals.py --allowlist validation/release_literal_allowlist.json`

Expected: zero failed/skipped tests and Ruff/literal gates pass. Stop the fresh isolated server session with Ctrl-C, wait for exit, verify the port no longer answers and temporary directory was removed, and record that concrete cleanup evidence.

Now update the readiness report with the resulting final test counts, model/optimizer report paths, data timestamps/versions, screenshot paths, detector findings/resolutions, reviewer `ship` disposition, isolated smoke results, verified cleanup, and known non-blocking limitations.

Run: `UV_CACHE_DIR=/tmp/eplpo-uv-cache uv run python scripts/check_release_secrets.py docs/release validation/reports .impeccable/review`

Expected: `PASS`. Add the exact command/result to the report, then rerun the same secret-scan command so the final report bytes containing that statement are also scanned and must pass.

Run: `git diff --check`

Expected: no whitespace errors after the final report update.

- [ ] **Step 8: Inspect staged evidence and commit**

Manually inspect staged screenshot/report diffs; do not stage credentials, headers, cookies, raw responses, or runtime `data/`. Before staging review fixes, run `git status --short`, verify each changed tracked app path belongs to a documented finding, and use `git add -u src/epl_prediction_optimizer/app` only for those reviewed tracked paths. Do not modify or stage browser tests during this review task; all existing browser suites were already rerun in Step 7.

```bash
git add scripts/release_smoke_server.py tests/unit/test_release_smoke_server.py docs/release/2026-27-readiness-report.md docs/release/test-results.xml docs/release/impeccable-detector.json docs/release/impeccable-finish-review.md docs/release/screenshots/2627-light-desktop.png docs/release/screenshots/2627-dark-desktop.png docs/release/screenshots/2627-light-mobile.png docs/release/screenshots/2627-dark-mobile.png .impeccable/review/desktop.png .impeccable/review/mobile.png .impeccable/review/hero-repro.png
git commit -m "chore: verify season-ready pick optimizer"
```

## Final Acceptance Checklist

- [ ] The default season is `2627` on 2026-08-21 and no runtime path silently selects `2526`.
- [ ] Live data is timestamped, source-auditable, atomic, last-known-good, and never silently sample-backed.
- [ ] The live snapshot contains exactly the official 20 clubs, including promoted Coventry City, Hull City, and Ipswich Town, and 380 unique fixtures.
- [ ] Every scenario is an exact re-solve tied to fixture, prediction, model, rule, ledger, and eligibility versions.
- [ ] The next open round alone can commit; future rounds clearly state their canonical-plan assumptions.
- [ ] A pick changes only before both involved kickoffs and locks authoritatively at kickoff.
- [ ] The UI provides fast round browsing, linked preview effects, forecast horizon, move tree, and complete lifecycle history.
- [ ] Light/dark/system, desktop/tablet/mobile, keyboard, non-color, and reduced-motion behavior pass.
- [ ] The visible model version and training cutoff are current and recorded in readiness/release evidence.
- [ ] The server-rendered decision form submits through the same pick service when JavaScript is unavailable.
- [ ] Deterministic, integration, browser, migration, optimizer replay, and—if applicable—model-change gates pass.
- [ ] The one Impeccable detector pass is recorded, every blocking finding has documented post-fix resolution, the independent finish reviewer returns `ship`, and post-fix desktop/mobile/hero plus light/dark screenshots are tracked.
- [ ] Light, dark, and system themes persist correctly, and disabling JavaScript still permits an accessible server-rendered pick submission.
- [ ] The isolated mutable-clock smoke test proves preview, pre-kickoff change, post-kickoff lock, lookahead, stale blocking, and cleanup without changing the live ledger.
- [ ] The final report shows zero skipped/failed tests, Ruff/literal/secret gates passing, and a nonzero feasible scenario count for the next decision round.
- [ ] The live 2026–27 readiness report contains current data/version/model evidence and no unresolved release blocker.
