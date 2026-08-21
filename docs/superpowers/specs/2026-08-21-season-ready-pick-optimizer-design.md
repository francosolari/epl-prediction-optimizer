# Season-Ready Pick Optimizer Design

Date: 2026-08-21

Status: Approved design, pending implementation planning

## 1. Objective

Turn the existing EPL Prediction Optimizer into a season-ready weekly decision tool for the 2026–27 Premier League season. The product must help one owner choose a team each matchweek, understand how that choice changes the best remaining-season plan, record the submitted pick, and revisit future schedules without exposing pipeline mechanics as the primary experience.

The release must preserve the confirmed contest rules:

- Only Saturday and Sunday fixtures are eligible.
- Exactly one team is selected per contest matchweek.
- Every Premier League team is selected at least once during the season.
- A team is selected at most twice: once at home and once away.
- A submitted pick can be changed until that team's match kicks off. It is immutable after kickoff.

The official Premier League schedule confirms that the 2026–27 season begins on Friday, August 21, 2026. The application therefore targets season code `2627`, while deriving the active season centrally so future migrations do not require scattered code edits.

## 2. Current Readiness Findings

The application is not ready for 2026–27 in its current state:

- Application routes, templates, pipeline functions, challenge defaults, CLI defaults, and current-season statistics contain hard-coded `2526` or 2025–26 labels.
- The current model, fixture exports, recommendations, and SQLite state were last generated in May 2026.
- The current refresh flow explicitly downloads and merges the 2025–26 season.
- Offline sample refresh can unexpectedly call the live football-data.org endpoint when it produces no `2526` fixtures.
- The test baseline is 8 passing and 3 failing tests. One test expects obsolete UI copy; two reveal the unintended network dependency in the offline pipeline.
- The current UI separates recommendations from actual pick entry and does not explain downstream opportunity cost.
- A stored actual pick is an annotation today; it does not constrain subsequent optimization.

No release-readiness claim is valid until the migration, data refresh, full test suite, model validation, and UI verification gates in this specification pass.

## 3. Product and Visual Direction

The approved visual direction is **Preparation Room + Forecast Horizon**, using the **Balanced Horizon** composition.

Approved comp: `.impeccable/mocks/pick-optimizer/ensemble-split-light.webp`

The first viewport is a working decision surface:

- A narrow context rail shows section, season, model status, and freshness.
- A ruled candidate comparison occupies the left side.
- A forecast horizon occupies the right side and plots one cumulative season path per feasible current-week choice.
- A continuous season move tree spans the lower band.
- Selecting a candidate updates the comparison, forecast paths, and move tree together before commitment.

Light mode uses warm analysis paper, charcoal, cobalt, amber, and muted red. Dark mode preserves the same hierarchy and behavior on a low-glare graphite drafting surface. Theme preference supports light, dark, and system modes and persists locally.

The product must never resemble a sportsbook. It excludes odds formatting, betting language, casino color semantics, artificial urgency, currency-based stakes, and fabricated performance claims.

## 4. System Boundaries

### 4.1 Season Context

Create one season-context unit responsible for:

- Active season code, start year, display label, and known matchweeks.
- An explicit environment/configuration override for testing and unusual calendar periods.
- Date-based default derivation: July through December maps to the current year/current year + 1; January through June maps to the previous year/current year.
- Validation against the season returned by the fixture source. A disagreement produces a readiness error instead of silently mixing seasons.

All routes, templates, refresh functions, CLI defaults, database queries, artifact names, and model reference seasons consume this unit. No presentation or pipeline file owns its own season literal.

### 4.2 Live Data Coordinator

Create a coordinator that owns current-season synchronization and source health. It has adapters for:

- Official football-data.org fixtures, kickoff changes, matchweek assignments, match status, and results.
- football-data.co.uk completed match history used for training.
- ClubElo team-strength history/current ratings.
- Locally cached Understat xG where available.

The coordinator returns a structured sync result per source: `source`, `season`, `started_at`, `completed_at`, `record_count`, `state`, and `error`. It writes new data atomically and retains the last-known-good dataset if a refresh fails.

Freshness policy:

- Trigger a non-blocking current-season refresh when the app starts if fixture data is stale.
- While the app is open, check fixture/result freshness every 15 minutes without starting overlapping refreshes.
- Treat fixture data as matchday-ready only after a successful refresh within 30 minutes when any eligible fixture begins within 24 hours.
- Refresh ClubElo at most once per 24 hours.
- Mark the model stale whenever newly completed matches are newer than its training artifact.
- Retrain after a successful refresh introduces completed matches, then regenerate predictions and scenarios.
- Expose exact per-source timestamps and errors. A generic `ready` label without timestamps is insufficient.

Live mode never falls back to sample fixtures. Sample data is allowed only when an explicit offline/test mode is selected.

### 4.3 Prediction Model Boundary

The existing calibrated match-outcome model remains the probability provider for this release. The season-readiness work may fix data selection, active-season context, and training orchestration, but it must not add unvalidated injury or lineup values to the feature vector.

Automated injuries, suspensions, and predicted lineups are deferred because the current provider does not supply them and available alternatives are premium third-party predictions. The UI states that team news is not modeled and lets the user attach a risk flag and override note to a committed pick. These notes do not silently alter model probabilities.

Every model-affecting change must pass the validation gate in Section 10 before it can ship.

### 4.4 Counterfactual Scenario Optimizer

Extend the optimizer to accept locked picks:

```text
optimize(candidates, contest_rules, locked_picks) -> SeasonPlan
```

`locked_picks` includes all prior committed picks and the candidate being previewed. The solver validates that locked picks are eligible, do not conflict, and still permit a feasible full-season plan.

For the next open matchweek:

1. Solve the baseline plan using prior committed picks and no current-week lock.
2. For every eligible current-week candidate, solve a counterfactual plan with that candidate locked.
3. Return a scenario record containing:
   - Immediate win probability and expected points.
   - Immediate rank by expected points.
   - Season-aware rank by counterfactual total expected points.
   - Projected remaining-season expected points.
   - `season_cost = baseline_total - scenario_total`.
   - First future matchweek whose selected team differs from baseline.
   - The displaced picks and replacement picks.
   - The complete revised season plan.
   - Feasibility state and a human-readable reason when infeasible.
4. Cache scenarios by season, week, fixture-data version, model version, rules version, and committed-pick version.
5. Invalidate the cache after data refresh, retraining, rule changes, or pick commitment.

This is exact counterfactual re-solving, not a heuristic. The optimizer does not perform Monte Carlo simulation in this release.

### 4.5 Forecast Horizon Semantics

The approved chart must remain mathematically honest. It is a **scenario forecast horizon**, not a statistical confidence ensemble.

- Each thin path represents the cumulative expected points of one feasible current-week candidate's re-optimized season plan.
- The highlighted amber path is the highest immediate-value choice.
- The highlighted cobalt path is the highest full-season-value choice.
- The selected path is identified by label and line treatment, not color alone.
- Event markers identify the first displaced week and later team-usage conflicts.
- Tooltips disclose that paths are optimizer scenarios based on current match probabilities.
- Confidence bands, percentile labels, and probability-of-season-outcome claims are prohibited until a separately validated simulation model exists.

### 4.6 Pick Ledger

Committed picks are durable optimizer inputs. Extend the existing pick record with lifecycle fields:

- `season`
- `contest_week`
- `match_id`
- `team`
- `venue`
- `kickoff_at`
- `committed_at`
- `updated_at`
- `locked_at`
- `state` (`committed`, `locked`, `scored`)
- `notes`
- `news_risk` (`none`, `watch`, `override`)
- `actual_points`

The server, not the browser, determines whether a pick can change. An update is rejected when current time is at or after the selected fixture kickoff. A background/read-time reconciliation transitions committed picks to locked and later scored states from official match status/results.

Changing a pre-kickoff pick replaces the current week's ledger entry and immediately invalidates/rebuilds remaining scenarios. Historical locked picks are never overwritten.

## 5. User Experience

### 5.1 Readiness Gate

On entry, the app renders the last-known-good decision surface immediately when available and shows a compact updating state. The primary commit action remains disabled until matchday freshness requirements pass.

States:

- **Ready:** fixtures, model, and scenarios are current.
- **Updating:** cached information remains readable; preview/commit communicates what is temporarily unavailable.
- **Stale:** information is visible with its age and source; commit is disabled when the matchday gate fails.
- **Error:** the failed source and recovery action are explicit. The UI never displays sample data as live.
- **No model/data:** a first-run checklist offers one action, `Prepare season`, with detailed pipeline controls under an advanced disclosure.

### 5.2 Weekly Decision

The default route opens the next uncommitted matchweek. It shows the top candidates in an integrated comparison rather than isolated recommendation cards.

Selecting a candidate is a reversible preview. It updates:

- Immediate and season-aware ranking.
- Plain-language trade-off summary, such as “0.08 expected season points lower; preserves Manchester City for Matchweek 7.”
- Scenario horizon.
- Season move tree.
- Affected-week annotations.

The commit action includes the team, opponent, kickoff, and consequence. Confirmation is inline rather than a generic modal. After commitment, the app shows the new remaining-season plan and the deadline for editing.

### 5.3 Week Navigation

The user can browse all known matchweeks with:

- Previous and next controls.
- A direct matchweek picker.
- Left/right keyboard shortcuts when focus is not inside a form control.
- Clickable nodes in the season move tree.
- Mobile swipe as a progressive enhancement, never the only control.

The next open week is marked `Decision due`; later weeks are `Look ahead`; committed/locked/scored weeks show their lifecycle state. Future weeks are preview-only. The user can inspect their current planned choice and alternatives but commits only the next open week, preventing contradictory out-of-order locks.

Navigation preserves the selected scenario per week during the browser session and never discards a committed pick.

### 5.4 Season Plan and History

The season view shows every week, the current optimal line, committed picks, team/venue usage, and constraint pressure. Selecting a week returns to its decision view. The history view shows submitted team, opponent, result, points, override note, and whether the model recommendation was followed.

Pipeline, model diagnostics, backtests, and raw data remain available as secondary expert surfaces, not primary navigation priorities.

### 5.5 Responsive and Accessible Behavior

Desktop uses the approved Balanced Horizon split. At narrower widths:

- Tablet stacks the forecast horizon beneath the candidate comparison while keeping the move tree horizontally navigable.
- Mobile leads with the candidate comparison, then the selected scenario summary, then a simplified horizontally scrollable forecast and move tree.
- Tables convert to aligned rows without hiding decision-critical values.

All actions are keyboard accessible, visible focus is mandatory, state never relies on color alone, chart paths have text summaries, and reduced-motion mode disables trace morphing while preserving instant updates.

## 6. Application Interfaces

The implementation plan may refine route names, but these contracts must remain distinct:

### Readiness

```text
GET /api/readiness?season=2627
-> season context, next open week, source freshness, model version/state,
   scenario state, and commit eligibility
```

### Matchweek Decision

```text
GET /api/seasons/{season}/weeks/{week}
-> fixtures, committed state, ranked scenario summaries, highlighted scenario ids,
   scenario-horizon series, and move-tree summary
```

### Scenario Detail

```text
GET /api/seasons/{season}/weeks/{week}/scenarios/{scenario_id}
-> complete counterfactual plan, displaced/replacement picks, explanation fields
```

### Commit or Change Pick

```text
PUT /api/seasons/{season}/weeks/{week}/pick
body: match_id, team, venue, notes, news_risk
-> validated pick lifecycle record, recalculated remaining plan, updated readiness
```

Expected client errors are structured and user-actionable: stale data, kickoff passed, ineligible fixture, constraint conflict, scenario expired, refresh in progress, and no feasible season plan.

### Refresh

```text
POST /api/refresh/current
-> accepted/in-progress state with per-source progress
```

Repeated requests are idempotent while a refresh is running.

## 7. Data Storage and Migration

- Add a forward-only SQLite migration mechanism rather than assuming a new database.
- Migrate existing actual-pick rows without deleting user history.
- Add lifecycle columns to the pick ledger with safe defaults derived from fixture data when possible.
- Store source-sync history in a dedicated table rather than one opaque status JSON value.
- Store experiment/model-run metadata with model version and training-data cutoff.
- Store scenario results as a bounded cache or regenerate them; they are derived data, not historical truth.
- Keep artifact writes atomic so a failed train/predict cycle cannot replace a valid model or export with a partial file.

The migration is idempotent and covered by tests against both a new database and a database using the current schema.

## 8. Error Handling and Recovery

- Network timeout or rate limit: retain cached data, record the source error and retry eligibility, and surface the exact stale age.
- Fixture-source season mismatch: stop the refresh before merge and mark readiness error.
- Partial source success: publish only coherent artifacts; optional ClubElo/xG failures may retain prior values with explicit warnings, while fixture failure blocks matchday commit.
- Training failure: retain the prior model and mark it stale/error; never publish a half-written model.
- No feasible optimizer plan: identify which locked pick/rule produces the conflict; preserve committed history and disable only infeasible preview choices.
- Scenario version mismatch during commit: reject with a refresh-required response and preserve the user's selection for retry.
- Kickoff race: recheck official kickoff and server time inside the commit transaction.
- Theme/JavaScript failure: core comparison, navigation links, and form submission remain usable with server-rendered HTML.

## 9. Non-Goals for This Release

- Automated injury, suspension, or predicted-lineup ingestion.
- Monte Carlo season simulation or statistical confidence bands.
- Betting odds, staking, monetary return, or sportsbook comparison.
- Multiple users, authentication, cloud synchronization, or notifications.
- Changing the confirmed contest rules.
- Replacing the existing calibrated classifier without validated evidence.

## 10. Testing and Model Validation

### 10.1 Deterministic Unit and Integration Tests

- Season-code/display derivation across calendar boundaries and override behavior.
- Offline refresh never performs a network request.
- Live refresh never substitutes sample fixtures.
- Source freshness, atomic-write, last-known-good, season-mismatch, and partial-failure behavior.
- Optimizer feasibility with zero, one, and many locked picks.
- Counterfactual season cost, first displaced week, immediate rank, season-aware rank, and cache invalidation.
- Every contest constraint after commitments and pick changes.
- Pick update before kickoff, rejection at/after kickoff, lifecycle reconciliation, and scoring.
- Database migration for new and existing SQLite files.
- API error schemas and idempotent refresh behavior.

### 10.2 Model Change Gate

Any change to features, labels, weighting, estimator, calibration, training population, or probability correction requires:

1. Deterministic sample-data tests.
2. Walk-forward backtests on at least three completed Premier League seasons.
3. A comparison with the checked-in/current baseline for:
   - Classification accuracy.
   - Multiclass log loss.
   - Calibration diagnostics.
   - Realized optimized contest points.
   - Expected-versus-realized pick points.
4. A saved experiment record containing code/model version, feature list, parameters, training cutoff, seasons, and metrics.
5. Explicit review of any trade-off. No model change ships merely because accuracy rises while probability quality or realized optimizer performance materially worsens.

Scenario-optimizer changes that do not alter match probabilities still require multi-season replay to confirm plan feasibility and realized points relative to the existing optimizer.

### 10.3 UX and Visual Verification

- Server-rendered and browser interaction tests for readiness, week navigation, preview, commit, change-before-kickoff, lock-after-kickoff, stale state, and failure recovery.
- Light, dark, and system theme behavior with persisted preference.
- Keyboard navigation, focus order, screen-reader labels, non-color chart summaries, and reduced motion.
- Desktop and mobile screenshots compared with the approved comp and responsive rules.
- Mechanical Impeccable detector pass on changed templates/styles.
- Independent Impeccable finish review using approved comp and screenshots.

### 10.4 Release Gate

Release requires:

- All existing and new tests passing, including repair of the present 3 failures.
- Ruff passing.
- A successful 2026–27 live refresh with official fixtures and correct promoted clubs.
- A trained model whose training cutoff and version are visible in readiness.
- Feasible counterfactual scenarios for the next open matchweek.
- Manual verification that a preview changes the chart/tree, a pre-kickoff pick can change, and an after-kickoff pick cannot.
- Desktop and mobile visual review in both themes.
- No hard-coded `2526` references in runtime paths or UI defaults except historical fixtures/tests explicitly labeled as past-season data.

## 11. Implementation Slices

The detailed implementation plan should preserve these dependency boundaries:

1. **Season and data readiness:** central season context, 2026–27 source refresh, freshness records, atomic artifacts, and repair of baseline tests.
2. **Committed optimizer state:** pick-ledger migration, locked-pick solver interface, exact counterfactual scenarios, explanations, caching, and backtest/replay validation.
3. **Decision APIs:** readiness, matchweek scenarios, detail, commit/change, refresh, and structured errors.
4. **Decision-desk UI:** approved Balanced Horizon composition, week browsing, preview/commit flow, forecast horizon, move tree, readiness states, and light/dark themes.
5. **Verification and release:** full automated suite, model reports, detector, desktop/mobile screenshots, independent design review, and season-ready smoke test.

Each slice must be independently testable. The prediction model, optimizer, data coordinator, pick ledger, and presentation layer communicate through the explicit contracts above rather than importing each other's internal implementation details.

## 12. Approved Decisions

- Primary user: the app owner making one weekly contest pick.
- Exact contest rules remain unchanged for 2026–27.
- Product focus: compare choices, understand future opportunity cost, commit, track, and re-optimize.
- Visual direction: Preparation Room.
- Required visual feature: Forecast Horizon.
- Approved composition: Balanced Horizon.
- Themes: equivalent light and dark modes.
- Optimization method: exact counterfactual re-solving now; simulation deferred.
- Pick lifecycle: editable before selected fixture kickoff, locked at kickoff.
- Future navigation: browse every known matchweek; commit only the next open week.
- Data policy: automatic verified refresh, visible freshness, last-known-good recovery, no silent sample fallback.
- Availability policy: no unvalidated automated injury/lineup feature in this release; manual news-risk note only.
- Validation policy: every model-affecting change requires sample tests and multi-season metric comparison.
