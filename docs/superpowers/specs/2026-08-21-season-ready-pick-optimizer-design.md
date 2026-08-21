# Season-Ready Pick Optimizer Design

Date: 2026-08-21

Status: Approved design, pending implementation planning

## 1. Objective

Turn the existing EPL Prediction Optimizer into a season-ready weekly decision tool for the 2026–27 Premier League season. The product must help one owner choose a team each eligible contest weekend, understand how that choice changes the best remaining-season plan, record the submitted pick, and revisit future schedules without exposing pipeline mechanics as the primary experience.

The release must preserve the confirmed contest rules:

- Only Saturday and Sunday fixtures are eligible.
- Exactly one team is selected per eligible contest weekend.
- Every Premier League team is selected at least once during the season.
- A team is selected at most twice: once at home and once away.
- A submitted pick can be changed until that team's match kicks off. It is immutable after kickoff.

The official Premier League schedule confirms that the 2026–27 season begins on Friday, August 21, 2026. The application therefore targets season code `2627`, while deriving the active season centrally so future migrations do not require scattered code edits.

### 1.1 Canonical Round Definitions

`league_matchweek` and `contest_round` are different concepts:

- `league_matchweek` is the Premier League provider's official round number, 1–38. It is metadata for browsing and may contain midweek fixtures.
- `contest_round_id` is the ISO date of a Saturday in `Europe/London`. Its eligibility window is that Saturday 00:00 through Sunday 23:59:59 in `Europe/London`.
- An **active contest round** is a weekend window containing at least one currently scheduled Premier League fixture. Exactly one pick is required for each active contest round.
- Midweek fixtures are visible in the schedule but never become pick candidates and do not create a contest round.
- `contest_round_number` is a display ordinal over active weekend windows. The stable identifier is the Saturday date, so later schedule changes do not rewrite historical keys.
- Fixtures from more than one `league_matchweek` may appear in the same contest round after rescheduling. The calendar window, not league-round metadata, determines eligibility.

If every fixture leaves a future weekend before any pick locks, that round becomes `void` and requires no pick. If a user fails to commit before all eligible fixtures in an active round have kicked off, the round becomes `missed`, earns zero points, consumes no team/venue usage, and is represented as a fixed synthetic `MISSED` choice so the remaining optimizer can advance without violating one-choice-per-round accounting.

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
- A scenario forecast horizon occupies the right side and plots one cumulative season path per feasible current-week choice.
- A continuous season move tree spans the lower band.
- Selecting a candidate updates the comparison, forecast paths, and move tree together before commitment.

Light mode uses warm analysis paper, charcoal, cobalt, amber, and muted red. Dark mode preserves the same hierarchy and behavior on a low-glare graphite drafting surface. Theme preference supports light, dark, and system modes and persists locally.

The product must never resemble a sportsbook. It excludes odds formatting, betting language, casino color semantics, artificial urgency, currency-based stakes, and fabricated performance claims.

All fixture names and numeric values inside generated comps are synthetic composition material, not product truth or acceptance-test fixtures. In particular, the comp's Arsenal–Coventry Friday fixture is ineligible and Liverpool–Nottingham Forest is not the official Matchweek 1 fixture. Implementation visual tests compare structure, density, hierarchy, materials, and interaction placement while runtime content always comes from normalized live data. Comp labels that imply statistical confidence or negative opportunity cost are also non-normative and must be replaced by the semantics in Sections 4.4 and 4.5.

## 4. System Boundaries

### 4.1 Season Context

Create one season-context unit responsible for:

- Active season code, start year, display label, league matchweeks, and derived contest rounds.
- An explicit environment/configuration override for testing and unusual calendar periods.
- Date-based default derivation: July through December maps to the current year/current year + 1; January through June maps to the previous year/current year.
- Validation against the season returned by the fixture source. A disagreement produces a readiness error instead of silently mixing seasons.

All routes, templates, refresh functions, CLI defaults, database queries, artifact names, and model reference seasons consume this unit. No presentation or pipeline file owns its own season literal.

### 4.2 Live Data Coordinator

Create a coordinator that owns current-season synchronization and source health. It has adapters for:

- football-data.org fixtures, kickoff changes, matchweek assignments, match status, and results. This is the application's current-season schedule source; it is not described as the Premier League's official API.
- football-data.co.uk completed match history used for training.
- ClubElo team-strength history/current ratings.
- Locally cached Understat xG where available.

Each adapter normalizes records before the coordinator can publish them:

```text
FixtureRecord {
  match_id: string                 # provider-stable id
  season_code: string              # four digits, e.g. 2627
  league_matchweek: int | null
  kickoff_utc: datetime
  home_team: canonical team id/name
  away_team: canonical team id/name
  status: scheduled | timed | in_play | finished | postponed | cancelled
  home_goals: int | null
  away_goals: int | null
  source_updated_at_utc: datetime
}

ContestRound {
  contest_round_id: YYYY-MM-DD     # Saturday in Europe/London
  contest_round_number: int        # derived display ordinal
  state: future | open | locked | complete | missed | void
  eligible_match_ids: list[string]
}
```

The current-season schedule source is authoritative for kickoff, status, league matchweek, and current results. football-data.co.uk supplies historical training results; ClubElo and xG are supplemental features and never redefine fixture identity, kickoff, or status. Canonical team mapping is applied before publishing. A conflict in current result/status is resolved in favor of the schedule source and recorded as a warning.

The coordinator returns a structured sync result per source: `source`, `season`, `started_at`, `completed_at`, `record_count`, `state`, and `error`. A successful atomic fixture publish also creates a monotonic `fixture_data_version` tied to the normalized snapshot. It writes new data atomically and retains the last-known-good dataset if a refresh fails.

Freshness policy:

- Trigger a non-blocking current-season refresh when the app starts if fixture data is stale.
- While the app is open, check fixture/result freshness every 15 minutes without starting overlapping refreshes.
- Treat fixture data as matchday-ready only after a successful refresh within 30 minutes when any eligible fixture begins within 24 hours.
- Refresh ClubElo at most once per 24 hours.
- Mark the model stale whenever newly completed matches are newer than its training artifact.
- Retrain after a successful refresh introduces completed matches, then regenerate predictions and scenarios.
- Expose exact per-source timestamps and errors. A generic `ready` label without timestamps is insufficient.

Candidate eligibility is time-versioned independently of source refresh. Each contest round stores an `eligibility_version`, incremented whenever server time crosses an eligible fixture kickoff or reconciliation changes a candidate status. The season also stores a monotonic `season_eligibility_version`, incremented by any round eligibility change; lookahead scenarios therefore change when an earlier assumed round changes. Scheduled reconciliation runs at each next kickoff while the server is active; every readiness, round, and commit request also performs lazy reconciliation so sleep/restart cannot preserve stale eligibility. Round/scenario responses include the local and season-wide versions, `as_of_utc`, and `valid_until_utc`, where `valid_until_utc` is the earliest upcoming boundary in any dependency round that can change the candidate set. Cached responses are never served after that instant.

Live mode never falls back to sample fixtures. Sample data is allowed only when an explicit offline/test mode is selected.

### 4.3 Prediction Model Boundary

The existing calibrated match-outcome model remains the probability provider for this release. The season-readiness work may fix data selection, active-season context, and training orchestration, but it must not add unvalidated injury or lineup values to the feature vector.

Every atomic probability publish creates a monotonic `prediction_version` derived from the trained `model_version`, feature configuration, normalized fixture snapshot, ClubElo/input snapshot, and current-season-statistics cutoff. Regenerating probabilities after an input refresh changes `prediction_version` even when the trained model artifact does not change. Readiness, scenario cache keys, responses, and commit tokens carry this version.

Automated injuries, suspensions, and predicted lineups are deferred because the current provider does not supply them and available alternatives are premium third-party predictions. The UI states that team news is not modeled and lets the user attach a risk flag and override note to a committed pick. These notes do not silently alter model probabilities.

Every model-affecting change must pass the validation gate in Section 10 before it can ship.

### 4.4 Counterfactual Scenario Optimizer

Extend the optimizer to accept locked picks:

```text
optimize(candidates, contest_rules, locked_picks) -> SeasonPlan
```

`locked_picks` includes all prior committed picks and the candidate being previewed. The solver validates that locked picks are eligible, do not conflict, and still permit a feasible full-season plan.

For the next open contest round:

1. Solve the baseline plan using prior committed/missed choices and no current-round lock.
2. For every eligible current-round candidate, solve a counterfactual plan with that candidate locked.
3. Return a scenario record containing:
   - Immediate win probability and expected points.
   - Immediate rank by expected points.
   - Season-aware rank by counterfactual total expected points.
   - Projected remaining-season expected points.
   - `season_cost = max(0, baseline_remaining_ev - scenario_remaining_ev)`.
   - First future contest round whose selected team differs from baseline.
   - The displaced picks and replacement picks.
   - The complete revised season plan.
   - Feasibility state and a human-readable reason when infeasible.
4. Cache scenarios by season, contest-round id, fixture-data version, prediction version, model version, rules version, pick-ledger version, local eligibility version, and season-wide eligibility version.
5. Invalidate the cache after data refresh, retraining, rule changes, or pick commitment.

This is exact counterfactual re-solving, not a heuristic. The optimizer does not perform Monte Carlo simulation in this release.

Mathematical contract:

- Match expected points are `3 × P(win) + 1 × P(draw) + 0 × P(loss)`.
- `remaining_ev` sums match expected points from the previewed contest round through the final active contest round. Prior realized points are excluded because they are identical for all current choices.
- Pending already-committed picks contribute their current match expected points; scored picks contribute only to `projected_season_points` as realized points.
- `projected_season_points = realized_points + pending_committed_ev + optimized_open_round_ev`.
- The baseline and every scenario use the same active-round horizon, model version, fixture-data version, and prior committed/missed choices.
- `season_delta = scenario_remaining_ev - baseline_remaining_ev` is non-positive apart from numerical tolerance; the primary UI displays the non-negative `season_cost`, not a negative opportunity cost.
- Values within `1e-6` expected points are treated as tied. Immediate and season-aware comparisons are independent: `immediate_tie_group` identifies candidates tied on match expected points, while `season_tie_group` identifies candidates tied on scenario remaining EV. A null group means no tie. Displayed ranks remain unique ordinals after tie-breaking. Scenario ordering uses season remaining EV descending, immediate expected points descending, normalized team name ascending, and match id ascending; immediate-rank ordering uses immediate expected points followed by the same stable identity keys.
- The solver first maximizes expected points. It then constrains the result to the optimal value within `1e-6` and performs chronological lexicographic canonicalization: for each contest round in order, test candidate keys sorted by normalized team name, venue, and match id; fix the first candidate that preserves global optimality, then continue. This yields one unique plan without scalar rank sums or objective-distorting epsilon weights.
- When a scenario matches the baseline in all later rounds, `first_displaced_round` is `null`, displaced/replacement lists are empty, and the explanation reads `No downstream plan change`.
- Infeasible candidates do not receive a numeric season cost or rank and sort after feasible candidates with an explicit rule-conflict reason.

Example summary contract:

```text
ScenarioSummary {
  scenario_id: string
  scenario_version: string
  candidate_match_id: string
  candidate_team: string
  immediate_expected_points: float
  immediate_rank: int
  scenario_remaining_ev: float
  projected_season_points: float
  season_cost: float
  season_aware_rank: int
  first_displaced_round: YYYY-MM-DD | null
  displaced: list[PickChange]
  feasible: bool
  infeasible_reason: string | null
  immediate_tie_group: string | null
  season_tie_group: string | null
  mode: decision | lookahead
  assumed_prior_picks: list[Pick]
  as_of_utc: datetime
  valid_until_utc: datetime | null
}
```

Future-round previews use the same solver with an explicit conditional contract. For a future round, every earlier uncommitted round is hypothetically fixed to the unique canonical baseline plan before that future round's candidates are counterfactually solved. The response uses `mode: lookahead` and returns those assumptions in `assumed_prior_picks`; the UI labels it `Look ahead · assumes current plan`. These are never commitments. Only the next open round uses `mode: decision` and can be committed.

### 4.5 Scenario Forecast Horizon Semantics

The approved chart must remain mathematically honest. It is a **scenario forecast horizon**, not a statistical confidence ensemble. Product copy, legends, the surface brief, and visual acceptance tests use `scenario`, `candidate path`, and `alternative choice`; they do not use `ensemble member`, `confidence path`, percentile, or uncertainty-band terminology.

- Each thin path represents the cumulative expected points of one feasible current-round candidate's re-optimized season plan.
- The highlighted amber path is the highest immediate-value choice.
- The highlighted cobalt path is the highest full-season-value choice.
- The selected path is identified by label and line treatment, not color alone.
- Event markers identify the first displaced contest round and later team-usage conflicts.
- Tooltips disclose that paths are optimizer scenarios based on current match probabilities.
- If the immediate and season-aware winners are the same candidate, one path receives both labels rather than inventing a second winner.
- Confidence bands, percentile labels, and probability-of-season-outcome claims are prohibited until a separately validated simulation model exists.

### 4.6 Pick Ledger and Round Lifecycle

Committed picks are durable optimizer inputs. Extend the existing pick record with lifecycle fields:

- `season`
- `contest_round_id`
- `league_matchweek` (informational, nullable)
- `match_id`
- `team`
- `venue`
- `kickoff_at`
- `committed_at`
- `updated_at`
- `locked_at`
- `state` (`committed`, `invalidated`, `locked`, `postponed`, `cancelled`, `scored`, `legacy_unverified`)
- `notes`
- `news_risk` (`none`, `watch`, `override`)
- `actual_points`

All datetimes are stored and compared as timezone-aware UTC. Weekend eligibility and contest-round grouping use `Europe/London`; the UI converts kickoff times to the user's display timezone. Naive datetimes are rejected at the boundary.

Allowed pick transitions are:

```text
committed -> committed   # replacement before the replacement fixture kickoff
committed -> invalidated # pre-lock reschedule removes weekend eligibility
committed -> locked      # current UTC reaches kickoff_utc
locked -> postponed      # provider marks the already-locked fixture postponed
postponed -> locked      # provider supplies a new kickoff; pick remains bound
committed -> cancelled   # provider cancels before lock; round reopens
locked|postponed -> cancelled # provider cancels after lock; terminal void pick
locked|postponed -> scored # finished result becomes available
```

Pre-lock schedule behavior is explicit:

- A kickoff change inside the same Saturday/Sunday contest window retains the committed pick and updates its kickoff, provided both the old and new kickoff are still in the future.
- A move to any different contest weekend invalidates the pick even if the new date is also Saturday/Sunday. The original round reopens; the fixture becomes a candidate in the destination round; the user must explicitly recommit rather than having a choice moved between rounds without consent.
- A move outside Saturday/Sunday invalidates the pick and makes that fixture ineligible.
- Any such change increments `fixture_data_version`, the affected rounds' `eligibility_version`, and `pick_ledger_version`, invalidating scenarios for both origin and destination rounds.

`invalidated` and pre-lock `cancelled` reopen the original contest round and consume no team/venue usage. After lock, a postponement does not release the chosen team: the pick remains bound to its original contest round, is excluded from becoming a second candidate when replayed on a later weekend, and is scored when completed. A locked/postponed pick with no result remains pending and is never silently converted to a miss or zero.

Post-lock cancellation is terminal: the pick becomes `cancelled`, earns zero points, consumes no team/venue usage, and remains as the original round's historical void choice; its contest round becomes `complete`, allowing next-open-round derivation to advance. A later replay with a new fixture id is a new schedule event and is not automatically rebound. Pre-lock cancellation simply invalidates the choice and reopens the round.

Contest rounds have their own lifecycle. A future round with no eligible fixtures is `void`. An active round with no committed pick becomes `missed` after every eligible fixture has kicked off; it contributes a fixed zero-point synthetic choice and no team/venue usage. This lets `next open round` advance and keeps optimizer accounting feasible.

The server, not the browser, determines whether a pick can change. A replacement is allowed only when both the currently committed fixture and the proposed replacement fixture remain strictly pre-kickoff at transaction time. A background/read-time reconciliation performs the transitions above from the schedule source.

Changing a pre-kickoff pick replaces the current round's ledger entry and immediately invalidates/rebuilds remaining scenarios. Historical locked picks are never overwritten.

Migration behavior for current rows:

- Match a row to normalized fixture data by `match_id` and season.
- Finished rows with known points become `scored`; started/finished rows without a result become `locked`; upcoming rows become `committed` with normalized `kickoff_at`.
- A row whose fixture cannot be resolved becomes `legacy_unverified`. It remains visible in history, is excluded from new-season optimization, and produces a migration warning for manual review.
- Migration never invents kickoff times or deletes a row and is idempotent.

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

The default route opens the next open contest round. It shows the top candidates in an integrated comparison rather than isolated recommendation cards.

Selecting a candidate is a reversible preview. It updates:

- Immediate and season-aware ranking.
- Plain-language trade-off summary, such as “0.08 expected season points lower; preserves Manchester City for the September 26 contest round.”
- Scenario horizon.
- Season move tree.
- Affected-round annotations.

The commit action includes the team, opponent, kickoff, and consequence. Confirmation is inline rather than a generic modal. After commitment, the app shows the new remaining-season plan and the deadline for editing.

### 5.3 Week Navigation

The user can browse all active, completed, missed, and void contest rounds with:

- Previous and next controls.
- A direct contest-round picker that also displays associated league matchweek metadata.
- Left/right keyboard shortcuts when focus is not inside a form control.
- Clickable nodes in the season move tree.
- Mobile swipe as a progressive enhancement, never the only control.

The next open round is marked `Decision due`; later rounds are `Look ahead`; committed/locked/scored/missed/void rounds show their lifecycle state. Midweek-only league rounds remain visible in the complete fixture schedule but do not appear as decision rounds. Future rounds are preview-only. The user can inspect their current planned choice and alternatives but commits only the next open round, preventing contradictory out-of-order locks.

Navigation preserves the selected scenario per round during the browser session and never discards a committed pick.

### 5.4 Season Plan and History

The season view shows every contest round, the current optimal line, committed picks, team/venue usage, and constraint pressure. Selecting a round returns to its decision view. A complete schedule also shows ineligible midweek fixtures. The history view shows submitted team, opponent, result, points, override note, and whether the model recommendation was followed.

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
-> season context, next open round, source freshness, fixture_data_version,
   model version/state, prediction_version, rules version, pick-ledger version,
   local/season eligibility versions, scenario state,
   and commit eligibility
```

### Contest-Round Decision

```text
GET /api/seasons/{season}/rounds/{contest_round_id}
-> fixtures, league-matchweek metadata, round lifecycle, committed state,
   scenario_version, ranked scenario summaries, highlighted scenario ids,
   scenario-horizon series, move-tree summary, mode, assumed prior picks,
   prediction_version, eligibility_version, season_eligibility_version,
   as_of_utc, and valid_until_utc
```

### Scenario Detail

```text
GET /api/seasons/{season}/rounds/{contest_round_id}/scenarios/{scenario_id}
-> complete counterfactual plan, displaced/replacement picks, explanation fields
```

### Commit or Change Pick

```text
PUT /api/seasons/{season}/rounds/{contest_round_id}/pick
body: scenario_id, scenario_version, expected_pick_version,
      match_id, team, venue, notes, news_risk
-> validated pick lifecycle record, recalculated remaining plan, updated readiness
```

`scenario_version` is a stable hash of `season + contest_round_id + fixture_data_version + prediction_version + eligibility_version + season_eligibility_version + model_version + rules_version + pick_ledger_version + mode`. `expected_pick_version` provides optimistic concurrency for replacing a pre-kickoff pick and is `null` when creating one. The server validates every dependency version inside the same transaction as kickoff, eligibility, and ledger writes. Diagnostic responses return the component versions so an expired token is auditable.

On mismatch the server returns HTTP 409:

```text
{
  code: "SCENARIO_EXPIRED" | "PICK_VERSION_CONFLICT",
  message: string,
  latest_scenario_version: string,
  refresh_url: string,
  candidate_identity: {match_id, team} | null
}
```

The client refetches the round and reselects the same candidate only if it remains eligible; it never retries a commit automatically.

Commit validation has deterministic precedence inside one transaction:

1. Reject stale `expected_pick_version` as `PICK_VERSION_CONFLICT`.
2. Reconcile kickoff/status boundaries and reject stale fixture freshness when the matchday gate fails.
3. Reject an existing committed pick whose kickoff has arrived as `PICK_LOCKED`.
4. Reject a proposed replacement that is ineligible, cancelled, or at/after kickoff as `CANDIDATE_UNAVAILABLE`.
5. Reject mismatched `scenario_version` as `SCENARIO_EXPIRED`.
6. Reject any `mode: lookahead` scenario or contest round other than the current next-open round as `ROUND_NOT_COMMITTABLE`.
7. Require `scenario_id` to resolve exactly to the submitted `match_id`, team, and venue; otherwise reject as `SCENARIO_IDENTITY_MISMATCH`.
8. Revalidate contest constraints and feasibility, then write.

This ordering ensures a user sees the irreversible lock condition before a secondary scenario-version error, while simultaneous PUT requests resolve through `expected_pick_version`.

Expected client errors are structured and user-actionable: stale data, kickoff passed, ineligible fixture, pick-version conflict, constraint conflict, scenario expired, refresh in progress, and no feasible season plan.

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
- Pre-lock reschedule: retain only changes inside the same contest window; invalidate and reopen when the fixture moves to another weekend or leaves Saturday/Sunday. Recompute origin/destination rounds and preserve the prior candidate identity so the UI can explain the change.
- Post-lock postponement: keep the pick bound to its original contest round and pending until the schedule source supplies a new kickoff/result; exclude it from later candidate pools.
- Cancellation: pre-lock reopens the round; post-lock records a terminal zero-point void pick with no usage consumption and never auto-binds a later replay.
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
- Contest-round derivation across BST/GMT boundaries, split league matchweeks, midweek-only rounds, eligible-weekend-to-eligible-weekend moves, moves out of eligibility, postponements, pre/post-lock cancellations, void weekends, and missed weekends.
- Offline refresh never performs a network request.
- Live refresh never substitutes sample fixtures.
- Source freshness, atomic-write, last-known-good, season-mismatch, and partial-failure behavior.
- Optimizer feasibility with zero, one, and many locked picks.
- Counterfactual season cost, first displaced round, immediate rank, season-aware rank, immediate-only and season-only tie groups with ordinal ranks, unique canonical plans under equal objectives, decision versus conditional lookahead, local/season eligibility-version token changes at kickoff, prediction-version changes after probability regeneration without retraining, and cache invalidation.
- Every contest constraint after commitments and pick changes.
- Pick update before kickoff, simultaneous PUT conflicts, both-fixtures-pre-kickoff replacement validation and error precedence, rejection at/after kickoff, all allowed lifecycle transitions, pre-lock invalidation, post-lock postponement/cancellation, missed/void rounds, timezone conversion, reconciliation, and scoring.
- Database migration for new and existing SQLite files.
- API error schemas, lookahead/non-next-round commit rejection, scenario/body identity mismatch, and idempotent refresh behavior.
- Kickoff-driven scenario expiry in both active servers and sleep/restart lazy reconciliation, including lookahead invalidation when an earlier dependency round changes and `as_of_utc`/`valid_until_utc` behavior.

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

- Server-rendered and browser interaction tests for readiness, round navigation, preview, commit, change-before-kickoff, lock-after-kickoff, missed/void rounds, stale state, and failure recovery.
- Light, dark, and system theme behavior with persisted preference.
- Keyboard navigation, focus order, screen-reader labels, non-color chart summaries, and reduced motion.
- Desktop and mobile screenshots compared with the approved comp's structure/materials and responsive rules using normalized live or explicitly labeled synthetic eligible fixtures; comp fixture copy and non-normative metrics are never asserted.
- Mechanical Impeccable detector pass on changed templates/styles.
- Independent Impeccable finish review using approved comp and screenshots.

### 10.4 Release Gate

Release requires:

- All existing and new tests passing, including repair of the present 3 failures.
- Ruff passing.
- A successful 2026–27 live refresh with official fixtures and correct promoted clubs.
- A trained model whose training cutoff and version are visible in readiness.
- Feasible counterfactual scenarios for the next open contest round.
- Manual verification that a preview changes the chart/tree, a pre-kickoff pick can change, and an after-kickoff pick cannot.
- Desktop and mobile visual review in both themes.
- No hard-coded `2526` references in runtime paths or UI defaults except historical fixtures/tests explicitly labeled as past-season data.

## 11. Implementation Slices

The detailed implementation plan should preserve these dependency boundaries:

1. **Season and data readiness:** central season context, 2026–27 source refresh, freshness records, atomic artifacts, and repair of baseline tests.
2. **Committed optimizer state:** pick-ledger migration, locked-pick solver interface, exact counterfactual scenarios, explanations, caching, and backtest/replay validation.
3. **Decision APIs:** readiness, contest-round scenarios, detail, commit/change, refresh, and structured errors.
4. **Decision-desk UI:** approved Balanced Horizon composition, round browsing, preview/commit flow, scenario forecast horizon, move tree, readiness states, and light/dark themes.
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
- Future navigation: browse every contest round and the complete fixture schedule; commit only the next open contest round.
- Data policy: automatic verified refresh, visible freshness, last-known-good recovery, no silent sample fallback.
- Availability policy: no unvalidated automated injury/lineup feature in this release; manual news-risk note only.
- Validation policy: every model-affecting change requires sample tests and multi-season metric comparison.
