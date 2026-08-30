# Evaluating changes to the model and the plan

## Why contest points cannot decide anything

A season is 38 picks. Swapping one pick from a draw to a win moves the season
total by 2 points, and the standard deviation of a season's total under a fixed
model is roughly 8–10 points. A feature that changes the two-season total by 9
points has told you nothing.

This matters because it already happened here. `scripts/ablation_test.py`
compared feature sets on combined contest points over 2023-24 and 2024-25 and
recorded, in `ml/features.py`:

```
# "market_prob_home", "market_prob_draw", "market_prob_away",  # +odds: 154 combined
```

Market odds were shelved on a 9-point gap across 76 picks. On probability
metrics over the same data, market information is the single strongest signal
available.

The reverse shows up too. Fixing the fixture-feature bug improved 2024-25 log
loss from 1.0466 to 1.0103 and accuracy from 0.466 to 0.500 — a clearly better
model — while the season's contest points fell from 78 to 74.

## What to use instead

Rank candidates on **log loss**, **RPS**, and **Brier**, computed over every
match in several held-out seasons. All three live in
`epl_prediction_optimizer.scoring`.

- **Log loss** punishes confident mistakes hardest. Primary metric.
- **RPS** respects the ordering away < draw < home, so backing a home win and
  getting a draw costs less than getting a defeat. This is the metric closest
  to how a pick actually fails.
- **Brier** is the stable tie-breaker.
- **Expected calibration error** tells you whether a stated 60% happens 60% of
  the time. Watch it whenever probabilities are pooled or corrected.

Report contest points alongside. Never select on them.

## Scripts

```bash
# Replay seasons through the path production actually runs.
uv run scripts/backtest_production_path.py --seasons 2223 2324 2425 2526

# Sweep the model/market blend weight on real closing prices.
uv run scripts/backtest_market_blend.py --devig power

# Feature-group ablation (now reported on probability metrics).
uv run scripts/ablation_test.py
```

## The trap that made the backtest look better than production

`pipeline.backtest_season` predicts straight from the training frame. The live
`predict` step went through `build_fixture_features`, which filled every form,
streak, and head-to-head feature with zero. The two paths disagreed on 20 of 26
features, so the backtest was measuring a model that never ran in production.

`scripts/backtest_production_path.py` exists to stop that recurring: it drives
the same `ModelRun.predict` call the pipeline does, week by week, with only the
history available at that point. Any change to feature building must be
validated there, not only in `backtest_season`.

## Results on record

### Fixture-feature fix (`build_fixture_features` accumulating from history)

Replayed week by week through `ModelRun.predict`, training on earlier seasons
only. `neutral` is the previous behaviour, `history` the current one.

| Season | log loss (neutral → history) | accuracy | contest points |
|---|---|---|---|
| 2020-21 | 1.0839 → 1.0610 | 0.442 → 0.458 | 58 → 56 |
| 2021-22 | 1.0037 → 0.9907 | 0.508 → 0.539 | 63 → 69 |
| 2022-23 | 0.9909 → 0.9807 | 0.529 → 0.561 | 53 → 52 |
| 2023-24 | 0.9805 → 0.9599 | 0.537 → 0.571 | 73 → 80 |
| 2024-25 | 1.0466 → 1.0103 | 0.466 → 0.500 | 78 → 74 |
| 2025-26 | 1.0733 → 1.0363 | 0.432 → 0.484 | 56 → 67 |
| **mean** | **1.0298 → 1.0065** | | 381 → 398 |

Log loss and accuracy improve in all six seasons. Contest points move the wrong
way in three of them — which is the argument for not selecting on points.

### Market blend weight (power de-vig, 2018-19..2025-26)

| weight | log loss | RPS | Brier | accuracy | ECE | points |
|---|---|---|---|---|---|---|
| 0.00 | 0.9968 | 0.4178 | 0.5944 | 0.528 | 0.0510 | 536 |
| 0.40 | 0.9716 | 0.4015 | 0.5764 | 0.544 | 0.0495 | 550 |
| 0.70 | 0.9610 | 0.3949 | 0.5693 | 0.548 | 0.0490 | 557 |
| 0.85 | 0.9586 | 0.3935 | 0.5677 | 0.551 | 0.0491 | 554 |
| 0.90 | 0.9582 | 0.3933 | 0.5675 | 0.552 | 0.0476 | 555 |
| 1.00 | 0.9584 | 0.3933 | 0.5676 | 0.551 | 0.0488 | 552 |

Blending improves log loss in 8 of 8 seasons
(`uv run scripts/summarize_blend_sweep.py --weights 0.0 0.9`). The curve is
flat from 0.8 to 1.0; `MARKET_BLEND_WEIGHT` is set to 0.85, inside that
plateau, so a stale or mismatched price cannot carry a fixture on its own.

### De-vig method

Shin and power land in the same place — 0.9588 vs 0.9586 mean log loss at
weight 0.85 over the same eight seasons — so the choice is not load-bearing.
`power` stays the default. `proportional` is available but leaves
favourite-longshot bias in the distribution and should not be used.

The honest reading is that the market is simply a better forecaster than this
model, and the model's remaining job is to cover the fixtures the market has
not priced yet — most of the season, at any given moment.

### What this does not fix

The 2026-27 week 1 pick (Manchester United away at Hull, lost 2-0) came out the
same under every version tested. The market had Manchester United at 1.36,
about 73% before de-vig, more confident than the model's 51.6%. A better model
and a market blend both back that team. One pick is not evidence of a broken
model in either direction.

## Where the edge actually is

The contest is won by finishing first, so the question is not "is this model
good" but "what raises the chance of clearing the winning score". Measured on
2020-21..2025-26 with the same probabilities throughout:

| Lever | Effect on the season | Verdict |
|---|---|---|
| Season optimizer vs greedy weekly picks | **+4.5 expected points** (63.1 vs 58.6) | biggest single edge; keep |
| Market blend on priced rounds | +2.4 actual points/season, −3.9% log loss | keep, weight 0.90 |
| Fixture-feature fix | +2.8 actual points/season, −2.3% log loss | keep |
| The "use every team" rule | costs 0.1 expected points | it is a rule, and it is nearly free |
| Risk-seeking toward the winning score | 0 to +5% relative win probability | not an edge |
| Sharpening the model toward market confidence | helps 2 of 4 held-out seasons | not adopted |

### The optimizer is the strongest layer

`uv run scripts/backtest_optimizer_edge.py` compares the season program against
picking the best available team each week and against dropping the coverage
rule. Greedy loses 4.5 expected points a season — more than the market blend
gains.

The uncapped run in that script is a measurement, not an option: selecting
every team is a rule of the contest and the solver enforces it hard. It is
worth knowing that it costs only 0.1 expected points (63.1 vs 63.2), because
that says the optimizer is already scheduling the weak teams into their best
available slots rather than paying a real price for the requirement.

The one state where the rule cannot be satisfied is a round that passed with no
pick recorded: that round has left the candidate set and some team has nowhere
left to go. `optimize_picks` raises there rather than returning an ineligible
plan. The decision desk retries once with `allow_incomplete_coverage=True` so
it can still show the best remaining plan, and marks the result as
rule-breaking with the teams it could not place.

### Risk-seeking does not work here

`optimizer/tournament.py` plans against a target score using the exact
distribution of a plan's total, and `scripts/backtest_tournament_objective.py`
scores it. The chosen risk weight comes back 0 or 0.25 and win probability is
unchanged to four decimals.

The reason is structural. A pick pays 3, 1, or 0, so its variance is at most
about 1.8 and a 33-round season has a standard deviation near 7.1. Trading mean
for spread moves it to about 7.2 — nowhere near enough to bridge a 15-to-20
point gap. Over a shortened endgame horizon
(`--endgame`) it is worth at most 5% relative with twelve rounds left, and
exactly zero with four to eight, because the constraints leave no alternative
plan to switch to.

**Mean is the only lever that moves win probability.** Each extra expected point
is worth roughly 0.14 standard deviations, so model quality and the optimizer
are where the contest is won.

### The ceiling, and the honest gap

`uv run scripts/hindsight_ceiling.py` re-solves each season with perfect
foresight: the ceiling is about 3.0 points per round (99 in 2024-25), against
recorded winning scores of 78-86 and a shipped-plan score of 58-73. The winning
scores are reachable under the encoded rules, so the benchmark is not wrong —
the gap is real.

Stacking every improvement above moves the expected season total from roughly
63 to 68.

### Correction: the field is much weaker than the benchmarks suggested

The estimates above were made against `WINNER_BENCHMARKS` (78-86) with no
knowledge of who actually enters. Importing the 2025-26 contest sheet replaced
that guess with the real thing:

| | |
|---|---|
| entrants | 65 |
| winning score | 77 |
| median | 57 |
| mean | 53.3 |
| standard deviation | 17.1 |

The current plan expects 65.6 points with a standard deviation of 7.0 — above
the field median by more than eight points. Against that field the chance of
finishing outright first is about **6%**, not the fraction of a percent implied
by planning against a score of 86. Restricting the field to entrants who played
at least 25 rounds, which removes people who stopped filling the sheet in, does
not move it: 6.3% either way.

`WINNER_BENCHMARKS` and this sheet disagree about what wins — 86 against 77 —
so at least one of them describes a different contest or ruleset. The imported
field is the better authority because it is the field being entered, and
`win_probability_against_field` in `optimizer/tournament.py` uses it directly
rather than assuming a target.

This does not change which levers matter — mean is still the only one that
moves — but it changes the stakes. At a 6% baseline, an extra expected point is
worth having rather than a rounding error on a hopeless cause.

### Model confidence vs market confidence

`uv run scripts/calibrate_to_market.py` measures how much sharper the market is
than the model. The gap is consistent — mean top-outcome probability 0.49
against 0.55, in all eight seasons — but correcting it does not help: fitting an
exponent on four seasons and applying it to four held-out ones improves log loss
in two and worsens it in two, while pushing expected calibration error from
0.011 to 0.048 in the seasons where the model was already well calibrated.

The model is not overcautious; it is **less informed**. It is calibrated
correctly for what it knows, and the missing confidence is missing information
that only the price carries. That is the argument for taking the market at 0.90
where a price exists rather than trying to imitate it where one does not.

## Should you deviate from what the field picks?

The intuition is that taking the same team as everyone else cannot gain ground,
so a pool is won by differentiating. `scripts/analyze_field_strategy.py` tests
that against the 2025-26 contest sheet (49 entrants who played 25+ rounds).

It does not hold here.

| | follow rate | points |
|---|---|---|
| top 10 finishers | 42% | 73.4 |
| bottom 10 finishers | 24% | 48.1 |
| winner (77 pts) | 45% | 77 |

Correlation between an entrant's consensus-follow rate and their final score is
**+0.475**. Contrarians finished lower, not higher.

Two things explain it. The field is not concentrated enough for crowding to
matter — a mean of 35% of entrants on the most-taken team, peaking at 63%, so
differentiation happens on its own. And the contest rules already force
variety: every team once, at most twice, once per venue. The deliberate
anti-consensus play that works in a survivor pool at 70%+ concentration has no
equivalent here.

Read the correlation carefully: it is not evidence that following the crowd
*causes* a good score. Strong entrants and the crowd both gravitate to strong
favourites. The finding is narrower and sufficient — there is no measurable
premium for being contrarian, so paying expected points to avoid the field is
not supported.

2026-27 week 1 is a clean illustration: 56% of the field took Manchester
United, and 76% of all entrants scored zero when Hull won. A consensus loss
costs almost nothing in the standings.

