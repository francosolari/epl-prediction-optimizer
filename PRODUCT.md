# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The primary user is the owner of the app, making one Premier League contest pick each matchweek. They need to make a confident weekly decision quickly while preserving strong options for the rest of the season.

## Product Purpose

The EPL Pick Optimizer turns match probabilities and contest constraints into an explainable season strategy. Success means the user can compare the best eligible picks for the current week, understand how each choice changes future expected points, commit the team they actually submit, and continue from a re-optimized plan.

## Positioning

Unlike a match-prediction dashboard that ranks this week's favorites in isolation, the product exposes the opportunity cost of every weekly choice across the remaining season. It distinguishes the best immediate pick from the best season-aware pick and makes the downstream trade-off legible before commitment.

## Operating Context

The product is used before the user's weekly submission deadline and revisited after results are known. The recurring workflow is: refresh live Premier League data, review the current matchweek's ranked choices, inspect future implications, commit the actual submitted pick, track the result, and re-optimize the remaining weeks. The user also needs to look ahead across future matchweeks and review team and venue usage.

## Capabilities and Constraints

- Current target season: Premier League 2026–27 (`2627`). The active season must be derived centrally rather than hard-coded across routes and templates.
- Eligible candidates play on Saturday or Sunday.
- Exactly one team is selected per contest matchweek.
- Every Premier League team must be selected at least once during the season.
- A team can be selected at most twice: no more than once at home and once away.
- Committed user picks are durable constraints for subsequent optimization, not presentation-only annotations.
- The interface must compare a choice's immediate expected points with its effect on the best achievable remaining-season plan.
- Light and dark themes must offer the same information, hierarchy, and interaction capability.
- Live-data freshness, model readiness, and calculation failures must be visible and actionable in the product.
- Existing Python, FastAPI, Jinja, SQLite, pandas, scikit-learn, and PuLP architecture remains the implementation base.

## Brand Commitments

- The product must feel like a rigorous data-analysis and prediction-modeling workspace, never a betting or sportsbook product.
- The chosen durable visual world is an analyst's preparation room: season decisions read as lines in an evolving plan, supported by an ensemble forecast horizon that makes uncertainty and downstream opportunity cost visible.

## Evidence on Hand

- Historical match and ClubElo data, trained-model artifacts, probability exports, and prior-season backtests exist under `data/`, but the current artifacts were last generated in May 2026 and are not valid evidence of 2026–27 readiness.
- The existing optimizer and contest rules live in `src/epl_prediction_optimizer/optimizer/`.
- Actual submitted picks and results are persisted in SQLite by `src/epl_prediction_optimizer/storage/database.py`.
- The existing FastAPI/Jinja interface provides operational functionality but is not the visual authority for this redesign.
- No testimonials, competitive claims, or externally validated performance claims are available and none should be fabricated.

## Product Principles

- Lead with the weekly decision; keep pipeline mechanics secondary.
- Explain opportunity cost, not only probability.
- Treat a committed pick as a consequential planning action with a clear before-and-after state.
- Keep the season plan inspectable so the optimizer never feels like a black box.
- Fail visibly and recoverably when data is stale, incomplete, or unavailable.

## Accessibility & Inclusion

The core decision flow must be fully keyboard operable, preserve visible focus, avoid color-only meaning, respect reduced-motion preferences, and remain usable on desktop and mobile web.
