---
version: 1
slug: "prediction-optimizer-app-templates-dashboard-html"
primary_target: "src/epl_prediction_optimizer/app/templates/dashboard.html"
related_targets: ["src/epl_prediction_optimizer/app/main.py","src/epl_prediction_optimizer/app/static/styles.css"]
---

## Scope and mode

- Primary target: `src/epl_prediction_optimizer/app/templates/dashboard.html`
- Related targets: dashboard routes, optimizer scenario endpoints, theme tokens, and responsive behavior.
- Mode: Operate.

## Audience, job, and action

The owner uses the surface each matchweek to compare eligible teams, understand immediate strength versus season-aware value, preview the downstream plan, and commit the team actually submitted. The primary action is `Commit <team>`. The committed pick becomes a hard input to every later optimization.

## Content and constraints

- Current season and matchweek, freshness, model readiness, and last successful refresh.
- Ranked candidates with win probability, expected points, immediate rank, season-aware rank, opportunity cost, and first affected future week.
- Forecast ensemble horizon comparing best-now and best-season trajectories under uncertainty.
- Continuous season move tree showing the active line, displaced choices, and alternates.
- Same content and interaction hierarchy in light and dark modes.
- Saturday/Sunday only; one pick per week; every club at least once and at most twice; each club at home at most once and away at most once.
- Never use sportsbook odds, betting language, casino states, or fabricated model claims.

## Chosen direction

Preparation Room + Forecast Ensemble, Balanced Horizon composition. The interface resembles a rigorous analyst's preparation sheet: warm paper and ebonized framing in light mode, graphite drafting slate in dark mode, fine rules, engine notation, sparse marginalia, cobalt for the season-aware line, amber for immediate pressure, and muted red for displaced constraints.

Approved comp: `.impeccable/mocks/pick-optimizer/ensemble-split-light.webp`

Memorable moment: selecting a candidate makes the ensemble traces and branching season move tree reform together before the user commits.

## Comp implementation inventory

| Ingredient | Commitment | Medium |
| --- | --- | --- |
| Slim navigation and context rail | Ebonized vertical frame; active section marked by a cobalt rule | Semantic HTML/CSS |
| Integrated candidate comparison | One ruled analytical surface, not a card collection | Semantic table/list + CSS |
| Forecast ensemble horizon | Dozens of fine uncertainty paths plus labeled best-now and best-season paths; event markers at affected weeks | Authored responsive SVG driven by scenario JSON |
| Season move tree | Continuous main line with visible displaced and alternate branches across the lower band | Authored responsive SVG/HTML hybrid driven by optimized plans |
| Paper/slate material | Subtle visible tooth over the workspace; survives both themes without reducing contrast | Generated seamless raster texture, theme-adjusted in CSS |
| Sparse marginalia | Small equations and analyst notes that clarify decisions and never carry required meaning | Semantic text with restrained handwritten face/fallback |
| Commit action | Compact cobalt action in light mode and equivalent high-contrast action in dark mode; confirmation communicates downstream effect | Semantic button + inline confirmation flow |
| Theme control | Light, dark, and system preference; no capability differences | Semantic control + CSS custom properties + persisted preference |

## Component grammar

Corners are square to 6px. Hairline rules separate regions; shadows are limited to one shallow elevation where a selected layer must lift. Primary UI uses a narrow grotesk/neutral sans; figures and notation use tabular mono; a restrained serif may appear only in the decision heading. Controls use explicit labels, visible focus, and state conveyed by label/shape as well as color.

## Unresolved decisions

Exact scenario confidence method and the final wording for irreversible versus editable pick commitment will be resolved in implementation planning and tests.
