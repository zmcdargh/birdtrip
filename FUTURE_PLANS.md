# birdtrip — future plans / backlog

Deferred features and known limitations, roughly in priority order.

## Flagged 2026-06 (US data live)
- **Single-state clustering** (likely a fluke — a later global search gave broad results). Related,
  expected behavior: with alpha>0, Hawaii dominates because the region-vs-elsewhere weight correctly
  flags its many endemics as maximally irreplaceable. That's *correct*, but a single endemic-rich
  region can swamp the national view — consider a per-state cap or diversity term for global searches.
- **Seasonal specialties ("temporally rare" — the TIME analog of spatial endemics).** NOT about overall
  rarity. These are birds that are *easy at a place in a narrow window but hard there the rest of the
  year* — e.g. Golden-winged & Cerulean Warblers at Harriman SP, NY in breeding season, otherwise very
  hard in NY. Idea: a TEMPORAL weight symmetric to the spatial w(s) — how concentrated a species'
  attainability is in week w vs. its year-round baseline at that place. The per-week occupancy/detection
  data already encodes it; surfacing it lets the planner say "late May is THE time here for this bird,"
  and feeds directly into the Big Year mode below (you must be there in that window or you miss it).
- **Big Year / Big Month itinerary mode.** Combine the greedy multi-stop itinerary with temporal
  constraints: some species require being at a specific place AND season ("be at X in mid-May for
  bird A, at Y in September for bird B"). Plan a route *over time* that maximizes species across the
  year, respecting these spatiotemporal windows. Likely a separate site mode with its own UI.

## Features
- **Per-hotspot score curves.** Plot each recommended hotspot's score across the 48 weeks,
  all curves on one axis, so the user can see *when* each spot peaks and compare seasonality
  at a glance. (Requested 2026-06; the recommender already computes per-(hotspot, week) scores,
  so the data is there — just needs an endpoint returning the weekly curve per hotspot + a chart.)
- **Target-species mechanism.** Let the user flag must-see birds (e.g. Snail Kite); the planner
  guarantees the best site/week for each target is included, separate from the expected-lifers
  optimization. The real answer to "I want to see *this* bird."
- **Greedy multi-stop itinerary.** Stitch several hotspots into one trip using submodular /
  marginal-gain selection (a species expected at stop A no longer counts at stop B). Includes
  travel-distance / time-budget constraints.

## Modeling
- **Checklist autocorrelation (v3).** `p_trip = 1-(1-d)^k` assumes the k checklists are
  independent; real repeat visits to a site/week are correlated, so the trip probability is
  over-optimistic at higher k (and saturates to ~100% too fast). Needs a correlation model and
  real data to calibrate.
- **Effort/observer-bias awareness.** Frequency conflates abundance with observer effort and
  intent (e.g. dedicated seawatches at Montauk inflate seabird rates). Consider surfacing the
  per-checklist rate alongside the trip probability, and/or normalizing by effort.
- **Species covariance in the range estimate.** The "x–y expected lifers" range uses a
  Poisson-binomial (independence). Real co-occurrence would tighten/loosen it.

## Hosting / production
- **SQL-pushdown for the recommender.** Currently the whole trusted-cells table is loaded into
  memory at startup (fast sliders, but RAM grows with dataset). For US-scale public hosting,
  push the per-request query into SQLite/DuckDB so memory stays flat on a small instance.
- **Keyed tile provider.** Swap the keyless CARTO basemap for a MapTiler/Mapbox key before any
  real traffic (CARTO's free tiles aren't meant for production load).
- **Dockerfile + deploy config** (Render/Fly) and a domain + HTTPS.
