# birdtrip

Plan birding trips from eBird data: given your life list and a region/season, find **where and
when** to go to maximize your expected number of new species (lifers) — with rare/local
specialties pulled in via a tunable rarity slider.

It answers three shapes of question over the full US eBird Basic Dataset:

- **Where should I go?** — rank hotspots in a region/season by expected lifers.
- **When should I go?** — omit the date and it sweeps the calendar for your best multi-day window.
- **Where's the best trip *anywhere*?** — omit the location and it searches the country for the
  top few distinct multi-day trips.

## How it works

The atom is a **detection probability** estimated from eBird complete checklists: the chance of
detecting a species on a visit of *t* hours at a given place in a given week of the year. It's
decomposed into two independent questions, estimated separately and then multiplied:

```
P(detect on a visit) = ψ        ×   (1 − e^(−λ·t))
                       ╰─ is it ╯     ╰─ if present, how fast ╯
                          there?         do you detect it?
```

- **ψ — occupancy** (is the species present that site/week in a typical year). Estimated with
  **empirical-Bayes shrinkage** (κ = 3) toward a regional prior, so a single over-wintering
  vagrant seen in one surveyed year doesn't get recommended as if it were reliable — while a
  genuinely recurring bird stays high.
- **λ — detection rate** per hour, given present (a time-to-detection rate). This is where
  **effort** (hours) enters, and it cleanly separates skulkers (λ ≈ 0.1/h) from conspicuous birds
  (λ ≈ 1/h), roughly orthogonally to rarity.

The **product** is then **recalibrated** with a monotone (isotonic) map fit on a held-out block,
so the displayed probability means what it says. Trips are scored by summing this probability over
species *not* on your life list.

A separate **rarity slider** (`alpha`) trades "most birds" against "most specialties" in the
*ranking*: a species' weight is its attainability inside your region versus the rest of the world,
gated on reliable presence so vagrants can't masquerade as specialties. It affects ordering, not
the calibrated probability.

The model and the held-out validation (calibration, detectability, range maps) are written up in
**[MODEL.md](MODEL.md)**.

## Package layout

```
birdtrip/                    core library (installable)
  taxonomy.py                name <-> species_code; resolve any taxon to its countable species
  lifelist.py                parse an eBird "Download My Data" export into a life list
  precompute.py              EBD -> per-(species, place, week) table (small-scale / reference)
  recommend.py               rank destinations by rarity-weighted expected lifers
  summary.py                 human-readable trip summary: expected lifers + likely birds
  itinerary.py               multi-day greedy planner (base camp + radius, over a soft life list)
  service.py                 query the store + run the recommend / itinerary / best-trips math
  store.py                   storage layer (DuckDB-over-Parquet at scale; SQLite for small data)
  api.py                     FastAPI app + endpoints; serves the frontend
  ask.py                     optional natural-language "fill the form" interface (off by default)
scripts/                     dev / data-pipeline tools (not part of the library)
  precompute_duckdb.py       build the full-US store (EBD -> Parquet) with EB κ=3 occupancy
  validate_holdout.py        temporal hold-out trials for calibration
  kappa_sweep.py             tune κ + fit the isotonic recalibration map on the shrunk product
  make_viz.py, range_maps*.py, lambda_*.py, ...   validation & range-map figures
frontend/index.html          single-file Leaflet + vanilla-JS UI (no build step)
tests/                       pytest suite
data/
  taxonomy/                  eBird taxonomy + Clements checklist (published reference data)
  precomputed.csv            small synthetic sample used by the tests
```

## Quickstart

```bash
pip install -e ".[api,viz,dev]"     # library + API + charts + tests
pytest                              # run the suite

# serve the small bundled sample (no EBD needed) and open the UI:
python -m birdtrip.store --precomputed data/precomputed.csv --db data/birdtrip.sqlite
uvicorn birdtrip.api:app --reload   # http://127.0.0.1:8000  (API docs at /docs)
```

Open the root URL for the **frontend**: a Leaflet map with one marker per region that has data.
Pick a spot, set season / effort / the rarity slider, optionally upload your eBird life-list CSV,
and it pins the top hotspots with the expected lifers and the birds driving each. (Map tiles load
from a CDN, so the UI needs internet at runtime.)

### Building the full-US store

The bundled sample is synthetic. For the real thing you need the [eBird Basic
Dataset](https://ebird.org/data/download) (free, requires an approved request; not redistributable
— see below). Then:

```bash
python scripts/precompute_duckdb.py --ebd data/ebd_US_relApr-2026.txt --current-year 2026 \
    --out data/birdtrip.sqlite --memory-limit 24GB --threads 4
python scripts/kappa_sweep.py --trials data/holdout_trials_NY --kappa-fig 3 \
    --save-recal-map data/birdtrip.recal.json
```

This writes `data/birdtrip.parquet` (the file the server actually queries) and the recal map.

## API

`GET /regions` · `POST /lifelist` (eBird CSV → species codes) · `POST /recommend` (region +
season + effort + α → ranked destinations) · `POST /summary` (one destination → expected lifers +
likely birds) · `POST /itinerary` (multi-day plan; omit the start date for best-window search) ·
`POST /best_trips` (top-N distinct trips anywhere) · `GET /config` + `POST /ask` (optional NL
interface — see below).

## Optional: natural-language search

If an LLM key is configured, an "Ask in plain English" box appears and the model simply **fills
the search form** ("warblers near the Hudson Valley in May") — it never touches the data. It's
**off by default**; enabling it and the key-safety/rate-limit setup are covered in
**[DEPLOY.md](DEPLOY.md)**.

## Deploying

The app is a read-only FastAPI service + static frontend backed by the Parquet store. Recommended
host is Fly.io with a persistent volume and the store fetched from object storage on first boot.
Full steps in **[DEPLOY.md](DEPLOY.md)**.

## Status

The estimation pipeline (EB occupancy, per-hour λ, recalibration), the recommender, the multi-day
itinerary planner, best-window / best-trips search, taxonomy + life-list parsing, and the frontend
are built and tested, with held-out calibration on real US eBird data (see MODEL.md). Deferred
backlog lives in [FUTURE_PLANS.md](FUTURE_PLANS.md).

## Data & terms

eBird data are provided by the Cornell Lab of Ornithology under a non-commercial
[Terms of Use](https://www.birds.cornell.edu/home/ebird-api-terms-of-use/). This project serves
only **derived** products (estimated frequencies, recommendations) and **never** redistributes the
raw dataset or any personal life-list export — both are gitignored. The bundled `data/taxonomy/`
files are eBird/Clements published reference checklists, distinct from the observation dataset.

## License

[MIT](LICENSE) — covers the source code only, not eBird data (see above).
