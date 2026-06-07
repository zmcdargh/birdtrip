# birdtrip

Plan birding trips from eBird data: given your life list and a region/season, find **where and when** to go to maximize your expected number of new species (lifers) — with rare/endemic specialties pulled in via a tunable rarity slider.

## How it works

The atom is a **detection probability** estimated from eBird complete checklists: the chance of seeing a species on one checklist at a given place in a given week of the year. Trips are scored by summing, over species *not* on your life list, the probability of finding each.

Two estimation ideas keep it honest:

- **Inter-year occupancy** separates "present in a typical year" from "detectable given present", so a single over-wintering vagrant (high pooled frequency, but seen in only one year) doesn't get recommended as if it were reliable.
- **Shrinkage** (Beta-Binomial, pooled toward the parent region) tames cells built on very few checklists, where a raw "1 of 1 = 100%" frequency would otherwise mislead. Real eBird data is dominated by such sparse cells.

The **rarity slider** (`alpha`) trades "most birds" against "most specialties". A species' irreplaceability weight is its attainability inside your region versus the rest of the world — gated on reliable presence so vagrants can't masquerade as specialties.

## Package layout

```
birdtrip/                core library (installable)
  taxonomy.py            name <-> species_code; resolve any taxon to its countable species
  lifelist.py            parse an eBird "Download My Data" export into a species-code life list
  precompute.py          EBD + Sampling Event Data -> per-(species, place, week) table
  recommend.py           rank destinations by rarity-weighted expected lifers
  summary.py             human-readable trip summary: expected lifers + likely birds
scripts/                 dev tools (not part of the library)
  generate_sample.py     write a synthetic EBD/SED pair for testing
  make_viz.py            Tufte-style charts of the precompute
tests/                   pytest suite
data/
  taxonomy/              eBird taxonomy + Clements checklist
  real/                  real EBD samples (e.g. the auk zerofill example)
  sample/                generated synthetic EBD/SED
```

## Quickstart

```bash
pip install -e .            # core
pip install -e ".[viz,dev]" # + charts + tests
pytest                      # run tests

# build the per-(species, place, week) table from an EBD + sampling file
python -m birdtrip.precompute --ebd data/real/zerofill-ex_ebd.txt \
    --sed data/real/zerofill-ex_sampling.txt --current-year 2012 --out data/precomputed_real.csv
```

```python
from birdtrip import Taxonomy, parse_life_list
tax = Taxonomy()
result = parse_life_list("MyEBirdData.csv", tax)   # your eBird export
print(result.summary())                            # species, dropped non-species taxa, unmatched
```

## Backend (storage + API)

The precomputed table is persisted in **SQLite** (`birdtrip/store.py`) and served by a
**FastAPI** app (`birdtrip/api.py`); a service layer (`birdtrip/service.py`) runs the
recommendation math over queried region/season slices. The store query methods return
DataFrames, so swapping to DuckDB/Parquet or Postgres at full-EBD scale touches only `store.py`.

```bash
pip install -e ".[api]"
python -m birdtrip.store --precomputed data/precomputed.csv --db data/birdtrip.sqlite
uvicorn birdtrip.api:app --reload          # open http://127.0.0.1:8000  (API docs at /docs)
```

Opening the root URL serves the **frontend** (`frontend/index.html`): a single-file vanilla-JS +
Leaflet map. The map drops one clickable marker per region that has data (at the centroid of its
hotspots); pick a region, set season/effort/the rarity slider, optionally upload your eBird life-list
CSV, and the planner pins the top recommended hotspots and lists each with its expected lifers and the
birds driving it. No build step — FastAPI serves it. (Map tiles/Leaflet load from a CDN, so the UI
needs internet at runtime even though everything else is local.)

Endpoints: `GET /regions` (selectable regions) · `POST /lifelist` (upload eBird CSV →
species codes) · `POST /recommend` (region + season + effort + α → ranked destinations with
the birds driving each) · `POST /summary` (one destination → expected lifers + likely birds).

Note: SQLite needs a normal local filesystem; some network/fuse mounts reject its file locking.

## Status

Estimation machinery, recommender, summary, taxonomy and life-list parsing are built and tested, validated on real eBird-format data. Parameter **calibration** (min-checklist thresholds, shrinkage strength, occupancy gate) awaits a multi-year regional EBD extract.

## Data & terms

eBird data are non-commercial (Cornell Lab of Ornithology Terms of Use). This tool serves only **derived** products (estimated frequencies, recommendations), never the raw dataset.
