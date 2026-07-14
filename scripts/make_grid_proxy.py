#!/usr/bin/env python3
"""Build the best_trips grid-proxy sidecars from an existing store Parquet.

The location-agnostic best_trips shortlist needs a per-(grid-cell, week, species) max-occupancy
table plus cell centroids. Computing that live scans the whole 66M-row store (~2 min). Precomputing
it once collapses it to a few-MB pair of sidecars the server reads in well under a second.

Run against the served store (full or slim — both carry latitude/longitude/week/species_code/
occupancy/state/locality_id/trusted):

    python scripts/make_grid_proxy.py --store data/birdtrip_serve.parquet

Writes  <base>.grid.parquet  (gy, gx, week, species_code, mo)   and
        <base>.gridcen.parquet (gy, gx, lat, lon, state, nhot).

Upload BOTH next to the store in object storage, as birdtrip.grid.parquet / birdtrip.gridcen.parquet
(the server derives their names from the store path; docker-entrypoint.sh fetches them on boot).
GRID_DEG must match the serving default; it is baked in here, not a request parameter.
"""
import argparse
import os
import duckdb

GRID_DEG = 0.9   # must match find_best_trips' grid resolution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="path to the store parquet (e.g. birdtrip.parquet)")
    ap.add_argument("--grid-deg", type=float, default=GRID_DEG)
    a = ap.parse_args()
    store = a.store
    base = store[:-len(".parquet")] if store.endswith(".parquet") else store
    grid, cen = base + ".grid.parquet", base + ".gridcen.parquet"
    g = float(a.grid_deg)
    con = duckdb.connect()
    print(f"building grid proxy (grid_deg={g}) from {store} …", flush=True)
    con.execute(f"""COPY (
        SELECT floor(latitude/{g}) gy, floor(longitude/{g}) gx, week, species_code, MAX(occupancy) mo
        FROM '{store}'
        WHERE trusted=1 AND latitude IS NOT NULL AND species_code IS NOT NULL
        GROUP BY 1, 2, 3, 4
    ) TO '{grid}' (FORMAT PARQUET)""")
    con.execute(f"""COPY (
        SELECT floor(latitude/{g}) gy, floor(longitude/{g}) gx,
               AVG(latitude) lat, AVG(longitude) lon,
               any_value(state) state, COUNT(DISTINCT locality_id) nhot
        FROM '{store}'
        WHERE trusted=1 AND latitude IS NOT NULL
        GROUP BY 1, 2
    ) TO '{cen}' (FORMAT PARQUET)""")
    ncells = con.execute(f"SELECT COUNT(*) FROM '{cen}'").fetchone()[0]
    print(f"  {grid}  ({os.path.getsize(grid)/1e6:.1f} MB)", flush=True)
    print(f"  {cen}  ({os.path.getsize(cen)/1e6:.2f} MB, {ncells} grid cells)", flush=True)


if __name__ == "__main__":
    main()
