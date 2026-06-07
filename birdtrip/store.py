"""
Storage layer: the precomputed (species, place, week) table, persisted in SQLite
and queried by region/season.

SQLite is the dev/default backend — a single file, zero setup, indexed for the
two access patterns the API needs: "give me all cells in this region for these
weeks" and "best attainability of each species outside this region" (for the
region-vs-elsewhere rarity weight). The query methods return pandas DataFrames,
so the scoring code is backend-agnostic; swapping in DuckDB/Parquet or Postgres
later means reimplementing this class, nothing downstream.
"""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path
import pandas as pd

from .taxonomy import Taxonomy

# precomputed-CSV column  ->  sqlite column
COLMAP = {
    "STATE": "state", "STATE CODE": "state_code", "COUNTY": "county",
    "LOCALITY": "locality", "LOCALITY ID": "locality_id",
    "latitude": "latitude", "longitude": "longitude", "week": "week",
    "SCIENTIFIC NAME": "sci_name", "COMMON NAME": "common_name",
    "n_checklists": "n_checklists", "n_detections": "n_detections",
    "freq_raw": "freq_raw", "freq_shrunk": "freq_shrunk",
    "years_surveyed": "years_surveyed", "years_present": "years_present",
    "occupancy": "occupancy", "detect_given_present": "detect_given_present",
    "p_lifer_1": "p_lifer_1", "trusted": "trusted",
}


def build_store(precomputed_csv: str | Path, db_path: str | Path,
                taxonomy: Taxonomy | None = None, verbose: bool = True) -> "Store":
    """(Re)build the SQLite store from a precomputed CSV, attaching species_code."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    t0 = time.time()
    tax = taxonomy or Taxonomy()

    log(f"[1/4] reading {precomputed_csv} …")
    t = time.time()
    df = pd.read_csv(precomputed_csv)
    df = df.rename(columns=COLMAP)[list(COLMAP.values())]
    df["trusted"] = df["trusted"].astype(int)
    log(f"      {len(df):,} rows in {time.time()-t:.1f}s")

    # attach the countable species code. Resolve ONCE per distinct species (a few hundred),
    # not once per row (millions) — the row-wise version dominated build time on a state file.
    log("[2/4] resolving species codes …")
    t = time.time()
    uniq = df[["sci_name", "common_name"]].drop_duplicates().copy()
    uniq["species_code"] = [tax.resolve_to_species(tax.code_for(sci=s, common=c))
                            for s, c in zip(uniq["sci_name"], uniq["common_name"])]
    df = df.merge(uniq, on=["sci_name", "common_name"], how="left")
    log(f"      {len(uniq):,} distinct taxa in {time.time()-t:.1f}s")

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")  # safe: rebuilt from CSV

    log(f"[3/4] writing {len(df):,} rows to {db_path.name} …")
    t = time.time()
    CHUNK = 100_000
    df.iloc[:0].to_sql("cells", con, index=False, if_exists="replace")     # create empty table
    for i in range(0, len(df), CHUNK):
        df.iloc[i:i + CHUNK].to_sql("cells", con, index=False, if_exists="append")
        done = min(i + CHUNK, len(df))
        log(f"      {done:,}/{len(df):,} rows ({100*done/len(df):.0f}%, {time.time()-t:.0f}s)")

    log("[4/4] indexing …")
    t = time.time()
    con.executescript("""
        CREATE INDEX idx_region ON cells(state, week);
        CREATE INDEX idx_locality ON cells(locality_id, week);
        CREATE INDEX idx_species ON cells(species_code);
        CREATE INDEX idx_week ON cells(week);
    """)
    con.commit()
    con.close()
    log(f"      indexed in {time.time()-t:.1f}s")
    log(f"done in {time.time()-t0:.1f}s -> {db_path}")
    return Store(db_path)


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

    def _con(self):
        return sqlite3.connect(self.db_path)

    # --- region selection -----------------------------------------------------
    def regions(self, level: str = "state") -> pd.DataFrame:
        """Regions available for selection, each with a centroid (mean of its hotspots)
        and counts, so the map can drop one clickable marker per region with data.
        level = 'state' (US state / country) or 'county'."""
        group = "state" if level == "state" else "state, county"
        with self._con() as con:
            return pd.read_sql(
                f"""SELECT {group},
                           AVG(latitude)  AS latitude,
                           AVG(longitude) AS longitude,
                           COUNT(DISTINCT locality_id) AS n_hotspots,
                           COUNT(*) AS n_cells
                    FROM cells
                    WHERE latitude IS NOT NULL
                    GROUP BY {group} ORDER BY {group}""", con)

    @staticmethod
    def _where(state=None, county=None, weeks=None, negate=False):
        clauses, params = [], []
        if state:
            clauses.append("state = ?"); params.append(state)
        if county:
            clauses.append("county = ?"); params.append(county)
        region = " AND ".join(clauses) if clauses else "1=1"
        region = f"NOT ({region})" if negate and clauses else region
        if weeks:
            qs = ",".join("?" * len(weeks))
            region += f" AND week IN ({qs})"; params += list(weeks)
        return region, params

    def query_cells(self, state=None, states=None, county=None, weeks=None,
                    locality_id=None, locality_ids=None, trusted_only=True) -> pd.DataFrame:
        """Cells matching a region/season/locality. Filtering happens in SQL (indexed) so we
        transfer only what's needed — the recommender never loads the whole table into RAM."""
        where, params = [], []
        if states:
            where.append(f"state IN ({','.join('?' * len(states))})"); params += list(states)
        elif state:
            where.append("state = ?"); params.append(state)
        if county:
            where.append("county = ?"); params.append(county)
        if locality_id:
            where.append("locality_id = ?"); params.append(locality_id)
        if locality_ids:
            where.append(f"locality_id IN ({','.join('?' * len(locality_ids))})"); params += list(locality_ids)
        if weeks:
            where.append(f"week IN ({','.join('?' * len(weeks))})"); params += list(weeks)
        if trusted_only:
            where.append("trusted = 1")
        clause = " AND ".join(where) if where else "1=1"
        with self._con() as con:
            return pd.read_sql(f"SELECT * FROM cells WHERE {clause}", con, params=params)

    def species_best_by_state(self) -> pd.DataFrame:
        """Per (state, species) best attainability and best occupancy across the whole
        dataset. Used to compute region-relative rarity weights — including in a
        no-region (global) search, where each state is weighed against the others."""
        with self._con() as con:
            return pd.read_sql(
                "SELECT state, species_code, MAX(p_lifer_1) AS p, MAX(occupancy) AS occ "
                "FROM cells WHERE species_code IS NOT NULL GROUP BY state, species_code", con)

    def elsewhere_best_p(self, state=None, county=None) -> dict[str, float]:
        """Best p_lifer of each species OUTSIDE the selected region (rest of dataset).
        This is the denominator of the region-vs-elsewhere irreplaceability weight."""
        where, params = self._where(state, county, negate=True)
        with self._con() as con:
            rows = con.execute(
                f"SELECT species_code, MAX(p_lifer_1) FROM cells WHERE {where} "
                f"AND species_code IS NOT NULL GROUP BY species_code", params).fetchall()
        return {code: val for code, val in rows if code is not None}


if __name__ == "__main__":
    import argparse
    ROOT = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Build the SQLite store from a precomputed CSV.")
    ap.add_argument("--precomputed", default=str(ROOT / "data" / "precomputed.csv"))
    ap.add_argument("--db", default=str(ROOT / "data" / "birdtrip.sqlite"))
    a = ap.parse_args()
    s = build_store(a.precomputed, a.db)
    n = s._con().execute("SELECT COUNT(*) FROM cells").fetchone()[0]
    print(f"Built store at {a.db}: {n} cells, regions:")
    print(s.regions("county").to_string(index=False))
