#!/usr/bin/env python3
"""
Out-of-core precompute for the REAL eBird Basic Dataset, via DuckDB.

Streams the (multi-GB) EBD TSV from disk and aggregates in SQL. Output schema matches
the store's `cells` table. Two output modes:

  --out something.sqlite   -> writes the SQLite store DIRECTLY (DuckDB's sqlite writer).
                              No CSV, no pandas, no row-by-row inserts. Resolves species_code
                              in SQL and stores only trusted cells. This is the fast path for
                              a state-sized file (millions of rows). Point the app straight at it.
  --out something.csv      -> writes a CSV in the uppercase schema that birdtrip.store ingests
                              (kept for small datasets / the pandas path).

Design notes (mirror birdtrip/precompute.py):
  * No Sampling Event Data file required: the complete-checklist denominator is derived from the
    EBD's own distinct complete (ALL SPECIES REPORTED=1) approved checklist IDs. Pass --sed for
    exact denominators if you have the sampling file.
  * eBird 48-week year (use floor, not CAST — DuckDB CAST rounds), min-checklist trust threshold,
    Beta-Binomial shrinkage toward the state mean, effort-aware multi-year occupancy,
    p_lifer = occupancy * detect_given_present.
  * species_code is resolved via the eBird taxonomy: species->itself, issf/form/etc.->REPORT_AS
    (parent species), spuh/slash/hybrid->dropped. Subspecies roll UP, so a cell has one row per
    countable species (more correct than keying on raw scientific name).

Run locally (minutes on a state file):
    pip install duckdb
    python scripts/precompute_duckdb.py --ebd data/ebd_US-NY_relApr-2026.txt \
        --out data/birdtrip.sqlite --current-year 2026 --memory-limit 8GB --threads 8
"""
import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
MIN_CHECKLISTS = 5
MIN_CHECKLISTS_YEAR = 3
PRIOR_STRENGTH = 10.0     # for freq_shrunk
DGP_PRIOR = 5.0          # Beta prior strength for the detection-rate posterior
OCC_GATE = 0.5           # min occupancy for a species to earn a region-relative weight > 1
W_FLOOR = 1e-3           # floor on elsewhere-attainability (ratio denominator)
LOOKBACK_YEARS = 10
DEFAULT_TAX = ROOT / "data" / "taxonomy" / "eBird_taxonomy_v2025-4.csv"

WEEK = "((month(d) - 1) * 4 + least(3, CAST(floor((day(d) - 1) / 7.0) AS INTEGER)) + 1)"


def read_csv(path):  # the EBD / SED are tab-delimited
    return (f"read_csv('{path}', delim='\\t', header=true, quote='', all_varchar=true, "
            f"ignore_errors=true, max_line_size=20000000)")


def read_comma(path):  # the eBird taxonomy is comma-delimited (and quoted)
    return f"read_csv('{path}', delim=',', header=true, all_varchar=true, ignore_errors=true)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebd", required=True)
    ap.add_argument("--sed", default=None, help="optional Sampling Event Data file")
    ap.add_argument("--out", required=True, help="*.sqlite (direct store) or *.csv")
    ap.add_argument("--current-year", type=int, required=True)
    ap.add_argument("--taxonomy", default=str(DEFAULT_TAX))
    ap.add_argument("--keep-untrusted", action="store_true",
                    help="keep cells with < MIN_CHECKLISTS (default: drop them)")
    ap.add_argument("--parquet-only", action="store_true",
                    help="write only the Parquet (the app's serve format); skip the redundant SQLite copy")
    ap.add_argument("--memory-limit", default="4GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temp-dir", default=None,
                    help="where DuckDB spills intermediates (needs lots of free space for a big EBD); "
                         "default is alongside --out. Point at a drive with room.")
    a = ap.parse_args()
    min_year, max_year = a.current_year - LOOKBACK_YEARS, a.current_year
    is_sqlite = a.out.endswith((".sqlite", ".db"))

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{a.memory_limit}'; PRAGMA threads={a.threads};")
    tmp = a.temp_dir or str(Path(a.out).resolve().parent / "duckdb_tmp")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp}';")
    print(f"DuckDB spill dir: {tmp}  (ensure the drive holding it has plenty of free space)", flush=True)

    # taxonomy: name -> countable species_code, and species_code -> canonical names
    con.execute(f"""
    CREATE TEMP TABLE tax AS
      SELECT "SCI_NAME" AS sci_name,
        CASE WHEN "CATEGORY"='species' THEN "SPECIES_CODE"
             WHEN "REPORT_AS" <> '' THEN "REPORT_AS" ELSE NULL END AS species_code
      FROM {read_comma(a.taxonomy)} WHERE "SCI_NAME" <> '';
    CREATE TEMP TABLE spinfo AS
      SELECT "SPECIES_CODE" AS species_code, "PRIMARY_COM_NAME" AS common_name, "SCI_NAME" AS sci_name
      FROM {read_comma(a.taxonomy)} WHERE "CATEGORY"='species';
    """)

    # PHASE 1: one streaming scan of the EBD -> a compact, filtered Parquet on disk.
    # Writing a column-compressed Parquet (not a DuckDB temp table) keeps scratch use tiny,
    # and every later aggregation is a cheap COUNT(*) GROUP BY over it — no DISTINCT, no big spill.
    obs_pq = str(Path(tmp) / "_obs.parquet")
    print(f"scanning EBD -> {obs_pq} (the slow part)…", flush=True)
    d_ebd = 'TRY_CAST("OBSERVATION DATE" AS DATE)'
    con.execute(f"""
    COPY (
      SELECT t.species_code, x.state, x.state_code, x.county, x.locality, x.locid,
             x.lat, x.lon, x.sei, x.yr, x.week
      FROM (
        SELECT "SCIENTIFIC NAME" AS sci, "STATE" AS state, "STATE CODE" AS state_code,
               "COUNTY" AS county, "LOCALITY" AS locality, "LOCALITY ID" AS locid,
               TRY_CAST("LATITUDE" AS DOUBLE) AS lat, TRY_CAST("LONGITUDE" AS DOUBLE) AS lon,
               "SAMPLING EVENT IDENTIFIER" AS sei, CAST(year(d) AS INTEGER) AS yr, {WEEK} AS week
        FROM (SELECT *, {d_ebd} AS d FROM {read_csv(a.ebd)})
        WHERE "ALL SPECIES REPORTED"='1' AND "APPROVED"='1' AND d IS NOT NULL
          AND "LOCALITY TYPE"='H' AND year(d) BETWEEN {min_year} AND {max_year}
      ) x JOIN tax t ON t.sci_name = x.sci WHERE t.species_code IS NOT NULL
    ) TO '{obs_pq}' (FORMAT PARQUET);
    """)

    # checklist table: one row per checklist. From the SED if provided, else dedupe obs by sei
    # (a GROUP BY, far cheaper than a wide DISTINCT) — so COUNT(*) gives checklist counts.
    if a.sed:
        d_sed = 'TRY_CAST("OBSERVATION DATE" AS DATE)'
        con.execute(f"""CREATE TEMP TABLE chk AS SELECT DISTINCT "SAMPLING EVENT IDENTIFIER" AS sei,
          "LOCALITY ID" AS locid, "STATE" AS state, CAST(year(d) AS INTEGER) AS yr, {WEEK} AS week
          FROM (SELECT *, {d_sed} AS d FROM {read_csv(a.sed)})
          WHERE "ALL SPECIES REPORTED"='1' AND d IS NOT NULL AND "LOCALITY TYPE"='H'
            AND year(d) BETWEEN {min_year} AND {max_year};""")
    else:
        con.execute(f"""CREATE TEMP TABLE chk AS SELECT sei, any_value(locid) locid,
          any_value("week") "week", any_value(yr) yr, any_value(state) state
          FROM '{obs_pq}' GROUP BY sei;""")
    con.execute(f"""CREATE TEMP TABLE locmeta AS SELECT locid, any_value(locality) locality,
        any_value(state) state, any_value(state_code) state_code, any_value(county) county,
        avg(lat) latitude, avg(lon) longitude FROM '{obs_pq}' GROUP BY locid;""")

    print("aggregating…", flush=True)
    # COUNT(*) everywhere: a species is listed once per checklist, so detections == row count.
    # (Rare issf+species double-listing can nudge a count high; freq is clamped to <=1 below.)
    con.execute(f"""
    CREATE TEMP TABLE den       AS SELECT locid, week, COUNT(*) n_checklists FROM chk GROUP BY 1,2;
    CREATE TEMP TABLE den_y     AS SELECT locid, week, yr, COUNT(*) nchk FROM chk GROUP BY 1,2,3;
    CREATE TEMP TABLE state_den AS SELECT state, week, COUNT(*) s_chk FROM chk GROUP BY 1,2;
    CREATE TEMP TABLE det       AS SELECT locid, week, species_code, COUNT(*) n_detections FROM '{obs_pq}' GROUP BY 1,2,3;
    CREATE TEMP TABLE det_y     AS SELECT locid, week, yr, species_code, COUNT(*) nd FROM '{obs_pq}' GROUP BY 1,2,3,4;
    CREATE TEMP TABLE state_det AS SELECT state, week, species_code, COUNT(*) s_det FROM '{obs_pq}' GROUP BY 1,2,3;
    CREATE TEMP TABLE surveyed AS SELECT locid, week, COUNT(*) years_surveyed FROM den_y
           WHERE nchk >= {MIN_CHECKLISTS_YEAR} GROUP BY 1,2;
    CREATE TEMP TABLE present AS SELECT dy.locid, dy.week, dy.species_code,
           COUNT(*) years_present, SUM(dy.nd) det_present, SUM(d.nchk) chk_present
           FROM det_y dy JOIN den_y d USING (locid, week, yr)
           WHERE d.nchk >= {MIN_CHECKLISTS_YEAR} GROUP BY 1,2,3;
    """)

    trusted_filter = "" if a.keep_untrusted else f"WHERE den.n_checklists >= {MIN_CHECKLISTS}"
    # core projection. ALL static per-cell work is done here (not per request): the Beta-shrunk
    # detection rate + interval params (det_a/det_b), and p_lifer_1. The region-relative weight w
    # is added afterwards (it needs cross-state aggregates). At serve time the app only filters +
    # does light scoring, so memory stays flat and queries are fast at US scale.
    C = DGP_PRIOR
    core = f"""
      SELECT lm.state, lm.state_code, lm.county, lm.locality, det.locid AS locality_id,
             lm.latitude, lm.longitude, det.week, sn.sci_name, sn.common_name, det.species_code,
             den.n_checklists, least(det.n_detections, den.n_checklists) AS n_detections,
             least(1.0, det.n_detections::DOUBLE / den.n_checklists) AS freq_raw,
             least(1.0, (det.n_detections + (sd.s_det::DOUBLE/sden.s_chk) * {PRIOR_STRENGTH})
                 / (den.n_checklists + {PRIOR_STRENGTH})) AS freq_shrunk,
             COALESCE(sv.years_surveyed,0) AS years_surveyed,
             COALESCE(p.years_present,0)  AS years_present,
             CASE WHEN COALESCE(sv.years_surveyed,0)>0
                  THEN COALESCE(p.years_present,0)::DOUBLE/sv.years_surveyed ELSE 0 END AS occupancy,
             -- Beta posterior on the per-checklist detection rate, prior = regional rate (s_det/s_chk)
             (COALESCE(p.det_present,0) + {C} * CASE WHEN COALESCE(sden.s_chk,0)>0 THEN sd.s_det::DOUBLE/sden.s_chk ELSE 0 END) AS det_a,
             (greatest(0, COALESCE(p.chk_present,0)-COALESCE(p.det_present,0)) + {C} * (1 - CASE WHEN COALESCE(sden.s_chk,0)>0 THEN sd.s_det::DOUBLE/sden.s_chk ELSE 0 END)) AS det_b,
             den.n_checklists AS _nchk_keep,
             CASE WHEN den.n_checklists >= {MIN_CHECKLISTS} THEN 1 ELSE 0 END AS trusted
      FROM det
      JOIN den USING (locid, week)
      JOIN locmeta lm USING (locid)
      JOIN spinfo sn USING (species_code)
      LEFT JOIN state_det sd ON sd.state=lm.state AND sd.week=det.week AND sd.species_code=det.species_code
      LEFT JOIN state_den sden ON sden.state=lm.state AND sden.week=det.week
      LEFT JOIN surveyed sv USING (locid, week)
      LEFT JOIN present p USING (locid, week, species_code)
      {trusted_filter}
    """
    # derive shrunk detection rate + p_lifer from det_a/det_b, then the region-relative weight w
    con.execute(f"CREATE TEMP TABLE cells0 AS SELECT *, least(1.0, det_a/(det_a+det_b)) AS detect_given_present, "
                f"occupancy * least(1.0, det_a/(det_a+det_b)) AS p_lifer_1 FROM ({core});")
    con.execute("CREATE TEMP TABLE sb AS SELECT state, species_code, MAX(p_lifer_1) f, MAX(occupancy) occ "
                "FROM cells0 GROUP BY 1,2;")
    con.execute(f"""CREATE TEMP TABLE wtab AS SELECT a.state, a.species_code,
        CASE WHEN a.occ < {OCC_GATE} THEN 1.0
             ELSE greatest(1.0, a.f / greatest(
                 COALESCE((SELECT MAX(b.f) FROM sb b WHERE b.species_code=a.species_code AND b.state<>a.state), 0.0),
                 {W_FLOOR})) END AS w
        FROM sb a;""")
    final = ("SELECT c.* EXCLUDE (_nchk_keep), wtab.w FROM cells0 c "
             "JOIN wtab USING (state, species_code)")

    if is_sqlite:
        con.execute(f"CREATE TEMP TABLE cells_final AS {final};")
        # Parquet sidecar — columnar; this is what the app SCORES against at serve time
        # (DuckDB GROUP BY over Parquet is ~100x faster than scanning SQLite). Written straight
        # from DuckDB's own data, so it's fast (no SQLite re-read).
        pq = a.out.rsplit(".", 1)[0] + ".parquet"
        print(f"writing Parquet store -> {pq}…", flush=True)
        con.execute(f"COPY cells_final TO '{pq}' (FORMAT PARQUET);")
        n = con.execute("SELECT COUNT(*) FROM cells_final").fetchone()[0]
        if a.parquet_only:
            con.close()
        else:
            print(f"writing SQLite store -> {a.out}…", flush=True)
            Path(a.out).unlink(missing_ok=True)
            con.execute("INSTALL sqlite; LOAD sqlite;")
            con.execute(f"ATTACH '{a.out}' AS s (TYPE SQLITE);")
            con.execute("CREATE TABLE s.cells AS SELECT * FROM cells_final;")
            con.execute("DETACH s;")
            con.close()
            import sqlite3
            sc = sqlite3.connect(a.out)
            sc.executescript("""CREATE INDEX idx_region ON cells(state, week);
                                CREATE INDEX idx_locality ON cells(locality_id, week);
                                CREATE INDEX idx_species ON cells(species_code);
                                CREATE INDEX idx_week ON cells(week);""")
            sc.commit(); sc.close()
    else:
        print(f"writing CSV -> {a.out}…", flush=True)
        up = f"""SELECT state AS "STATE", state_code AS "STATE CODE", county AS "COUNTY",
            locality AS "LOCALITY", locality_id AS "LOCALITY ID", latitude, longitude, week,
            sci_name AS "SCIENTIFIC NAME", common_name AS "COMMON NAME", n_checklists, n_detections,
            freq_raw, freq_shrunk, years_surveyed, years_present, occupancy,
            detect_given_present, p_lifer_1, det_a, det_b, w, trusted FROM ({final})"""
        con.execute(f"COPY ({up}) TO '{a.out}' (HEADER, DELIMITER ',');")
        n = con.execute(f"SELECT COUNT(*) FROM read_csv('{a.out}', delim=',', header=true)").fetchone()[0]
    Path(obs_pq).unlink(missing_ok=True)   # remove the intermediate scan file
    print(f"done: {n:,} cells -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
