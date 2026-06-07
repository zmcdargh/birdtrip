"""
Temporal hold-out validation of the detection model.

Train on the early years, hold out the last N years, then test the model's PER-CHECKLIST detection
prediction on the held-out checklists. For each test checklist (with its real duration) and each
candidate species at that hotspot/week (= a species the TRAINING data knew about there), the model
predicts P(detect on this checklist) = occupancy_train * P(detect | present, duration), and we score
it against whether the species was actually on the list.

Two models are compared head-to-head on the SAME held-out data:
  * lambda (hours)    : occupancy * (1 - exp(-lambda_train * T))           — uses the test list's duration
  * duration-blind    : occupancy * detect_given_present_train             — the old per-checklist rate
If modelling duration genuinely helps, the lambda model has lower Brier / log-loss out of sample.

Metrics: Brier score, log-loss (lower=better), a reliability/calibration table, and a coarse
occupancy calibration. All aggregated in DuckDB so it runs out-of-core at US scale.

Usage:
  python scripts/validate_holdout.py --ebd data/ebd_US_relApr-2026.txt.gz --current-year 2026 \
      --holdout-years 3 --temp-dir data/duckdb_tmp --memory-limit 24GB --threads 4
"""
import argparse
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precompute_duckdb as pc   # reuse read_csv/read_comma, WEEK, DUR_BUCKETS, _dur_hist_sql, _lambda_sql, constants

EPS = 1e-6


def bucket_case(col="dur"):
    parts = []
    for i, (lo, hi) in enumerate(pc.DUR_BUCKETS, 1):
        cond = f"{col}>{lo} AND {col}<={hi}" if i > 1 else f"{col}<={hi}"
        parts.append(f"WHEN {cond} THEN {i}")
    return "CASE " + " ".join(parts) + " END"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebd", required=True)
    ap.add_argument("--current-year", type=int, required=True)
    ap.add_argument("--holdout-years", type=int, default=3, help="number of most-recent years to test on")
    ap.add_argument("--taxonomy", default=str(pc.DEFAULT_TAX))
    ap.add_argument("--min-checklists", type=int, default=pc.MIN_CHECKLISTS)
    ap.add_argument("--memory-limit", default="4GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temp-dir", default=None)
    a = ap.parse_args()

    test_lo = a.current_year - a.holdout_years + 1
    train_max = test_lo - 1
    train_min = a.current_year - pc.LOOKBACK_YEARS
    MINY, MINC, C, PS = pc.MIN_CHECKLISTS_YEAR, a.min_checklists, pc.DGP_PRIOR, pc.PRIOR_STRENGTH
    print(f"train years {train_min}-{train_max}   |   test years {test_lo}-{a.current_year}", flush=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{a.memory_limit}'; PRAGMA threads={a.threads};")
    tmp = a.temp_dir or "/tmp"
    Path(tmp).mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp}';")

    con.execute(f"""
    CREATE TEMP TABLE tax AS SELECT "SCI_NAME" AS sci_name,
      CASE WHEN "CATEGORY"='species' THEN "SPECIES_CODE"
           WHEN "REPORT_AS" <> '' THEN "REPORT_AS" ELSE NULL END AS species_code
      FROM {pc.read_comma(a.taxonomy)} WHERE "SCI_NAME" <> '';""")

    obs_pq = str(Path(tmp) / "_obs_val.parquet")
    d = 'TRY_CAST("OBSERVATION DATE" AS DATE)'
    print("scanning EBD (train+test years)…", flush=True)
    con.execute(f"""
    COPY (
      SELECT t.species_code, x.state, x.locid, x.sei, x.yr, x.week, x.dur
      FROM (
        SELECT "SCIENTIFIC NAME" sci, "STATE" state, "LOCALITY ID" locid,
               "SAMPLING EVENT IDENTIFIER" sei, CAST(year(d) AS INTEGER) yr, {pc.WEEK} "week",
               TRY_CAST("DURATION MINUTES" AS DOUBLE) dur
        FROM (SELECT *, {d} AS d FROM {pc.read_csv(a.ebd)})
        WHERE "ALL SPECIES REPORTED"='1' AND "APPROVED"='1' AND d IS NOT NULL
          AND "LOCALITY TYPE"='H' AND year(d) BETWEEN {train_min} AND {a.current_year}
          AND "PROTOCOL NAME" IN ('Traveling','Stationary')
      ) x JOIN tax t ON t.sci_name = x.sci WHERE t.species_code IS NOT NULL
    ) TO '{obs_pq}' (FORMAT PARQUET);""")

    # one row per checklist, tagged train/test
    con.execute(f"""CREATE TEMP TABLE chk_all AS SELECT sei, any_value(locid) locid,
      any_value("week") "week", any_value(yr) yr, any_value(state) state, any_value(dur) dur
      FROM '{obs_pq}' GROUP BY sei;""")
    con.execute(f"CREATE TEMP TABLE chk AS SELECT * FROM chk_all WHERE yr <= {train_max};")     # TRAIN
    con.execute(f"CREATE TEMP TABLE chk_te AS SELECT * FROM chk_all WHERE yr >= {test_lo};")     # TEST

    # ---- TRAIN model (mirrors precompute, restricted to train years) ----
    print("fitting training model…", flush=True)
    con.execute(f"""
    CREATE TEMP TABLE den AS SELECT locid,"week",COUNT(*) n_checklists FROM chk GROUP BY 1,2;
    CREATE TEMP TABLE den_y AS SELECT locid,"week",yr,COUNT(*) nchk FROM chk GROUP BY 1,2,3;
    CREATE TEMP TABLE state_den AS SELECT state,"week",COUNT(*) s_chk FROM chk GROUP BY 1,2;
    CREATE TEMP TABLE det AS SELECT locid,"week",species_code,COUNT(*) n_det
        FROM '{obs_pq}' WHERE yr <= {train_max} GROUP BY 1,2,3;
    CREATE TEMP TABLE det_y AS SELECT locid,"week",yr,species_code,COUNT(*) nd
        FROM '{obs_pq}' WHERE yr <= {train_max} GROUP BY 1,2,3,4;
    CREATE TEMP TABLE state_det AS SELECT state,"week",species_code,COUNT(*) s_det
        FROM '{obs_pq}' WHERE yr <= {train_max} GROUP BY 1,2,3;
    CREATE TEMP TABLE surveyed AS SELECT locid,"week",COUNT(*) years_surveyed FROM den_y
        WHERE nchk >= {MINY} GROUP BY 1,2;
    CREATE TEMP TABLE present AS SELECT dy.locid, dy."week", dy.species_code,
        COUNT(*) years_present, SUM(dy.nd) det_present, SUM(d.nchk) chk_present
        FROM det_y dy JOIN den_y d USING (locid,"week",yr) WHERE d.nchk >= {MINY} GROUP BY 1,2,3;
    CREATE TEMP TABLE dur_h AS {pc._dur_hist_sql()};
    CREATE TEMP TABLE locst AS SELECT locid, any_value(state) state FROM chk_all GROUP BY locid;
    """)
    core = f"""
      SELECT det.locid, det."week", det.species_code, den.n_checklists,
             COALESCE(sv.years_surveyed,0) AS years_surveyed, COALESCE(p.years_present,0) AS years_present,
             CASE WHEN COALESCE(sv.years_surveyed,0)>0
                  THEN COALESCE(p.years_present,0)::DOUBLE/sv.years_surveyed ELSE 0 END AS occupancy,
             (COALESCE(p.det_present,0) + {C}*CASE WHEN COALESCE(sden.s_chk,0)>0 THEN sd.s_det::DOUBLE/sden.s_chk ELSE 0 END) AS det_a,
             (greatest(0, COALESCE(p.chk_present,0)-COALESCE(p.det_present,0)) + {C}*(1-CASE WHEN COALESCE(sden.s_chk,0)>0 THEN sd.s_det::DOUBLE/sden.s_chk ELSE 0 END)) AS det_b
      FROM det JOIN den USING (locid,"week")
      JOIN locst ls USING (locid)
      LEFT JOIN state_det sd ON sd.state=ls.state AND sd."week"=det."week" AND sd.species_code=det.species_code
      LEFT JOIN state_den sden ON sden.state=ls.state AND sden."week"=det."week"
      LEFT JOIN surveyed sv ON sv.locid=det.locid AND sv."week"=det."week"
      LEFT JOIN present p ON p.locid=det.locid AND p."week"=det."week" AND p.species_code=det.species_code
      WHERE den.n_checklists >= {MINC}
    """
    con.execute(f"""CREATE TEMP TABLE cells0 AS
        SELECT *, least(1.0, det_a/(det_a+det_b)) AS detect_given_present FROM ({core});""")
    nb = len(pc.DUR_BUCKETS)
    hist = ", ".join(["dh.mean_dur_min", "dh.n_dur"] + [f"dh.n{i}, dh.t{i}" for i in range(1, nb + 1)])
    con.execute(f"""CREATE TEMP TABLE cellsh AS SELECT c.*, {hist}
        FROM cells0 c LEFT JOIN dur_h dh ON dh.locid=c.locid AND dh."week"=c."week";""")
    con.execute(f"CREATE TEMP TABLE model AS {pc._lambda_sql('cellsh')};")

    # ---- TEST aggregates ----
    print("aggregating held-out years…", flush=True)
    bc = bucket_case("dur")
    con.execute(f"""CREATE TEMP TABLE test_cb AS
        SELECT locid,"week",{bc} AS bk, COUNT(*) n_chk, avg(dur) tb
        FROM chk_te WHERE dur>0 AND dur<=600 GROUP BY locid,"week",{bc};""")
    con.execute(f"""CREATE TEMP TABLE test_det AS
        SELECT locid,"week",species_code,{bc} AS bk, COUNT(*) n_det
        FROM '{obs_pq}' WHERE yr >= {test_lo} AND dur>0 AND dur<=600 GROUP BY locid,"week",species_code,{bc};""")
    # cross every trained (cell, species) with that cell's test duration buckets; missing detections = 0
    con.execute(f"""CREATE TEMP TABLE ev AS
        SELECT m.occupancy AS occ, m.detect_given_present AS dgp, m.lambda_hr AS lam,
               tcb.tb/60.0 AS th, tcb.n_chk AS n_chk, COALESCE(td.n_det,0) AS n_det
        FROM model m JOIN test_cb tcb ON tcb.locid=m.locid AND tcb."week"=m."week"
        LEFT JOIN test_det td ON td.locid=m.locid AND td."week"=m."week"
             AND td.species_code=m.species_code AND td.bk=tcb.bk
        WHERE m.lambda_hr IS NOT NULL;""")

    # ---- metrics ----
    pl = f"least(1-{EPS}, greatest({EPS}, occ*(1-exp(-lam*th))))"
    pb = f"least(1-{EPS}, greatest({EPS}, occ*dgp))"
    agg = con.execute(f"""SELECT SUM(n_chk) N, SUM(n_det) D,
        SUM(n_det*pow(1-{pl},2)+(n_chk-n_det)*pow({pl},2)) bl,
        SUM(n_det*pow(1-{pb},2)+(n_chk-n_det)*pow({pb},2)) bb,
        -SUM(n_det*ln({pl})+(n_chk-n_det)*ln(1-{pl})) ll,
        -SUM(n_det*ln({pb})+(n_chk-n_det)*ln(1-{pb})) lb
        FROM ev""").fetchone()
    N, D = agg[0] or 0, agg[1] or 0
    if not N:
        print("no held-out trials — check year range / data"); return
    base_rate = D / N
    brier_l, brier_b = agg[2] / N, agg[3] / N
    ll_l, ll_b = agg[4] / N, agg[5] / N
    # climatology reference: predict the global base rate for everything
    brier_clim = base_rate * (1 - base_rate)

    print("\n" + "=" * 64)
    print(f"  HELD-OUT TRIALS: {int(N):,} (checklist x candidate-species)   detections: {int(D):,}  ({base_rate:.3%})")
    print("=" * 64)
    print(f"  {'model':<22}{'Brier':>10}{'log-loss':>12}{'Brier skill':>14}")
    print(f"  {'climatology (base)':<22}{brier_clim:>10.4f}{'—':>12}{0.0:>13.1%}")
    print(f"  {'duration-blind':<22}{brier_b:>10.4f}{ll_b:>12.4f}{1-brier_b/brier_clim:>13.1%}")
    print(f"  {'lambda (hours)':<22}{brier_l:>10.4f}{ll_l:>12.4f}{1-brier_l/brier_clim:>13.1%}")
    print(f"\n  lambda vs duration-blind:  Brier {(brier_b-brier_l)/brier_b:+.2%}   "
          f"log-loss {(ll_b-ll_l)/ll_b:+.2%}   (positive = lambda better)")

    # reliability / calibration for the lambda model
    cal = con.execute(f"""SELECT floor(least(0.999,{pl})*10) AS bin, SUM(n_chk) n, SUM(n_det) d,
        SUM({pl}*n_chk)/SUM(n_chk) pred FROM ev GROUP BY 1 ORDER BY 1""").fetchdf()
    cal["observed"] = cal["d"] / cal["n"]
    print("\n  calibration (lambda model): predicted vs observed detection rate")
    print(f"  {'pred-bin':<10}{'n_trials':>12}{'pred':>9}{'observed':>10}")
    for r in cal.itertuples():
        print(f"  {f'{int(r.bin)*10}-{int(r.bin)*10+10}%':<10}{int(r.n):>12,}{r.pred:>9.3f}{r.observed:>10.3f}")
    con.close()
    Path(obs_pq).unlink(missing_ok=True)
    print("\ndone.")


if __name__ == "__main__":
    main()
