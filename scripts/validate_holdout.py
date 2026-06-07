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


def fit_platt(p, d, n):
    """Platt scaling: fit calibrated = sigmoid(a + b*logit(p)) by weighted logistic regression
    (IRLS) on binned calibration data — p=bin predicted prob, d=detections, n=trials."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)); n = np.asarray(n, float); d = np.asarray(d, float)
    X = np.column_stack([np.ones_like(x), x]); beta = np.array([0.0, 1.0])
    for _ in range(60):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        W = np.clip(n * mu * (1 - mu), 1e-9, None)
        z = X @ beta + (d - n * mu) / W
        beta = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2), X.T @ (W * z))
    return float(beta[0]), float(beta[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebd", required=True)
    ap.add_argument("--current-year", type=int, required=True)
    ap.add_argument("--holdout-years", type=int, default=3, help="number of most-recent years to test on")
    ap.add_argument("--calib-years", type=int, default=0,
                    help="if >0, reserve this many years (just before the test block) to FIT a Platt "
                         "recalibration; the model trains on the years before that, and calibration is "
                         "reported on the test block only — a true 3-way temporal split.")
    ap.add_argument("--taxonomy", default=str(pc.DEFAULT_TAX))
    ap.add_argument("--min-checklists", type=int, default=pc.MIN_CHECKLISTS)
    ap.add_argument("--memory-limit", default="4GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--temp-dir", default=None)
    ap.add_argument("--label", default="data", help="tag for output files, e.g. US or NY")
    ap.add_argument("--out-dir", default="viz", help="where to write the calibration plot + CSV")
    ap.add_argument("--bins", type=int, default=15, help="reliability-curve bins")
    a = ap.parse_args()

    recal = a.calib_years > 0
    test_lo = a.current_year - a.holdout_years + 1
    calib_hi = test_lo - 1
    calib_lo = calib_hi - a.calib_years + 1               # calib block (only used when recal)
    train_max = (calib_lo - 1) if recal else (test_lo - 1)
    train_min = a.current_year - pc.LOOKBACK_YEARS
    MINY, MINC, C, PS = pc.MIN_CHECKLISTS_YEAR, a.min_checklists, pc.DGP_PRIOR, pc.PRIOR_STRENGTH
    if recal:
        print(f"train {train_min}-{train_max} | calib {calib_lo}-{calib_hi} | test {test_lo}-{a.current_year}", flush=True)
    else:
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

    # ---- held-out aggregates: cross each trained (cell, species) with that cell's duration buckets ----
    print("aggregating held-out years…", flush=True)
    bc = bucket_case("dur")

    def build_trials(name, ylo, yhi):
        con.execute(f"""CREATE TEMP TABLE {name}_cb AS
            SELECT locid,"week",{bc} AS bk, COUNT(*) n_chk, avg(dur) tb
            FROM chk_all WHERE yr BETWEEN {ylo} AND {yhi} AND dur>0 AND dur<=600 GROUP BY locid,"week",{bc};""")
        con.execute(f"""CREATE TEMP TABLE {name}_det AS
            SELECT locid,"week",species_code,{bc} AS bk, COUNT(*) n_det
            FROM '{obs_pq}' WHERE yr BETWEEN {ylo} AND {yhi} AND dur>0 AND dur<=600
            GROUP BY locid,"week",species_code,{bc};""")
        con.execute(f"""CREATE TEMP TABLE {name} AS
            SELECT m.occupancy AS occ, m.detect_given_present AS dgp, m.lambda_hr AS lam,
                   cb.tb/60.0 AS th, cb.n_chk AS n_chk, COALESCE(d.n_det,0) AS n_det
            FROM model m JOIN {name}_cb cb ON cb.locid=m.locid AND cb."week"=m."week"
            LEFT JOIN {name}_det d ON d.locid=m.locid AND d."week"=m."week"
                 AND d.species_code=m.species_code AND d.bk=cb.bk
            WHERE m.lambda_hr IS NOT NULL;""")

    build_trials("ev", test_lo, a.current_year)           # TEST block (final evaluation)
    if recal:
        build_trials("cal", calib_lo, calib_hi)           # CALIBRATION block (fits the Platt map only)

    # ---- metrics on the TEST block ----
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
    brier_clim = base_rate * (1 - base_rate)               # climatology: always predict the base rate

    # optional Platt recalibration: fit (a,b) on the CALIB block, apply to TEST (never fit on test)
    pl_recal = None; brier_r = ll_r = aP = bP = None
    if recal:
        cbdf = con.execute(f"""SELECT round(least(1-{EPS},greatest({EPS},{pl}))*400)/400.0 AS p,
            SUM(n_chk) n, SUM(n_det) d FROM cal GROUP BY 1 HAVING SUM(n_chk)>0""").fetchdf()
        aP, bP = fit_platt(cbdf["p"].values, cbdf["d"].values, cbdf["n"].values)
        pl_recal = (f"least(1-{EPS}, greatest({EPS}, "
                    f"1.0/(1.0+exp(-({aP}+({bP})*ln(({pl})/(1.0-({pl}))))))))")
        ar = con.execute(f"""SELECT SUM(n_det*pow(1-{pl_recal},2)+(n_chk-n_det)*pow({pl_recal},2)) br,
            -SUM(n_det*ln({pl_recal})+(n_chk-n_det)*ln(1-{pl_recal})) lr FROM ev""").fetchone()
        brier_r, ll_r = ar[0] / N, ar[1] / N

    print("\n" + "=" * 64)
    print(f"  HELD-OUT TRIALS: {int(N):,} (checklist x candidate-species)   detections: {int(D):,}  ({base_rate:.3%})")
    print("=" * 64)
    print(f"  {'model':<24}{'Brier':>10}{'log-loss':>12}{'Brier skill':>14}")
    print(f"  {'climatology (base)':<24}{brier_clim:>10.4f}{'—':>12}{0.0:>13.1%}")
    print(f"  {'duration-blind':<24}{brier_b:>10.4f}{ll_b:>12.4f}{1-brier_b/brier_clim:>13.1%}")
    print(f"  {'lambda (hours)':<24}{brier_l:>10.4f}{ll_l:>12.4f}{1-brier_l/brier_clim:>13.1%}")
    if recal:
        print(f"  {'lambda + Platt recal':<24}{brier_r:>10.4f}{ll_r:>12.4f}{1-brier_r/brier_clim:>13.1%}")
        print(f"  (Platt a={aP:.3f}, b={bP:.3f}; fit on calib {calib_lo}-{calib_hi}, scored on test {test_lo}-{a.current_year})")
    print(f"\n  lambda vs duration-blind:  Brier {(brier_b-brier_l)/brier_b:+.2%}   "
          f"log-loss {(ll_b-ll_l)/ll_b:+.2%}   (positive = lambda better)")

    # reliability tables (predicted vs observed), weighted by trials
    def reliability(expr):
        df = con.execute(f"""SELECT least({a.bins-1}, floor(least(1-1e-9,{expr})*{a.bins}))::INT AS bin,
            SUM(n_chk) n, SUM(n_det) d, SUM(({expr})*n_chk)/SUM(n_chk) pred
            FROM ev GROUP BY 1 ORDER BY 1""").fetchdf()
        df["observed"] = df["d"] / df["n"]
        return df
    rel_l, rel_b = reliability(pl), reliability(pb)
    rel_r = reliability(pl_recal) if recal else None
    con.close()
    Path(obs_pq).unlink(missing_ok=True)

    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    parts = [rel_l.assign(model="lambda"), rel_b.assign(model="duration_blind")]
    if recal:
        parts.append(rel_r.assign(model="lambda_recal"))
    rel = pd.concat(parts, ignore_index=True)
    csv_path = outd / f"holdout_{a.label}_reliability.csv"
    rel.to_csv(csv_path, index=False)
    print(f"\n  reliability table -> {csv_path}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.6, 6.4))
        ax.plot([0, 1], [0, 1], ls=":", color="#999", lw=1.2, label="perfect calibration")
        curves = [(rel_b, "#2c6fa6", "duration-blind", brier_b),
                  (rel_l, "#c0392b", "lambda (raw)" if recal else "lambda (hours)", brier_l)]
        if recal:
            curves.append((rel_r, "#27ae60", "lambda + recal", brier_r))
        for df, col, nm, br in curves:
            sz = 12 + 90 * (df["n"] / df["n"].max()) ** 0.5     # marker size ~ trials in bin
            ax.plot(df["pred"], df["observed"], "-", color=col, lw=1.6, alpha=.85)
            ax.scatter(df["pred"], df["observed"], s=sz, color=col, zorder=3,
                       label=f"{nm}  (Brier {br:.3f}, skill {1-br/brier_clim:.0%})")
        ax.axhline(base_rate, color="#bbb", lw=.8, ls="--")
        ax.text(0.02, base_rate + .01, f"base rate {base_rate:.1%}", fontsize=8, color="#888")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("predicted detection probability"); ax.set_ylabel("observed detection rate")
        ax.set_title(f"Hold-out calibration — {a.label}\n"
                     f"test {test_lo}–{a.current_year} · {int(N):,} trials · point size ∝ √(trials in bin)",
                     fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#eee", lw=.6)
        png = outd / f"holdout_{a.label}_calibration.png"
        fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  calibration curve -> {png}")
    except Exception as e:
        print(f"  (plot skipped: {e}; reliability CSV is written — plot it yourself)")
    print("\ndone.")


if __name__ == "__main__":
    main()
