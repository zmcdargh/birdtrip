#!/usr/bin/env python3
"""Model-predicted species-per-visit vs. what birders ACTUALLY record per checklist.

For each (hotspot, week) the Find-spots score predicts, at 1 h of birding:
    model = Σ_species occupancy * (1 - exp(-lambda_hr))          # what the ranking sorts on
The honest observed comparison, straight from the data, is the mean species per COMPLETE checklist:
    emp   = Σ_species (n_detections / n_checklists)              # avg species actually logged per visit
residual = model - emp. If the model is well-calibrated at the site level, residual ~ 0.

Shows: (A) the top sites by model score with model/emp/residual, (B) the biggest over-predictors
among well-sampled sites, (C) residual by survey-years, (D) named spots. Run on the FULL store:

    python scripts/diag_residual.py --store data/birdtrip.parquet --state "New York" --month 5
"""
import argparse
import duckdb

MONTH_WEEKS = lambda m: [(m - 1) * 4 + i for i in (1, 2, 3, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--month", type=int, required=True)
    ap.add_argument("--min-chk", type=int, default=30, help="min checklists for a reliable empirical estimate")
    ap.add_argument("--look", nargs="*", default=["central park", "doodletown", "montezuma", "edgemere"])
    a = ap.parse_args()
    weeks = ",".join(str(w) for w in MONTH_WEEKS(a.month))
    st = a.state.replace("'", "''")
    pdet1 = "(1 - CASE WHEN lambda_hr IS NOT NULL THEN exp(-lambda_hr) ELSE (1-detect_given_present) END)"
    # model at each cell's ACTUAL mean checklist duration — apples-to-apples with emp (per-checklist)
    pdd = ("(1 - CASE WHEN lambda_hr IS NOT NULL THEN exp(-lambda_hr*(COALESCE(mean_dur_min,60)/60.0)) "
           "ELSE (1-detect_given_present) END)")
    con = duckdb.connect()
    con.execute(f"""CREATE TEMP TABLE rk AS
      WITH cell AS (
        SELECT locality_id, any_value(locality) loc, week,
               SUM(occupancy*{pdet1}) model,
               SUM(occupancy*{pdd}) model_dur,
               SUM(occupancy) occ_sum,
               SUM(n_detections)::DOUBLE / NULLIF(any_value(n_checklists),0) emp,
               COUNT(*) n_species, any_value(n_checklists) chk, any_value(years_surveyed) yrs,
               any_value(mean_dur_min) dur
        FROM '{a.store}'
        WHERE trusted=1 AND state='{st}' AND week IN ({weeks})
        GROUP BY locality_id, week),
      peak AS (   -- the week the ranking would pick for each hotspot
        SELECT locality_id, arg_max(loc,model) locality, MAX(model) model,
               arg_max(model_dur,model) model_dur, arg_max(occ_sum,model) occ_sum,
               arg_max(emp,model) emp, arg_max(n_species,model) n_species,
               arg_max(chk,model) chk, arg_max(yrs,model) yrs, arg_max(dur,model) dur
        FROM cell GROUP BY locality_id)
      SELECT *, model-emp AS resid, row_number() OVER (ORDER BY model DESC) rank FROM peak""")

    print(f"\n=== A. top 15 by MODEL score — {a.state}, month {a.month} ===")
    print("  model=score@1h · model_dur=score@site's mean checklist duration · occ_sum=Σoccupancy(cap) · emp=species/checklist")
    print(con.execute("""SELECT rank, round(model,1) model, round(model_dur,1) model_dur, round(occ_sum,1) occ_sum,
                                round(emp,1) emp, chk, round(dur) dur_min, yrs, locality
                         FROM rk ORDER BY model DESC LIMIT 15""").df().to_string(index=False))

    print(f"\n=== B. biggest OVER-predictors among well-sampled sites (>= {a.min_chk} checklists) ===")
    print(con.execute(f"""SELECT rank, round(model,1) model, round(emp,1) emp, round(resid,1) over_pred,
                                chk, yrs, locality FROM rk WHERE chk>={a.min_chk}
                          ORDER BY resid DESC LIMIT 15""").df().to_string(index=False))

    print(f"\n=== C. residual (model - emp) by survey-years, well-sampled sites (>= {a.min_chk} chk) ===")
    print(con.execute(f"""SELECT yrs, COUNT(*) n, round(median(model),1) med_model,
                                round(median(emp),1) med_emp, round(median(resid),1) med_over_pred
                          FROM rk WHERE chk>={a.min_chk} GROUP BY yrs ORDER BY yrs""").df().to_string(index=False))

    print("\n=== D. named spots: model vs emp ===")
    for name in a.look:
        df = con.execute(f"""SELECT rank, round(model,1) model, round(model_dur,1) model_dur, round(occ_sum,1) occ_sum,
                                    round(emp,1) emp, chk, round(dur) dur_min, yrs, locality
                              FROM rk WHERE lower(locality) LIKE '%{name.lower()}%'
                              ORDER BY model DESC LIMIT 4""").df()
        print(f"\n  '{name}':")
        print(df.to_string(index=False) if len(df) else "    (none)")


if __name__ == "__main__":
    main()
