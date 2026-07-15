#!/usr/bin/env python3
"""Diagnose the Find-spots ranking for a state/season: why does hotspot X rank where it does?

Reproduces the server's per-hotspot score (alpha=0, 1 h of birding, NO life list):
  score(hotspot) = MAX over the chosen weeks of  Σ_species occupancy * (1 - exp(-lambda_hr))
and prints the top 20 with checklist volume + species count, then shows where a few named
hotspots actually rank. Run on the FULL store (needs n_checklists):

    python scripts/diag_rank.py --store data/birdtrip.parquet --state "New York" --month 5
"""
import argparse
import duckdb

MONTH_WEEKS = lambda m: [(m - 1) * 4 + i for i in (1, 2, 3, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--month", type=int, required=True, help="1-12")
    ap.add_argument("--hours", type=float, default=1.0, help="hours of birding assumed in the score")
    ap.add_argument("--look", nargs="*", default=["central park", "doodletown", "montezuma"],
                    help="substrings of hotspot names to locate in the ranking")
    a = ap.parse_args()
    weeks = ",".join(str(w) for w in MONTH_WEEKS(a.month))
    con = duckdb.connect()
    h = float(a.hours); kfb = max(1, round(h))
    pdet = f"(1 - CASE WHEN lambda_hr IS NOT NULL THEN exp(-lambda_hr*{h}) ELSE pow(1-detect_given_present,{kfb}) END)"
    con.execute(f"""CREATE TEMP TABLE rk AS
      WITH cell AS (
        SELECT locality_id, any_value(locality) loc, week,
               SUM(occupancy*{pdet}) score, COUNT(*) n_species,
               any_value(n_checklists) chk, any_value(years_surveyed) yrs
        FROM '{a.store}'
        WHERE trusted=1 AND state='{a.state.replace("'","''")}' AND week IN ({weeks})
        GROUP BY locality_id, week),
      peak AS (
        SELECT locality_id, arg_max(loc,score) locality, MAX(score) score,
               arg_max(n_species,score) n_species, arg_max(chk,score) checklists,
               arg_max(yrs,score) yrs, arg_max(week,score) wk
        FROM cell GROUP BY locality_id)
      SELECT *, row_number() OVER (ORDER BY score DESC) AS rank FROM peak""")

    print(f"\n=== top 20 hotspots — {a.state}, month {a.month} (weeks {weeks}), alpha=0, no life list ===")
    print(con.execute("""SELECT rank, round(score,2) score, n_species, checklists, yrs, wk, locality
                         FROM rk ORDER BY rank LIMIT 20""").df().to_string(index=False))

    ntot = con.execute("SELECT COUNT(*) FROM rk").fetchone()[0]
    print(f"\n(out of {ntot} NY hotspots with data in those weeks)")

    print("\n=== where the famous spots land ===")
    for name in a.look:
        df = con.execute(f"""SELECT rank, round(score,2) score, n_species, checklists, yrs, locality
                             FROM rk WHERE lower(locality) LIKE '%{name.lower()}%' ORDER BY rank LIMIT 5""").df()
        print(f"\n  '{name}':")
        print(df.to_string(index=False) if len(df) else "    (no hotspot with that name in the data)")


if __name__ == "__main__":
    main()
