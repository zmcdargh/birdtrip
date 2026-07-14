#!/usr/bin/env python3
"""Evaluate the grid-proxy shortlist against the exact ranking, on the real store.

Run on the FULL store (it needs n_checklists for the weighted proxy and all serving columns for
the exact scan):

    python scripts/eval_proxy.py --store data/birdtrip.parquet

Two parts:
  A. Cheap — top-15 shortlist cells (and their states) under three proxy definitions:
     MAX(occ)  vs  unweighted mean(occ)  vs  observation-weighted mean(occ).
     This shows directly whether averaging de-biases the "all-Texas" shortlist.
  B. Expensive (one full-scan) — the EXACT nationwide recommend (ground truth) vs the grid-proxy
     FUNNEL, at a few alpha/week settings: recall of the true top-K and the state spread of each.
     This is the real test: does the shortlist actually recover the hotspots the exact ranking picks?
"""
import argparse
import os
import subprocess
import duckdb


def top_cells(con, store, agg_sql, g=0.9, limit=15):
    """Top `limit` grid cells by Σ_species(agg occupancy), with each cell's state."""
    return con.execute(f"""
        WITH cell AS (
            SELECT floor(latitude/{g}) gy, floor(longitude/{g}) gx, week, species_code,
                   {agg_sql} mo, any_value(state) st
            FROM '{store}' WHERE trusted=1 AND latitude IS NOT NULL AND species_code IS NOT NULL
            GROUP BY 1,2,3,4),
        perweek AS (SELECT gy, gx, week, SUM(mo) proxy, any_value(st) st FROM cell GROUP BY 1,2,3)
        SELECT gy, gx, any_value(st) state, MAX(proxy) best
        FROM perweek GROUP BY gy, gx ORDER BY best DESC LIMIT {limit}
    """).fetchdf()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--k", type=int, default=20, help="top-K hotspots to compare")
    a = ap.parse_args()
    store = a.store
    con = duckdb.connect()

    print("\n===== A. shortlist cells by proxy definition (states of the top 15) =====")
    variants = {
        "MAX(occ)              ": "MAX(occupancy)",
        "mean(occ) unweighted  ": "AVG(occupancy)",
        "mean(occ) obs-weighted": "SUM(occupancy*n_checklists)/NULLIF(SUM(n_checklists),0)",
    }
    for label, agg in variants.items():
        df = top_cells(con, store, agg)
        counts = df["state"].value_counts().to_dict()
        print(f"  {label}: {counts}")

    print("\n===== B. exact ranking vs grid funnel (recall of true top-K, state spread) =====")
    # make sure the (weighted) sidecar sits next to the store
    base = store[:-len(".parquet")] if store.endswith(".parquet") else store
    if not os.path.exists(base + ".grid.parquet"):
        subprocess.run(["python", "scripts/make_grid_proxy.py", "--store", store], check=True)

    from birdtrip.store import Store
    from birdtrip import service
    st = Store(base + ".sqlite")            # sqlite need not exist; the parquet path is used
    pq = service._parquet(st)
    assert pq, "no parquet sidecar found next to the store"
    has_lambda = service._has_lambda(st)

    def states_of(recs):
        from collections import Counter
        return dict(Counter(r["state"] for r in recs))

    for alpha in (0.0, 0.5, 1.0):
        for weeks in (None, [18]):
            gt = service._recommend_via_parquet(st, pq, [], None, weeks, 1, alpha, 0.5, a.k,
                                                None, 4.0, has_lambda, False, None,
                                                cand_locality_ids=None)               # EXACT full scan
            fn = service.recommend_trips(st, states=None, weeks=weeks, alpha=alpha,
                                         hours=4.0, topn=a.k)                          # FUNNEL
            gset, fset = {r["locality_id"] for r in gt}, {r["locality_id"] for r in fn}
            recall = len(gset & fset) / max(1, len(gset))
            wl = "all" if weeks is None else str(weeks)
            print(f"\n  alpha={alpha} weeks={wl}: recall@{a.k}={recall:.0%}  "
                  f"(exact {len(gt)}, funnel {len(fn)})")
            print(f"     exact  states: {states_of(gt)}")
            print(f"     funnel states: {states_of(fn)}")


if __name__ == "__main__":
    main()
