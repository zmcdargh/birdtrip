"""
Swarm / intra-day-correlation correction for detection rates — FULL-SCALE runner.

Run on your machine against the checklist-level EBD (the in-sandbox NY zip is 30 GB
uncompressed, too big there). It recomputes, per cell (species x locality x eBird-week),
the per-checklist detection frequency three ways and writes a CSV + figure, so we can see
how much the swarm bias moves real cells (e.g. the Dosoris Pond geese):

  naive       Σ k_day / Σ n_day                            each checklist counts 1   (current)
  subsample   mean over the cell's days of k_day/n_day     one checklist per hotspot-day
  design-eff  Σ w_day (k_day/n_day) / Σ w_day,             keeps partial within-day info
              w_day = n_eff(n_day, ρ),  n_eff(n,ρ) = n/(1+(n-1)ρ)

Cluster = (locality, calendar day); cell = (species, locality, eBird-week) pooling its days.

ρ is a SINGLE global intra-day detection ICC (a property of the checklist process, ~species
independent), estimated by pooling the per-species ICC over the data-rich species and taking
the median. ρ = (r-p)/(1-p) in conditional-prob terms — the correlation net of the base rate.
Estimated on days where the species was detected (given-present redundancy), so it does not
double-count the occupancy/presence draw the main model already conditions on.

DENOMINATORS come from the EBD itself: the set of complete checklists (ALL SPECIES REPORTED=1)
per locality-day is recovered from the distinct sampling-event IDs in the EBD. No SED needed.
--sed is OPTIONAL and only adds the rare complete checklists that recorded zero species
(no EBD observation rows), giving exact rather than near-exact denominators.

Usage:
  python scripts/swarm_lambda_compare.py --ebd data/ebd_US-NY_relApr-2026.txt \
      --current-year 2026 --out data/swarm_compare_NY.csv         # SED not required
"""
import argparse, numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

WEEK = "((month(d)-1)*4 + least(3, CAST(floor((day(d)-1)/7.0) AS INTEGER)) + 1)"
LOOKBACK_YEARS = 10

def rd(p):  # tab-delimited EBD / SED
    return f"read_csv('{p}', delim='\\t', header=true, quote='', all_varchar=true, ignore_errors=true)"

def n_eff(n, rho):
    return n / (1.0 + (n - 1.0) * rho)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebd", required=True)
    ap.add_argument("--sed", default=None, help="OPTIONAL sampling-event file (exact denominators only)")
    ap.add_argument("--current-year", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    miny = a.current_year - LOOKBACK_YEARS
    DATE = "strptime(\"OBSERVATION DATE\",'%Y-%m-%d')"

    con = duckdb.connect(); con.execute(f"PRAGMA threads={a.threads}")

    # ---- complete-checklist denominators reconstructed from the EBD (no SED needed) ----
    con.execute(f"""CREATE TEMP TABLE cl AS
      WITH base AS (
        SELECT DISTINCT "SAMPLING EVENT IDENTIFIER" sei, "LOCALITY ID" loc, {DATE} d
        FROM {rd(a.ebd)} WHERE "ALL SPECIES REPORTED"='1' AND "LOCALITY TYPE"='H')
      SELECT sei, loc, d, {WEEK} AS wk FROM base
      WHERE year(d) BETWEEN {miny} AND {a.current_year}""")
    if a.sed:   # optional: fold in zero-species complete checklists that have no EBD rows
        con.execute(f"""INSERT INTO cl
          WITH base AS (
            SELECT DISTINCT "SAMPLING EVENT IDENTIFIER" sei, "LOCALITY ID" loc, {DATE} d
            FROM {rd(a.sed)} WHERE "ALL SPECIES REPORTED"='1' AND "LOCALITY TYPE"='H')
          SELECT sei, loc, d, {WEEK} AS wk FROM base
          WHERE year(d) BETWEEN {miny} AND {a.current_year} AND sei NOT IN (SELECT sei FROM cl)""")
    print("complete checklists:", con.execute("SELECT count(*) FROM cl").fetchone()[0])

    # ---- detections: distinct (sei, species) over approved species-level obs ----
    con.execute(f"""CREATE TEMP TABLE det AS
      SELECT DISTINCT "SAMPLING EVENT IDENTIFIER" sei, "COMMON NAME" cn
      FROM {rd(a.ebd)} WHERE "ALL SPECIES REPORTED"='1' AND "APPROVED"='1'
        AND "CATEGORY" IN ('species','issf','form','domestic')""")

    # ---- per (species, locality, calendar-day): n_day complete checklists, k_day detecting ----
    #  Built and aggregated IN DUCKDB (spills to disk, not RAM) so it scales to full NY.
    con.execute("""CREATE TEMP TABLE dday AS
      WITH day_n AS (SELECT loc, d, any_value(wk) AS wk, count(*) n_day FROM cl GROUP BY loc, d),
           day_k AS (SELECT t.cn, c.loc, c.d, count(DISTINCT c.sei) k_day
                     FROM cl c JOIN det t USING(sei) GROUP BY t.cn, c.loc, c.d)
      SELECT k.cn, k.loc, n.wk AS week, k.k_day, n.n_day
      FROM day_k k JOIN day_n n ON k.loc=n.loc AND k.d=n.d""")
    con.execute("CREATE TEMP TABLE foot AS SELECT cn, count(DISTINCT loc) nsite FROM dday GROUP BY cn")

    # ---- single global intra-day ICC rho, pooled from data-rich species ----
    #  per species: between-day variance tau2 of the detection rate -> rho = tau2/(mu(1-mu)).
    #  All heavy aggregation in SQL; only ~one row per data-rich species pulled to pandas.
    sp = con.execute("""
      SELECT y.cn, sum(n_day) N, count(*) S, sum(k_day) K,
             sum(k_day*k_day*1.0/n_day) Snp2_raw, sum(n_day*n_day) Sn2
      FROM dday y JOIN foot f USING(cn)
      WHERE y.n_day >= 2 AND f.nsite >= 50
      GROUP BY y.cn HAVING sum(n_day) >= 8 AND count(*) >= 4""").df()
    mu = sp.K / sp.N
    Snp2 = sp.Snp2_raw - mu*mu*sp.N
    D = sp.N - sp.Sn2/sp.N
    tau2 = (Snp2 - (sp.S - 1)*mu*(1-mu)) / D
    rho_s = (tau2 / (mu*(1-mu))).where((mu > 0) & (mu < 1) & (D > 0)).clip(0, 1).dropna()
    rho = float(rho_s.median()) if len(rho_s) >= 5 else 0.5
    print(f"global intra-day ICC  rho = {rho:.3f}   (pooled from {len(rho_s)} data-rich species)")

    # ---- three detection-frequency estimates per cell, aggregated in SQL ----
    con.execute(f"""CREATE TEMP TABLE res AS
      WITH agg AS (
        SELECT cn, loc, week, count(*) n_days, max(n_day) max_day_checklists,
          sum(k_day) ksum, sum(n_day) nsum, avg(k_day*1.0/n_day) p_sub,
          sum((n_day/(1+(n_day-1)*{rho}))*(k_day*1.0/n_day)) kwsum,
          sum(n_day/(1+(n_day-1)*{rho})) wsum
        FROM dday GROUP BY cn, loc, week)
      SELECT cn AS common_name, loc AS locality_id, week, n_days, max_day_checklists,
        round(ksum*1.0/nsum, 4) p_naive,        -- weight = n_day  (current)
        round(p_sub, 4)        p_subsample,     -- weight = 1 per day
        round(kwsum/wsum, 4)   p_deff,          -- weight = n_eff(n_day, rho)
        round((ksum*1.0/nsum)/greatest(p_sub,1e-6), 2) infl_ratio
      FROM agg""")
    con.execute(f"COPY (SELECT * FROM res ORDER BY infl_ratio DESC) TO '{a.out}' (HEADER, DELIMITER ',')")
    ncells = con.execute("SELECT count(*) FROM res").fetchone()[0]
    print(f"wrote {a.out}  ({ncells} cells);  rho={rho:.3f}")

    # surface the inflated-goose cells we've been arguing about, real numbers front-and-center
    watch = con.execute("""SELECT * FROM res WHERE regexp_matches(common_name,
        'Bean-Goose|Ross''s Goose|Greater White-fronted|Barnacle|Pink-footed|Cackling|Short-eared Owl')
        ORDER BY max_day_checklists DESC LIMIT 40""").df()
    if len(watch):
        print("\n=== watchlist (real day-structure of the cells in question) ===")
        print(watch.to_string(index=False))

    # ---- figure: cells where the swarm bias is largest (most replicated days) ----
    top = con.execute("SELECT * FROM res WHERE max_day_checklists>=5 ORDER BY infl_ratio DESC LIMIT 20").df().iloc[::-1]
    if len(top):
        yv = np.arange(len(top))
        fig, ax = plt.subplots(figsize=(11, max(4, .42*len(top)+1)))
        ax.hlines(yv, top.p_subsample, top.p_naive, color="#c0392b", lw=2, alpha=.5)
        ax.scatter(top.p_naive, yv, s=46, facecolors="white", edgecolors="#c0392b", lw=1.8, label="naive (count all)")
        ax.scatter(top.p_deff, yv, s=40, c="#e08a1e", label=f"design-effect (ρ={rho:.2f})")
        ax.scatter(top.p_subsample, yv, s=52, c="#000", label="one-per-day subsample")
        ax.set_yticks(yv); ax.set_yticklabels([f"{r.common_name[:24]} · wk{r.week} · ≤{r.max_day_checklists}/day"
                                               for r in top.itertuples()], fontsize=8)
        ax.set_xlabel("per-checklist detection frequency  (feeds λ)"); ax.set_xlim(0, 1.02)
        ax.set_title(f"Swarm correction on real cells (ρ={rho:.2f}): naive vs subsample vs design-effect", loc="left")
        ax.legend(fontsize=8, loc="lower right")
        for s in ("top","right"): ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(a.out.replace(".csv",".png"), dpi=145)
        print("wrote", a.out.replace(".csv",".png"))

if __name__ == "__main__":
    main()
