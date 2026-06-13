"""Compute step: write per-cell (yp, ns, mu_state, m) to parquet + the kappa(mu) curve. Fast (DuckDB COPY)."""
import duckdb, numpy as np, pandas as pd
PARQ="/sessions/clever-epic-johnson/mnt/outputs/ny.parquet"; O="/sessions/clever-epic-johnson/mnt/outputs"; KAPPA1=20.0
con=duckdb.connect(); con.execute("PRAGMA threads=4")
con.execute(f"""CREATE TEMP TABLE ss AS SELECT locality_id, any_value(county) county, week, max(years_surveyed) ns FROM '{PARQ}' GROUP BY locality_id, week""")
con.execute("CREATE TEMP TABLE wk AS SELECT week, sum(ns) Sn, sum(ns*ns) Sn2, count(*) S FROM ss GROUP BY week")
con.execute("CREATE TEMP TABLE cwk AS SELECT county, week, sum(ns) Snc FROM ss GROUP BY county, week")
con.execute(f"CREATE TEMP TABLE sp AS SELECT species_code, week, sum(years_present) tp, sum(years_present*years_present*1.0/years_surveyed) q FROM '{PARQ}' WHERE years_surveyed>0 GROUP BY 1,2")
con.execute(f"CREATE TEMP TABLE spc AS SELECT species_code, county, week, sum(years_present) tpc FROM '{PARQ}' GROUP BY 1,2,3")
con.execute(f"CREATE TEMP TABLE foot AS SELECT species_code, count(distinct locality_id) nsite FROM '{PARQ}' GROUP BY 1")
rich=con.execute("""SELECT s.tp,s.q,w.Sn,w.Sn2,w.S FROM sp s JOIN wk w USING(week) JOIN foot f USING(species_code) WHERE f.nsite>=50""").df()
mu=rich.tp/rich.Sn; Snp2=rich.q-mu*mu*rich.Sn; D=rich.Sn-rich.Sn2/rich.Sn; tau2=(Snp2-(rich.S-1)*mu*(1-mu))/D
rich["mu"]=mu; rich["k"]=(mu*(1-mu)/tau2.where(tau2>0)-1).clip(0.5,500); rich.loc[tau2<=0,"k"]=500; rich=rich.dropna(subset=["k"])
edges=np.linspace(0,1,21)
kbin=[float(np.median(rich[(rich.mu>=edges[i])&(rich.mu<edges[i+1])].k)) if len(rich[(rich.mu>=edges[i])&(rich.mu<edges[i+1])])>=8 else np.nan for i in range(20)]
kbin=pd.Series(kbin).ffill().bfill().to_numpy()
np.savez(f"{O}/loo_kappa.npz", kbin=kbin, edges=edges)
con.execute(f"""COPY (
  WITH base AS (
    SELECT c.years_present yp, c.years_surveyed ns, sp.tp/wk.Sn AS mu_state,
           COALESCE(spc.tpc/NULLIF(cwk.Snc,0), sp.tp/wk.Sn) AS mu_county, COALESCE(cwk.Snc,0) Nc
    FROM '{PARQ}' c JOIN sp ON c.species_code=sp.species_code AND c.week=sp.week JOIN wk ON c.week=wk.week
    LEFT JOIN spc ON c.species_code=spc.species_code AND c.county=spc.county AND c.week=spc.week
    LEFT JOIN cwk ON c.county=cwk.county AND c.week=cwk.week)
  SELECT yp, ns, mu_state, (Nc*mu_county+{KAPPA1}*mu_state)/(Nc+{KAPPA1}) AS m FROM base)
  TO '{O}/loo_cells.parquet' (FORMAT PARQUET)""")
print("wrote loo_cells.parquet + loo_kappa.npz")
