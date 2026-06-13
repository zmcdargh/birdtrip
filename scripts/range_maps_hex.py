"""True hex-binned occupancy heatmaps over NY, month by month. Every surveyed hex gets a value;
absence (surveyed hotspots with no detections) renders as 0 (dark), not grey. Hexes with no
surveyed hotspot at all are not drawn, so the NY footprint shows."""
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P="/sessions/clever-epic-johnson/mnt/outputs/ny3.parquet"; OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"
con=duckdb.connect()
hs=con.execute(f"SELECT DISTINCT locality_id, lat, lon FROM '{P}'").df()
MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
EXT=[-79.9,-71.7,40.45,45.05]

def hexmap(name, fname):
    df=con.execute(f"""SELECT locality_id, ((week-1)/4)::INT mo, max(occupancy) occ
        FROM '{P}' WHERE common_name='{name.replace("'","''")}' GROUP BY 1,2""").df()
    if not len(df): print("  no data:",name); return
    fig,axes=plt.subplots(3,4,figsize=(15,9.2)); sc=None
    for mo in range(12):
        ax=axes[mo//4][mo%4]
        m=hs.merge(df[df.mo==mo][["locality_id","occ"]],on="locality_id",how="left")
        m["occ"]=m["occ"].fillna(0.0)                      # surveyed but absent -> 0
        sc=ax.hexbin(m.lon, m.lat, C=m.occ.values, gridsize=26, reduce_C_function=np.max,
                     cmap="inferno", vmin=0, vmax=1, mincnt=1, linewidths=0.1, extent=EXT)
        ax.set_title(MON[mo],fontsize=10,loc="left"); ax.set_xlim(EXT[0],EXT[1]); ax.set_ylim(EXT[2],EXT[3])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
    fig.colorbar(sc,ax=axes,shrink=.5,label="occupancy (best hotspot in hex; 0=surveyed, absent)",location="right")
    fig.suptitle(f"{name} — monthly occupancy heatmap across NY",fontsize=14,weight="bold",x=0.5,y=0.98)
    fig.savefig(f"{OUT}/{fname}",dpi=140,bbox_inches="tight"); print("  WROTE",fname)

for nm,fn in [("Snowy Owl","hexmap_snowy_owl.png"),("Short-eared Owl","hexmap_short_eared_owl.png"),
              ("Blackburnian Warbler","hexmap_blackburnian.png"),("Saltmarsh Sparrow","hexmap_saltmarsh_sparrow.png")]:
    hexmap(nm,fn)
