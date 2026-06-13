"""Month-by-month geographic occupancy maps for NY species — does the model put birds in the
right places at the right times? Background = all surveyed hotspots; foreground = where the
species is present, colored by occupancy. One 3x4 (12-month) panel per species."""
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P="/sessions/clever-epic-johnson/mnt/outputs/ny3.parquet"; OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"
con=duckdb.connect()
# all surveyed hotspot locations (background)
hs=con.execute(f"SELECT DISTINCT locality_id, lat, lon FROM '{P}'").df()
print("hotspots:",len(hs))
MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def species_map(name, fname):
    df=con.execute(f"""SELECT lat, lon, ((week-1)/4)::INT AS mo, max(occupancy) occ
        FROM '{P}' WHERE common_name='{name.replace("'","''")}' GROUP BY lat,lon,((week-1)/4)::INT""").df()
    if not len(df): print("  no data:",name); return
    fig,axes=plt.subplots(3,4,figsize=(15,9.2))
    for mo in range(12):
        ax=axes[mo//4][mo%4]
        ax.scatter(hs.lon, hs.lat, s=1.2, color="#e3e3e3", alpha=.5, zorder=1)   # all hotspots
        d=df[df.mo==mo]
        if len(d):
            sc=ax.scatter(d.lon, d.lat, c=d.occ, s=8+22*d.occ, cmap="inferno", vmin=0, vmax=1,
                          zorder=3, edgecolors="none")
        ax.set_title(MON[mo], fontsize=10, loc="left")
        ax.set_xlim(-79.9,-71.7); ax.set_ylim(40.45,45.05); ax.set_xticks([]); ax.set_yticks([])
        for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
    fig.colorbar(sc, ax=axes, shrink=.5, label="occupancy (prob. present that month)", location="right")
    fig.suptitle(f"{name} — monthly occupancy across NY  (grey = all surveyed hotspots)", fontsize=14, weight="bold", x=0.5, y=0.98)
    fig.savefig(f"{OUT}/{fname}", dpi=140, bbox_inches="tight"); print("  WROTE",fname)

for nm,fn in [("Snowy Owl","map_snowy_owl.png"),
              ("Short-eared Owl","map_short_eared_owl.png"),
              ("Blackburnian Warbler","map_blackburnian.png"),
              ("Saltmarsh Sparrow","map_saltmarsh_sparrow.png")]:
    species_map(nm, fn)
