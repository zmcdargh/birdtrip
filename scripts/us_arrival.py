"""Continental spring-arrival sweep: first eBird week a migrant's occupancy crosses a threshold,
hex-binned across the (eastern) US. Shows the wave moving Gulf Coast -> Canada over ~8 weeks."""
import duckdb, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P="/sessions/clever-epic-johnson/mnt/outputs/us_migrants.parquet"; OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"
con=duckdb.connect(); EXT=[-100.5,-66.5,24,49.5]; THR=0.10
from datetime import date,timedelta
def wk2date(w): return (date(2025,1,1)+timedelta(days=int((w-1)*7.61))).strftime("%b %d")
def arrival(ax,name):
    d=con.execute(f"""SELECT lat, lon, min(week) aw FROM '{P}'
        WHERE common_name='{name.replace("'","''")}' AND occupancy>={THR} AND week BETWEEN 8 AND 28
        GROUP BY lat,lon""").df()
    sc=ax.hexbin(d.lon,d.lat,C=d.aw.values,gridsize=55,reduce_C_function=np.min,cmap="turbo",
                 vmin=9,vmax=24,mincnt=1,extent=EXT)
    ax.set_title(name,fontsize=12,loc="left",weight="bold")
    ax.set_xlim(EXT[0],EXT[1]);ax.set_ylim(EXT[2],EXT[3]);ax.set_xticks([]);ax.set_yticks([])
    for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
    return sc
fig,axes=plt.subplots(2,2,figsize=(15,10)); fig.subplots_adjust(right=0.88,wspace=0.04,hspace=0.12)
for ax,nm in zip(axes.ravel(),["Ruby-throated Hummingbird","Baltimore Oriole","Rose-breasted Grosbeak","Scarlet Tanager"]):
    sc=arrival(ax,nm)
cax=fig.add_axes([0.90,0.2,0.014,0.6])
cb=fig.colorbar(sc,cax=cax,label="spring arrival: first week occupancy ≥ 10%")
cb.set_ticks([9,12,15,18,21,24]); cb.set_ticklabels([f"~{wk2date(w)}" for w in [9,12,15,18,21,24]])
fig.suptitle("Continental spring-arrival sweep — Gulf Coast (blue, March) → Canada (red, June)",x=0.46,y=0.95,fontsize=14,weight="bold")
fig.savefig(f"{OUT}/us_arrival_sweep.png",dpi=140,bbox_inches="tight"); print("WROTE us_arrival_sweep.png")
