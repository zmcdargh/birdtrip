"""Spring-arrival sweep: for each NY hex, the first eBird week a migrant's occupancy crosses a
threshold. Shows the wave of arrival moving across the state."""
import duckdb, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P="/sessions/clever-epic-johnson/mnt/outputs/ny3.parquet"; OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"
con=duckdb.connect(); hs=con.execute(f"SELECT DISTINCT locality_id, lat, lon FROM '{P}'").df()
EXT=[-79.9,-71.7,40.45,45.05]; THR=0.15
def wk2date(w):  # eBird week -> approx calendar
    from datetime import date,timedelta; return (date(2025,1,1)+timedelta(days=int((w-1)*7.61))).strftime("%b %d")
def arrival(ax,name):
    d=con.execute(f"""SELECT locality_id, min(week) aw FROM '{P}'
        WHERE common_name='{name.replace("'","''")}' AND occupancy>={THR} AND week BETWEEN 12 AND 30
        GROUP BY 1""").df()
    m=hs.merge(d,on="locality_id",how="inner")
    sc=ax.hexbin(m.lon,m.lat,C=m.aw.values,gridsize=40,reduce_C_function=np.min,cmap="turbo",
                 vmin=14,vmax=24,mincnt=1,extent=EXT)
    ax.set_title(f"{name}",fontsize=12,loc="left",weight="bold")
    ax.set_xlim(EXT[0],EXT[1]);ax.set_ylim(EXT[2],EXT[3]);ax.set_xticks([]);ax.set_yticks([])
    for s in ("top","right","bottom","left"): ax.spines[s].set_visible(False)
    return sc
fig,ax=plt.subplots(1,2,figsize=(15.5,5.4)); fig.subplots_adjust(right=0.86,wspace=0.05)
for a,nm in zip(ax,["Ruby-throated Hummingbird","Baltimore Oriole"]): sc=arrival(a,nm)
cax=fig.add_axes([0.89,0.18,0.015,0.64])
cb=fig.colorbar(sc,cax=cax,label="first week occupancy ≥ 15%  (spring arrival)")
cb.set_ticks([14,16,18,20,22,24]); cb.set_ticklabels([f"~{wk2date(w)}" for w in [14,16,18,20,22,24]])
fig.suptitle("Spring arrival sweep across NY — earlier (blue) coast/south, later (red) upstate",x=0.46,y=0.98,fontsize=13,weight="bold")
fig.savefig(f"{OUT}/arrival_sweep.png",dpi=145,bbox_inches="tight"); print("WROTE arrival_sweep.png")
