"""Does lambda capture birder intuition about detectability? NY store analysis.
Figures: (1) violins of lambda for classic skulkers vs conspicuous birds,
(2) a detectability x commonness map, (3) seasonal lambda for skulkers.
Also tests whether lambda correlates with rarity (it shouldn't — rarity lives in occupancy)."""
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P="/sessions/clever-epic-johnson/mnt/outputs/ny2.parquet"; OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"
con=duckdb.connect()
cells=con.execute(f"SELECT common_name cn, week, locality_id loc, lambda_hr lam, occupancy occ, n_checklists nc FROM '{P}' WHERE n_checklists>=10 AND lambda_hr>0").df()
print("stable cells:",len(cells))

SKULK=["Virginia Rail","Sora","Clapper Rail","King Rail","Least Bittern","American Bittern",
       "Connecticut Warbler","Mourning Warbler","Sedge Wren","Marsh Wren","Yellow-billed Cuckoo",
       "Black-billed Cuckoo","Eastern Whip-poor-will","American Woodcock","Nelson's Sparrow",
       "Saltmarsh Sparrow","Swamp Sparrow","Mourning Warbler"]
VIS=["Canada Goose","Mallard","American Robin","Red-winged Blackbird","Northern Cardinal","Blue Jay",
     "American Crow","Mourning Dove","European Starling","Common Grackle","Turkey Vulture",
     "Great Blue Heron","Double-crested Cormorant","Osprey","Red-tailed Hawk","Song Sparrow","Mute Swan"]
grp={**{s:"skulker" for s in SKULK},**{s:"conspicuous" for s in VIS}}

# per-species aggregates (all species, for the map + correlation)
sp=cells.groupby("cn").agg(med_lam=("lam","median"), med_occ=("occ","median"),
     nsite=("loc","nunique"), ncells=("lam","size")).reset_index()
sp=sp[sp.ncells>=5]
sp["half_min"]=60*np.log(2)/sp.med_lam            # minutes of birding to 50% detection (given present)
print("species with >=5 stable cells:",len(sp))
# orthogonality: lambda vs rarity
def spearman(a,b):
    ra=pd.Series(a).rank(); rb=pd.Series(b).rank(); return np.corrcoef(ra,rb)[0,1]
print("Spearman(median lambda, log nsite)  = %.3f"%spearman(np.log(sp.med_lam), np.log(sp.nsite)))
print("Spearman(median lambda, median occ) = %.3f"%spearman(sp.med_lam, sp.med_occ))

# ---- demo set for violins ----
demo=cells[cells.cn.isin(grp)].copy(); demo["grp"]=demo.cn.map(grp)
order=demo.groupby("cn").lam.median().sort_values()
order=order[order.index.isin(sp.cn)]      # keep only species with enough cells
names=list(order.index)
print("\nskulker vs conspicuous — median lambda (per-hr) and 50%-detection time:")
for nm in names:
    s=sp[sp.cn==nm].iloc[0]; print(f"  {grp[nm][:4]:>4}  {nm:<26} λ={s.med_lam:5.2f}/h  t50={s.half_min:6.1f} min  ncells={int(s.ncells)}")

fig,ax=plt.subplots(figsize=(12.5,6.6))
data=[demo[demo.cn==nm].lam.values for nm in names]
parts=ax.violinplot(data,positions=np.arange(len(names)),showmedians=True,widths=0.85)
cols=["#c0392b" if grp[nm]=="skulker" else "#2471a3" for nm in names]
for pc,c in zip(parts["bodies"],cols): pc.set_facecolor(c); pc.set_alpha(.6)
for key in ("cmedians","cbars","cmins","cmaxes"):
    if key in parts: parts[key].set_color("#555"); parts[key].set_linewidth(1)
ax.set_yscale("log"); ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(names,rotation=55,ha="right",fontsize=8.5)
ax.set_ylabel("detection rate λ  (per hour, given present)  — log scale")
# secondary axis: detection half-life
def lam2half(l): return 60*np.log(2)/np.clip(l,1e-6,None)
sec=ax.secondary_yaxis("right",functions=(lam2half,lam2half)); sec.set_ylabel("≈ minutes of birding to 50% chance (given present)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#c0392b",alpha=.6,label="classic skulker / cryptic"),Patch(color="#2471a3",alpha=.6,label="conspicuous")],fontsize=10,loc="upper left")
ax.set_title("Does λ know which birds are hard to see?  NY hotspots, cells with ≥10 checklists",fontsize=12,weight="bold",loc="left")
for s in ("top",): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/lambda_skulkers_violin.png",dpi=145,bbox_inches="tight"); print("\nWROTE lambda_skulkers_violin.png")

# ---- 2D map: detectability (lambda) x commonness (nsite) ----
fig2,bx=plt.subplots(figsize=(11,8))
bx.scatter(sp.nsite, sp.med_lam, s=8, color="#ccc", alpha=.5, zorder=1)
mx=sp.med_lam.median(); mn=np.exp(np.log(sp.nsite).median())
bx.axhline(mx,color="#ddd",lw=1); bx.axvline(mn,color="#ddd",lw=1)
lab=set(SKULK[:10]+VIS[:10]+["Snowy Owl","Snow Goose","Bald Eagle","Ruby-throated Hummingbird","Winter Wren","House Wren","Cedar Waxwing","Golden-crowned Kinglet"])
for _,r in sp[sp.cn.isin(lab)].iterrows():
    c="#c0392b" if grp.get(r.cn)=="skulker" else "#2471a3" if grp.get(r.cn)=="conspicuous" else "#27ae60"
    bx.scatter(r.nsite,r.med_lam,s=30,color=c,zorder=3)
    bx.annotate(r.cn,(r.nsite,r.med_lam),fontsize=7.5,xytext=(3,3),textcoords="offset points",color=c)
bx.set_xscale("log"); bx.set_yscale("log")
bx.set_xlabel("commonness  →  number of NY hotspots present  (log)"); bx.set_ylabel("detectability  →  λ per hour given present  (log)")
bx.set_title("Two independent axes: how widespread (occupancy/range) vs how detectable (λ)\nskulkers=red, conspicuous=blue, other labeled=green",fontsize=11,weight="bold",loc="left")
for s in ("top","right"): bx.spines[s].set_visible(False)
fig2.text(0.5,0.005,f"Spearman(λ, range) = {spearman(np.log(sp.med_lam),np.log(sp.nsite)):.2f}  →  near zero: detectability is NOT rarity. A rare bird can be obvious-when-present; a common one can be a skulker.",ha="center",fontsize=9,color="#444")
fig2.tight_layout(rect=[0,0.03,1,1]); fig2.savefig(f"{OUT}/lambda_detectability_map.png",dpi=145,bbox_inches="tight"); print("WROTE lambda_detectability_map.png")

# ---- seasonal lambda for a few skulkers (detectability phenology) ----
seas=["Virginia Rail","Sora","Marsh Wren","American Bittern","Eastern Whip-poor-will"]
seas=[s for s in seas if s in set(cells.cn)]
fig3,cx=plt.subplots(figsize=(11,5))
for nm in seas:
    g=cells[cells.cn==nm].groupby("week").lam.median()
    if len(g)>=8: cx.plot(g.index, g.values, "-o", ms=3, label=nm)
cx.set_xlabel("eBird week (1=early Jan … 48=late Dec)"); cx.set_ylabel("median λ per hour (given present)")
cx.set_title("Detectability has a season too: skulkers' λ peaks when they call (spring/summer)",fontsize=11,weight="bold",loc="left")
cx.legend(fontsize=9);
for s in ("top","right"): cx.spines[s].set_visible(False)
fig3.tight_layout(); fig3.savefig(f"{OUT}/lambda_seasonal_skulkers.png",dpi=145,bbox_inches="tight"); print("WROTE lambda_seasonal_skulkers.png")
