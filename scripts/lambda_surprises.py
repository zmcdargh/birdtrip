"""'Biggest surprises': species whose detectability λ is far from what their commonness predicts.
Negative residual = harder to find than its abundance suggests (true skulkers).
Positive residual = easier than its rarity suggests (conspicuous-when-present, OR swarm artifacts)."""
import duckdb, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P="/sessions/clever-epic-johnson/mnt/outputs/ny2.parquet"; OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"
con=duckdb.connect()
c=con.execute(f"SELECT common_name cn, locality_id loc, lambda_hr lam FROM '{P}' WHERE n_checklists>=10 AND lambda_hr>0").df()
sp=c.groupby("cn").agg(med_lam=("lam","median"),nsite=("loc","nunique"),ncells=("lam","size")).reset_index()
sp=sp[(sp.ncells>=8)&(sp.nsite>=3)].copy()
sp["x"]=np.log(sp.nsite); sp["y"]=np.log(sp.med_lam)
b1,b0=np.polyfit(sp.x,sp.y,1); sp["resid"]=sp.y-(b0+b1*sp.x)
sp["t50_min"]=60*np.log(2)/sp.med_lam
srt=sp.sort_values("resid")
cols=["cn","med_lam","t50_min","nsite","resid"]
print("=== HARDER to find than their commonness predicts (cryptic) ===")
print(srt.head(16)[cols].round(2).to_string(index=False))
print("\n=== EASIER to find than their rarity predicts (conspicuous-when-present / possible swarm artifacts) ===")
print(srt.tail(16)[cols].iloc[::-1].round(2).to_string(index=False))

fig,ax=plt.subplots(figsize=(11,7.5))
ax.scatter(sp.nsite,sp.med_lam,s=7,color="#ccc",alpha=.5)
xx=np.linspace(sp.x.min(),sp.x.max(),50); ax.plot(np.exp(xx),np.exp(b0+b1*xx),"--",color="#888",label="expected λ vs commonness")
for _,r in pd.concat([srt.head(10),srt.tail(10)]).iterrows():
    col="#c0392b" if r.resid<0 else "#2471a3"
    ax.scatter(r.nsite,r.med_lam,s=28,color=col,zorder=3)
    ax.annotate(r.cn,(r.nsite,r.med_lam),fontsize=7.3,color=col,xytext=(3,2),textcoords="offset points")
ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("commonness → NY hotspots present (log)")
ax.set_ylabel("detectability → median λ per hour (log)")
ax.set_title("Biggest surprises: detectability vs what commonness predicts\nred = cryptic for how common · blue = obvious for how rare",fontsize=12,weight="bold",loc="left")
ax.legend(fontsize=9);
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/lambda_surprises.png",dpi=145,bbox_inches="tight"); print("\nWROTE lambda_surprises.png")
