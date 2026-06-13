"""Figures + metrics for the LOO occupancy calibration, from cached loo_cells.npz (fast, no SQL)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"; EPS=1e-4
import duckdb
c=duckdb.connect().execute("SELECT yp,ns,mu_state,m FROM '/sessions/clever-epic-johnson/mnt/outputs/loo_cells.parquet'").df()
k=c.yp.to_numpy(float); n=c.ns.to_numpy(float); m=c.m.to_numpy(float); mus=c.mu_state.to_numpy(float)
kp=np.load("/sessions/clever-epic-johnson/mnt/outputs/loo_kappa.npz"); kbin=kp["kbin"]; edges=kp["edges"]
kmu=np.interp(mus,(edges[:-1]+edges[1:])/2,kbin)
sel=n>=2; ks_,ns_,ms_,kmus_=k[sel],n[sel],m[sel],kmu[sel]

def inst(kap):
    kk=kap[sel] if hasattr(kap,"__len__") else kap
    p_pres=np.clip((ks_-1+kk*ms_)/(ns_-1+kk),EPS,1-EPS); p_abs=np.clip((ks_+kk*ms_)/(ns_-1+kk),EPS,1-EPS)
    pred=np.concatenate([p_pres,p_abs]); out=np.concatenate([np.ones(len(ks_)),np.zeros(len(ks_))]); w=np.concatenate([ks_,ns_-ks_])
    return pred,out,w
def metrics(pred,out,w):
    return (np.sum(w*-(out*np.log(pred)+(1-out)*np.log(1-pred)))/w.sum(), np.sum(w*(pred-out)**2)/w.sum())
def reliability(pred,out,w,bins):
    idx=np.clip(np.digitize(pred,bins)-1,0,len(bins)-2)
    tw=np.bincount(idx,weights=w,minlength=len(bins)-1); ty=np.bincount(idx,weights=w*out,minlength=len(bins)-1); tp=np.bincount(idx,weights=w*pred,minlength=len(bins)-1)
    ok=tw>0; return tp[ok]/tw[ok], ty[ok]/tw[ok]
def roc(pred,out,w):
    o=np.argsort(-pred); y=out[o]; ww=w[o]; P=(w*out).sum(); N=(w*(1-out)).sum()
    tpr=np.cumsum(ww*y)/P; fpr=np.cumsum(ww*(1-y))/N; return fpr,tpr,np.trapezoid(tpr,fpr)

variants={"no shrink":0.0,"κ=1":1.0,"κ=2":2.0,"κ=3":3.0,"κ=4":4.0,"κ=5":5.0,"κ=8":8.0,"κ(μ)":kmu}
summary=pd.DataFrame([(nm,*metrics(*inst(kp))) for nm,kp in variants.items()],columns=["variant","logloss","brier"])
print(summary.round(4).to_string(index=False))

show=["no shrink","κ=3","κ=5","κ(μ)"]; colmap={"no shrink":"#999","κ=3":"#27ae60","κ=5":"#2471a3","κ(μ)":"#c0392b"}
fig,ax=plt.subplots(1,3,figsize=(16,5)); binsf=np.linspace(0,1,21); binslow=np.linspace(0,0.1,21)
for nm in show:
    pred,out,w=inst(variants[nm])
    px,py=reliability(pred,out,w,binsf); ax[0].plot(px,py,"-o",ms=3,color=colmap[nm],label=nm)
    lx,ly=reliability(pred,out,w,binslow); ax[1].plot(lx,ly,"-o",ms=3,color=colmap[nm],label=nm)
    fp,tp,auc=roc(pred,out,w); ax[2].plot(fp,tp,"-",color=colmap[nm],label=f"{nm} (AUC {auc:.3f})")
for a in ax[:2]: a.plot([0,1],[0,1],"--",color="#bbb",lw=1)
ax[0].set_xlim(0,1);ax[0].set_ylim(0,1);ax[0].set_xlabel("predicted occupancy");ax[0].set_ylabel("observed held-out frequency");ax[0].set_title("A. Reliability (full range)",loc="left");ax[0].legend(fontsize=8)
ax[1].set_xlim(0,0.1);ax[1].set_ylim(0,0.2);ax[1].set_xlabel("predicted occupancy");ax[1].set_ylabel("observed held-out frequency");ax[1].set_title("B. Reliability — low-probability tail (p<0.1)",loc="left");ax[1].legend(fontsize=8)
ax[2].plot([0,1],[0,1],"--",color="#bbb",lw=1);ax[2].set_xlabel("false positive rate");ax[2].set_ylabel("true positive rate");ax[2].set_title("C. ROC",loc="left");ax[2].legend(fontsize=8,loc="lower right")
for a in ax:
    for s in("top","right"): a.spines[s].set_visible(False)
fig.suptitle("Leave-one-year-out calibration of raw EB occupancy (no recalibration) — full NY store, cells with ≥2 surveyed years",y=1.02,fontsize=12.5,weight="bold")
fig.tight_layout();fig.savefig(f"{OUT}/eb_calibration_NY.png",dpi=145,bbox_inches="tight");print("WROTE eb_calibration_NY.png")

# stratified
def strat(kap,lo,hi):
    s=sel&(n>=lo)&(n<=hi); kk=kap[s] if hasattr(kap,"__len__") else kap
    pp=np.clip((k[s]-1+kk*m[s])/(n[s]-1+kk),EPS,1-EPS); pa=np.clip((k[s]+kk*m[s])/(n[s]-1+kk),EPS,1-EPS)
    return ((k[s]*-np.log(pp)).sum()+((n-k)[s]*-np.log(1-pa)).sum())/n[s].sum()
fig2,bx=plt.subplots(1,2,figsize=(12,4.6)); ks=[0,1,2,3,4,5,8]; names=["no shrink","κ=1","κ=2","κ=3","κ=4","κ=5","κ=8"]
lls=[summary[summary.variant==v].logloss.iloc[0] for v in names]
bx[0].plot(ks,lls,"-o",color="#2471a3"); kmll=summary[summary.variant=="κ(μ)"].logloss.iloc[0]
bx[0].axhline(kmll,color="#c0392b",ls="--",label=f"κ(μ): {kmll:.3f}"); best=ks[int(np.argmin(lls))]
bx[0].axvline(best,color="#27ae60",ls=":",label=f"best flat κ={best}")
bx[0].set_xlabel("flat κ");bx[0].set_ylabel("LOO log-loss (lower=better)");bx[0].set_title("D. Overall: best-calibrated κ",loc="left");bx[0].legend(fontsize=8)
for lab,lo,hi in [("2 yr",2,2),("3 yr",3,3),("4-7 yr",4,7),("8+ yr",8,99)]:
    bx[1].plot(ks,[strat(kp,lo,hi) for kp in ks],"-o",ms=3,label=lab)
bx[1].set_xlabel("flat κ");bx[1].set_ylabel("LOO log-loss");bx[1].set_title("E. By years_surveyed (thin wants more, thick wants less)",loc="left");bx[1].legend(fontsize=8)
for a in bx:
    for s in("top","right"): a.spines[s].set_visible(False)
fig2.suptitle("Selecting κ by out-of-sample log-loss",y=1.02,fontsize=12.5,weight="bold")
fig2.tight_layout();fig2.savefig(f"{OUT}/eb_calibration_kappa_NY.png",dpi=145,bbox_inches="tight");print("WROTE eb_calibration_kappa_NY.png")
