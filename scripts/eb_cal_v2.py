"""Re-analysis from cached loo_cells.parquet: actual reliability numbers, stratified log-loss
(by predicted regime, by rarity mu, by recurrence), and isotonic-recalibration check on the low bin."""
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/sessions/clever-epic-johnson/mnt/Birding Trip Planner"; EPS=1e-4
c=duckdb.connect().execute("SELECT yp,ns,mu_state,m FROM '/sessions/clever-epic-johnson/mnt/outputs/loo_cells.parquet'").df()
k=c.yp.to_numpy(float); n=c.ns.to_numpy(float); m=c.m.to_numpy(float); mus=c.mu_state.to_numpy(float)
kp=np.load("/sessions/clever-epic-johnson/mnt/outputs/loo_kappa.npz"); kbin=kp["kbin"]; edges=kp["edges"]
kmu=np.interp(mus,(edges[:-1]+edges[1:])/2,kbin)
sel=n>=2
def inst(kap, extra=None):
    kk=kap[sel] if hasattr(kap,"__len__") else kap
    pp=np.clip((k[sel]-1+kk*m[sel])/(n[sel]-1+kk),EPS,1-EPS); pa=np.clip((k[sel]+kk*m[sel])/(n[sel]-1+kk),EPS,1-EPS)
    pred=np.concatenate([pp,pa]); out=np.concatenate([np.ones(sel.sum()),np.zeros(sel.sum())]); w=np.concatenate([k[sel],(n-k)[sel]])
    ex=None if extra is None else np.concatenate([extra[sel],extra[sel]])
    return pred,out,w,ex
def ll(pred,out,w,mask=None):
    if mask is not None: pred,out,w=pred[mask],out[mask],w[mask]
    if w.sum()==0: return np.nan
    return np.sum(w*-(out*np.log(pred)+(1-out)*np.log(1-pred)))/w.sum()
variants={"κ=2":2.0,"κ=3":3.0,"κ=4":4.0,"κ=5":5.0,"κ=8":8.0,"κ(μ)":kmu}

# ---- log-loss stratified by PREDICTED regime ----
print("log-loss by predicted-probability regime:")
rows=[]
for nm,kp_ in variants.items():
    pred,out,w,_=inst(kp_)
    rows.append((nm, ll(pred,out,w), ll(pred,out,w,pred<0.1), ll(pred,out,w,pred<0.05), ll(pred,out,w,pred>=0.5)))
print(pd.DataFrame(rows,columns=["variant","all","pred<0.1","pred<0.05","pred>=0.5"]).round(4).to_string(index=False))

# ---- log-loss stratified by RARITY mu ----
print("\nlog-loss by species rarity mu:")
rows=[]
for nm,kp_ in variants.items():
    pred,out,w,mm=inst(kp_, mus)
    rows.append((nm, ll(pred,out,w,mm<0.01), ll(pred,out,w,(mm>=0.01)&(mm<0.1)),
                 ll(pred,out,w,(mm>=0.1)&(mm<0.5)), ll(pred,out,w,mm>=0.5)))
print(pd.DataFrame(rows,columns=["variant","mu<0.01(rare)","0.01-0.1","0.1-0.5","mu>=0.5(common)"]).round(4).to_string(index=False))

# ---- actual reliability numbers, lowest bins, kappa=5 vs kappa(mu) ----
def reliab(pred,out,w,bins):
    idx=np.clip(np.digitize(pred,bins)-1,0,len(bins)-2)
    tw=np.bincount(idx,weights=w,minlength=len(bins)-1); ty=np.bincount(idx,weights=w*out,minlength=len(bins)-1); tp=np.bincount(idx,weights=w*pred,minlength=len(bins)-1)
    return tp,ty,tw
lb=np.array([0,.002,.005,.01,.02,.05,.1,.2,.4,.7,1.0])
print("\nlowest-bin reliability (mean_pred -> observed_freq, weight):")
for nm in ["κ=5","κ(μ)"]:
    pred,out,w,_=inst(variants[nm]); tp,ty,tw=reliab(pred,out,w,lb)
    print(f"  {nm}: "+" | ".join(f"[{lb[i]:.3f}-{lb[i+1]:.3f}] p={tp[i]/tw[i]:.3f} obs={ty[i]/tw[i]:.3f} w={tw[i]/1e3:.0f}k" for i in range(len(lb)-1) if tw[i]>0))

# ---- isotonic recalibration check on kappa=5 (does it fix the low bin? log-loss after?) ----
try:
    from sklearn.isotonic import IsotonicRegression
    pred,out,w,_=inst(5.0)
    # fit on binned summary to keep it fast
    order=np.argsort(pred); iso=IsotonicRegression(out_of_bounds="clip").fit(pred,out,sample_weight=w)
    rec=np.clip(iso.predict(pred),EPS,1-EPS)
    print(f"\nisotonic recal on κ=5:  log-loss {ll(pred,out,w):.4f} -> {ll(rec,out,w):.4f}")
    tp,ty,tw=reliab(pred,out,w,lb); tpr,tyr,twr=reliab(rec,out,w,lb)
    print("  lowest 3 bins after recal (pred->obs):")
    for i in range(3):
        if tw[i]>0: print(f"   raw p={tp[i]/tw[i]:.3f} obs={ty[i]/tw[i]:.3f}  | recal p={tpr[i]/twr[i]:.3f} obs={tyr[i]/twr[i]:.3f}")
    HAVE_ISO=True
except Exception as e:
    print("sklearn isotonic unavailable:",e); HAVE_ISO=False

# ================= FIGURES (fixed) =================
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
colmap={"κ=2":"#27ae60","κ=5":"#2471a3","κ=8":"#8e44ad","κ(μ)":"#c0392b"}
binsf=np.linspace(0,1,21)
for nm in ["κ=2","κ=5","κ=8","κ(μ)"]:
    pred,out,w,_=inst(variants[nm]); tp,ty,tw=reliab(pred,out,w,binsf)
    ok=tw>0; ax[0].plot((tp/np.where(tw>0,tw,1))[ok],(ty/np.where(tw>0,tw,1))[ok],"-o",ms=3,color=colmap[nm],label=nm)
ax[0].plot([0,1],[0,1],"--",color="#bbb"); ax[0].set_xlabel("predicted");ax[0].set_ylabel("observed held-out freq");ax[0].set_title("A. Reliability (full)",loc="left");ax[0].legend(fontsize=8)
# low tail, LOG x, auto y
logb=np.logspace(-4,np.log10(0.3),16)
for nm in ["κ=2","κ=5","κ=8","κ(μ)"]:
    pred,out,w,_=inst(variants[nm]); tp,ty,tw=reliab(pred,out,w,logb)
    ok=tw>0; ax[1].plot((tp/np.where(tw>0,tw,1))[ok],(ty/np.where(tw>0,tw,1))[ok],"-o",ms=3,color=colmap[nm],label=nm)
ax[1].plot([1e-4,0.3],[1e-4,0.3],"--",color="#bbb"); ax[1].set_xscale("log")
ax[1].set_xlabel("predicted (log)");ax[1].set_ylabel("observed held-out freq");ax[1].set_title("B. Low-prob tail (log x) — the real problem",loc="left");ax[1].legend(fontsize=8)
# log-loss vs kappa, stratified, kappa=0 EXCLUDED, with rare stratum
ks=[2,3,4,5,8]
def llk(kap,mask_fn):
    pred,out,w,mm=inst(kap,mus); mk=mask_fn(pred,mm); return ll(pred,out,w,mk)
ax[2].plot(ks,[llk(x,lambda p,mm:np.ones_like(p,bool)) for x in ks],"-o",color="#2471a3",label="all")
ax[2].plot(ks,[llk(x,lambda p,mm:mm<0.01) for x in ks],"-o",color="#c0392b",label="rare μ<0.01")
ax[2].plot(ks,[llk(x,lambda p,mm:p<0.1) for x in ks],"-o",color="#e08a1e",label="pred<0.1")
kmll_all=ll(*inst(kmu)[:3]); ax[2].axhline(kmll_all,ls="--",color="#2471a3",lw=1)
ax[2].set_xlabel("flat κ (κ(μ)=dashed)");ax[2].set_ylabel("LOO log-loss");ax[2].set_title("C. log-loss by regime (κ=0 excluded)",loc="left");ax[2].legend(fontsize=8)
for a in ax:
    for s in("top","right"): a.spines[s].set_visible(False)
fig.suptitle("Occupancy calibration, re-examined: the low-probability regime",y=1.02,fontsize=12.5,weight="bold")
fig.tight_layout();fig.savefig(f"{OUT}/eb_calibration_v2_NY.png",dpi=145,bbox_inches="tight");print("\nWROTE eb_calibration_v2_NY.png")
