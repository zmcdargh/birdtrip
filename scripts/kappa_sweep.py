"""
Occupancy-kappa sweep over the held-out trials saved by validate_holdout.py --save-trials.
Scores BOTH the occupancy term (vs held-out presence, from ev_occ) and the served detection
probability (vs held-out detections, from ev), under each kappa, raw and recalibrated (isotonic
fit on the calib block). Prints stratified log-loss/Brier and writes reliability figures.

Usage:
  python scripts/kappa_sweep.py --trials data/holdout_trials_NY --out viz --kappas 0,3,4,5,8,mu
"""
import argparse, json, numpy as np, pandas as pd, duckdb
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EPS=1e-4

def isotonic(x, y, w):
    o=np.argsort(x, kind="mergesort"); x=x[o]; y=y[o].astype(float); w=w[o].astype(float)
    val=[]; wt=[]; cnt=[]
    for i in range(len(y)):
        val.append(y[i]); wt.append(w[i]); cnt.append(1)
        while len(val)>1 and val[-2]>val[-1]:
            v2,w2,c2=val.pop(),wt.pop(),cnt.pop(); v1,w1,c1=val.pop(),wt.pop(),cnt.pop()
            val.append((v1*w1+v2*w2)/(w1+w2)); wt.append(w1+w2); cnt.append(c1+c2)
    yh=np.empty(len(y)); k=0
    for v,c in zip(val,cnt): yh[k:k+c]=v; k+=c
    return x, np.clip(yh, EPS, 1-EPS)

def occ_eb(yp, ns, prior, kap):
    return np.clip((yp + kap*prior)/(ns + kap), EPS, 1-EPS)

def kappa_mu_curve(uc):                       # fit kappa(mu) from unique cells (yp,ns,prior)
    edges=np.linspace(0,1,21); kbin=[]
    yp=uc.yp.to_numpy(float); ns=uc.ns.to_numpy(float); pr=uc.prior.to_numpy(float)
    for i in range(20):
        s=(pr>=edges[i])&(pr<edges[i+1])&(ns>=2)
        if s.sum()<8: kbin.append(np.nan); continue
        kk,nn=yp[s],ns[s]; N=nn.sum(); S=s.sum(); mu=kk.sum()/N
        if not (0<mu<1): kbin.append(np.nan); continue
        Snp2=(kk*kk/nn).sum()-mu*mu*N; Sn2=(nn*nn).sum(); D=N-Sn2/N
        if D<=0: kbin.append(np.nan); continue
        tau2=(Snp2-(S-1)*mu*(1-mu))/D
        kbin.append(500.0 if tau2<=0 else float(np.clip(mu*(1-mu)/tau2-1,0.5,500)))
    kbin=pd.Series(kbin).ffill().bfill().to_numpy()
    return (edges[:-1]+edges[1:])/2, kbin

def wll(p,y,w):  return float(np.sum(w*-(y*np.log(p)+(1-y)*np.log(1-p)))/w.sum())
def wbr(p,y,w):  return float(np.sum(w*(p-y)**2)/w.sum())

def reliab(p,y,w,bins):
    idx=np.clip(np.digitize(p,bins)-1,0,len(bins)-2)
    tw=np.bincount(idx,weights=w,minlength=len(bins)-1); ty=np.bincount(idx,weights=w*y,minlength=len(bins)-1); tp=np.bincount(idx,weights=w*p,minlength=len(bins)-1)
    ok=tw>0; return (tp/np.where(tw>0,tw,1))[ok], (ty/np.where(tw>0,tw,1))[ok]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--trials",required=True); ap.add_argument("--out",default="viz")
    ap.add_argument("--kappas",default="0,3,4,5,8,mu")
    ap.add_argument("--kappa-fig",default="3",help="which kappa to draw the served-prob calibration figure for")
    ap.add_argument("--save-recal-map",default=None,help="write the isotonic recal map (fit on the SHRUNK served product, calib block) as JSON for service.py")
    a=ap.parse_args(); T=Path(a.trials); outd=Path(a.out); outd.mkdir(parents=True,exist_ok=True)
    meta=json.load(open(T/"meta.json")); recal=meta.get("recal",False)
    con=duckdb.connect()
    rd=lambda f:con.execute(f"SELECT * FROM '{T/f}'").df()
    occ=rd("ev_occ.parquet"); ev=rd("ev.parquet")
    occ_c=rd("cal_occ.parquet") if recal else None; ev_c=rd("cal.parquet") if recal else None
    for df in [occ,ev,occ_c,ev_c]:
        if df is not None: df["prior"]=df["prior"].fillna(0.0)   # missing regional prior -> 0 (rare)
    uc=occ.drop_duplicates(["locid","week","species_code"]) if {"locid","week","species_code"}<=set(occ.columns) else occ
    cx,cy=kappa_mu_curve(uc)
    def kvec(df,kap): return np.interp(df.prior.to_numpy(float),cx,cy) if kap=="mu" else float(kap)
    kaps=a.kappas.split(",")

    # ---------- OCCUPANCY (vs held-out presence) ----------
    print("="*70,"\nOCCUPANCY held-out calibration (rows = trained cell x surveyed test-year)")
    print(f"{'kappa':>6}{'logloss':>10}{'brier':>9}{'ll(p<0.1)':>11}{'ll(p>=.5)':>11}", end="")
    print(f"{'  ll_recal':>11}" if recal else "")
    yO=occ.present.to_numpy(float); wO=np.ones(len(occ))
    occ_rel={}
    for kp in kaps:
        kv=kvec(occ,kp); p=occ_eb(occ.yp.to_numpy(float),occ.ns.to_numpy(float),occ.prior.to_numpy(float),kv)
        row=f"{kp:>6}{wll(p,yO,wO):>10.4f}{wbr(p,yO,wO):>9.4f}{wll(p[p<0.1],yO[p<0.1],wO[p<0.1]) if (p<0.1).any() else float('nan'):>11.4f}{wll(p[p>=0.5],yO[p>=0.5],wO[p>=0.5]) if (p>=0.5).any() else float('nan'):>11.4f}"
        if recal:
            kc=kvec(occ_c,kp); pc_=occ_eb(occ_c.yp.to_numpy(float),occ_c.ns.to_numpy(float),occ_c.prior.to_numpy(float),kc)
            xs,yh=isotonic(pc_,occ_c.present.to_numpy(float),np.ones(len(occ_c)))
            pr=np.clip(np.interp(p,xs,yh),EPS,1-EPS); row+=f"{wll(pr,yO,wO):>11.4f}"
        print(row); occ_rel[kp]=p
    # ---------- SERVED detection prob (vs held-out detections) ----------
    print("\nSERVED detection-prob held-out calibration (weighted by checklist trials)")
    print(f"{'kappa':>6}{'logloss':>10}{'brier':>9}{'ll(p<0.1)':>11}", end=""); print(f"{'  ll_recal':>11}" if recal else "")
    nck=ev.n_chk.to_numpy(float); ndt=ev.n_det.to_numpy(float)
    def sll(p): return float(-np.sum(ndt*np.log(p)+(nck-ndt)*np.log(1-p))/nck.sum())
    def sbr(p): return float(np.sum(ndt*(1-p)**2+(nck-ndt)*p**2)/nck.sum())
    serv_rel={}
    for kp in kaps:
        kv=kvec(ev,kp); occv=occ_eb(ev.yp.to_numpy(float),ev.ns.to_numpy(float),ev.prior.to_numpy(float),kv)
        p=np.clip(occv*(1-np.exp(-ev.lam.to_numpy(float)*ev.th.to_numpy(float))),EPS,1-EPS)
        lowm=p<0.1
        row=f"{kp:>6}{sll(p):>10.4f}{sbr(p):>9.4f}{(float(-np.sum((ndt*np.log(p)+(nck-ndt)*np.log(1-p))[lowm])/nck[lowm].sum()) if lowm.any() and nck[lowm].sum()>0 else float('nan')):>11.4f}"
        if recal:
            kc=kvec(ev_c,kp); occc=occ_eb(ev_c.yp.to_numpy(float),ev_c.ns.to_numpy(float),ev_c.prior.to_numpy(float),kc)
            pc_=np.clip(occc*(1-np.exp(-ev_c.lam.to_numpy(float)*ev_c.th.to_numpy(float))),EPS,1-EPS)
            g=pd.DataFrame({"p":np.round(pc_*400)/400.0,"n":ev_c.n_chk.to_numpy(float),"d":ev_c.n_det.to_numpy(float)}).groupby("p").sum()
            xs,yh=isotonic(g.index.to_numpy(float),(g.d/g.n).to_numpy(float),g.n.to_numpy(float))
            pr=np.clip(np.interp(p,xs,yh),EPS,1-EPS); row+=f"{sll(pr):>11.4f}"
        print(row); serv_rel[kp]=p

    # ---------- SERVED-PROB CALIBRATION FIGURE for chosen kappa: raw vs recal (+ raw-occupancy ref) ----------
    KF=a.kappa_fig
    def served_p(df,kap):
        kv=kvec(df,kap); occv=occ_eb(df.yp.to_numpy(float),df.ns.to_numpy(float),df.prior.to_numpy(float),kv)
        return np.clip(occv*(1-np.exp(-df.lam.to_numpy(float)*df.th.to_numpy(float))),EPS,1-EPS)
    base=ndt.sum()/nck.sum(); bclim=base*(1-base)
    p_shr=served_p(ev,KF)
    p_rawocc=np.clip(ev.occ.to_numpy(float)*(1-np.exp(-ev.lam.to_numpy(float)*ev.th.to_numpy(float))),EPS,1-EPS)
    p_recal=p_shr.copy()
    if recal:
        pc_=served_p(ev_c,KF)
        g=pd.DataFrame({"p":np.round(pc_*400)/400.0,"n":ev_c.n_chk.to_numpy(float),"d":ev_c.n_det.to_numpy(float)}).groupby("p").sum()
        xs,yh=isotonic(g.index.to_numpy(float),(g.d/g.n).to_numpy(float),g.n.to_numpy(float))
        p_recal=np.clip(np.interp(p_shr,xs,yh),EPS,1-EPS)
        if a.save_recal_map:   # 1001-point grid in service.py's format: index i -> predicted p=i/1000
            gi=np.arange(1001); calg=np.clip(np.interp(gi/1000.0,xs,yh),EPS,1-EPS)
            json.dump({"method":"isotonic","grid_n":1000,"kappa":KF,
                       "fit_on":f"calib {meta.get('calib_lo')}-{meta.get('calib_hi')} (shrunk product)",
                       "calibrated":[round(float(x),6) for x in calg]}, open(a.save_recal_map,"w"))
            print("wrote recal map ->",a.save_recal_map)
    def servrel(p,bins):
        idx=np.clip(np.digitize(p,bins)-1,0,len(bins)-2)
        tn=np.bincount(idx,weights=nck,minlength=len(bins)-1); td=np.bincount(idx,weights=ndt,minlength=len(bins)-1); tp=np.bincount(idx,weights=p*nck,minlength=len(bins)-1)
        ok=tn>0; return (tp/np.where(tn>0,tn,1))[ok],(td/np.where(tn>0,tn,1))[ok]
    def sbrier(p): return float(np.sum(ndt*(1-p)**2+(nck-ndt)*p**2)/nck.sum())
    series=[("raw occupancy × detect","#999",p_rawocc),(f"shrunk κ={KF} × detect","#c0392b",p_shr)]
    if recal: series.append((f"shrunk κ={KF} + recal","#27ae60",p_recal))
    figS,axS=plt.subplots(1,2,figsize=(13,5.2)); bf=np.linspace(0,1,16); bl=np.logspace(-3,np.log10(0.5),16)
    for lab,col,p in series:
        px,py=servrel(p,bf); axS[0].plot(px,py,"-o",ms=4,color=col,label=f"{lab}  (Brier {sbrier(p):.3f}, skill {1-sbrier(p)/bclim:.0%})")
        lx,ly=servrel(p,bl); axS[1].plot(lx,ly,"-o",ms=4,color=col,label=lab)
    axS[0].plot([0,1],[0,1],"--",color="#bbb"); axS[0].axhline(base,color="#ccc",lw=.8,ls=":")
    axS[0].set_xlim(0,1);axS[0].set_ylim(0,1);axS[0].set_aspect("equal")
    axS[0].set_xlabel("predicted served probability");axS[0].set_ylabel("observed detection rate");axS[0].set_title("Served calibration (full)",loc="left");axS[0].legend(fontsize=8,loc="upper left")
    _ref=np.logspace(-3,np.log10(.5),200); axS[1].plot(_ref,_ref,"--",color="#bbb");axS[1].set_xscale("log")
    axS[1].set_xlabel("predicted (log)");axS[1].set_ylabel("observed detection rate");axS[1].set_title("Served calibration — low-p tail (log x)",loc="left");axS[1].legend(fontsize=8)
    for x in axS:
        for s in ("top","right"): x.spines[s].set_visible(False)
    figS.suptitle(f"SERVED probability calibration (occupancy×detection) — shrunk κ={KF}, raw vs recal — test {meta['test_lo']}-{meta['current_year']}",y=1.02,weight="bold")
    figS.tight_layout(); figS.savefig(outd/"kappa_sweep_served.png",dpi=145,bbox_inches="tight"); print(f"wrote {outd/'kappa_sweep_served.png'}")
    # served reliability CSV
    rows=[]
    for lab,_,p in series:
        px,py=servrel(p,bf)
        for a_,b_ in zip(px,py): rows.append((lab,float(a_),float(b_)))
    pd.DataFrame(rows,columns=["series","predicted","observed"]).to_csv(outd/"kappa_sweep_served_reliability.csv",index=False)
    print(f"wrote {outd/'kappa_sweep_served_reliability.csv'}")

    # ---------- figures ----------
    show=[k for k in kaps if k in ("0","5","mu")] or kaps[:3]
    cm={"0":"#999","3":"#27ae60","4":"#16a085","5":"#2471a3","8":"#8e44ad","mu":"#c0392b"}
    fig,ax=plt.subplots(1,2,figsize=(12,5)); bf=np.linspace(0,1,21); bl=np.logspace(-4,np.log10(0.3),16)
    for kp in show:
        p=occ_rel[kp]; px,py=reliab(p,yO,wO,bf); ax[0].plot(px,py,"-o",ms=3,color=cm.get(kp,"#333"),label=f"κ={kp}")
        lx,ly=reliab(p,yO,wO,bl); ax[1].plot(lx,ly,"-o",ms=3,color=cm.get(kp,"#333"),label=f"κ={kp}")
    ax[0].plot([0,1],[0,1],"--",color="#bbb"); ax[0].set_title("Occupancy reliability (full)",loc="left"); ax[0].set_xlabel("predicted"); ax[0].set_ylabel("observed held-out presence"); ax[0].legend(fontsize=8)
    _refo=np.logspace(-4,np.log10(.3),200); ax[1].plot(_refo,_refo,"--",color="#bbb"); ax[1].set_xscale("log"); ax[1].set_title("Occupancy reliability — low-p tail (log x)",loc="left"); ax[1].set_xlabel("predicted (log)"); ax[1].set_ylabel("observed"); ax[1].legend(fontsize=8)
    for x in ax:
        for s in ("top","right"): x.spines[s].set_visible(False)
    fig.suptitle(f"Occupancy κ sweep — held-out test {meta['test_lo']}-{meta['current_year']}",y=1.02,weight="bold")
    fig.tight_layout(); fig.savefig(outd/"kappa_sweep_occupancy.png",dpi=145,bbox_inches="tight"); print(f"\nwrote {outd/'kappa_sweep_occupancy.png'}")

if __name__=="__main__":
    main()
