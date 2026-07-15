#!/usr/bin/env python3
"""Plot ranking calibration: model-predicted species/visit vs. observed species/checklist.

Three panels, over all (hotspot, week) cells in a state with >= --min-chk complete checklists:
  1. model @ fixed 1 h        vs emp   (what the ranking uses today)
  2. model @ site's actual mean checklist duration vs emp   (matched effort)
  3. residual (model@1h - emp) vs mean checklist duration   (shows the effort bias)

    python scripts/plot_calibration.py --store data/birdtrip.parquet --state "New York"
"""
import argparse
import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--min-chk", type=int, default=20)
    ap.add_argument("--out", default="birdtrip_calibration.png")
    a = ap.parse_args()
    st = a.state.replace("'", "''")
    pdet1 = "(1 - CASE WHEN lambda_hr IS NOT NULL THEN exp(-lambda_hr) ELSE (1-detect_given_present) END)"
    pdd = ("(1 - CASE WHEN lambda_hr IS NOT NULL THEN exp(-lambda_hr*(COALESCE(mean_dur_min,60)/60.0)) "
           "ELSE (1-detect_given_present) END)")
    q = f"""SELECT locality_id, week,
        SUM(occupancy*{pdet1}) model1, SUM(occupancy*{pdd}) model_dur,
        SUM(n_detections)::DOUBLE/NULLIF(any_value(n_checklists),0) emp,
        any_value(mean_dur_min) dur, any_value(years_surveyed) yrs
      FROM '{a.store}' WHERE trusted=1 AND state='{st}' GROUP BY locality_id, week
      HAVING any_value(n_checklists) >= {a.min_chk}"""
    df = duckdb.connect().execute(q).df().dropna(subset=["emp", "model1", "model_dur", "dur"])
    print(f"{a.state}: {len(df)} hotspot-weeks with >= {a.min_chk} checklists")
    mad = lambda x, y: float(np.median(np.abs(x - y)))
    print(f"median|model@1h - emp|  = {mad(df.model1, df.emp):.1f}")
    print(f"median|model@dur - emp| = {mad(df.model_dur, df.emp):.1f}")
    c1 = np.corrcoef(df.model1, df.emp)[0, 1]
    c2 = np.corrcoef(df.model_dur, df.emp)[0, 1]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    mx = max(df.emp.max(), df.model1.max(), df.model_dur.max()) * 1.05
    for a_, x, ttl in [(ax[0], df.model1, f"model @ fixed 1 h  (r={c1:.2f})"),
                       (ax[1], df.model_dur, f"model @ mean checklist duration  (r={c2:.2f})")]:
        sc = a_.scatter(df.emp, x, c=np.log10(df.dur.clip(5)), s=10, alpha=.5, cmap="viridis")
        a_.plot([0, mx], [0, mx], "r--", lw=1); a_.set_xlim(0, mx); a_.set_ylim(0, mx)
        a_.set_xlabel("observed species / checklist"); a_.set_ylabel("model-predicted species / visit"); a_.set_title(ttl)
    cb = fig.colorbar(sc, ax=ax[1]); cb.set_label("log10 mean checklist minutes")
    ax[2].scatter(df.dur, df.model1 - df.emp, s=10, alpha=.5, color="#c0392b")
    ax[2].axhline(0, color="k", lw=.8); ax[2].set_xscale("log")
    ax[2].set_xlabel("mean checklist duration (min, log)"); ax[2].set_ylabel("model@1h − emp")
    ax[2].set_title("1-hour model over-predicts short visits,\nunder-predicts long ones")
    fig.suptitle(f"birdtrip ranking calibration — {a.state}, all weeks (≥{a.min_chk} checklists/site-week)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
