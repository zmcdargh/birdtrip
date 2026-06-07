"""
Verification report: confirm the precompute behaves correctly on the planted
patterns, and demonstrate the 'no life list' recommendation (every species is a
potential lifer, so the planner just ranks sites/weeks by expected yield).

Run after pipeline.py.
"""
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

DATA = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA / "precomputed.csv")

def show(title, sub):
    print("\n" + "=" * 96 + f"\n{title}\n" + "=" * 96)
    cols = ["COUNTY", "LOCALITY", "week", "COMMON NAME", "n_checklists", "n_detections",
            "freq_raw", "freq_shrunk", "years_surveyed", "years_present", "occupancy",
            "detect_given_present", "p_lifer_1", "trusted"]
    print(sub[cols].to_string(index=False,
          formatters={"freq_raw": "{:.2f}".format, "freq_shrunk": "{:.2f}".format,
                      "occupancy": "{:.2f}".format, "detect_given_present": "{:.2f}".format,
                      "p_lifer_1": "{:.3f}".format}))

# 1) COMMON WIDESPREAD: Northern Cardinal, Central Park, a winter week -> stable, occupancy 1.0
show("1) COMMON (Northern Cardinal, Central Park): raw~shrunk, occupancy=1.0, stable",
     df[(df["COMMON NAME"] == "Northern Cardinal") & (df["LOCALITY"] == "Central Park")
        & (df["week"].isin([2, 20]))])

# 2) VAGRANT: Varied Thrush, Central Park, winter -> decent raw freq but occupancy ~0.2 -> p_lifer crushed
show("2) VAGRANT (Varied Thrush, Central Park, winter): pooled freq looks findable, "
     "but occupancy~0.2 (1 of 5 yrs) crushes p_lifer",
     df[(df["COMMON NAME"] == "Varied Thrush")].sort_values("freq_raw", ascending=False).head(4))

# 3) REGIONAL SPECIALTY: Saltmarsh Sparrow -> only coastal Suffolk, absent New York county
show("3) SPECIALTY (Saltmarsh Sparrow): present only in coastal Suffolk, breeding weeks",
     df[(df["COMMON NAME"] == "Saltmarsh Sparrow")].sort_values("p_lifer_1", ascending=False).head(4))

# 4) LOW-SAMPLE: Painted Bunting, 1 of 1 -> raw freq 1.0, shrunk toward parent mean, untrusted
show("4) LOW-SAMPLE (Painted Bunting, 1-of-1 checklist): raw freq=1.00 but shrunk down, trusted=False",
     df[(df["COMMON NAME"] == "Painted Bunting")])

# --- irreplaceability weight w(s): region (Suffolk coast) vs elsewhere (rest of state) ---
print("\n" + "=" * 96 + "\nIrreplaceability w(s) = attainability_in_region / attainability_elsewhere\n" + "=" * 96)
def f_star(d):  # best attainability of a species across cells in a slice
    return d.groupby("COMMON NAME")["p_lifer_1"].max()
region = df[df["COUNTY"] == "Suffolk"]          # user picked the coast
elsewhere = df[df["COUNTY"] != "Suffolk"]
fin, fout = f_star(region), f_star(elsewhere)
FLOOR = 0.01
w = (fin / fout.reindex(fin.index).fillna(0).clip(lower=FLOOR)).clip(lower=1.0)
wt = pd.DataFrame({"f*_region": fin, "f*_elsewhere": fout.reindex(fin.index).fillna(0),
                   "w(s)": w}).sort_values("w(s)", ascending=False)
print(wt.to_string(formatters={c: "{:.3f}".format for c in wt.columns}))
print("\n-> Saltmarsh Sparrow scores a high w(s): attainable in-region, ~unattainable elsewhere "
      "in the data. That is exactly the endemic/specialty pull the alpha slider amplifies.")

# --- NO LIFE LIST mode: every species is a candidate lifer; rank sites/weeks by expected yield ---
print("\n" + "=" * 96 + "\nNO-LIFE-LIST recommendation: rank (locality, week) by rarity-weighted expected species\n" + "=" * 96)
# global w(s) across the whole sample region for weighting
allbest = df.groupby("COMMON NAME")["p_lifer_1"].max()
wmap = {sp: 1.0 / max(allbest.get(sp, 0.0), 0.05) for sp in allbest.index}  # rarer -> bigger weight
trusted = df[df["trusted"]].copy()
for alpha, label in [(0.0, "alpha=0  (raw expected count)"), (1.5, "alpha=1.5 (rarity-weighted)")]:
    trusted["score"] = trusted.apply(
        lambda r: (wmap[r["COMMON NAME"]] ** alpha) * r["p_lifer_1"], axis=1)
    top = (trusted.groupby(["COUNTY", "LOCALITY", "week"])["score"].sum()
           .sort_values(ascending=False).head(5).reset_index())
    print(f"\n  {label}:")
    print(top.to_string(index=False, formatters={"score": "{:.3f}".format}))
print("\n-> alpha=0 favors the busy migrant trap (most birds); raising alpha pulls the coastal "
      "specialty site up the ranking. One knob, continuous, as designed.")
