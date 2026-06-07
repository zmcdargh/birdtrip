"""
Trip summary for a chosen destination + effort: the human-readable answer.

Given a (locality, week), a life list, and an effort level (k complete checklists
you expect to do there), report:
  - expected number of LIFERS  = sum over candidate species of p_lifer(k)
  - P(at least one lifer), and the full distribution of how many you'll get
    (Poisson-binomial: each species is an independent Bernoulli with prob p_lifer(k))
  - the likely birds, bucketed by how probable each is on the trip

p_lifer(k) = occupancy * (1 - (1 - detect_given_present)^k)
  occupancy            = P(species present in a typical year, this place/week)
  detect_given_present = P(detect on one checklist | present)

NOTE: numbers below come from the multi-year SYNTHETIC sample, so they are a
format demonstration, not calibrated predictions. Point --pre at a real precompute
(from the EBD) to get real answers; the code is identical.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def p_lifer_k(occupancy, detect_given_present, k):
    return occupancy * (1 - (1 - detect_given_present) ** k)


def poisson_binomial(probs):
    """Distribution of the count of successes among independent Bernoulli(probs)."""
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return dist  # dist[i] = P(exactly i successes)


def trip_summary(pre, locality, week, k, life_list=frozenset()):
    cell = pre[(pre["LOCALITY"] == locality) & (pre["week"] == week) & pre["trusted"]].copy()
    cell = cell[~cell["COMMON NAME"].isin(life_list)]
    cell["p_trip"] = p_lifer_k(cell["occupancy"], cell["detect_given_present"], k)
    cell = cell[cell["p_trip"] > 0.001].sort_values("p_trip", ascending=False)

    exp_lifers = cell["p_trip"].sum()
    dist = poisson_binomial(cell["p_trip"].values)
    p_at_least_1 = 1 - dist[0]
    cum = np.cumsum(dist)
    lo = int(np.searchsorted(cum, 0.10)); hi = int(np.searchsorted(cum, 0.90))

    print("=" * 74)
    print(f"  {locality} — {MONTHS[(week-1)//4]} (week {week}),  effort = {k} checklists")
    print("=" * 74)
    print(f"  Expected new lifers:        {exp_lifers:.1f}")
    print(f"  Chance of >=1 lifer:        {p_at_least_1:.0%}")
    print(f"  Likely range (10-90%):      {lo}-{hi} lifers")
    if len(life_list):
        print(f"  (excluding {len(life_list)} species already on your life list)")
    print("\n  Most likely lifers (chance of seeing on this trip):")
    buckets = [("Near-certain (>=80%)", 0.80, 1.01), ("Likely (50-80%)", 0.50, 0.80),
               ("Possible (20-50%)", 0.20, 0.50), ("Long shot (<20%)", 0.0, 0.20)]
    for label, lo_p, hi_p in buckets:
        grp = cell[(cell["p_trip"] >= lo_p) & (cell["p_trip"] < hi_p)]
        if len(grp):
            print(f"\n    {label}")
            for _, r in grp.iterrows():
                print(f"       {r['p_trip']:>4.0%}  {r['COMMON NAME']}")
    return exp_lifers


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(ROOT / "data/precomputed.csv"))
    ap.add_argument("--locality", default="Central Park")
    ap.add_argument("--week", type=int, default=18)   # mid-May migration
    ap.add_argument("--k", type=int, default=6)        # ~2 mornings of birding
    a = ap.parse_args()
    pre = pd.read_csv(a.pre)

    print("\n### New birder — empty life list (everything is a lifer)\n")
    trip_summary(pre, a.locality, a.week, a.k)

    seen = {"Northern Cardinal", "Song Sparrow", "American Robin"}
    print("\n\n### Experienced birder — common residents already ticked\n")
    trip_summary(pre, a.locality, a.week, a.k, life_list=seen)
