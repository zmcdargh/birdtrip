"""
Rank destinations by rarity-weighted expected lifers, and SHOW THE WORK: for every
recommended (place, week), the species driving the score and how much each contributes.

Scoring for one cell c (a place x week):
    score(c) = sum_{s not on life list}  w(s)^alpha  *  p_lifer(s, c)
where p_lifer = occupancy * detect_given_present (chance of seeing s on one checklist
there that week), and w(s) is the rarity / irreplaceability weight.

The pure functions (rarity_weights_from, score_cells, rank_destinations) take plain
DataFrames + an elsewhere-best mapping, so the API service and this CLI share one
implementation. Column names are parameterized (the CLI keys on 'COMMON NAME'; the
SQLite-backed service keys on 'species_code').
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# floor on a species' attainability OUTSIDE the region, used as the ratio denominator.
# Must be small: a true endemic (≈0 elsewhere) should yield a large weight, not be capped.
# At 1e-3, a bird absent elsewhere with in-region frequency f gets w ≈ f/1e-3.
ELSEWHERE_FLOOR = 1e-3
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def month_of_week(w):
    return MONTHS[(w - 1) // 4]


# --- pure scoring (shared by CLI and API) ------------------------------------
def rarity_weights_from(region_df, elsewhere_best: dict, key="species_code", occ_gate=0.5) -> dict:
    """w(s) = f*(s, region) / max(f*(s, elsewhere), floor), gated on reliable presence.

    region_df:       cells inside the selected region
    elsewhere_best:  {species: best p_lifer outside the region} (the 'rest of world')
    A species earns w>1 only if attainable in-region AND reliably present
    (best occupancy >= occ_gate); otherwise w=1 so vagrants can't pose as specialties."""
    fin = region_df.groupby(key)["p_lifer_1"].max()
    occ = region_df.groupby(key)["occupancy"].max()
    w = {}
    for sp in fin.index:
        if occ.get(sp, 0) < occ_gate:
            w[sp] = 1.0
        else:
            w[sp] = max(1.0, fin[sp] / max(elsewhere_best.get(sp, 0.0), ELSEWHERE_FLOOR))
    return w


def score_cells(region_df, weights: dict, alpha: float, life_list=frozenset(),
                key="species_code", k: int = 1) -> pd.DataFrame:
    """Score each candidate species in each cell at effort k:

        contrib = w(s)^alpha * occupancy * (1 - (1 - detect_given_present)^k)

    Ranking on this k-aware probability (not the per-checklist rate) is what lets a
    rarity-rich site overtake a common-but-quickly-saturated one as effort grows:
    each species' detection curve bends at its own rate, so the summed expectation
    reorders sites with k. With k=1 this reduces to the per-checklist p_lifer_1."""
    cand = region_df[~region_df[key].isin(life_list)].copy()
    cand["w"] = cand[key].map(weights).fillna(1.0)
    cand["p_eff"] = cand["occupancy"] * (1 - (1 - cand["detect_given_present"]) ** k)
    cand["contrib"] = (cand["w"] ** alpha) * cand["p_eff"]
    return cand


def rank_destinations(scored: pd.DataFrame, topn=5, loc="locality"):
    scores = scored.groupby([loc, "week"])["contrib"].sum().sort_values(ascending=False)
    out = []
    for (l, wk), sc in scores.head(topn).items():
        rows = scored[(scored[loc] == l) & (scored["week"] == wk)].sort_values("contrib", ascending=False)
        out.append((l, wk, sc, rows))
    return out


# --- CLI-flavored convenience wrappers (key on COMMON NAME, in-memory mask) ---
def rarity_weights(df_all, region_mask, occ_gate=0.5):
    region, elsewhere = df_all[region_mask], df_all[~region_mask]
    elsewhere_best = elsewhere.groupby("COMMON NAME")["p_lifer_1"].max().to_dict()
    w = rarity_weights_from(region, elsewhere_best, key="COMMON NAME", occ_gate=occ_gate)
    fin = region.groupby("COMMON NAME")["p_lifer_1"].max()
    occ = region.groupby("COMMON NAME")["occupancy"].max()
    meta = pd.DataFrame({"f_region": fin,
                         "f_elsewhere": pd.Series(elsewhere_best).reindex(fin.index).fillna(0),
                         "best_occ": occ.reindex(fin.index).fillna(0)})
    return w, meta


def recommend(df, alpha, region_mask=None, life_list=frozenset(), occ_gate=0.5, topn=5):
    if region_mask is None:
        region_mask = pd.Series(True, index=df.index)
    region = df[region_mask]
    elsewhere_best = df[~region_mask].groupby("COMMON NAME")["p_lifer_1"].max().to_dict()
    w = rarity_weights_from(region, elsewhere_best, key="COMMON NAME", occ_gate=occ_gate)
    scored = score_cells(region, w, alpha, life_list, key="COMMON NAME")
    return rank_destinations(scored, topn, loc="LOCALITY")


def show(title, recs):
    print("\n" + "#" * 92 + f"\n# {title}\n" + "#" * 92)
    for i, (loc, wk, sc, rows) in enumerate(recs, 1):
        print(f"\n{i}. {loc} — {month_of_week(wk)} (wk{wk:>2})   score {sc:.2f}")
        for _, r in rows.head(5).iterrows():
            print(f"     {r['COMMON NAME']:<20} p(see on a checklist)={r['p_lifer_1']:.2f}  "
                  f"occupancy={r['occupancy']:.2f}  rarity_w={r['w']:.1f}  -> contributes {r['contrib']:.2f}")


if __name__ == "__main__":
    pre = pd.read_csv(Path(__file__).resolve().parent.parent / "data/precomputed.csv")
    pre = pre[pre["trusted"]].copy()
    NY = pre["COUNTY"] == "New York"    # selecting Central Park's county; coast acts as 'elsewhere'
    show("alpha = 0   (maximize expected new species) — region: New York Co.",
         recommend(pre, alpha=0.0, region_mask=NY))
    show("alpha = 1.5 (favor specialties, occupancy-gated) — region: New York Co.",
         recommend(pre, alpha=1.5, region_mask=NY))
    w, meta = rarity_weights(pre, NY, occ_gate=0.5)
    meta = meta.assign(w=[w[s] for s in meta.index]).sort_values("w", ascending=False)
    print("\nregion-vs-elsewhere irreplaceability weights (occupancy-gated at 0.5):")
    print(meta.to_string(formatters={c: "{:.3f}".format for c in meta.columns}))
