"""
EBD precompute pipeline.

Reads an eBird Basic Dataset (EBD) observations file + its Sampling Event Data
(SED) file and produces a tidy per-(species, locality, week) table of the
quantities the trip planner needs:

    n_checklists          complete checklists at this locality/week (the denominator)
    n_detections          complete checklists on which the species was reported
    freq_raw              n_detections / n_checklists           (the naive eBird "bar chart" number)
    freq_shrunk           Beta-Binomial empirical-Bayes estimate, pooled toward the parent (state) mean
    years_surveyed        # of years with adequate effort at this locality/week
    years_present         # of those years the species was detected at least once
    occupancy             years_present / years_surveyed        (inter-year consistency; vagrant filter)
    detect_given_present  detection rate restricted to present-years (P(detect | present, 1 checklist))
    p_lifer_1             occupancy * detect_given_present       (P see it on ONE checklist this year)

p_lifer(k) = occupancy * (1 - (1 - detect_given_present)^k) is exposed as a function for k checklists.

Design notes
------------
* Two-factor model. Pooled frequency conflates "present in a typical year" with
  "detectable given present". A one-winter vagrant has a high pooled freq but
  near-zero occupancy, so splitting the two crushes it. See `occupancy`.
* Shrinkage handles low sample size (the "1 of 1 = 1.0" pathology) by pooling a
  noisy local estimate toward the parent-region mean. Strength = PRIOR_STRENGTH
  pseudo-checklists. It does NOT fix the vagrant problem (that's occupancy's job).
* Everything keys on (SCIENTIFIC NAME); join the eBird taxonomy later for codes.

The only eBird-specific assumptions are column names and the completeness flag,
so pointing this at a real regional EBD extract is a path swap.
"""
from __future__ import annotations
import argparse, datetime as dt
from pathlib import Path
import pandas as pd

# --- tunable parameters -------------------------------------------------------
MIN_CHECKLISTS = 5      # a (locality, week) cell needs >= this many complete checklists to be trusted
MIN_CHECKLISTS_YEAR = 3 # a (locality, week, year) needs >= this many checklists to count as "surveyed"
PRIOR_STRENGTH = 10.0   # Beta-Binomial prior pseudo-count (higher = more shrinkage toward parent mean)
LOOKBACK_YEARS = 5      # inter-year occupancy window
# -----------------------------------------------------------------------------

EBD_COLS = ["SCIENTIFIC NAME", "COMMON NAME", "CATEGORY", "COUNTY", "COUNTY CODE",
            "STATE", "STATE CODE", "LOCALITY", "LOCALITY ID", "OBSERVATION DATE",
            "ALL SPECIES REPORTED", "APPROVED", "SAMPLING EVENT IDENTIFIER"]
SED_COLS = ["COUNTY", "COUNTY CODE", "STATE", "STATE CODE", "LOCALITY", "LOCALITY ID",
            "LATITUDE", "LONGITUDE", "OBSERVATION DATE", "ALL SPECIES REPORTED",
            "APPROVED", "SAMPLING EVENT IDENTIFIER"]


def ebird_week(date_series: pd.Series) -> pd.Series:
    """eBird pseudo-week: 4 per month, 48 per year (1..48)."""
    d = pd.to_datetime(date_series)
    wm = ((d.dt.day - 1) // 7).clip(upper=3)
    return (d.dt.month - 1) * 4 + wm + 1


def _read(path, usecols):
    # real EBD files have ~50 cols and a trailing tab; usecols by name tolerates extras
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols,
                     dtype=str, na_filter=False)
    return df


def load(ebd_path, sed_path, current_year=None):
    current_year = current_year or dt.date.today().year
    min_year = current_year - LOOKBACK_YEARS

    sed = _read(sed_path, SED_COLS)
    ebd = _read(ebd_path, EBD_COLS)

    for df in (sed, ebd):
        # the eBird Sampling Event file omits APPROVED/REVIEWED (they are per-observation,
        # carried only in the EBD); default to approved when the column is absent.
        if "APPROVED" not in df.columns:
            df["APPROVED"] = "1"
        df["ALL SPECIES REPORTED"] = df["ALL SPECIES REPORTED"].astype(int)
        df["APPROVED"] = df["APPROVED"].replace("", "1").astype(int)
        df["date"] = pd.to_datetime(df["OBSERVATION DATE"])
        df["year"] = df["date"].dt.year
        df["week"] = ebird_week(df["OBSERVATION DATE"])

    # complete + approved checklists only, within the look-back window
    def keep(df):
        return df[(df["ALL SPECIES REPORTED"] == 1) & (df["APPROVED"] == 1)
                  & (df["year"] >= min_year) & (df["year"] <= current_year)].copy()

    sed, ebd = keep(sed), keep(ebd)
    # observations only count if they sit on a complete checklist present in the SED
    valid_sei = set(sed["SAMPLING EVENT IDENTIFIER"])
    ebd = ebd[ebd["SAMPLING EVENT IDENTIFIER"].isin(valid_sei)]
    # species-level taxa only (drop spuhs, slashes, hybrids, domestics) -- exact-match rule
    ebd = ebd[ebd["CATEGORY"].isin(["species", "issf"])]
    return sed, ebd


def precompute(sed, ebd):
    LOC = ["STATE", "STATE CODE", "COUNTY", "LOCALITY", "LOCALITY ID"]

    # --- denominators: complete checklists per locality/week (and per year) ---
    den = (sed.groupby(LOC + ["week"])["SAMPLING EVENT IDENTIFIER"]
              .nunique().rename("n_checklists").reset_index())
    den_year = (sed.groupby(LOC + ["week", "year"])["SAMPLING EVENT IDENTIFIER"]
                   .nunique().rename("n_checklists_year").reset_index())

    # --- numerators: distinct complete checklists per species/locality/week ---
    sp_cols = ["SCIENTIFIC NAME", "COMMON NAME"]
    det = (ebd.groupby(LOC + ["week"] + sp_cols)["SAMPLING EVENT IDENTIFIER"]
              .nunique().rename("n_detections").reset_index())
    det_year = (ebd.groupby(LOC + ["week", "year"] + sp_cols)["SAMPLING EVENT IDENTIFIER"]
                   .nunique().rename("n_det_year").reset_index())

    # cartesian-ish join: every species that occurs in a (locality,week) gets its denominator
    df = det.merge(den, on=LOC + ["week"], how="left")
    df["freq_raw"] = df["n_detections"] / df["n_checklists"]

    # --- Beta-Binomial shrinkage toward the parent (state) mean for that species/week ---
    # parent mean m(species, state, week): pooled detections / pooled checklists across the state
    state_den = (sed.groupby(["STATE", "week"])["SAMPLING EVENT IDENTIFIER"]
                    .nunique().rename("state_checklists").reset_index())
    state_det = (ebd.groupby(["STATE", "week"] + sp_cols)["SAMPLING EVENT IDENTIFIER"]
                    .nunique().rename("state_detections").reset_index())
    parent = state_det.merge(state_den, on=["STATE", "week"], how="left")
    parent["m"] = parent["state_detections"] / parent["state_checklists"]
    df = df.merge(parent[["STATE", "week", "SCIENTIFIC NAME", "m"]],
                  on=["STATE", "week", "SCIENTIFIC NAME"], how="left")
    alpha = df["m"] * PRIOR_STRENGTH
    beta = (1 - df["m"]) * PRIOR_STRENGTH
    df["freq_shrunk"] = (df["n_detections"] + alpha) / (df["n_checklists"] + alpha + beta)

    # --- inter-year occupancy (effort-aware) ---
    # years "surveyed" at a locality/week = years with >= MIN_CHECKLISTS_YEAR checklists
    surveyed = (den_year[den_year["n_checklists_year"] >= MIN_CHECKLISTS_YEAR]
                .groupby(LOC + ["week"])["year"].nunique().rename("years_surveyed").reset_index())
    # of those, years the species was present (>=1 detection)
    present = det_year.merge(
        den_year[den_year["n_checklists_year"] >= MIN_CHECKLISTS_YEAR][LOC + ["week", "year"]],
        on=LOC + ["week", "year"], how="inner")
    years_present = (present.groupby(LOC + ["week"] + sp_cols)["year"]
                     .nunique().rename("years_present").reset_index())
    # detection rate restricted to present-years: pooled det / pooled checklists in those years
    pres_keys = present[LOC + ["week", "year"] + sp_cols].drop_duplicates()
    pres_den = pres_keys.merge(den_year, on=LOC + ["week", "year"], how="left")
    pres_den = (pres_den.groupby(LOC + ["week"] + sp_cols)["n_checklists_year"]
                .sum().rename("checklists_present_years").reset_index())
    pres_det = (present.groupby(LOC + ["week"] + sp_cols)["n_det_year"]
                .sum().rename("det_present_years").reset_index())

    df = df.merge(surveyed, on=LOC + ["week"], how="left")
    df = df.merge(years_present, on=LOC + ["week"] + sp_cols, how="left")
    df = df.merge(pres_den, on=LOC + ["week"] + sp_cols, how="left")
    df = df.merge(pres_det, on=LOC + ["week"] + sp_cols, how="left")

    df["years_surveyed"] = df["years_surveyed"].fillna(0).astype(int)
    df["years_present"] = df["years_present"].fillna(0).astype(int)
    df["occupancy"] = (df["years_present"] / df["years_surveyed"]).where(df["years_surveyed"] > 0, 0.0)
    df["detect_given_present"] = (df["det_present_years"] / df["checklists_present_years"]).fillna(0.0)
    # for planning, use the shrunk detection rate to tame low-sample present-year cells
    df["p_lifer_1"] = df["occupancy"] * df["detect_given_present"]

    df["trusted"] = df["n_checklists"] >= MIN_CHECKLISTS

    # locality coordinates (mean position per locality) so the map can place hotspots
    xy = sed.copy()
    xy["LATITUDE"] = pd.to_numeric(xy["LATITUDE"], errors="coerce")
    xy["LONGITUDE"] = pd.to_numeric(xy["LONGITUDE"], errors="coerce")
    xy = xy.groupby("LOCALITY ID")[["LATITUDE", "LONGITUDE"]].mean().reset_index()
    xy.columns = ["LOCALITY ID", "latitude", "longitude"]
    df = df.merge(xy, on="LOCALITY ID", how="left")

    cols = (LOC + ["latitude", "longitude", "week", "SCIENTIFIC NAME", "COMMON NAME",
            "n_checklists", "n_detections", "freq_raw", "freq_shrunk", "years_surveyed",
            "years_present", "occupancy", "detect_given_present", "p_lifer_1", "trusted"])
    return df[cols].sort_values(LOC + ["week", "SCIENTIFIC NAME"]).reset_index(drop=True)


def p_lifer(occupancy, detect_given_present, k=1):
    """Probability of getting the species at least once given k complete checklists this trip."""
    return occupancy * (1 - (1 - detect_given_present) ** k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent.parent / "data" / "sample"
    ap.add_argument("--ebd", default=str(base / "ebd_sample.txt"))
    ap.add_argument("--sed", default=str(base / "ebd_sample_sampling.txt"))
    ap.add_argument("--current-year", type=int, default=2026)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "data" / "precomputed.parquet"))
    a = ap.parse_args()
    sed, ebd = load(a.ebd, a.sed, a.current_year)
    table = precompute(sed, ebd)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    csv_out = a.out.replace(".parquet", ".csv")
    table.to_csv(csv_out, index=False)
    try:
        table.to_parquet(a.out, index=False)   # fast columnar format for the real (large) dataset
        print(f"Precomputed {len(table):,} cells -> {a.out} (and {csv_out})")
    except Exception:
        print(f"Precomputed {len(table):,} cells -> {csv_out} (parquet skipped: pyarrow not installed)")
