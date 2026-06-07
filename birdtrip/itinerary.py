"""
Greedy day-by-day itinerary builder (base-camp model).

The user picks a base (a pin + radius), a start date, and a number of days. We take the
hotspots reachable from the base as the candidate set and fill the trip one day at a time,
each day picking the reachable hotspot with the largest *marginal* expected number of new
lifers, then discounting the species we probably just got. Because the value of a stop is
conditioned on what you've likely already seen, a near-certain bird at day 1's stop barely
counts on day 2 -- so the route doesn't waste days re-finding the same species, and a second
day at one rich site (depth) vs. a new site (breadth) falls out of the greedy automatically.

Math (the soft / probabilistic life list)
------------------------------------------
For species s, P(present in the accessible area this week) is occupancy O_s, and given
present, one day of effort k at a site detects it with D = 1 - (1 - detect_given_present)^k.
Crucially, occupancy is ONE shared draw for the place/week -- it must not be re-rolled each
day -- so only the detection part D compounds across repeat visits:

    P(see s at a site over the days spent there) = O_s * (1 - prod_days (1 - D_day))

Treating different sites' presence as independent, the probability you still NEED s after the
days chosen so far is the product over visited sites of  (1 - O_{s,site} * (1 - F_{s,site})),
where F_{s,site} = prod over days at that site of (1 - D). The marginal gain of spending a day
at site L (week w) then works out to, per species,

    gain_s = stillNeeded_without_L(s) * O_{s,L} * F_{s,L} * D_{s,L,w}

(for a site's first visit F=1 and stillNeeded_without_L = stillNeeded, so this reduces to the
familiar stillNeeded * O * D). The day's site is the argmax of  sum_s gain_s * w(s)^alpha; we
report each day's biggest gain_s as "what this stop adds." Greedy on a submodular coverage
objective, so it carries the standard 1-1/e guarantee on expected coverage.
"""
from __future__ import annotations
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .recommend import month_of_week

_SUB = ["early", "mid", "late", "late"]


def date_to_week(d: date) -> int:
    """eBird 48-week index for a date, matching the precompute's WEEK expression
    ((month-1)*4 + least(3, floor((day-1)/7)) + 1)."""
    return (d.month - 1) * 4 + min(3, (d.day - 1) // 7) + 1


def week_label(w: int) -> str:
    return f"{_SUB[(w - 1) % 4]} {month_of_week(w)}"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Scalars or numpy arrays (lat2/lon2 may be arrays)."""
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(np.asarray(lat2) - lat1)
    dlon = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def trip_weeks(start: date, n_days: int) -> list[int]:
    """The distinct eBird weeks a trip of n_days starting on `start` touches."""
    return sorted({date_to_week(start + timedelta(days=i)) for i in range(n_days)})


def _parse_date(d) -> date:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def plan(cells: pd.DataFrame, sites: pd.DataFrame, start_date, n_days: int,
         hours_per_day: float = 4.0, alpha: float = 0.0, has_lambda: bool = False,
         max_per_species_report: int = 8) -> dict:
    """Greedy itinerary over a prepared candidate set.

    cells : per-(locality_id, week, species) rows with columns
            locality_id, week, species_code, common_name, occupancy, detect_given_present, w
            (and lambda_hr when has_lambda) — restricted to candidate localities, trip weeks, off
            the life list.
    sites : one row per candidate locality with locality_id, locality, state, latitude,
            longitude, dist_km (distance from base).
    Effort is hours_per_day: detection given present over a day = 1-exp(-lambda*hours) where lambda
    is available, else the k-checklist fallback with k≈round(hours). Returns a JSON dict.
    """
    kfb = max(1, int(round(hours_per_day)))
    start = _parse_date(start_date)
    n_days = int(n_days)
    days_dates = [start + timedelta(days=i) for i in range(n_days)]
    day_weeks = [date_to_week(d) for d in days_dates]

    site_meta = {r.locality_id: r for r in sites.itertuples()}

    if cells.empty or not site_meta:
        return _empty(start, n_days, hours_per_day, sites)

    # --- index the candidate data: per (locality_id, week) -> aligned arrays over a species universe
    cells = cells.dropna(subset=["occupancy", "detect_given_present"]).copy()
    cells["w"] = pd.to_numeric(cells.get("w", 1.0), errors="coerce").fillna(1.0)
    species = cells["species_code"].astype(str).to_numpy()
    universe = pd.Index(pd.unique(species))
    S = len(universe)
    sidx = {code: i for i, code in enumerate(universe)}
    names = dict(zip(cells["species_code"].astype(str), cells["common_name"]))
    # rarity weight per species (max across cells; w is a species/region property here)
    wvec = np.ones(S)
    for code, wv in cells.groupby(cells["species_code"].astype(str))["w"].max().items():
        wvec[sidx[code]] = float(wv)
    walpha = wvec ** float(alpha)

    # detection-given-present per day of effort, and occupancy, for each (locality, week)
    # D_cell[(loc, wk)] and O_cell[(loc, wk)] are length-S arrays (0 where species absent there).
    D_cell: dict = {}
    O_cell: dict = {}
    for (loc, wk), g in cells.groupby(["locality_id", "week"]):
        idx = np.array([sidx[c] for c in g["species_code"].astype(str)])
        occ = np.zeros(S); occ[idx] = np.clip(g["occupancy"].to_numpy(float), 0, 1)
        dgpv = np.clip(g["detect_given_present"].to_numpy(float), 0, 1)
        if has_lambda and "lambda_hr" in g.columns:           # time-to-detection, per-row fallback
            lam = pd.to_numeric(g["lambda_hr"], errors="coerce").to_numpy()
            dpos = np.where(np.isnan(lam), 1.0 - (1.0 - dgpv) ** kfb,
                            1.0 - np.exp(-np.nan_to_num(lam) * float(hours_per_day)))
        else:
            dpos = 1.0 - (1.0 - dgpv) ** kfb
        D = np.zeros(S); D[idx] = np.clip(dpos, 0, 1)
        O_cell[(loc, int(wk))] = occ
        D_cell[(loc, int(wk))] = D

    # --- greedy state ---------------------------------------------------------
    # F[loc] = per-species product of (1 - D) over days already spent at loc (failure-given-present).
    # A loc absent from F has been unvisited (F implicitly 1).  factor(loc) = 1 - O_loc*(1 - F_loc).
    F: dict = {}                      # loc -> length-S array
    cell_O: dict = {}                 # loc -> occupancy array used for that loc (week it was visited)
    still = np.ones(S)                # P(not yet seen) across visited sites

    def recompute_still():
        nonlocal still
        s = np.ones(S)
        for loc, f in F.items():
            s = s * (1.0 - cell_O[loc] * (1.0 - f))
        still = s

    days_out = []
    stops: dict = {}
    cum = 0.0
    for di, (the_date, wk) in enumerate(zip(days_dates, day_weeks), start=1):
        # candidate localities that actually have data this week
        cand = [loc for loc in site_meta if (loc, wk) in D_cell]
        best_loc, best_obj, best_gain = None, -1.0, None
        for loc in cand:
            O = O_cell[(loc, wk)]
            D = D_cell[(loc, wk)]
            if loc in F:                       # revisit: presence already partly "spent"
                f_cur = F[loc]
                still_wo = np.ones(S)
                for l2, f2 in F.items():
                    if l2 != loc:
                        still_wo = still_wo * (1.0 - cell_O[l2] * (1.0 - f2))
            else:
                f_cur = np.ones(S)
                still_wo = still
            gain = still_wo * O * f_cur * D     # per-species marginal P(see) increase
            obj = float(np.dot(gain, walpha))
            if obj > best_obj:
                best_obj, best_loc, best_gain = obj, loc, gain
        if best_loc is None:                    # no reachable data this week; rest day
            days_out.append({"day": di, "date": the_date.isoformat(), "week": wk,
                             "week_label": week_label(wk), "locality": None,
                             "expected_new": 0.0, "cumulative_expected": round(cum, 2), "birds": []})
            continue

        # commit the day at best_loc
        O = O_cell[(best_loc, wk)]
        D = D_cell[(best_loc, wk)]
        F[best_loc] = (F.get(best_loc, np.ones(S))) * (1.0 - D)
        cell_O[best_loc] = O                    # occupancy for this site (week of first visit)
        recompute_still()

        day_gain = float(best_gain.sum())
        cum += day_gain
        m = site_meta[best_loc]
        order = np.argsort(best_gain)[::-1]
        birds = []
        for j in order[:max_per_species_report]:
            if best_gain[j] <= 1e-4:
                break
            code = universe[j]
            birds.append({"species_code": code, "common_name": names.get(code, code),
                          "p_new_here": round(float(best_gain[j]), 3),
                          "rarity_weight": round(float(wvec[j]), 2)})
        days_out.append({
            "day": di, "date": the_date.isoformat(), "week": wk, "week_label": week_label(wk),
            "locality": m.locality, "locality_id": best_loc, "state": getattr(m, "state", None),
            "latitude": _f(m.latitude), "longitude": _f(m.longitude),
            "dist_km": round(float(m.dist_km), 1),
            "expected_new": round(day_gain, 2), "cumulative_expected": round(cum, 2),
            "revisit": best_loc in stops, "birds": birds,
        })
        stops.setdefault(best_loc, {
            "locality": m.locality, "locality_id": best_loc, "state": getattr(m, "state", None),
            "latitude": _f(m.latitude), "longitude": _f(m.longitude),
            "dist_km": round(float(m.dist_km), 1), "days": []})
        stops[best_loc]["days"].append(di)

    total = float((1.0 - still).sum())          # expected distinct lifers over whole trip
    return {
        "start_date": start.isoformat(), "n_days": n_days, "hours_per_day": round(float(hours_per_day), 1),
        "alpha": float(alpha), "n_candidate_sites": len(site_meta),
        "expected_lifers_total": round(total, 1),
        "days": days_out,
        "stops": list(stops.values()),
    }


def _empty(start, n_days, hours_per_day, sites):
    return {"start_date": start.isoformat(), "n_days": int(n_days),
            "hours_per_day": round(float(hours_per_day), 1),
            "n_candidate_sites": 0 if sites is None else int(len(sites)),
            "expected_lifers_total": 0.0, "days": [], "stops": [],
            "message": "No hotspots with data were reachable from this base. Try a larger radius."}


def _f(v):
    return None if pd.isna(v) else round(float(v), 5)
