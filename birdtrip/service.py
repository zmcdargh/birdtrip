"""
Service layer: query the store for a region/season and run the scoring + summary.
Returns plain JSON-serializable dicts, so the API is a thin wrapper over this.
"""
from __future__ import annotations
import os
import re
import threading
import numpy as np
import pandas as pd

from .store import Store
from .recommend import rank_destinations, month_of_week, ELSEWHERE_FLOOR
from .summary import p_lifer_k, poisson_binomial
from . import itinerary as _itin

_DUCK = None            # shared DuckDB connection (created lazily)
_DUCK_LOCK = threading.Lock()   # serialize access: FastAPI serves requests on multiple threads,
                                # and a single DuckDB connection isn't safe for concurrent queries


def _parquet(store: Store):
    """Path to the Parquet sidecar next to the SQLite store, if it exists."""
    base = store.db_path
    pq = (base[:-7] if base.endswith(".sqlite") else base[:-3] if base.endswith(".db") else base) + ".parquet"
    return pq if os.path.exists(pq) else None


def _duck():
    global _DUCK
    if _DUCK is None:
        import duckdb
        _DUCK = duckdb.connect()
        _DUCK.execute("PRAGMA threads=4;")
    return _DUCK


def _q(sql):
    """Run a DuckDB query and return a DataFrame. Uses a fresh cursor under a lock so concurrent
    requests (FastAPI threadpool) can't corrupt each other's result sets on the shared connection.
    Values are inlined (not bound) — see _lit/_inlist."""
    with _DUCK_LOCK:
        return _duck().cursor().execute(sql).fetchdf()


def _lit(v) -> str:
    """SQL string literal, single-quote-escaped (values are our own controlled codes/names)."""
    return "'" + str(v).replace("'", "''") + "'"


def _inlist(vals) -> str:
    return "(" + ",".join(_lit(v) for v in vals) + ")"


_SPECIES_CACHE: dict = {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())   # drop spaces/punctuation for forgiving match


def species_search(store: Store, q: str, limit: int = 12) -> list[dict]:
    """Autocomplete over species that actually exist in the store, matched forgivingly
    (case/space/hyphen/apostrophe-insensitive) so 'florida scrub jay' finds 'Florida Scrub-Jay'."""
    qn = _norm(q)
    if not qn:
        return []
    key = store.db_path
    if key not in _SPECIES_CACHE:
        pq = _parquet(store)
        df = (_q(f"SELECT DISTINCT species_code, common_name FROM '{pq}' WHERE species_code IS NOT NULL")
              if pq else store.query_cells()[["species_code", "common_name"]].drop_duplicates())
        df = df.dropna(subset=["common_name"])
        df["norm"] = df["common_name"].map(_norm)
        _SPECIES_CACHE[key] = df
    df = _SPECIES_CACHE[key]
    pre = df[df["norm"].str.startswith(qn)]
    sub = df[df["norm"].str.contains(qn, regex=False) & ~df.index.isin(pre.index)]
    hits = pd.concat([pre, sub]).head(limit)
    return [{"species_code": r.species_code, "common_name": r.common_name} for r in hits.itertuples()]


def _cell_scores(pq, alpha, k, states=None, weeks=None, life_list=(), locality_ids=None,
                 targets=None) -> pd.DataFrame:
    """Per-(hotspot, week) score, computed in DuckDB over the Parquet. Returns a small frame.
    `targets` restricts the candidate species to those (search FOR those birds); otherwise the
    candidate pool is all species minus the life list."""
    where = ["trusted=1"]
    if states:
        where.append(f"state IN {_inlist(states)}")
    if weeks:
        where.append(f"week IN ({','.join(str(int(w)) for w in weeks)})")
    if locality_ids:
        where.append(f"locality_id IN {_inlist(locality_ids)}")
    if targets:
        where.append(f"species_code IN {_inlist(targets)}")
    elif life_list:
        where.append(f"species_code NOT IN {_inlist(life_list)}")
    sql = (f"SELECT locality_id, week, SUM(pow(w,{float(alpha)}) * occupancy * "
           f"(1 - pow(1 - detect_given_present, {int(k)}))) AS score "
           f"FROM '{pq}' WHERE {' AND '.join(where)} GROUP BY locality_id, week")
    return _q(sql)

_SUB = ["early", "mid", "late", "late"]   # eBird has 4 pseudo-weeks per month


def week_label(w: int) -> str:
    return f"{_SUB[(w - 1) % 4]} {month_of_week(w)}"


def week_range_label(lo: int, hi: int) -> str:
    if hi - lo >= 40:                       # strong almost all year
        return "year-round"
    return week_label(lo) if lo == hi else f"{week_label(lo)} – {week_label(hi)}"


def _detect_ci(occ, a, b, k, z=1.2816):
    """80% credible interval for the trip detection probability, propagating the Beta
    posterior on the per-checklist rate (normal approx) through 1-(1-q)^k (monotonic in q)."""
    s = a + b
    mean = a / s
    sd = (a * b / (s * s * (s + 1))) ** 0.5
    qlo = max(0.0, min(1.0, mean - z * sd))
    qhi = max(0.0, min(1.0, mean + z * sd))
    lo = max(0.0, min(1.0, occ * (1 - (1 - qlo) ** k)))
    hi = max(0.0, min(1.0, occ * (1 - (1 - qhi) ** k)))   # never exceeds 100%
    return round(lo, 3), round(hi, 3)


# Static caches only (small): the region-relative weights and the region selector list.
# Both depend on the whole dataset but reduce to small aggregates. The per-request cell slice
# is queried from SQL (region-bounded), so memory stays flat at US scale — nothing holds the
# full table. warm() pays these one-time scans at startup.
_WEIGHTS_CACHE: dict = {}
_REGIONS_CACHE: dict = {}
_C = 5.0   # Beta prior strength


def weights(store: Store, occ_gate: float = 0.5) -> dict:
    key = (store.db_path, occ_gate)
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE[key] = _regionwise_weights(store.species_best_by_state(), occ_gate)
    return _WEIGHTS_CACHE[key]


_FAST: dict = {}        # does this store have the precomputed w/det_a/det_b columns?
_FULL: dict = {}        # cached fully-prepared frame, ONLY for old stores lacking those columns


def _is_fast(store: Store) -> bool:
    if store.db_path not in _FAST:
        import sqlite3
        cols = {r[1] for r in sqlite3.connect(store.db_path).execute("PRAGMA table_info(cells)")}
        _FAST[store.db_path] = {"w", "det_a", "det_b"}.issubset(cols)
    return _FAST[store.db_path]


def _slice(store, occ_gate=0.5, states=None, state=None, county=None,
           weeks=None, locality_id=None, locality_ids=None) -> pd.DataFrame:
    """Return a prepared cell slice for the given filter. Fast stores query SQL per request
    (flat memory); old stores fall back to a cached prepared frame sliced in memory (so they
    stay responsive without a rebuild)."""
    if _is_fast(store):
        df = store.query_cells(states=states, state=state, county=county, weeks=weeks,
                               locality_id=locality_id, locality_ids=locality_ids)
        return _prepare(df, store, occ_gate) if not df.empty else df
    key = (store.db_path, occ_gate)
    if key not in _FULL:
        _FULL[key] = _prepare(store.query_cells(), store, occ_gate)
    df = _FULL[key]
    if states:
        df = df[df["state"].isin(states)]
    elif state:
        df = df[df["state"] == state]
    if county:
        df = df[df["county"] == county]
    if weeks:
        df = df[df["week"].isin(weeks)]
    if locality_id:
        df = df[df["locality_id"] == locality_id]
    if locality_ids:
        df = df[df["locality_id"].isin(locality_ids)]
    return df.copy()


def warm(store: Store) -> None:
    """Warm the caches at startup. Parquet stores skip the SQLite weight/cell scans entirely
    (the Parquet path uses precomputed weights and queries columnar data directly)."""
    if _parquet(store):
        regions(store, "state")    # fast DuckDB GROUP BY; also primes the Parquet read
        return
    weights(store)
    regions(store, "state")
    if not _is_fast(store):
        _slice(store)


def _prepare(df: pd.DataFrame, store: Store, occ_gate: float) -> pd.DataFrame:
    """Make a region/locality slice ready for scoring. If the store was built by the DuckDB
    precompute it already has det_a/det_b and w (static work done at build time) — so this is
    just a numeric coercion. Older/CSV stores fall back to computing those here."""
    for c in ["occupancy", "detect_given_present", "p_lifer_1", "n_detections", "n_checklists",
              "latitude", "longitude", "week", "trusted", "det_a", "det_b", "w",
              "det_present", "chk_present", "prior_freq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "trusted" in df.columns:
        df = df[df["trusted"] == 1]
    df = df.copy()
    if {"det_a", "det_b", "w"}.issubset(df.columns):
        return df                                   # fast path: everything precomputed in the store
    # --- fallback for stores without precomputed weight/interval columns ---
    df = (df.sort_values("p_lifer_1", ascending=False)
            .drop_duplicates(["locality_id", "week", "species_code"], keep="first"))
    if {"det_present", "chk_present", "prior_freq"}.issubset(df.columns):
        m = df["prior_freq"].clip(0, 1)
        a = df["det_present"] + _C * m
        b = (df["chk_present"] - df["det_present"]).clip(lower=0) + _C * (1 - m)
        df["det_a"], df["det_b"] = a, b
        df["detect_given_present"] = a / (a + b)
        df["p_lifer_1"] = df["occupancy"] * df["detect_given_present"]
    else:
        M = df["n_checklists"].clip(lower=1) + _C
        dgp = df["detect_given_present"].clip(0, 1)
        df["det_a"], df["det_b"] = dgp * M, (1 - dgp) * M
    W = weights(store, occ_gate)
    idx = pd.MultiIndex.from_arrays([df["state"], df["species_code"]])
    df["w"] = pd.Series(W).reindex(idx).fillna(1.0).to_numpy()
    return df


def _regionwise_weights(state_best, occ_gate: float) -> dict:
    """w[(state, species)] = f*(species, state) / max(f*(species, other states), floor),
    gated on the species being reliably present in that state (occupancy >= occ_gate).

    Always region-relative: with one state selected this is region-vs-rest; in a global
    search every state is weighed against the others, so a state's exclusive specialties
    lift its cells as alpha rises. A species attainable everywhere -> w≈1 everywhere."""
    W = {}
    for sp, grp in state_best.groupby("species_code"):
        p = dict(zip(grp.state, grp.p))
        occ = dict(zip(grp.state, grp.occ))
        for st in p:
            others = [p[s2] for s2 in p if s2 != st]
            f_else = max(others) if others else 0.0
            W[(st, sp)] = (1.0 if occ[st] < occ_gate
                           else max(1.0, p[st] / max(f_else, ELSEWHERE_FLOOR)))
    return W


def recommend_trips(store: Store, life_list=(), states=None, state=None, county=None, weeks=None,
                    k=1, alpha=1.0, occ_gate=0.5, topn=5, targets=None) -> list[dict]:
    """Rank (locality, week) destinations by rarity-weighted expected lifers at effort k.
    `states` is a list (multi-select); empty/None means search everywhere. `targets` (species
    codes) restricts the search to those birds — every other filter is just a restriction too."""
    targets = list(targets) if targets else None
    pq = _parquet(store)
    if pq:   # scalable path: score in DuckDB over Parquet, fetch details for top hotspots only
        return _recommend_via_parquet(store, pq, list(life_list), states or ([state] if state else None),
                                      weeks, k, alpha, occ_gate, topn, targets)

    region = _slice(store, occ_gate, states=states, state=state, county=county, weeks=weeks)
    if region.empty:
        return []
    if targets:
        region = region[region["species_code"].isin(set(targets))]
    elif life_list:
        region = region[~region["species_code"].isin(set(life_list))]
    if region.empty:
        return []
    cand = region.assign(
        p_eff=region["occupancy"] * (1 - (1 - region["detect_given_present"]) ** k))
    cand["contrib"] = (cand["w"] ** alpha) * cand["p_eff"]
    # per-(hotspot, week) score, then DE-DUPLICATE to one card per hotspot: a single hotspot
    # would otherwise fill the top of the list with its consecutive good weeks. We keep each
    # hotspot's best week as the representative and report the span of its strong weeks.
    cell_scores = cand.groupby(["locality_id", "week"], as_index=False)["contrib"].sum()
    peak_idx = cell_scores.groupby("locality_id")["contrib"].idxmax()
    peaks = cell_scores.loc[peak_idx].sort_values("contrib", ascending=False).head(topn)

    out = []
    for prow in peaks.itertuples():
        locid, best_wk, peak = prow.locality_id, int(prow.week), float(prow.contrib)
        strong = cell_scores[(cell_scores["locality_id"] == locid)
                             & (cell_scores["contrib"] >= 0.8 * peak)]["week"]
        lo, hi = int(strong.min()), int(strong.max())
        rows = cand[(cand["locality_id"] == locid) & (cand["week"] == best_wk)] \
            .sort_values("contrib", ascending=False)
        birds = [{
            "species_code": r.species_code, "common_name": r.common_name,
            "p_lifer_per_checklist": round(float(r.p_lifer_1), 3),
            "p_lifer_trip": round(float(p_lifer_k(r.occupancy, r.detect_given_present, k)), 3),
            "occupancy": round(float(r.occupancy), 3),
            "rarity_weight": round(float(r.w), 2),
            "contribution": round(float(r.contrib), 3),
        } for r in rows.head(8).itertuples()]
        head = rows.iloc[0]
        probs = [p_lifer_k(r.occupancy, r.detect_given_present, k) for r in rows.itertuples()]
        mean = sum(probs)
        sd = sum(p * (1 - p) for p in probs) ** 0.5
        out.append({
            "locality": head.locality, "locality_id": locid, "state": head.state,
            "week": best_wk, "month": month_of_week(best_wk),
            "recommended_weeks": [lo, hi], "recommended_label": week_range_label(lo, hi),
            "latitude": (None if pd.isna(head.latitude) else round(float(head.latitude), 5)),
            "longitude": (None if pd.isna(head.longitude) else round(float(head.longitude), 5)),
            "score": round(peak, 3),
            "lifers_low": max(0, round(mean - sd)), "lifers_high": round(mean + sd),
            "expected_lifers_trip": round(mean, 2),
            "top_birds": birds,
        })

    # seasonal curves: each recommended hotspot's score across ALL 48 weeks (not just the
    # selected season), so the user can see when each peaks. One extra bounded query.
    top_ids = [d["locality_id"] for d in out]
    if top_ids:
        full = _slice(store, occ_gate, locality_ids=top_ids)
        if life_list:
            full = full[~full["species_code"].isin(set(life_list))]
        full = full.assign(
            p_eff=full["occupancy"] * (1 - (1 - full["detect_given_present"]) ** k))
        full["contrib"] = (full["w"] ** alpha) * full["p_eff"]
        cs = full.groupby(["locality_id", "week"])["contrib"].sum()
        for d in out:
            wk = cs.loc[d["locality_id"]] if d["locality_id"] in cs.index.get_level_values(0) else None
            weekly = [0.0] * 48
            if wk is not None:
                for w_, v in wk.items():
                    weekly[int(w_) - 1] = round(float(v), 3)
            d["weekly"] = weekly
    return out


def _recommend_via_parquet(store, pq, life_list, states, weeks, k, alpha, occ_gate, topn, targets=None):
    cs = _cell_scores(pq, alpha, k, states=states, weeks=weeks, life_list=life_list, targets=targets)
    if cs.empty:
        return []
    peaks = cs.loc[cs.groupby("locality_id")["score"].idxmax()].sort_values("score", ascending=False).head(topn)
    top_ids = peaks["locality_id"].tolist()
    full = _cell_scores(pq, alpha, k, life_list=life_list, locality_ids=top_ids, targets=targets)  # full-year curves
    curves = {}
    for lid, g in full.groupby("locality_id"):
        wk = [0.0] * 48
        for _, r in g.iterrows():
            wk[int(r.week) - 1] = round(float(r.score), 3)
        curves[lid] = wk
    # bird-level detail for the few top hotspots only — from Parquet via DuckDB (fast, no index needed)
    if targets:
        ll = f"AND species_code IN {_inlist(targets)}"
    else:
        ll = f"AND species_code NOT IN {_inlist(life_list)}" if life_list else ""
    detail = _q(
        f"SELECT locality, locality_id, state, week, species_code, common_name, occupancy, "
        f"detect_given_present, det_a, det_b, p_lifer_1, w, latitude, longitude FROM '{pq}' "
        f"WHERE locality_id IN {_inlist(top_ids)} AND trusted=1 {ll}")
    out = []
    for prow in peaks.itertuples():
        lid, best_wk, peak = prow.locality_id, int(prow.week), float(prow.score)
        strong = cs[(cs["locality_id"] == lid) & (cs["score"] >= 0.8 * peak)]["week"]
        lo, hi = int(strong.min()), int(strong.max())
        rows = detail[(detail["locality_id"] == lid) & (detail["week"] == best_wk)].copy()
        if rows.empty:
            continue
        rows["contrib"] = (rows["w"] ** alpha) * rows["occupancy"] * (1 - (1 - rows["detect_given_present"]) ** k)
        rows = rows.sort_values("contrib", ascending=False)
        birds = [{
            "species_code": r.species_code, "common_name": r.common_name,
            "p_lifer_per_checklist": round(float(r.p_lifer_1), 3),
            "p_lifer_trip": round(float(p_lifer_k(r.occupancy, r.detect_given_present, k)), 3),
            "occupancy": round(float(r.occupancy), 3), "rarity_weight": round(float(r.w), 2),
            "contribution": round(float(r.contrib), 3),
        } for r in rows.head(8).itertuples()]
        head = rows.iloc[0]
        probs = [p_lifer_k(r.occupancy, r.detect_given_present, k) for r in rows.itertuples()]
        mean = sum(probs); sd = sum(p * (1 - p) for p in probs) ** 0.5
        out.append({
            "locality": head.locality, "locality_id": lid, "state": head.state,
            "week": best_wk, "month": month_of_week(best_wk),
            "recommended_weeks": [lo, hi], "recommended_label": week_range_label(lo, hi),
            "latitude": (None if pd.isna(head.latitude) else round(float(head.latitude), 5)),
            "longitude": (None if pd.isna(head.longitude) else round(float(head.longitude), 5)),
            "score": round(peak, 3),
            "lifers_low": max(0, round(mean - sd)), "lifers_high": round(mean + sd),
            "expected_lifers_trip": round(mean, 2), "top_birds": birds, "weekly": curves.get(lid, [0.0] * 48),
        })
    return out


def target_sites(store: Store, target_codes, states=None, k=1, life_list=(), occ_gate=0.5) -> list[dict]:
    """For each must-see species, the best hotspot/week to find it in the region (all weeks
    considered, so we can tell the user WHEN to go). Independent of the expected-lifers ranking."""
    if not target_codes:
        return []
    pq = _parquet(store)
    out = []
    for code in target_codes:
        if pq:   # find the single best (hotspot, week) for this species in DuckDB
            where = [f"species_code = {_lit(code)}", "trusted=1"]
            if states:
                where.append(f"state IN {_inlist(states)}")
            best_df = _q(
                f"SELECT locality, locality_id, state, week, occupancy, detect_given_present, det_a, det_b, "
                f"latitude, longitude, common_name, "
                f"occupancy*(1-pow(1-detect_given_present,{int(k)})) AS p FROM '{pq}' "
                f"WHERE {' AND '.join(where)} ORDER BY p DESC LIMIT 1")
            if best_df.empty:
                out.append({"species_code": code, "found": False}); continue
            best = best_df.iloc[0]
        else:
            region = _slice(store, occ_gate, states=states)
            sub = region[region["species_code"] == code] if not region.empty else region
            if len(sub) == 0:
                out.append({"species_code": code, "found": False}); continue
            sub = sub.assign(p=sub["occupancy"] * (1 - (1 - sub["detect_given_present"]) ** k))
            best = sub.loc[sub["p"].idxmax()]
        lo, hi = _detect_ci(best.occupancy, best.det_a, best.det_b, k)
        out.append({
            "species_code": code, "common_name": best.common_name, "found": True,
            "locality": best.locality, "locality_id": best.locality_id, "state": best.state,
            "week": int(best.week), "recommended_label": week_label(int(best.week)),
            "p_trip": round(float(best.p), 3), "p_low": lo, "p_high": hi,
            "latitude": (None if pd.isna(best.latitude) else round(float(best.latitude), 5)),
            "longitude": (None if pd.isna(best.longitude) else round(float(best.longitude), 5)),
        })
    return out


def _candidate_sites(store, pq, base_lat, base_lon, radius_km, max_sites=80) -> pd.DataFrame:
    """Hotspots reachable from the base pin: bbox prefilter (cheap at US scale) then exact
    haversine, nearest-first, capped at max_sites."""
    import math
    dlat = radius_km / 111.0
    dlon = radius_km / max(1e-6, 111.0 * math.cos(math.radians(base_lat)))
    lat0, lat1, lon0, lon1 = base_lat - dlat, base_lat + dlat, base_lon - dlon, base_lon + dlon
    if pq:
        df = _q(f"SELECT locality_id, any_value(locality) locality, any_value(state) state, "
                f"avg(latitude) latitude, avg(longitude) longitude FROM '{pq}' "
                f"WHERE trusted=1 AND latitude BETWEEN {lat0} AND {lat1} "
                f"AND longitude BETWEEN {lon0} AND {lon1} GROUP BY locality_id")
    else:
        d = _slice(store)
        if d.empty:
            return d
        d = d.dropna(subset=["latitude", "longitude"])
        d = d[d["latitude"].between(lat0, lat1) & d["longitude"].between(lon0, lon1)]
        df = (d.groupby("locality_id")
                .agg(locality=("locality", "first"), state=("state", "first"),
                     latitude=("latitude", "mean"), longitude=("longitude", "mean"))
                .reset_index())
    if df.empty:
        return df
    df["dist_km"] = _itin.haversine_km(base_lat, base_lon,
                                       df["latitude"].to_numpy(float), df["longitude"].to_numpy(float))
    return df[df["dist_km"] <= radius_km].sort_values("dist_km").head(max_sites).reset_index(drop=True)


def plan_itinerary(store: Store, base_lat, base_lon, radius_km, start_date, n_days,
                   k_per_day=4, alpha=0.0, life_list=(), targets=None, occ_gate=0.5,
                   max_sites=80) -> dict:
    """Base-camp itinerary: pick + radius + dates -> a greedy day-by-day plan. See itinerary.plan."""
    pq = _parquet(store)
    sites = _candidate_sites(store, pq, float(base_lat), float(base_lon), float(radius_km), max_sites)
    start = _itin._parse_date(start_date)
    if sites.empty:
        return _itin._empty(start, n_days, k_per_day, sites)
    locids = sites["locality_id"].tolist()
    wks = _itin.trip_weeks(start, int(n_days))
    targets = list(targets) if targets else None
    life_list = list(life_list)
    cols = ["locality_id", "week", "species_code", "common_name", "occupancy",
            "detect_given_present", "w"]
    if pq:
        where = [f"locality_id IN {_inlist(locids)}",
                 f"week IN ({','.join(str(int(w)) for w in wks)})", "trusted=1"]
        if targets:
            where.append(f"species_code IN {_inlist(targets)}")
        elif life_list:
            where.append(f"species_code NOT IN {_inlist(life_list)}")
        cells = _q(f"SELECT {', '.join(cols)} FROM '{pq}' WHERE {' AND '.join(where)}")
    else:
        cells = _slice(store, occ_gate, locality_ids=locids, weeks=wks)
        if not cells.empty:
            if targets:
                cells = cells[cells["species_code"].isin(set(targets))]
            elif life_list:
                cells = cells[~cells["species_code"].isin(set(life_list))]
            cells = cells[cols].copy()
    return _itin.plan(cells, sites, start, n_days, k_per_day=k_per_day, alpha=alpha)


def regions(store: Store, level="state") -> list[dict]:
    """Region selectors (name + centroid + counts), cached. Uses Parquet+DuckDB when present."""
    key = (store.db_path, level)
    if key not in _REGIONS_CACHE:
        pq = _parquet(store)
        if pq:
            grp = "state" if level == "state" else "state, county"
            df = _q(
                f"SELECT {grp}, AVG(latitude) latitude, AVG(longitude) longitude, "
                f"COUNT(DISTINCT locality_id) n_hotspots, COUNT(*) n_cells "
                f"FROM '{pq}' WHERE latitude IS NOT NULL GROUP BY {grp} ORDER BY {grp}")
            _REGIONS_CACHE[key] = df.to_dict("records")
        else:
            _REGIONS_CACHE[key] = store.regions(level).to_dict("records")
    return _REGIONS_CACHE[key]


def trip_summary(store: Store, locality_id: str, week: int, k=6, life_list=()) -> dict:
    """Expected lifers + likely-bird breakdown for one destination and effort level."""
    pq = _parquet(store)
    if pq:
        ll = f"AND species_code NOT IN {_inlist(life_list)}" if life_list else ""
        cell = _q(
            f"SELECT locality, locality_id, week, species_code, common_name, occupancy, "
            f"detect_given_present, det_a, det_b FROM '{pq}' "
            f"WHERE locality_id={_lit(locality_id)} AND week={int(week)} AND trusted=1 {ll}")
    else:
        cell = _slice(store, 0.5, locality_id=locality_id, weeks=[week])
        cell = cell[~cell["species_code"].isin(set(life_list))] if not cell.empty else cell
    cell["p_trip"] = p_lifer_k(cell["occupancy"], cell["detect_given_present"], k)
    cell = cell[cell["p_trip"] > 0.001].sort_values("p_trip", ascending=False)
    name = cell["locality"].iloc[0] if len(cell) else locality_id

    if cell.empty:
        return {"locality": name, "locality_id": locality_id, "week": week,
                "month": month_of_week(week), "effort_checklists": k, "expected_lifers": 0.0,
                "p_at_least_one": 0.0, "likely_range": [0, 0], "birds": []}

    probs = cell["p_trip"].values
    dist = poisson_binomial(probs)
    mean = float(probs.sum())
    sd = float((probs * (1 - probs)).sum()) ** 0.5
    birds = []
    for r in cell.itertuples():
        lo, hi = _detect_ci(r.occupancy, r.det_a, r.det_b, k)
        birds.append({"common_name": r.common_name, "species_code": r.species_code,
                      "p_trip": round(float(r.p_trip), 3), "p_low": lo, "p_high": hi})
    return {
        "locality": name, "locality_id": locality_id, "week": int(week), "month": month_of_week(week),
        "effort_checklists": k,
        "expected_lifers": round(mean, 2),
        "lifers_low": max(0, round(mean - sd)), "lifers_high": round(mean + sd),
        "p_at_least_one": round(float(1 - dist[0]), 3),
        "birds": birds,
    }
