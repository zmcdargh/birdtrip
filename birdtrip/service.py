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


# --- effort model: P(detect | present) at a given number of HOURS of birding ---------------
# When the store carries a per-hour rate lambda_hr (built by the duration-aware precompute) we use
# the time-to-detection form 1 - exp(-lambda*hours); otherwise (older store / synthetic test store)
# we fall back to the abstract k-checklist model 1 - (1-dgp)^k, so nothing breaks before a rebuild.
_HAS_LAMBDA: dict = {}


_COLS: dict = {}


def _store_cols(store: Store) -> set:
    if store.db_path not in _COLS:
        pq = _parquet(store)
        if pq:
            cols = set(_q(f"SELECT * FROM '{pq}' LIMIT 0").columns)
        else:
            import sqlite3
            cols = {r[1] for r in sqlite3.connect(store.db_path).execute("PRAGMA table_info(cells)")}
        _COLS[store.db_path] = cols
    return _COLS[store.db_path]


def _has_lambda(store: Store) -> bool:
    return "lambda_hr" in _store_cols(store)


def _pdetect_sql(hours, k, has_lambda) -> str:
    """SQL fragment for P(detect | present). Per-row fallback to the k-model where lambda_hr is NULL."""
    if has_lambda:
        h = float(hours); kfb = max(1, int(round(h)))
        return (f"(1 - CASE WHEN lambda_hr IS NOT NULL THEN exp(-lambda_hr*{h}) "
                f"ELSE pow(1 - detect_given_present, {kfb}) END)")
    return f"(1 - pow(1 - detect_given_present, {int(k)}))"


def _pdetect(df: pd.DataFrame, hours, k, has_lambda):
    """Vectorized P(detect | present) over a frame (numpy array). Mirrors _pdetect_sql."""
    if has_lambda and "lambda_hr" in df.columns:
        h = float(hours); kfb = max(1, int(round(h)))
        lam = pd.to_numeric(df["lambda_hr"], errors="coerce")
        dgp = df["detect_given_present"].astype(float)
        return np.where(lam.notna(), 1 - np.exp(-lam.fillna(0.0) * h), 1 - (1 - dgp) ** kfb)
    return (1 - (1 - df["detect_given_present"].astype(float)) ** int(k)).to_numpy()


# --- post-hoc recalibration: map raw predicted detection probs onto calibrated ones --------------
# The isotonic map (a 1001-point predicted->calibrated grid) is produced by validate_holdout
# --save-recal-map and lives next to the store as <db>.recal.json. Monotone, so rankings are
# preserved; it just makes the displayed probabilities honest (tempers the high-end over-confidence).
_RECAL: dict = {}


def _recal_map(store: Store):
    if store.db_path not in _RECAL:
        import json
        pq = _parquet(store)
        paths = ([(pq[:-8] + ".recal.json")] if pq else []) + [os.environ.get("BIRDTRIP_RECAL", "")]
        arr = None
        for p in paths:
            if p and os.path.exists(p):
                arr = np.asarray(json.load(open(p))["calibrated"], dtype=float)
                break
        _RECAL[store.db_path] = arr
    return _RECAL[store.db_path]


def _calibrate(p, store):
    """Map predicted detection prob(s) through the recalibration grid (no-op if no map)."""
    m = _recal_map(store)
    if m is None:
        return p
    a = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    out = m[np.rint(a * (len(m) - 1)).astype(int)]
    return float(out) if np.ndim(p) == 0 else out


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
                 targets=None, hours=None, has_lambda=False) -> pd.DataFrame:
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
    pdet = _pdetect_sql(k if hours is None else hours, k, has_lambda)
    sql = (f"SELECT locality_id, week, SUM(pow(w,{float(alpha)}) * occupancy * {pdet}) AS score "
           f"FROM '{pq}' WHERE {' AND '.join(where)} GROUP BY locality_id, week")
    return _q(sql)

_SUB = ["early", "mid", "late", "late"]   # eBird has 4 pseudo-weeks per month


def week_label(w: int) -> str:
    return f"{_SUB[(w - 1) % 4]} {month_of_week(w)}"


def week_range_label(lo: int, hi: int) -> str:
    if hi - lo >= 40:                       # strong almost all year
        return "year-round"
    return week_label(lo) if lo == hi else f"{week_label(lo)} – {week_label(hi)}"


def _season_window(weeks, peak: int, n: int = 48):
    """The CONTIGUOUS block of strong weeks containing the peak week, circular over 1..48. Stops a
    hotspot that's good in two separate seasons (with a dead gap between) from being labeled
    'year-round' — we report only the season the shown peak-week birds actually belong to."""
    s = {int(w) for w in weeks if 1 <= int(w) <= n}
    peak = int(peak)
    if peak not in s:
        return peak, peak, 1
    if len(s) >= 44:                        # genuinely strong almost all year
        return 1, n, len(s)
    lo = peak
    while True:
        prev = n if lo == 1 else lo - 1
        if prev in s and prev != peak:
            lo = prev
        else:
            break
    hi = peak
    while True:
        nxt = 1 if hi == n else hi + 1
        if nxt in s and nxt != peak:
            hi = nxt
        else:
            break
    return lo, hi, (hi - lo) % n + 1


def season_label(weeks, peak: int) -> str:
    lo, hi, size = _season_window(weeks, peak)
    if size >= 40:
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
                    k=1, alpha=1.0, occ_gate=0.5, topn=5, targets=None, hours=None,
                    exclude_restricted=False, user_restricted=None) -> list[dict]:
    """Rank (locality, week) destinations by rarity-weighted expected lifers. Effort is `hours` of
    birding when the store has per-hour rates (lambda_hr), else `k` checklists. `states` is a list
    (multi-select); empty/None means everywhere. `targets` (species codes) restricts the search.
    Each result carries a `restricted` flag (name heuristic OR user_restricted); exclude_restricted
    drops those from the ranking."""
    targets = list(targets) if targets else None
    has_lambda = _has_lambda(store)
    ur = set(user_restricted or [])
    if hours is None:
        hours = float(k)
    pq = _parquet(store)
    if pq:   # scalable path: score in DuckDB over Parquet, fetch details for top hotspots only
        return _recommend_via_parquet(store, pq, list(life_list), states or ([state] if state else None),
                                      weeks, k, alpha, occ_gate, topn, targets, hours, has_lambda,
                                      exclude_restricted, ur)

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
    peaks = cell_scores.loc[peak_idx].sort_values("contrib", ascending=False).head(topn + 40)

    out = []
    for prow in peaks.itertuples():
        if len(out) >= topn:
            break
        locid, best_wk, peak = prow.locality_id, int(prow.week), float(prow.contrib)
        strong = cell_scores[(cell_scores["locality_id"] == locid)
                             & (cell_scores["contrib"] >= 0.8 * peak)]["week"]
        lo, hi, _sz = _season_window(strong, best_wk)
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
        restricted = _itin._restricted(head.locality) or (locid in ur)
        if exclude_restricted and restricted:
            continue
        probs = [p_lifer_k(r.occupancy, r.detect_given_present, k) for r in rows.itertuples()]
        mean = sum(probs)
        sd = sum(p * (1 - p) for p in probs) ** 0.5
        out.append({
            "locality": head.locality, "locality_id": locid, "state": head.state,
            "week": best_wk, "month": month_of_week(best_wk),
            "recommended_weeks": [lo, hi], "recommended_label": season_label(strong, best_wk),
            "latitude": (None if pd.isna(head.latitude) else round(float(head.latitude), 5)),
            "longitude": (None if pd.isna(head.longitude) else round(float(head.longitude), 5)),
            "score": round(peak, 3), "restricted": bool(restricted),
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


def _recommend_via_parquet(store, pq, life_list, states, weeks, k, alpha, occ_gate, topn, targets=None,
                           hours=None, has_lambda=False, exclude_restricted=False, user_restricted=None):
    if hours is None:
        hours = float(k)
    ur = set(user_restricted or [])
    cs = _cell_scores(pq, alpha, k, states=states, weeks=weeks, life_list=life_list, targets=targets,
                      hours=hours, has_lambda=has_lambda)
    if cs.empty:
        return []
    # take a buffer beyond topn so excluding restricted hotspots still leaves a full list
    peaks = cs.loc[cs.groupby("locality_id")["score"].idxmax()].sort_values("score", ascending=False).head(topn + 40)
    top_ids = peaks["locality_id"].tolist()
    full = _cell_scores(pq, alpha, k, life_list=life_list, locality_ids=top_ids, targets=targets,
                        hours=hours, has_lambda=has_lambda)  # full-year curves
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
    lamcol = ", lambda_hr" if has_lambda else ""
    detail = _q(
        f"SELECT locality, locality_id, state, week, species_code, common_name, occupancy, "
        f"detect_given_present, det_a, det_b, p_lifer_1, w, latitude, longitude{lamcol} FROM '{pq}' "
        f"WHERE locality_id IN {_inlist(top_ids)} AND trusted=1 {ll}")
    out = []
    for prow in peaks.itertuples():
        if len(out) >= topn:
            break
        lid, best_wk, peak = prow.locality_id, int(prow.week), float(prow.score)
        strong = cs[(cs["locality_id"] == lid) & (cs["score"] >= 0.8 * peak)]["week"]
        lo, hi, _sz = _season_window(strong, best_wk)
        rows = detail[(detail["locality_id"] == lid) & (detail["week"] == best_wk)].copy()
        if rows.empty:
            continue
        restricted = _itin._restricted(rows.iloc[0].locality) or (lid in ur)
        if exclude_restricted and restricted:
            continue
        rows["p_eff"] = _calibrate(rows["occupancy"].astype(float) * _pdetect(rows, hours, k, has_lambda), store)
        rows["contrib"] = (rows["w"] ** alpha) * rows["p_eff"]
        rows = rows.sort_values("contrib", ascending=False)
        birds = [{
            "species_code": r.species_code, "common_name": r.common_name,
            "p_lifer_per_checklist": round(float(r.p_lifer_1), 3),
            "p_lifer_trip": round(float(r.p_eff), 3),
            "occupancy": round(float(r.occupancy), 3), "rarity_weight": round(float(r.w), 2),
            "contribution": round(float(r.contrib), 3),
        } for r in rows.head(8).itertuples()]
        head = rows.iloc[0]
        probs = rows["p_eff"].tolist()
        mean = sum(probs); sd = sum(p * (1 - p) for p in probs) ** 0.5
        out.append({
            "locality": head.locality, "locality_id": lid, "state": head.state,
            "week": best_wk, "month": month_of_week(best_wk),
            "recommended_weeks": [lo, hi], "recommended_label": season_label(strong, best_wk),
            "latitude": (None if pd.isna(head.latitude) else round(float(head.latitude), 5)),
            "longitude": (None if pd.isna(head.longitude) else round(float(head.longitude), 5)),
            "score": round(peak, 3), "restricted": bool(restricted),
            "lifers_low": max(0, round(mean - sd)), "lifers_high": round(mean + sd),
            "expected_lifers_trip": round(mean, 2), "top_birds": birds, "weekly": curves.get(lid, [0.0] * 48),
        })
    return out


def target_sites(store: Store, target_codes, states=None, k=1, life_list=(), occ_gate=0.5,
                 hours=None) -> list[dict]:
    """For each must-see species, the best hotspot/week to find it in the region (all weeks
    considered, so we can tell the user WHEN to go). Independent of the expected-lifers ranking."""
    if not target_codes:
        return []
    has_lambda = _has_lambda(store)
    if hours is None:
        hours = float(k)
    kci = max(1, int(round(hours))) if has_lambda else int(k)
    pq = _parquet(store)
    out = []
    for code in target_codes:
        if pq:   # find the single best (hotspot, week) for this species in DuckDB
            where = [f"species_code = {_lit(code)}", "trusted=1"]
            if states:
                where.append(f"state IN {_inlist(states)}")
            lamcol = ", lambda_hr" if has_lambda else ""
            best_df = _q(
                f"SELECT locality, locality_id, state, week, occupancy, detect_given_present, det_a, det_b, "
                f"latitude, longitude, common_name{lamcol}, "
                f"occupancy*{_pdetect_sql(hours, k, has_lambda)} AS p FROM '{pq}' "
                f"WHERE {' AND '.join(where)} ORDER BY p DESC LIMIT 1")
            if best_df.empty:
                out.append({"species_code": code, "found": False}); continue
            best = best_df.iloc[0]
        else:
            region = _slice(store, occ_gate, states=states)
            sub = region[region["species_code"] == code] if not region.empty else region
            if len(sub) == 0:
                out.append({"species_code": code, "found": False}); continue
            sub = sub.assign(p=sub["occupancy"].astype(float) * _pdetect(sub, hours, k, has_lambda))
            best = sub.loc[sub["p"].idxmax()]
        lo, hi = _detect_ci(best.occupancy, best.det_a, best.det_b, kci)
        out.append({
            "species_code": code, "common_name": best.common_name, "found": True,
            "locality": best.locality, "locality_id": best.locality_id, "state": best.state,
            "week": int(best.week), "recommended_label": week_label(int(best.week)),
            "p_trip": round(float(best.p), 3), "p_low": lo, "p_high": hi,
            "latitude": (None if pd.isna(best.latitude) else round(float(best.latitude), 5)),
            "longitude": (None if pd.isna(best.longitude) else round(float(best.longitude), 5)),
        })
    return out


def _candidate_sites(store, pq, base_lat, base_lon, radius_km) -> pd.DataFrame:
    """ALL hotspots within the radius of the base pin (bbox prefilter then exact haversine).
    Distance is ONLY the reachability cutoff here — there is no nearest-first preference; any
    capping to a manageable count is done later by expected richness, not proximity."""
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
    return df[df["dist_km"] <= radius_km].sort_values("dist_km").reset_index(drop=True)


def plan_itinerary(store: Store, base_lat, base_lon, radius_km, start_date, n_days,
                   hours_per_day=4.0, alpha=0.0, life_list=(), targets=None, occ_gate=0.5,
                   max_sites=80, exclude_restricted=False, user_restricted=None) -> dict:
    """Base-camp itinerary: pick + radius + dates + hours/day -> a greedy day-by-day plan.
    Uses per-hour rates (lambda_hr) when the store has them, else the k-checklist fallback.
    `max_sites` caps how many hotspots the greedy considers — by EXPECTED RICHNESS within the
    radius, never by proximity. Distance is only the reachability cutoff (the radius).
    A site is 'restricted' if its name trips the heuristic OR its locality_id is in
    `user_restricted`; with `exclude_restricted` those sites are dropped from the candidates."""
    has_lambda = _has_lambda(store)
    pq = _parquet(store)
    ur = set(user_restricted or [])
    sites = _candidate_sites(store, pq, float(base_lat), float(base_lon), float(radius_km))
    start = _itin._parse_date(start_date)
    if sites.empty:
        return _itin._empty(start, n_days, hours_per_day, sites)
    if exclude_restricted:
        keep = ~(sites["locality"].map(_itin._restricted) | sites["locality_id"].isin(ur))
        sites = sites[keep].reset_index(drop=True)
        if sites.empty:
            return _itin._empty(start, n_days, hours_per_day, sites)
    wks = _itin.trip_weeks(start, int(n_days))
    wkin = ",".join(str(int(w)) for w in wks)
    targets = list(targets) if targets else None
    life_list = list(life_list)

    # If more hotspots are in range than we'll consider, keep the RICHEST (most expected lifers on
    # the best trip day), NOT the nearest — proximity must never decide which spots make the cut.
    if pq and len(sites) > max_sites:
        sp = (f"AND species_code IN {_inlist(targets)}" if targets
              else f"AND species_code NOT IN {_inlist(life_list)}" if life_list else "")
        pdet = _pdetect_sql(hours_per_day, max(1, int(round(hours_per_day))), has_lambda)
        rich = _q(f"""SELECT locality_id, MAX(s) rich FROM (
            SELECT locality_id, week, SUM(occupancy*{pdet}) s FROM '{pq}'
            WHERE locality_id IN {_inlist(sites['locality_id'].tolist())} AND week IN ({wkin})
              AND trusted=1 {sp} GROUP BY locality_id, week) GROUP BY locality_id
            ORDER BY rich DESC LIMIT {int(max_sites)}""")
        sites = sites[sites["locality_id"].isin(set(rich["locality_id"]))].reset_index(drop=True)
    elif not pq and len(sites) > max_sites:
        sites = sites.head(max_sites).reset_index(drop=True)        # synthetic store is tiny; harmless

    locids = sites["locality_id"].tolist()
    cols = ["locality_id", "week", "species_code", "common_name", "occupancy",
            "detect_given_present", "w"] + (["lambda_hr"] if has_lambda else []) \
        + (["taxon_order"] if "taxon_order" in _store_cols(store) else [])
    if pq:
        where = [f"locality_id IN {_inlist(locids)}", f"week IN ({wkin})", "trusted=1"]
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
            cells = cells[[c for c in cols if c in cells.columns]].copy()
    return _itin.plan(cells, sites, start, n_days, hours_per_day=hours_per_day, alpha=alpha,
                      has_lambda=has_lambda, user_restricted=ur, recal=_recal_map(store))


def plan_itinerary_window(store: Store, base_lat, base_lon, radius_km, n_days,
                          hours_per_day=4.0, alpha=0.0, life_list=(), targets=None, occ_gate=0.5,
                          max_sites=80, exclude_restricted=False, user_restricted=None,
                          step_days=14, ref_year=2026) -> dict:
    """Pick-a-time: no start date given. Sweep candidate start dates across the year over the SAME
    base+radius, run the greedy N-day plan for each, and return the best window's plan plus a
    seasonal expected-lifers curve (so the user sees why that window won)."""
    from datetime import date, timedelta
    has_lambda = _has_lambda(store)
    pq = _parquet(store)
    ur = set(user_restricted or [])
    ref0 = date(ref_year, 1, 1)
    sites = _candidate_sites(store, pq, float(base_lat), float(base_lon), float(radius_km))
    if sites.empty:
        return _itin._empty(ref0, n_days, hours_per_day, sites)
    if exclude_restricted:
        keep = ~(sites["locality"].map(_itin._restricted) | sites["locality_id"].isin(ur))
        sites = sites[keep].reset_index(drop=True)
        if sites.empty:
            return _itin._empty(ref0, n_days, hours_per_day, sites)
    targets = list(targets) if targets else None
    life_list = list(life_list)
    sp = (f"AND species_code IN {_inlist(targets)}" if targets
          else f"AND species_code NOT IN {_inlist(life_list)}" if life_list else "")
    # richness prefilter over ALL weeks (stable candidate set across windows), by richness not proximity
    if pq and len(sites) > max_sites:
        pdet = _pdetect_sql(hours_per_day, max(1, int(round(hours_per_day))), has_lambda)
        rich = _q(f"""SELECT locality_id, MAX(s) rich FROM (
            SELECT locality_id, week, SUM(occupancy*{pdet}) s FROM '{pq}'
            WHERE locality_id IN {_inlist(sites['locality_id'].tolist())} AND trusted=1 {sp}
            GROUP BY locality_id, week) GROUP BY locality_id ORDER BY rich DESC LIMIT {int(max_sites)}""")
        sites = sites[sites["locality_id"].isin(set(rich["locality_id"]))].reset_index(drop=True)
    elif not pq and len(sites) > max_sites:
        sites = sites.head(max_sites).reset_index(drop=True)
    locids = sites["locality_id"].tolist()
    cols = ["locality_id", "week", "species_code", "common_name", "occupancy",
            "detect_given_present", "w"] + (["lambda_hr"] if has_lambda else []) \
        + (["taxon_order"] if "taxon_order" in _store_cols(store) else [])
    if pq:   # fetch cells for ALL weeks once, then sweep windows in memory
        where = [f"locality_id IN {_inlist(locids)}", "trusted=1"]
        if targets: where.append(f"species_code IN {_inlist(targets)}")
        elif life_list: where.append(f"species_code NOT IN {_inlist(life_list)}")
        cells_all = _q(f"SELECT {', '.join(cols)} FROM '{pq}' WHERE {' AND '.join(where)}")
    else:
        cells_all = _slice(store, occ_gate, locality_ids=locids)
        if not cells_all.empty:
            if targets: cells_all = cells_all[cells_all["species_code"].isin(set(targets))]
            elif life_list: cells_all = cells_all[~cells_all["species_code"].isin(set(life_list))]
            cells_all = cells_all[[c for c in cols if c in cells_all.columns]].copy()
    recal = _recal_map(store)
    best = None; curve = []
    for off in range(0, 364, int(step_days)):
        d = ref0 + timedelta(days=off)
        wks = _itin.trip_weeks(d, int(n_days))
        cw = cells_all[cells_all["week"].isin(wks)] if not cells_all.empty else cells_all
        plan = _itin.plan(cw, sites, d, int(n_days), hours_per_day=hours_per_day, alpha=alpha,
                          has_lambda=has_lambda, user_restricted=ur, recal=recal)
        el = float(plan.get("expected_lifers_total", 0.0))
        wk = _itin.date_to_week(d)
        curve.append({"start_date": d.isoformat(), "week": wk, "month": month_of_week(wk),
                      "expected_lifers": el})
        if best is None or el > best[0]:
            best = (el, plan)
    out = best[1]
    out["auto_window"] = True
    out["window_curve"] = curve
    return out


def find_best_trips(store: Store, n_days, hours_per_day=4.0, alpha=0.0, life_list=(), targets=None,
                    states=None, week=None, n_trips=3, grid_deg=0.9, shortlist=20, radius_km=75.0,
                    exclude_restricted=False, user_restricted=None, ref_year=2026) -> dict:
    """Pick-a-place (and optionally pick-a-time): with no pin given, find the most rewarding trips.
    Two-stage funnel: (1) grid hotspots into clusters; a CHEAP occupancy proxy per (cluster, week)
    = Σ_species max-over-the-cluster's-hotspots(occupancy) ['expected lifers present']; argmax the
    week per cluster so each area appears once; (2) run the full greedy plan only on the top
    `shortlist` clusters at their best week, and return the top `n_trips` distinct trips.
    If `week` is given, that week is used for every cluster instead of the argmax (pick-a-place at a
    fixed time). KNOWN LIMITATION: a cluster that peaks in two different seasons is collapsed to one."""
    from datetime import date, timedelta
    pq = _parquet(store); life_list = list(life_list); targets = list(targets) if targets else None
    g = float(grid_deg)
    sp = (f"AND species_code IN {_inlist(targets)}" if targets
          else f"AND species_code NOT IN {_inlist(life_list)}" if life_list else "")
    stf = f"AND state IN {_inlist(states)}" if states else ""
    wf = f"AND week={int(week)}" if week else ""
    if pq:
        prox = _q(f"""WITH c AS (
            SELECT floor(latitude/{g}) gy, floor(longitude/{g}) gx, week, species_code, MAX(occupancy) mo
            FROM '{pq}' WHERE trusted=1 AND latitude IS NOT NULL {sp} {stf} {wf} GROUP BY 1,2,3,4)
          SELECT gy, gx, week, SUM(mo) proxy FROM c GROUP BY 1,2,3""")
        cen = _q(f"""SELECT floor(latitude/{g}) gy, floor(longitude/{g}) gx,
            AVG(latitude) lat, AVG(longitude) lon, any_value(state) state, COUNT(DISTINCT locality_id) nhot
            FROM '{pq}' WHERE trusted=1 AND latitude IS NOT NULL {stf} GROUP BY 1,2""")
    else:                                   # pandas fallback (synthetic / tiny SQLite store)
        df = _slice(store, 0.0, states=([s for s in states] if states else None))
        if df is None or df.empty:
            return {"trips": [], "n_clusters": 0}
        if targets: df = df[df["species_code"].isin(set(targets))]
        elif life_list: df = df[~df["species_code"].isin(set(life_list))]
        if week: df = df[df["week"] == int(week)]
        df["gy"] = np.floor(df["latitude"] / g); df["gx"] = np.floor(df["longitude"] / g)
        mo = df.groupby(["gy", "gx", "week", "species_code"])["occupancy"].max().reset_index()
        prox = mo.groupby(["gy", "gx", "week"])["occupancy"].sum().reset_index().rename(columns={"occupancy": "proxy"})
        cen = df.groupby(["gy", "gx"]).agg(lat=("latitude", "mean"), lon=("longitude", "mean"),
                                           state=("state", "first"), nhot=("locality_id", "nunique")).reset_index()
    if prox is None or len(prox) == 0:
        return {"trips": [], "n_clusters": 0}
    best = prox.loc[prox.groupby(["gy", "gx"])["proxy"].idxmax()]          # best week per cluster
    best = best.merge(cen, on=["gy", "gx"]).sort_values("proxy", ascending=False).head(int(shortlist))
    trips = []
    for r in best.itertuples():
        d = date(ref_year, 1, 1) + timedelta(days=int((int(r.week) - 1) * 7.61))
        plan = plan_itinerary(store, float(r.lat), float(r.lon), float(radius_km), d.isoformat(),
                              int(n_days), hours_per_day=hours_per_day, alpha=alpha, life_list=life_list,
                              targets=targets, exclude_restricted=exclude_restricted,
                              user_restricted=user_restricted)
        trips.append({"region": (None if pd.isna(r.state) else r.state),
                      "base_lat": round(float(r.lat), 4), "base_lon": round(float(r.lon), 4),
                      "week": int(r.week), "month": month_of_week(int(r.week)), "start_date": d.isoformat(),
                      "expected_lifers_total": plan.get("expected_lifers_total", 0.0),
                      "n_stops": len(plan.get("stops", [])), "n_hotspots_in_range": plan.get("n_candidate_sites", 0),
                      "plan": plan})
    trips.sort(key=lambda t: t["expected_lifers_total"], reverse=True)
    return {"trips": trips[:int(n_trips)], "n_clusters": int(len(best)), "n_days": int(n_days)}


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


def trip_summary(store: Store, locality_id: str, week: int, k=6, life_list=(), hours=None) -> dict:
    """Expected lifers + likely-bird breakdown for one destination and effort level (hours of
    birding when the store has lambda_hr, else k checklists)."""
    has_lambda = _has_lambda(store)
    if hours is None:
        hours = float(k)
    pq = _parquet(store)
    if pq:
        ll = f"AND species_code NOT IN {_inlist(life_list)}" if life_list else ""
        lamcol = ", lambda_hr, mean_dur_min" if has_lambda else ""
        taxcol = ", taxon_order" if "taxon_order" in _store_cols(store) else ""
        cell = _q(
            f"SELECT locality, locality_id, week, species_code, common_name, occupancy, "
            f"detect_given_present, det_a, det_b{lamcol}{taxcol} FROM '{pq}' "
            f"WHERE locality_id={_lit(locality_id)} AND week={int(week)} AND trusted=1 {ll}")
    else:
        cell = _slice(store, 0.5, locality_id=locality_id, weeks=[week])
        cell = cell[~cell["species_code"].isin(set(life_list))] if not cell.empty else cell
    if not cell.empty:
        cell["p_trip"] = _calibrate(cell["occupancy"].astype(float) * _pdetect(cell, hours, k, has_lambda), store)
    cell = cell[cell["p_trip"] > 0.001].sort_values("p_trip", ascending=False) if not cell.empty else cell
    name = cell["locality"].iloc[0] if len(cell) else locality_id

    if cell.empty:
        return {"locality": name, "locality_id": locality_id, "week": week,
                "month": month_of_week(week), "effort_checklists": k, "effort_hours": round(hours, 1),
                "expected_lifers": 0.0, "p_at_least_one": 0.0, "likely_range": [0, 0], "birds": []}

    probs = cell["p_trip"].values
    dist = poisson_binomial(probs)
    mean = float(probs.sum())
    sd = float((probs * (1 - probs)).sum()) ** 0.5
    # effort for the CI band: match the point estimate. With lambda, the point uses `hours`, so the
    # CI must use effective effort hours/mean_dur (NOT round(hours) checklists) or it won't bracket it.
    if has_lambda and "mean_dur_min" in cell.columns and pd.notna(cell["mean_dur_min"].iloc[0]) \
            and float(cell["mean_dur_min"].iloc[0]) > 0:
        kci = float(hours) / (float(cell["mean_dur_min"].iloc[0]) / 60.0)
    else:
        kci = int(k)
    has_tax = "taxon_order" in cell.columns
    birds = []
    for r in cell.itertuples():
        # CI as a RELATIVE band (from the Beta posterior's width) around the calibrated point, so it
        # always brackets the point and still widens for thin-data cells. occ cancels in the ratio.
        clo, chi = _detect_ci(r.occupancy, r.det_a, r.det_b, kci)
        qm = float(r.det_a) / (float(r.det_a) + float(r.det_b))
        gm = float(r.occupancy) * (1 - (1 - qm) ** kci)           # detection at the same effort, mean rate
        pt = float(r.p_trip)
        lo = pt * (clo / gm) if gm > 1e-9 else pt
        hi = pt * (chi / gm) if gm > 1e-9 else pt
        lo, hi = max(0.0, min(lo, pt)), min(1.0, max(hi, pt))
        birds.append({"common_name": r.common_name, "species_code": r.species_code,
                      "p_trip": round(pt, 3), "p_low": round(lo, 3), "p_high": round(hi, 3),
                      "taxon_order": (float(r.taxon_order) if has_tax and pd.notna(r.taxon_order) else None)})
    return {
        "locality": name, "locality_id": locality_id, "week": int(week), "month": month_of_week(week),
        "effort_checklists": k, "effort_hours": round(hours, 1),
        "expected_lifers": round(mean, 2),
        "lifers_low": max(0, round(mean - sd)), "lifers_high": round(mean + sd),
        "p_at_least_one": round(float(1 - dist[0]), 3),
        "birds": birds,
    }
