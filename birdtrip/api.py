"""
FastAPI app exposing the planner.

Endpoints
  GET  /healthz              liveness + which store is loaded
  GET  /regions?level=       regions available for selection (state | county)
  POST /lifelist             upload an eBird "Download My Data" CSV -> species-code life list
  POST /recommend            ranked destinations for a region/season/effort/alpha
  POST /summary              expected lifers + likely birds for one destination

The store path comes from $BIRDTRIP_DB (default data/birdtrip.sqlite). Build it once with
  python -m birdtrip.store  (see __main__ there) or birdtrip.store.build_store(...).

Run:  uvicorn birdtrip.api:app --reload
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .store import Store
from .taxonomy import Taxonomy
from .lifelist import parse_life_list
from . import service
from . import ask as ask_mod
import collections

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
app = FastAPI(title="birdtrip", version="0.1.0",
              description="Plan birding trips from eBird data to maximize expected lifers.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("birdtrip")


@app.middleware("http")
async def _timing(request: Request, call_next):
    """Log per-request latency (and expose it as a header) so we can watch performance."""
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.0f}"
    if request.url.path not in ("/", "/healthz"):   # skip noise
        log.info("%-7s %-12s %3d  %6.0f ms", request.method, request.url.path,
                 response.status_code, ms)
    return response


@app.on_event("startup")
def _warm_caches():
    """Warm the hotspot-coordinate cache in the background so the first trip search isn't cold
    (one ~7 GB read). Non-blocking: the app is ready immediately; warming finishes shortly after."""
    import threading

    def go():
        try:
            s = _store()
            service._loc_coords(s, service._parquet(s))
            log.info("coordinate cache warmed")
        except Exception as e:                       # no store yet / still downloading — fine
            log.info("cache warm skipped: %s", e)
    threading.Thread(target=go, daemon=True).start()


_tax: Taxonomy | None = None


def _db_path() -> str:
    return os.environ.get("BIRDTRIP_DB", str(ROOT / "data" / "birdtrip.sqlite"))


def _taxonomy() -> Taxonomy:
    global _tax
    if _tax is None:
        _tax = Taxonomy()
    return _tax


def _store() -> Store:
    db = _db_path()
    # the app serves from the Parquet sidecar when present; the SQLite file is optional.
    pq = (db[:-7] if db.endswith(".sqlite") else db[:-3] if db.endswith(".db") else db) + ".parquet"
    if not Path(db).exists() and not Path(pq).exists():
        raise HTTPException(503, f"no store at {db} or {pq}; run scripts/precompute_duckdb.py")
    return Store(db)


class RecommendReq(BaseModel):
    life_list: list[str] = Field(default_factory=list, description="species codes already seen")
    targets: list[str] | None = Field(default=None, description="species codes to search FOR (restricts candidates)")
    states: list[str] | None = Field(default=None, description="selected states (multi-select); empty=everywhere")
    state: str | None = None   # back-compat single
    county: str | None = None
    weeks: list[int] | None = Field(default=None, description="eBird weeks 1-48 (the season)")
    k: int = Field(1, ge=1, description="complete checklists of effort (fallback when no lambda)")
    hours: float | None = Field(None, gt=0, le=24, description="hours of birding (preferred effort unit; uses per-hour rates when the store has them)")
    alpha: float = Field(1.0, ge=0, description="rarity slider: 0=most birds, higher=specialties")
    occ_gate: float = Field(0.5, ge=0, le=1)
    topn: int = Field(5, ge=1, le=50)
    exclude_restricted: bool = Field(False, description="drop hotspots flagged as restricted-access")
    user_restricted: list[str] = Field(default_factory=list,
                                        description="locality IDs the user has marked restricted")


class SummaryReq(BaseModel):
    locality_id: str
    week: int = Field(..., ge=1, le=48)
    k: int = Field(6, ge=1)
    hours: float | None = Field(None, gt=0, le=24, description="hours of birding (preferred effort unit)")
    life_list: list[str] = Field(default_factory=list)


@app.on_event("startup")
def _warm():
    db = _db_path()
    if Path(db).exists():
        try:
            service.warm(Store(db))   # precompute small static caches (weights, region list)
        except Exception:
            pass


@app.get("/healthz")
def healthz():
    db = _db_path()
    return {"status": "ok", "db": db, "db_exists": Path(db).exists()}


@app.get("/regions")
def regions(level: str = "state"):
    if level not in ("state", "county"):
        raise HTTPException(400, "level must be 'state' or 'county'")
    return service.regions(_store(), level)


@app.post("/lifelist")
async def lifelist(file: UploadFile = File(...)):
    try:
        res = parse_life_list(file.file, _taxonomy())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"summary": res.summary(), "n_species": res.n_species,
            "species_codes": sorted(res.species_codes),
            "dropped": res.dropped, "unmatched_count": len(res.unmatched)}


@app.get("/species")
def species(q: str = ""):
    """Autocomplete: species in the store matching the (forgiving) query -> {species_code, common_name}."""
    return service.species_search(_store(), q)


@app.post("/recommend")
def recommend(req: RecommendReq):
    return service.recommend_trips(
        _store(), life_list=req.life_list, targets=req.targets, states=req.states, state=req.state,
        county=req.county, weeks=req.weeks, k=req.k, hours=req.hours, alpha=req.alpha,
        occ_gate=req.occ_gate, topn=req.topn,
        exclude_restricted=req.exclude_restricted, user_restricted=req.user_restricted)


@app.post("/summary")
def summary(req: SummaryReq):
    return service.trip_summary(_store(), req.locality_id, req.week, k=req.k, hours=req.hours,
                                life_list=req.life_list)


class ItineraryReq(BaseModel):
    base_lat: float = Field(..., ge=-90, le=90, description="base-camp pin latitude")
    base_lon: float = Field(..., ge=-180, le=180, description="base-camp pin longitude")
    radius_km: float = Field(75.0, gt=0, le=1000, description="how far you'll day-trip from the base")
    start_date: str | None = Field(None, description="trip start, YYYY-MM-DD; omit to let the model "
                                    "find the best N-day window of the year")
    n_days: int = Field(..., ge=1, le=30, description="trip length in days")
    hours_per_day: float = Field(4.0, gt=0, le=24, description="hours of birding per day at a stop")
    alpha: float = Field(0.0, ge=0, description="favor local specialties (0 = max expected lifers)")
    life_list: list[str] = Field(default_factory=list, description="species codes already seen")
    targets: list[str] | None = Field(default=None, description="restrict to these species codes")
    max_sites: int = Field(80, ge=1, le=300, description="cap on candidate hotspots considered")
    exclude_restricted: bool = Field(False, description="drop hotspots flagged as restricted-access")
    user_restricted: list[str] = Field(default_factory=list,
                                        description="locality IDs the user has marked restricted")


@app.post("/itinerary")
def itinerary(req: ItineraryReq):
    common = dict(base_lat=req.base_lat, base_lon=req.base_lon, radius_km=req.radius_km,
                  n_days=req.n_days, hours_per_day=req.hours_per_day, alpha=req.alpha,
                  exclude_restricted=req.exclude_restricted, user_restricted=req.user_restricted,
                  life_list=req.life_list, targets=req.targets, max_sites=req.max_sites)
    if req.start_date:                                  # explicit date -> single-window plan
        return service.plan_itinerary(_store(), start_date=req.start_date, **common)
    return service.plan_itinerary_window(_store(), **common)   # pick-a-time: best N-day window


class BestTripsReq(BaseModel):
    n_days: int = Field(..., ge=1, le=30, description="trip length in days")
    hours_per_day: float = Field(4.0, gt=0, le=24)
    radius_km: float = Field(75.0, gt=0, le=1000, description="day-trip radius from the auto-chosen base")
    alpha: float = Field(0.0, ge=0)
    week: int | None = Field(None, ge=1, le=48, description="fix the time of year; omit to also pick the best week per area")
    states: list[str] | None = Field(None, description="restrict the search to these states; omit for nationwide")
    n_trips: int = Field(3, ge=1, le=10, description="how many distinct trips to return")
    min_sep_km: float | None = Field(None, gt=0, description="minimum spacing between distinct trips (km); "
                                     "default ≈ max(3x radius, 350) so trips are different destinations")
    life_list: list[str] = Field(default_factory=list)
    targets: list[str] | None = None
    exclude_restricted: bool = False
    user_restricted: list[str] = Field(default_factory=list)


@app.post("/best_trips")
def best_trips(req: BestTripsReq):
    return service.find_best_trips(
        _store(), n_days=req.n_days, hours_per_day=req.hours_per_day, radius_km=req.radius_km,
        alpha=req.alpha, week=req.week, states=req.states, n_trips=req.n_trips, min_sep_km=req.min_sep_km,
        life_list=req.life_list, targets=req.targets,
        exclude_restricted=req.exclude_restricted, user_restricted=req.user_restricted)


# ---- optional natural-language interface (LLM fills the search form) ----------------------
# Cost guards: the LLM key lives ONLY server-side (never sent to the client); requests are rate
# limited per-IP and capped globally per day so the endpoint can't be hammered into a big bill.
_ask_ip: dict = collections.defaultdict(list)
_ask_day = {"day": None, "n": 0}


def _ask_ratelimit(ip: str):
    now = time.time(); today = time.strftime("%Y-%m-%d")
    if _ask_day["day"] != today:
        _ask_day.update(day=today, n=0)
    per_min = int(os.environ.get("ASK_RATE_PER_MIN", "6"))
    per_day_ip = int(os.environ.get("ASK_RATE_PER_DAY", "60"))
    glob = int(os.environ.get("ASK_GLOBAL_DAILY", "1500"))
    if _ask_day["n"] >= glob:
        raise HTTPException(429, "daily query budget reached — please try again tomorrow")
    hits = [t for t in _ask_ip[ip] if now - t < 86400]
    if sum(1 for t in hits if now - t < 60) >= per_min:
        raise HTTPException(429, "too many requests — please slow down")
    if len(hits) >= per_day_ip:
        raise HTTPException(429, "daily limit reached for this address")
    hits.append(now); _ask_ip[ip] = hits; _ask_day["n"] += 1


@app.get("/config")
def config():
    """Feature flags for the frontend (so the NL box only shows when the LLM is configured)."""
    return {"ask_enabled": ask_mod.ask_enabled()}


class AskReq(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="natural-language trip request")
    life_list: list[str] = Field(default_factory=list)


@app.post("/ask")
def ask(req: AskReq, request: Request):
    """Parse a free-text request into a search config the frontend applies to the form."""
    if not ask_mod.ask_enabled():
        raise HTTPException(503, "natural-language search is not enabled on this server")
    _ask_ratelimit(request.client.host if request.client else "unknown")
    try:
        return ask_mod.configure(req.query, _taxonomy(), req.life_list)
    except ask_mod.AskError as e:
        raise HTTPException(422, str(e))
    except Exception as e:                              # never leak internals (incl. the key)
        log.warning("ask failed: %s", type(e).__name__)
        raise HTTPException(502, "language model unavailable")


class TargetsReq(BaseModel):
    names: list[str] = Field(default_factory=list, description="must-see birds (common or scientific)")
    states: list[str] | None = None
    k: int = Field(1, ge=1)
    life_list: list[str] = Field(default_factory=list)


@app.post("/targets")
def targets(req: TargetsReq):
    tax = _taxonomy()
    resolved, unknown = [], []
    for nm in req.names:
        code = tax.resolve_to_species(tax.code_for(common=nm, sci=nm))
        (resolved.append((code, nm)) if code else unknown.append(nm))
    hits = service.target_sites(_store(), [c for c, _ in resolved],
                                states=req.states, k=req.k, life_list=req.life_list)
    return {"targets": hits, "unrecognized": unknown}


# --- serve the single-page frontend (a plain route, so it can't shadow the API) ---
@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
