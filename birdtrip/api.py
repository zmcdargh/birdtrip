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
    k: int = Field(1, ge=1, description="complete checklists of effort")
    alpha: float = Field(1.0, ge=0, description="rarity slider: 0=most birds, higher=specialties")
    occ_gate: float = Field(0.5, ge=0, le=1)
    topn: int = Field(5, ge=1, le=50)


class SummaryReq(BaseModel):
    locality_id: str
    week: int = Field(..., ge=1, le=48)
    k: int = Field(6, ge=1)
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
        county=req.county, weeks=req.weeks, k=req.k, alpha=req.alpha, occ_gate=req.occ_gate, topn=req.topn)


@app.post("/summary")
def summary(req: SummaryReq):
    return service.trip_summary(_store(), req.locality_id, req.week, k=req.k, life_list=req.life_list)


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
