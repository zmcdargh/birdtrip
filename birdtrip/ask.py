"""Optional natural-language interface: turn a birder's sentence into search parameters.

The LLM ONLY fills a form (a single `configure_search` tool call) — it never touches data or the
key. Provider-agnostic: any OpenAI-compatible chat/completions endpoint with tool-calling, chosen
by env (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY). Defaults to DeepSeek (cheap). Places are resolved by
a real geocoder server-side (never trusting model coordinates). The key is read here and NOWHERE
else; it is never returned to the client, logged, or shipped in the image.
"""
from __future__ import annotations
import json
import os
import urllib.parse
import urllib.request

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}


class AskError(Exception):
    """User-facing problem (bad query / couldn't geocode) — surfaced as a 4xx, not a 500."""


def ask_enabled() -> bool:
    return bool(os.environ.get("LLM_API_KEY"))


_TOOL = {
    "type": "function",
    "function": {
        "name": "configure_search",
        "description": "Configure a birding trip search from the user's request. Only set fields the "
                       "user actually implies; leave the rest unset.",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["best_trips", "find_spots", "plan_trip"],
                         "description": "best_trips: find the most rewarding trips over a region or "
                         "nationwide. find_spots: rank hotspots within a region for a time of year. "
                         "plan_trip: a trip based near a specific named place the user gives."},
                "near": {"type": "string", "description": "a place to center the trip on, e.g. "
                         "'Denver', 'Acadia National Park', 'the Hudson Valley'. Only for plan_trip."},
                "radius_hours": {"type": "number", "description": "how far the user will travel from "
                                 "'near', in hours of driving (e.g. 3)"},
                "states": {"type": "array", "items": {"type": "string"},
                           "description": "US state names to restrict to, e.g. ['Arizona','New Mexico']"},
                "month": {"type": "string", "description": "month of interest, e.g. 'May', if given"},
                "auto_time": {"type": "boolean", "description": "true if the user wants the best time "
                              "of year chosen for them"},
                "n_days": {"type": "integer", "description": "trip length in days"},
                "hours_per_day": {"type": "number", "description": "hours of birding per day"},
                "alpha": {"type": "number", "description": "0 = maximize number of new birds; higher "
                          "(up to ~2) favors rare/specialty birds"},
                "target_birds": {"type": "array", "items": {"type": "string"},
                                 "description": "specific birds the user wants, common or scientific names"},
                "exclude_restricted": {"type": "boolean"},
            },
            "required": ["mode"],
        },
    },
}

_SYS = ("You convert a birder's request into parameters by calling configure_search. "
        "Pick mode: 'plan_trip' when they name a place to go near; 'best_trips' to find top "
        "destinations over a region or the whole country; 'find_spots' to rank hotspots in a region "
        "for a season. Only set fields the user implies. If they say things like 'find the best time' "
        "set auto_time. Never invent a location or coordinates — put any place they mention in 'near' "
        "as plain text. Keep it to a single tool call.")

_GEO_CACHE: dict[str, tuple[float, float] | None] = {}


def _geocode(place: str) -> tuple[float, float] | None:
    """Resolve a place name to (lat, lon) via a geocoder (Nominatim by default), cached. Returns
    None if not found. Never raises — geocoding failure downgrades to a region/nationwide search."""
    key = place.strip().lower()
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    base = os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
    q = urllib.parse.urlencode({"q": place, "format": "json", "limit": 1, "countrycodes": "us"})
    req = urllib.request.Request(f"{base}?{q}", headers={"User-Agent": "birdtrip/0.1 (trip planner)"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            hits = json.load(r)
        out = (float(hits[0]["lat"]), float(hits[0]["lon"])) if hits else None
    except Exception:
        out = None
    _GEO_CACHE[key] = out
    return out


def _llm_configure(query: str, timeout: int = 20) -> dict:
    """One tool-forcing chat call to an OpenAI-compatible endpoint. Returns the tool arguments dict.
    Reads LLM_API_KEY here and only here."""
    key = os.environ["LLM_API_KEY"]                      # presence checked by caller (ask_enabled)
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    body = {
        "model": model, "max_tokens": 400, "temperature": 0,
        "tools": [_TOOL], "tool_choice": {"type": "function", "function": {"name": "configure_search"}},
        "messages": [{"role": "system", "content": _SYS}, {"role": "user", "content": query[:1000]}],
    }
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    try:
        args = resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        return json.loads(args)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        raise AskError("couldn't understand that request — try rephrasing") from e


def configure(query: str, tax, life_list=(), llm=_llm_configure) -> dict:
    """Parse `query` into a normalized search config the frontend can apply. `tax` is a Taxonomy for
    resolving bird names to species codes (server-side, never trusting the model to emit codes).
    `llm` is injectable for testing."""
    if not (query or "").strip():
        raise AskError("empty query")
    a = llm(query)
    mode = a.get("mode") or "best_trips"
    cfg: dict = {"mode": mode, "note": None, "unresolved_birds": []}
    if a.get("states"):
        cfg["states"] = [str(s) for s in a["states"]]
    if a.get("month"):
        m = MONTHS.get(str(a["month"]).strip().lower())
        if m:
            cfg["month"] = m
            cfg["week"] = (m - 1) * 4 + 2                 # a representative mid-month eBird week
    if a.get("auto_time"):
        cfg["auto_time"] = True
    for f in ("n_days", "hours_per_day", "alpha", "exclude_restricted"):
        if a.get(f) is not None:
            cfg[f] = a[f]
    # resolve target birds -> species codes (drop what we can't match, report them)
    targets, unresolved = [], []
    for nm in (a.get("target_birds") or []):
        code = tax.resolve_to_species(tax.code_for(common=nm, sci=nm))
        (targets.append(code) if code else unresolved.append(nm))
    if targets:
        cfg["targets"] = targets
    cfg["unresolved_birds"] = unresolved
    # geocode a named place -> pin + radius (only meaningful for plan_trip)
    if a.get("near"):
        loc = _geocode(str(a["near"]))
        if loc:
            cfg["near"] = str(a["near"]); cfg["base_lat"], cfg["base_lon"] = loc
            hrs = float(a.get("radius_hours") or 2.0)
            cfg["radius_km"] = round(max(20.0, min(400.0, hrs * 80.0)), 0)
        else:
            cfg["note"] = f"couldn't locate “{a['near']}” — searching more broadly instead"
            if cfg["mode"] == "plan_trip":
                cfg["mode"] = "best_trips"
    elif mode == "plan_trip":
        cfg["mode"] = "best_trips"                        # plan_trip needs a place; fall back
    return cfg
