"""
End-to-end API tests against the synthetic store, via FastAPI's TestClient.
Builds a fresh SQLite store in a temp dir, points the app at it, and exercises
every endpoint.
"""
import csv
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PRECOMP = ROOT / "data" / "precomputed.csv"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    pytest.importorskip("fastapi")
    from birdtrip.store import build_store
    db = tmp_path_factory.mktemp("store") / "birdtrip.sqlite"
    build_store(PRECOMP, db)
    os.environ["BIRDTRIP_DB"] = str(db)
    from fastapi.testclient import TestClient
    from birdtrip.api import app
    return TestClient(app)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["db_exists"] is True


def test_regions(client):
    counties = {row["county"] for row in client.get("/regions?level=county").json()}
    assert {"New York", "Suffolk"} <= counties


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "birdtrip" in r.text.lower()


def test_region_centroids(client):
    states = client.get("/regions?level=state").json()
    ny = next(g for g in states if g["state"] == "New York")
    assert ny["latitude"] is not None and ny["n_hotspots"] >= 2


def test_recommend_includes_coords(client):
    recs = client.post("/recommend", json={"state": "New York", "alpha": 1.0, "topn": 2}).json()
    assert recs and recs[0]["latitude"] is not None and recs[0]["longitude"] is not None
    assert "weekly" in recs[0] and len(recs[0]["weekly"]) == 48   # seasonal curve attached


def test_targets(client):
    # target a real species in the synthetic data; expect its best site + an unrecognized flag
    sp = client.post("/recommend", json={"state": "New York", "topn": 1}).json()[0]["top_birds"][0]["common_name"]
    r = client.post("/targets", json={"names": [sp, "Definitely Not A Bird"], "k": 1}).json()
    assert "Definitely Not A Bird" in r["unrecognized"]
    hit = next(h for h in r["targets"] if h.get("found"))
    assert hit["locality"] and 1 <= hit["week"] <= 48 and 0 <= hit["p_trip"] <= 1


def test_recommend_alpha_promotes_may(client):
    # selecting New York county, high alpha -> Central Park spring-migration weeks should appear
    body = {"county": "New York", "alpha": 1.5, "k": 6, "topn": 5}
    recs = client.post("/recommend", json=body).json()
    assert recs and all("top_birds" in d for d in recs)
    cp_may = [d for d in recs if d["locality"] == "Central Park" and 16 <= d["week"] <= 22]
    assert cp_may, "expected a Central Park May/June migration week in the top recommendations"
    # every recommendation exposes its driving birds
    assert recs[0]["top_birds"][0]["contribution"] >= recs[0]["top_birds"][-1]["contribution"]


def test_summary_excludes_life_list(client):
    rec = client.post("/recommend", json={"county": "New York", "alpha": 1.0, "topn": 1}).json()[0]
    locid, wk = rec["locality_id"], rec["week"]
    base = client.post("/summary", json={"locality_id": locid, "week": wk, "k": 6}).json()
    assert base["expected_lifers"] > 0 and 0 < base["p_at_least_one"] <= 1
    # remove a likely species from the candidate pool -> expected lifers should not increase
    seen = [base["birds"][0]["species_code"]]
    fewer = client.post("/summary",
                        json={"locality_id": locid, "week": wk, "k": 6, "life_list": seen}).json()
    assert fewer["expected_lifers"] <= base["expected_lifers"]


# --- itinerary builder -------------------------------------------------------
# Central Park (40.78, -73.97) and Montauk Point (41.07, -71.86) are the two synthetic
# hotspots; a base at Central Park with a 200 km radius reaches both. Starting May 1 for
# a few days keeps every day inside one eBird week (17), so marginal gains must be monotone.
CP = dict(base_lat=40.78, base_lon=-73.97)


def test_itinerary_basic(client):
    body = {**CP, "radius_km": 200, "start_date": "2026-05-01", "n_days": 3, "k_per_day": 4}
    r = client.post("/itinerary", json=body)
    assert r.status_code == 200
    plan = r.json()
    assert len(plan["days"]) == 3 and plan["n_candidate_sites"] >= 2
    assert plan["expected_lifers_total"] > 0
    cum = [d["cumulative_expected"] for d in plan["days"]]
    assert cum == sorted(cum)                                  # cumulative never decreases
    assert all(d["locality"] for d in plan["days"])            # every day got a stop
    # cumulative of the last day equals the trip total, and equals the sum of daily gains
    assert abs(cum[-1] - plan["expected_lifers_total"]) < 0.05
    assert abs(sum(d["expected_new"] for d in plan["days"]) - plan["expected_lifers_total"]) < 0.05


def test_itinerary_marginal_gains_monotone(client):
    # submodular coverage under greedy: each day's marginal expected-new can't exceed the prior day's
    body = {**CP, "radius_km": 200, "start_date": "2026-05-01", "n_days": 4}
    days = client.post("/itinerary", json=body).json()["days"]
    gains = [d["expected_new"] for d in days]
    assert all(gains[i] <= gains[i - 1] + 1e-6 for i in range(1, len(gains)))


def test_itinerary_revisit_diminishes(client):
    # tiny radius -> only Central Park reachable, so a 2-day trip must revisit it.
    body = {**CP, "radius_km": 5, "start_date": "2026-05-01", "n_days": 2}
    plan = client.post("/itinerary", json=body).json()
    assert plan["n_candidate_sites"] == 1
    d1, d2 = plan["days"]
    assert d1["locality_id"] == d2["locality_id"] and d2["revisit"] is True
    # occupancy-once conditioning: the second day adds strictly less, and the total is below 2x day-1
    assert d2["expected_new"] < d1["expected_new"]
    assert plan["expected_lifers_total"] < 2 * d1["expected_new"]


def test_itinerary_life_list_reduces_total(client):
    base = {**CP, "radius_km": 200, "start_date": "2026-05-01", "n_days": 2}
    full = client.post("/itinerary", json=base).json()
    seen = [full["days"][0]["birds"][0]["species_code"]]       # remove the top expected bird
    fewer = client.post("/itinerary", json={**base, "life_list": seen}).json()
    assert fewer["expected_lifers_total"] <= full["expected_lifers_total"]


def test_itinerary_exclude_restricted(client):
    base = {**CP, "radius_km": 200, "start_date": "2026-05-01", "n_days": 3}
    plan = client.post("/itinerary", json=base).json()
    victim = plan["stops"][0]["locality_id"]
    out = client.post("/itinerary",
                      json={**base, "user_restricted": [victim], "exclude_restricted": True}).json()
    assert victim not in [s["locality_id"] for s in out["stops"]]


def test_recommend_exclude_restricted(client):
    recs = client.post("/recommend", json={"state": "New York", "topn": 5}).json()
    assert recs and all("restricted" in r for r in recs)          # flag exposed on Find-spots results
    victim = recs[0]["locality_id"]
    out = client.post("/recommend", json={"state": "New York", "topn": 5,
                                          "user_restricted": [victim], "exclude_restricted": True}).json()
    assert victim not in [r["locality_id"] for r in out]


def test_season_window_not_year_round_with_gap():
    from birdtrip.service import _season_window, season_label
    weeks = [45, 46, 47, 48, 1, 2, 3] + [20, 21, 22, 23]   # winter block + summer block, gap between
    lo, hi, size = _season_window(weeks, peak=1)            # peak in the winter block
    assert size <= 12 and season_label(weeks, 1) != "year-round"   # only the winter season, not all year
    assert season_label(list(range(1, 49)), 24) == "year-round"    # genuinely strong all year -> year-round


def test_restricted_access_heuristic():
    from birdtrip.itinerary import _restricted
    assert _restricted("MacDill AFB")
    assert _restricted("Fort Morgan--restricted access")
    assert _restricted("Smith Preserve (by appointment)")
    assert _restricted("Naval Air Station Pensacola")
    assert not _restricted("Dauphin Island")
    assert not _restricted("Central Park")
    assert not _restricted(None)


def test_itinerary_auto_window(client):
    # omit start_date -> the model sweeps the year and returns the best N-day window + a curve
    from birdtrip.itinerary import date_to_week, _parse_date
    plan = client.post("/itinerary", json={**CP, "radius_km": 200, "n_days": 3}).json()
    assert plan.get("auto_window") is True
    assert len(plan["window_curve"]) >= 10 and plan["expected_lifers_total"] > 0 and plan["days"]
    peak = max(plan["window_curve"], key=lambda c: c["expected_lifers"])     # returned plan IS the argmax
    assert abs(peak["expected_lifers"] - plan["expected_lifers_total"]) < 0.05
    assert 14 <= date_to_week(_parse_date(plan["start_date"])) <= 24         # spring-migration peak


def test_best_trips(client):
    # no pin, no date -> the model finds the best trips (region + week + plan), ranked.
    # small min_sep_km so the two far-apart synthetic hotspots count as distinct trips.
    from birdtrip.itinerary import haversine_km
    r = client.post("/best_trips", json={"n_days": 3, "n_trips": 2, "min_sep_km": 100}).json()
    assert r["trips"] and len(r["trips"]) >= 1
    t = r["trips"][0]
    assert t["expected_lifers_total"] > 0 and "plan" in t and 1 <= t["week"] <= 48
    assert t["base_lat"] is not None and t["base_lon"] is not None
    els = [x["expected_lifers_total"] for x in r["trips"]]
    assert els == sorted(els, reverse=True)               # ranked best-first
    assert abs(t["plan"]["expected_lifers_total"] - t["expected_lifers_total"]) < 1e-6
    # any two returned trips are at least min_sep_km apart (distinct destinations)
    for a, b in [(r["trips"][i], r["trips"][j]) for i in range(len(r["trips"])) for j in range(i + 1, len(r["trips"]))]:
        assert haversine_km(a["base_lat"], a["base_lon"], b["base_lat"], b["base_lon"]) >= 100 - 1e-6


def test_itinerary_out_of_range_base(client):
    # a base in the middle of the Pacific reaches nothing -> graceful empty plan
    body = {"base_lat": 0.0, "base_lon": -150.0, "radius_km": 50,
            "start_date": "2026-05-01", "n_days": 2}
    plan = client.post("/itinerary", json=body).json()
    assert plan["expected_lifers_total"] == 0.0 and plan["days"] == [] and "message" in plan


def test_lifelist_upload(client, tmp_path):
    p = tmp_path / "MyEBirdData.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Common Name", "Scientific Name"])
        w.writerow(["Northern Cardinal", "Cardinalis cardinalis"])
        w.writerow(["gull sp.", "Larus sp."])           # spuh -> dropped
    with open(p, "rb") as f:
        r = client.post("/lifelist", files={"file": ("MyEBirdData.csv", f, "text/csv")})
    body = r.json()
    assert r.status_code == 200
    assert "norcar" in body["species_codes"] and body["n_species"] == 1
    assert any(cat == "spuh" for _, cat in body["dropped"])
