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
