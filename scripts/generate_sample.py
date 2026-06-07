"""
Generate a faithful synthetic eBird sample: an EBD observations file and a
matching Sampling Event Data (SED) file, both tab-delimited, following eBird's
real column names. The real EBD has ~50 columns; we emit the subset the pipeline
uses (selection is by column name, so extra columns in the real file are fine).

Planted patterns (so the pipeline's behavior is observable):
  - COMMON WIDESPREAD : "Northern Cardinal"  - present every year, high freq.    -> should stay stable.
  - VAGRANT           : "Varied Thrush"       - one winter only, hyper-reported.  -> should be crushed by low inter-year occupancy.
  - REGIONAL SPECIALTY: "Saltmarsh Sparrow"   - only in coastal Suffolk county.   -> should surface as a specialty (high w(s)).
  - LOW-SAMPLE        : "Painted Bunting"      - 1 detection on 1 checklist.       -> raw freq 1.0, should shrink toward parent mean.
"""
import csv, random, datetime as dt
from pathlib import Path

random.seed(42)
OUT = Path(__file__).resolve().parent.parent / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2018, 2026))   # 8-year look-back relative to "today" 2026

# (county, county_code, locality, locality_id, lat, lon, base_checklists_per_week)
COUNTIES = {
    "New York": ("US-NY-061", "Central Park", "L191106", 40.78, -73.97, 14),  # heavily birded migrant trap
    "Suffolk":  ("US-NY-103", "Montauk Point", "L160603", 41.07, -71.86, 4),  # coastal, less birded
}

# species: (common, scientific, category)
SP = {
    "card": ("Northern Cardinal", "Cardinalis cardinalis", "species"),
    "vath": ("Varied Thrush", "Ixoreus naevius", "species"),
    "sals": ("Saltmarsh Sparrow", "Ammospiza caudacuta", "species"),
    "pabu": ("Painted Bunting", "Passerina ciris", "species"),
    "sosp": ("Song Sparrow", "Melospiza melodia", "species"),     # filler common resident
    "amro": ("American Robin", "Turdus migratorius", "species"),  # filler common resident
    # reliable spring migrants -> high occupancy, sharply seasonal (the Central Park "May" signal)
    "amre": ("American Redstart", "Setophaga ruticilla", "species"),
    "magw": ("Magnolia Warbler", "Setophaga magnolia", "species"),
    "bkbw": ("Blackburnian Warbler", "Setophaga fusca", "species"),
}

def week_of_year(d):
    """eBird pseudo-week: 4 per month, 48 per year (1..48)."""
    wm = min(3, (d.day - 1) // 7)
    return (d.month - 1) * 4 + wm + 1

def date_in_week(year, woy):
    month = (woy - 1) // 4 + 1
    wm = (woy - 1) % 4
    day = wm * 7 + random.randint(1, 7)
    day = min(day, 28)
    return dt.date(year, month, day)

# detection probability model: returns P(species on a complete checklist) for a given context
def detect_prob(code, county, year, woy):
    winter = woy <= 8 or woy >= 44
    if code == "card":      # common resident everywhere, high & stable
        return 0.55
    if code == "sosp":
        return 0.45
    if code == "amro":
        return 0.40 if not winter else 0.20
    if code == "vath":      # VAGRANT: New York county, one winter only (2022), realistic reporting rate
        if county == "New York" and year == 2022 and winter:
            return 0.30     # a findable-but-not-mega bird, present 1 of 8 years -> occupancy ~0.12
        return 0.0
    if code == "sals":      # SPECIALTY: only coastal Suffolk, breeding season, present all years
        if county == "Suffolk" and 16 <= woy <= 36:
            return 0.35
        return 0.0
    if code in ("amre", "magw", "bkbw"):   # reliable spring migrants, every year, sharp May peak
        peak = {"amre": (18, 0.55), "magw": (19, 0.45), "bkbw": (19, 0.30)}[code]
        center, amp = peak
        if 16 <= woy <= 22:
            seasonal = amp * 2.71828 ** (-((woy - center) ** 2) / 4.0)  # gaussian-ish window
            return seasonal if county == "New York" else seasonal * 0.35  # CP is the trap
        return 0.0
    if code == "pabu":      # handled specially as a low-sample plant
        return 0.0
    return 0.0

ebd_rows, sed_rows = [], []
sei_counter = 0

for county, (ccode, loc, locid, lat, lon, base) in COUNTIES.items():
    for year in YEARS:
        for woy in range(1, 49):
            # number of complete checklists this county/week/year (Poisson-ish)
            n = max(0, int(random.gauss(base, base * 0.35)))
            for _ in range(n):
                sei_counter += 1
                sei = f"S{sei_counter:07d}"
                d = date_in_week(year, woy)
                obs = f"obsr{random.randint(1, 200):04d}"
                dur = random.choice([20, 30, 45, 60, 90, 120])
                # SED row (one per complete checklist) -> the denominator + the zeros
                sed_rows.append({
                    "SAMPLING EVENT IDENTIFIER": sei, "COUNTRY": "United States",
                    "COUNTRY CODE": "US", "STATE": "New York", "STATE CODE": "US-NY",
                    "COUNTY": county, "COUNTY CODE": ccode, "LOCALITY": loc,
                    "LOCALITY ID": locid, "LATITUDE": lat, "LONGITUDE": lon,
                    "OBSERVATION DATE": d.isoformat(), "OBSERVER ID": obs,
                    "PROTOCOL TYPE": "Traveling", "DURATION MINUTES": dur,
                    "ALL SPECIES REPORTED": 1, "NUMBER OBSERVERS": 1,
                    "APPROVED": 1, "REVIEWED": 0,
                })
                # which species are on this checklist
                for code, (common, sci, cat) in SP.items():
                    if random.random() < detect_prob(code, county, year, woy):
                        ebd_rows.append({
                            "GLOBAL UNIQUE IDENTIFIER": f"URN:CLO:{sei_counter}:{code}",
                            "CATEGORY": cat, "COMMON NAME": common, "SCIENTIFIC NAME": sci,
                            "OBSERVATION COUNT": random.randint(1, 6),
                            "COUNTRY": "United States", "COUNTRY CODE": "US",
                            "STATE": "New York", "STATE CODE": "US-NY",
                            "COUNTY": county, "COUNTY CODE": ccode, "LOCALITY": loc,
                            "LOCALITY ID": locid, "LATITUDE": lat, "LONGITUDE": lon,
                            "OBSERVATION DATE": d.isoformat(), "OBSERVER ID": obs,
                            "PROTOCOL TYPE": "Traveling", "DURATION MINUTES": dur,
                            "ALL SPECIES REPORTED": 1, "APPROVED": 1, "REVIEWED": 0,
                            "SAMPLING EVENT IDENTIFIER": sei,
                        })

# LOW-SAMPLE plant: a brand-new week/locality cell with exactly ONE complete checklist,
# on which Painted Bunting was (improbably) reported -> raw frequency = 1/1 = 1.0.
sei_counter += 1
sei = f"S{sei_counter:07d}"
d = dt.date(2025, 5, 3)              # week ~ 17
lowcell = dict(county="New York", ccode="US-NY-061", loc="Central Park",
               locid="L191106", lat=40.78, lon=-73.97)
sed_rows.append({
    "SAMPLING EVENT IDENTIFIER": sei, "COUNTRY": "United States", "COUNTRY CODE": "US",
    "STATE": "New York", "STATE CODE": "US-NY", "COUNTY": lowcell["county"],
    "COUNTY CODE": lowcell["ccode"], "LOCALITY": lowcell["loc"],
    "LOCALITY ID": lowcell["locid"], "LATITUDE": lowcell["lat"], "LONGITUDE": lowcell["lon"],
    "OBSERVATION DATE": d.isoformat(), "OBSERVER ID": "obsr9999",
    "PROTOCOL TYPE": "Stationary", "DURATION MINUTES": 15,
    "ALL SPECIES REPORTED": 1, "NUMBER OBSERVERS": 1, "APPROVED": 1, "REVIEWED": 0,
})
common, sci, cat = SP["pabu"]
ebd_rows.append({
    "GLOBAL UNIQUE IDENTIFIER": f"URN:CLO:{sei_counter}:pabu", "CATEGORY": cat,
    "COMMON NAME": common, "SCIENTIFIC NAME": sci, "OBSERVATION COUNT": 1,
    "COUNTRY": "United States", "COUNTRY CODE": "US", "STATE": "New York",
    "STATE CODE": "US-NY", "COUNTY": lowcell["county"], "COUNTY CODE": lowcell["ccode"],
    "LOCALITY": lowcell["loc"], "LOCALITY ID": lowcell["locid"], "LATITUDE": lowcell["lat"],
    "LONGITUDE": lowcell["lon"], "OBSERVATION DATE": d.isoformat(), "OBSERVER ID": "obsr9999",
    "PROTOCOL TYPE": "Stationary", "DURATION MINUTES": 15, "ALL SPECIES REPORTED": 1,
    "APPROVED": 1, "REVIEWED": 0, "SAMPLING EVENT IDENTIFIER": sei,
})

def write_tsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

ebd_cols = ["GLOBAL UNIQUE IDENTIFIER", "CATEGORY", "COMMON NAME", "SCIENTIFIC NAME",
            "OBSERVATION COUNT", "COUNTRY", "COUNTRY CODE", "STATE", "STATE CODE",
            "COUNTY", "COUNTY CODE", "LOCALITY", "LOCALITY ID", "LATITUDE", "LONGITUDE",
            "OBSERVATION DATE", "OBSERVER ID", "PROTOCOL TYPE", "DURATION MINUTES",
            "ALL SPECIES REPORTED", "APPROVED", "REVIEWED", "SAMPLING EVENT IDENTIFIER"]
sed_cols = ["SAMPLING EVENT IDENTIFIER", "COUNTRY", "COUNTRY CODE", "STATE", "STATE CODE",
            "COUNTY", "COUNTY CODE", "LOCALITY", "LOCALITY ID", "LATITUDE", "LONGITUDE",
            "OBSERVATION DATE", "OBSERVER ID", "PROTOCOL TYPE", "DURATION MINUTES",
            "ALL SPECIES REPORTED", "NUMBER OBSERVERS", "APPROVED", "REVIEWED"]

write_tsv(OUT / "ebd_sample.txt", ebd_rows, ebd_cols)
write_tsv(OUT / "ebd_sample_sampling.txt", sed_rows, sed_cols)
print(f"Wrote {len(ebd_rows):,} observation rows and {len(sed_rows):,} checklist (SED) rows to {OUT}")
