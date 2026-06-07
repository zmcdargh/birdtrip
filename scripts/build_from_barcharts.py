#!/usr/bin/env python3
"""
Build a realistic multi-state precomputed table from REAL eBird bar charts.

Bar charts give real per-checklist frequency by week for a region, but (a) at the
region level and (b) pooled across years. So we:
  1. parse each state's chart -> {species: 48 weekly frequencies} + weekly sample sizes
     (resolving taxa to countable species via the eBird taxonomy; dropping spuh/slash/hybrid),
  2. spin up a few synthetic HOTSPOTS per state, each a perturbed draw from the state's
     real distribution (hotspots concentrate birds, so a species can exceed the state mean),
  3. derive a principled OCCUPANCY from frequency: at a hotspot birded ~`intensity` times
     per week per year, P(present in a typical year) = 1 - (1 - f)^intensity. Common birds
     -> ~1; the rare tail and vagrant-frequency taxa -> low, exactly as the planner wants.
  4. keep the planner consistent with the data: p_lifer_1 = occupancy * detect_given_present = f.

What's REAL here: species pools, richness, seasonality, abundance, relative rarity, and
cross-state endemism (FL specialties present, absent elsewhere). What's SYNTHESIZED: the
split into hotspots and the per-year occupancy layer (bar charts have no year resolution).
"""
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from birdtrip.taxonomy import Taxonomy

BARCHARTS = ROOT / "data" / "barcharts"
OUT = ROOT / "data" / "precomputed_states.csv"

N_YEARS = 15
PRIOR_STRENGTH = 10.0
MIN_CHECKLISTS = 5
HOTSPOTS = [   # (suffix, overall richness/intensity factor, checklists per week per year)
    ("Premier Reserve", 1.25, 40),
    ("County Park", 1.00, 15),
    ("Riverside Trail", 0.85, 6),
    ("Outlying Marsh", 0.70, 3),
]
RNG = np.random.default_rng(7)

# state code -> (name, lat, lon centroid)
STATE_META = {
    "US-AL":("Alabama",32.8,-86.8),"US-AK":("Alaska",64.2,-152.0),"US-AZ":("Arizona",34.3,-111.7),
    "US-AR":("Arkansas",34.9,-92.4),"US-CA":("California",37.2,-119.5),"US-CO":("Colorado",39.0,-105.5),
    "US-CT":("Connecticut",41.6,-72.7),"US-DE":("Delaware",39.0,-75.5),"US-DC":("District of Columbia",38.90,-77.04),
    "US-FL":("Florida",27.8,-81.7),"US-GA":("Georgia",32.6,-83.4),"US-HI":("Hawaii",20.3,-156.4),
    "US-ID":("Idaho",44.4,-114.6),"US-IL":("Illinois",40.0,-89.2),"US-IN":("Indiana",39.9,-86.3),
    "US-IA":("Iowa",42.0,-93.5),"US-KS":("Kansas",38.5,-98.4),"US-KY":("Kentucky",37.5,-85.3),
    "US-LA":("Louisiana",31.0,-92.0),"US-ME":("Maine",45.4,-69.2),"US-MD":("Maryland",39.0,-76.8),
    "US-MA":("Massachusetts",42.3,-71.8),"US-MI":("Michigan",44.3,-85.4),"US-MN":("Minnesota",46.3,-94.3),
    "US-MS":("Mississippi",32.7,-89.7),"US-MO":("Missouri",38.4,-92.5),"US-MT":("Montana",47.0,-109.6),
    "US-NE":("Nebraska",41.5,-99.8),"US-NV":("Nevada",39.3,-116.6),"US-NH":("New Hampshire",43.7,-71.6),
    "US-NJ":("New Jersey",40.2,-74.7),"US-NM":("New Mexico",34.4,-106.1),"US-NY":("New York",42.9,-75.5),
    "US-NC":("North Carolina",35.6,-79.4),"US-ND":("North Dakota",47.5,-100.5),"US-OH":("Ohio",40.3,-82.8),
    "US-OK":("Oklahoma",35.6,-97.5),"US-OR":("Oregon",44.0,-120.5),"US-PA":("Pennsylvania",40.9,-77.8),
    "US-RI":("Rhode Island",41.7,-71.6),"US-SC":("South Carolina",33.9,-80.9),"US-SD":("South Dakota",44.4,-100.2),
    "US-TN":("Tennessee",35.9,-86.4),"US-TX":("Texas",31.5,-99.3),"US-UT":("Utah",39.3,-111.7),
    "US-VT":("Vermont",44.1,-72.7),"US-VA":("Virginia",37.5,-78.9),"US-WA":("Washington",47.4,-120.5),
    "US-WV":("West Virginia",38.6,-80.6),"US-WI":("Wisconsin",44.6,-89.9),"US-WY":("Wyoming",43.0,-107.6),
}

LABEL_RE = re.compile(r"^(.*?)\s*\(<em[^>]*>(.*?)</em>\)\s*$")


def parse_barchart(path, tax):
    """-> (sample_sizes[48], list of (species_code, common, sci, freqs[48]))."""
    sample = None
    rows = []
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        cells = ln.split("\t")
        if cells[0].startswith("Sample Size:"):
            sample = [float(x) if x else 0.0 for x in cells[1:49]]
        elif len(cells) >= 49 and "<em" in cells[0]:
            m = LABEL_RE.match(cells[0])
            common = m.group(1) if m else cells[0]
            sci = m.group(2) if m else None
            code = tax.resolve_to_species(tax.code_for(common=common, sci=sci))
            if code is None:
                continue  # spuh / slash / hybrid -> not a countable species
            freqs = [float(x) if x else 0.0 for x in cells[1:49]]
            rows.append((code, tax.common_name(code), tax.scientific_name(code), freqs))
    return sample, rows


def state_code_from_name(fname):
    m = re.search(r"US-([A-Z]{2})", fname)
    return f"US-{m.group(1)}" if m else None


def build():
    tax = Taxonomy()
    files = sorted(BARCHARTS.glob("*.txt"))
    if not files:
        sys.exit(f"No bar charts in {BARCHARTS}")
    out_rows = []
    for f in files:
        code = state_code_from_name(f.name)
        if code not in STATE_META:
            print(f"  skip {f.name}: unknown state code"); continue
        name, clat, clon = STATE_META[code]
        sample, species = parse_barchart(f, tax)
        print(f"  {code} {name}: {len(species)} countable species")
        # collapse duplicate species codes (issf rows rolling into one species): take max freq/week
        by_code = {}
        for scode, common, sci, freqs in species:
            if scode in by_code:
                by_code[scode] = (common, sci, np.maximum(by_code[scode][2], freqs))
            else:
                by_code[scode] = (common, sci, np.array(freqs))
        for h, (suffix, richness, intensity) in enumerate(HOTSPOTS):
            lat = clat + float(RNG.uniform(-0.6, 0.6))
            lon = clon + float(RNG.uniform(-0.6, 0.6))
            locality = f"{name} – {suffix}"
            locid = f"L{code[-2:]}{h:02d}"
            n_chk = intensity * N_YEARS
            for scode, (common, sci, sfreq) in by_code.items():
                # per-hotspot, per-species multiplier: hotspots concentrate (or miss) birds
                noise = float(RNG.lognormal(0.0, 0.55))
                for wk in range(48):
                    sf = sfreq[wk]
                    if sf <= 0:
                        continue
                    f = min(0.95, sf * richness * noise)
                    if f < 5e-4:
                        continue
                    occ = 1 - (1 - f) ** intensity            # P(present in a typical year)
                    dgp = min(1.0, f / occ) if occ > 0 else 0.0
                    n_det = int(round(f * n_chk))
                    a = sf * PRIOR_STRENGTH; b = (1 - sf) * PRIOR_STRENGTH
                    f_shrunk = (n_det + a) / (n_chk + a + b)   # toward the real state mean
                    out_rows.append((
                        name, code, "", locality, locid, round(lat, 4), round(lon, 4), wk + 1,
                        sci, common, n_chk, n_det, round(f, 6), round(f_shrunk, 6),
                        N_YEARS, int(round(occ * N_YEARS)), round(occ, 4), round(dgp, 4),
                        round(occ * dgp, 6), int(n_chk >= MIN_CHECKLISTS)))
    cols = ["STATE", "STATE CODE", "COUNTY", "LOCALITY", "LOCALITY ID", "latitude", "longitude",
            "week", "SCIENTIFIC NAME", "COMMON NAME", "n_checklists", "n_detections",
            "freq_raw", "freq_shrunk", "years_surveyed", "years_present", "occupancy",
            "detect_given_present", "p_lifer_1", "trusted"]
    df = pd.DataFrame(out_rows, columns=cols)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df):,} cells across {df['STATE'].nunique()} states, "
          f"{df['LOCALITY'].nunique()} hotspots -> {OUT}")


if __name__ == "__main__":
    build()
