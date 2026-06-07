#!/usr/bin/env python3
"""
Download eBird bar-chart (histogram) data for every US state — run LOCALLY with
your own eBird login. Saves one TSV per state into data/barcharts/US-XX.txt.

eBird gates the histogram download behind your session, so the script needs your
logged-in cookie. Two ways to provide it (pick one):

  A) Quick: in a browser logged into ebird.org, open DevTools -> Network, click any
     request to ebird.org, copy the full "cookie:" request header, then:
         export EBIRD_COOKIE='<paste the whole cookie string>'
         python scripts/download_barcharts.py

  B) cookies.txt: export a Netscape cookies.txt for ebird.org and:
         python scripts/download_barcharts.py --cookies /path/to/cookies.txt

Be polite: there's a delay between requests. This is your own non-commercial data
use; don't hammer the server. Verify one file looks right (it should start with
"Frequency of observations...") before trusting the batch.

Sanity-check the URL first by pasting this in your logged-in browser:
  https://ebird.org/barchart?byr=1900&eyr=2025&bmo=1&emo=12&r=US-CA&fmt=tsv
It should download a TSV, not show a web page. If the param names differ in your
region, edit URL_TEMPLATE below to match what your browser's "Download Histogram
Data" link uses.
"""
import argparse
import http.cookiejar
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "barcharts"
URL_TEMPLATE = "https://ebird.org/barchart?byr=1900&eyr=2025&bmo=1&emo=12&r={region}&fmt=tsv"

STATES = {
    "US-AL": "Alabama", "US-AK": "Alaska", "US-AZ": "Arizona", "US-AR": "Arkansas",
    "US-CA": "California", "US-CO": "Colorado", "US-CT": "Connecticut", "US-DE": "Delaware",
    "US-FL": "Florida", "US-GA": "Georgia", "US-HI": "Hawaii", "US-ID": "Idaho",
    "US-IL": "Illinois", "US-IN": "Indiana", "US-IA": "Iowa", "US-KS": "Kansas",
    "US-KY": "Kentucky", "US-LA": "Louisiana", "US-ME": "Maine", "US-MD": "Maryland",
    "US-MA": "Massachusetts", "US-MI": "Michigan", "US-MN": "Minnesota", "US-MS": "Mississippi",
    "US-MO": "Missouri", "US-MT": "Montana", "US-NE": "Nebraska", "US-NV": "Nevada",
    "US-NH": "New Hampshire", "US-NJ": "New Jersey", "US-NM": "New Mexico", "US-NY": "New York",
    "US-NC": "North Carolina", "US-ND": "North Dakota", "US-OH": "Ohio", "US-OK": "Oklahoma",
    "US-OR": "Oregon", "US-PA": "Pennsylvania", "US-RI": "Rhode Island", "US-SC": "South Carolina",
    "US-SD": "South Dakota", "US-TN": "Tennessee", "US-TX": "Texas", "US-UT": "Utah",
    "US-VT": "Vermont", "US-VA": "Virginia", "US-WA": "Washington", "US-WV": "West Virginia",
    "US-WI": "Wisconsin", "US-WY": "Wyoming",
}


def build_opener(args):
    if args.cookies:
        jar = http.cookiejar.MozillaCookieJar(args.cookies)
        jar.load(ignore_discard=True, ignore_expires=True)
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    else:
        opener = urllib.request.build_opener()
        cookie = os.environ.get("EBIRD_COOKIE")
        if not cookie:
            sys.exit("No auth: set EBIRD_COOKIE='<cookie header>' or pass --cookies cookies.txt")
        opener.addheaders = [("Cookie", cookie),
                             ("User-Agent", "Mozilla/5.0 (birdtrip personal data download)")]
    return opener


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cookies", help="Netscape cookies.txt for ebird.org")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between requests")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    opener = build_opener(args)
    ok = bad = 0
    for code, name in STATES.items():
        dest = OUT / f"{code}.txt"
        if dest.exists() and not args.overwrite:
            print(f"  skip {code} ({name}) — already downloaded")
            ok += 1
            continue
        url = URL_TEMPLATE.format(region=code)
        try:
            with opener.open(url, timeout=60) as r:
                data = r.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  FAIL {code} ({name}): {e}")
            bad += 1
            continue
        if "Frequency of observations" not in data:
            print(f"  WARN {code} ({name}): response doesn't look like a bar chart "
                  f"(not logged in? wrong URL params?) — got {len(data)} chars; not saving")
            bad += 1
            continue
        dest.write_text(data, encoding="utf-8")
        n_taxa = next((ln.split("\t")[1] for ln in data.splitlines()
                       if ln.startswith("Number of taxa")), "?")
        print(f"  ok   {code} ({name}): {n_taxa} taxa -> {dest.name}")
        ok += 1
        time.sleep(args.delay)

    print(f"\nDone: {ok} present, {bad} failed. Files in {OUT}")
    if bad:
        print("If many failed/warned: verify the sanity-check URL in your browser, "
              "refresh your cookie, or adjust URL_TEMPLATE.")


if __name__ == "__main__":
    main()
