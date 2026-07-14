# birdtrip — model & validation

How birdtrip predicts the chance of finding a species, and the evidence that the prediction
is honest and captures real field knowledge. All figures and numbers below are from the
**New York** eBird Basic Dataset (eBird week buckets; ~3.66M species×hotspot×week cells).

---

## 1. The prediction

For a species at a hotspot in a given eBird week, the chance a birder detects it on a visit of
`t` hours is decomposed into **two independent questions**:

```
P(detect on a visit) = ψ        ×   (1 − e^(−λ·t))
                       ╰─ is it ╯     ╰─ if it's there, how fast ╯
                          there?         do you detect it?
```

- **ψ — occupancy.** Probability the species is present at that site/week in a given year.
- **λ — detection rate** (per hour, given present). A time-to-detection (Poisson) rate.

These are estimated separately from the training data, multiplied, and then the **product** is
recalibrated. Rarity lives in ψ; difficulty-of-detection lives in λ (see §4).

### 1.1 Occupancy ψ — empirical-Bayes shrinkage (κ = 3)

From the training years, each cell has `k` = years present and `n` = years surveyed. The naive
rate `k/n` is unusable for thin histories (a bird seen in the only surveyed year → 1.0). We
shrink toward a regional prior:

```
ψ̂ = (k + κ·m) / (n + κ),     κ = 3
```

where `m` is the **regional occupancy** for that species×week (state level, computed *with
absences*: Σ present-years / Σ surveyed-years across the region). κ=3 was selected by held-out
log-loss (§3). This removes the "present 1-of-1 year → 100%" vagrant artifact while leaving
well-surveyed cells essentially unchanged, and it correctly respects genuine recurrence (a bird
present 9 of 11 years stays high).

> Note: a single global κ is used. Rarity-scaling κ(μ) was prototyped and **rejected** — it
> over-shrinks genuinely recurring rarities and was no better than flat κ on held-out data.

### 1.2 Detection λ — time-to-detection

Per cell, the per-checklist detection frequency is deconvolved against that cell's **duration
histogram** (cloglog with a log-duration offset) to recover a per-hour rate `λ`, with Beta
shrinkage on the rate and a regional fallback for thin cells. Then
`P(detect | present, t h) = 1 − e^(−λ·t)`. Effort enters only here, as hours.

### 1.3 Recalibration

A monotone **isotonic** map is fit on a held-out calibration block and applied to the **final
product** ψ̂·(1−e^(−λt)) — not to ψ or λ individually. It corrects residual over-/under-dispersion
so the displayed number means what it says.

> **Order matters:** shrinkage is applied *inside* the model, the product is formed, *then* the
> recal map is fit on that shrunk product. A map fit on raw-occupancy predictions is invalid for
> a shrunk-occupancy pipeline.

---

## 2. Rarity weight w(s) (separate, for ranking)

`w(s)` = region-vs-elsewhere ratio, occupancy-gated, is used to **emphasize local specialties in
ranking**. It is not part of the likelihood and does not affect the calibrated probability.

---

## 3. Is the probability honest? — held-out calibration

Temporal hold-out: **train 2006–2020 · calibrate 2021–2023 · test 2024–2026**, scored per
held-out checklist (27,751,578 trials, base detection rate 25.6%). Metrics: Brier and log-loss
(lower better), Brier skill vs a climatology baseline.

**Served probability (occupancy × detection):**

| model | Brier | skill |
|---|---|---|
| climatology (base rate) | 0.191 | 0% |
| duration-blind | 0.141 | 26% |
| λ (raw occupancy) | 0.137 | 28% |
| **shrunk κ=3 × detect + isotonic recal** | **0.131** | **31%** |

Shrinkage alone is roughly Brier-neutral but converts the raw model's over-confidence into mild
under-confidence; recalibration then lands the curve on the diagonal. The combination is the
best-calibrated served probability we obtained. See `viz/kappa_sweep_served.png` (raw vs recal,
full range + low-p tail) and `viz/holdout_data_calibration.png`.

**Why log-loss, not just Brier:** shrinkage barely moves Brier (0.1371 → 0.1368) but hugely
improves log-loss (0.62 → 0.44), because log-loss punishes the confident-and-wrong predictions
(raw occupancy = 1.0 cells that miss) that shrinkage cures. That tail behavior is the whole point.

**κ selection (held-out occupancy log-loss):** κ=3 minimizes it (0.566), rising monotonically to
κ=8 (0.602); κ(μ) ≈ κ=4 (0.575), not better. After recalibration the differences compress to
~0.003, so the exact value is not fragile. `viz/kappa_sweep_occupancy.png`.

---

## 4. Does λ capture "skulkiness"? — detectability validation

λ recovers field knowledge with no labels. Among NY hotspot cells with ≥10 checklists:

- **Clean skulker/conspicuous separation** (`figures/lambda_skulkers_violin.png`). Every classic
  skulker sits at λ = 0.05–0.29/h, every conspicuous bird at 0.35–1.34/h, with a gap near 0.3.
  In intuitive units (minutes to a 50/50 detection, given present): **Connecticut Warbler ≈ 13.5
  h**, rails/bitterns 4–7 h, vs **American Robin / Canada Goose ≈ 31 min**.
- **Detectability is its own axis, distinct from rarity** (`figures/lambda_detectability_map.png`).
  Spearman(λ, range) = 0.30 (weak). Common-but-cryptic species (Marsh Wren, Swamp Sparrow, at
  ~18k hotspots) sit low on λ; scarce-but-obvious species (Snowy Owl) sit higher. (λ vs *local*
  occupancy is 0.64 — shared abundance — but λ carries real crypsis signal beyond abundance.)
- **Detectability has a season** (`figures/lambda_seasonal_skulkers.png`). λ for marsh skulkers peaks
  when they are vocal — Marsh Wren May–Jul, American Bittern early spring, Whip-poor-will after
  spring arrival.

**"Biggest surprises"** (`figures/lambda_surprises.png`): species whose λ is far from what their
commonness predicts. This does triple duty: it (a) **validates** real cryptic species (Connecticut
& Kentucky Warbler, Yellow-breasted Chat, Gray-cheeked Thrush, Olive-sided & Yellow-bellied
Flycatcher — harder to find than their numbers suggest), (b) **flags data artifacts** (escaped
exotics like Helmeted Guineafowl, Budgerigar, Chukar look "cryptic" because their reports are
erratic), and (c) **surfaces the swarm/observer-intent λ inflation** of §6 — western vagrants
(Western Meadowlark/Grebe, Scissor-tailed Flycatcher, Mountain Bluebird) and Pink-footed Goose show
inflated λ because a stray that turns up gets twitched and reported by everyone. So this plot is
also a built-in QA tool for the deferred swarm problem.

> **Product idea (roadmap):** turn λ into a user-facing **"tough to find" flag** — e.g. a
> "Mourning Warbler · 85% · hard to find, learn its song" badge driven by the detection half-life
> `ln2/λ`, instead of a bare probability. The data clearly supports it.

---

## 5. Does it put birds in the right place at the right time? — range maps

Month-by-month occupancy, hex-binned across NY (true heatmap: every surveyed hex is colored;
dark = surveyed-but-absent = 0; bright = present; un-surveyed hexes are not drawn so the state
footprint shows). No geography or habitat is given to the model; the patterns emerge from the data.

- `figures/hexmap_snowy_owl.png` — winter only (Nov–Mar) on the coast and northern plains; absent in summer.
- `figures/hexmap_short_eared_owl.png` — winter only, on inland grasslands/open country (cleanly distinct
  from the Snowy Owl's coastal pattern).
- `figures/hexmap_blackburnian.png` — absent Nov–Apr; statewide in May/Sep migration; breeds Jun–Jul in the
  Adirondack/Catskill highland forests.
- `figures/hexmap_saltmarsh_sparrow.png` — only on the Long Island / NYC coastal salt marshes, May–Nov;
  never inland. A habitat obligate, pinned to the coast.

**Migration timing** (`figures/arrival_sweep.png` NY, `figures/us_arrival_sweep.png` continental): per-hex first
week occupancy crosses a threshold = spring arrival. At continental scale the wave is unmistakable
— Ruby-throated Hummingbird, Baltimore Oriole, Rose-breasted Grosbeak and Scarlet Tanager all
sweep from the Gulf Coast (March) to Canada (June). Within NY alone the gradient is real but
subtle (~1–2 weeks), so the continental view is the clearer demonstration. Built from the full-US
store (`scripts/us_arrival.py`).

---

## 6. Known limitations / deferred work

- **Intra-day "swarm" inflation of λ.** Checklists are treated as independent; a rare-bird event
  with many same-day checklists can inflate λ (observer-intent bias). A design-effect / one-per-day
  correction is built (`scripts/swarm_lambda_compare.py`) but **not yet evaluated or deployed** —
  on the few cells inspected it was second-order and bidirectional, so it is a later refinement,
  not the cause of the headline over-confidence (that was occupancy).
- **The n=1 floor.** For a site with a single surveyed year, no within-data signal distinguishes a
  one-off from a future recurrer; the estimate is the prior. The held-out test addresses this in
  aggregate but it remains the least-certain regime.
- **Deployment.** Wired in (2026-06): `precompute_duckdb.py` now stores the κ=3 EB occupancy in the
  `occupancy` column (raw kept as `occupancy_raw`, prior as `occ_prior`), so the serve layer uses it
  unchanged. Going live requires (1) **rebuilding the store** and (2) **regenerating the recal map on
  the shrunk product** (the old `*.recal.json` was fit on raw occupancy and is now invalid). See §7.

---

## 7. Reproduce

```bash
# held-out calibration + save trials (run on the EBD; ~minutes)
python scripts/validate_holdout.py --ebd data/ebd_US-NY_relApr-2026.txt --current-year 2026 \
    --starting-year 2006 --calib-years 3 --holdout-years 3 \
    --save-trials data/holdout_trials_NY --temp-dir data/duckdb_tmp --memory-limit 24GB

# κ sweep + occupancy/served calibration figures (instant, from saved trials)
python scripts/kappa_sweep.py --trials data/holdout_trials_NY --out viz --kappa-fig 3
```

### Deploy the EB occupancy model

```bash
# 1. rebuild the store with κ=3 EB occupancy (writes data/birdtrip{.sqlite,.parquet}
#    + data/birdtrip_occ_diag.{png,csv}: occupancy shrinkage US-vs-NY sanity check)
python scripts/precompute_duckdb.py --ebd data/ebd_US_relApr-2026.txt.gz --current-year 2026 \
    --out data/birdtrip.sqlite --temp-dir data/duckdb_tmp --memory-limit 24GB --threads 4

# 2. regenerate the recal map ON THE SHRUNK PRODUCT (overwrites the now-invalid raw-occupancy map).
#    Quick: reuse NY held-out trials. For a US-fit map, first re-run validate_holdout on the US EBD
#    with --save-trials, then point --trials at that.
python scripts/kappa_sweep.py --trials data/holdout_trials_NY --out viz --kappa-fig 3 \
    --save-recal-map data/birdtrip.recal.json
```

The serve layer (`service.py`) reads `occupancy` (now EB-shrunk) and applies `data/birdtrip.recal.json`
automatically — no serve-code changes.

Validation analyses (NY store): `lambda_intuition.py` (§4), `range_maps.py` (§5),
`eb_calibration_loo.py` / `eb_cal_v2.py` (in-sandbox occupancy LOO, superseded by §3).
