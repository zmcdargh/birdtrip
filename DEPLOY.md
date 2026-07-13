# Deploying birdtrip

The app is a read-only FastAPI service + a static frontend, backed by a ~7 GB DuckDB/parquet
store. No database, no user accounts (uploaded life lists are processed in memory and discarded).
The only deployment wrinkle is the store: too big for the image or git, it lives on a **persistent
volume** and is **fetched from object storage on first boot**.

Recommended host: **Fly.io** (always-on VM + volume + automatic HTTPS, cheap). Render works too —
see the bottom. Target here: public, modest traffic.

## 0. One-time: build the store + recal map

On your machine (needs the EBD; see `MODEL.md` §7):

```bash
python scripts/precompute_duckdb.py --ebd data/ebd_US_relApr-2026.txt --current-year 2026 \
    --out data/birdtrip.sqlite --temp-dir data/duckdb_tmp --memory-limit 24GB --threads 4
python scripts/kappa_sweep.py --trials data/holdout_trials_NY --out viz --kappa-fig 3 \
    --save-recal-map data/birdtrip.recal.json
```

You now have `data/birdtrip.parquet` (~7 GB, the only file the server queries) and
`data/birdtrip.recal.json`. The `.sqlite` is **not** needed in production.

## 1. Put the store in object storage

Use any S3-compatible store; **Cloudflare R2** or **Backblaze B2** are cheapest (~$0.10/GB-mo,
no egress on R2). Upload the two files and get URLs the container can `curl`:

```bash
# example with rclone (configure an R2 remote once), or use the provider's web UI
rclone copy data/birdtrip.parquet     r2:birdtrip/      # ~7 GB
rclone copy data/birdtrip.recal.json  r2:birdtrip/
```

Make them readable by the server — either public-read on the bucket, or a long-lived signed URL.
Note the resulting `STORE_URL` (…/birdtrip.parquet) and `RECAL_URL` (…/birdtrip.recal.json).

## 2. Deploy to Fly

```bash
# install flyctl, then from the repo root:
fly launch --no-deploy                       # names the app, fills app/primary_region in fly.toml
fly volume create birdtrip_data --size 15 --region <your-region>   # holds the store + duckdb spill
fly secrets set STORE_URL="https://…/birdtrip.parquet" \
                RECAL_URL="https://…/birdtrip.recal.json"
fly deploy
```

First boot downloads the store to the volume (a few minutes; the health check has a 120 s grace
and `/healthz` doesn't touch the store). Subsequent deploys reuse the volume — instant. The
coordinate cache warms in the background on startup, so the first trip search isn't cold.

Custom domain + HTTPS:

```bash
fly certs add birds.example.com      # then add the CNAME/A records it prints
```

## 3. Sizing & cost

- `shared-cpu-2x` / 4 GB RAM (in `fly.toml`) handles modest traffic. `BIRDTRIP_DUCK_MEM=3GB`
  keeps the heavy best-trips query spilling to disk instead of OOMing.
- Cost ≈ a few $/mo for the VM (always-on) + ~$1/mo for the 15 GB volume + ~$0.70/mo R2 storage.
- The **nationwide best-trips** search is the one heavy endpoint (scans the 7 GB store a few
  times). Fine for modest traffic; if it gets popular, the next steps are: precompute/cache the
  base occupancy proxy per store, cap `shortlist`, or move best-trips to a small job queue.

## 4. Updating the store

Rebuild locally, re-upload to object storage, then `fly ssh console -C "rm /data/birdtrip.parquet"`
and restart (it re-downloads), or `fly volume` swap. Always regenerate the recal map alongside the
store (it must be fit on the same shrunk product — see `MODEL.md` §6).

## Optional: natural-language search (`/ask`)

Off by default. When an LLM key is set, an "Ask in plain English" box appears (the frontend checks
`/config`); the model only fills the search form — it never touches data.

```bash
fly secrets set LLM_API_KEY="sk-…"          # DeepSeek by default (cheap, tool-calling)
# optional provider override (any OpenAI-compatible endpoint):
#   fly secrets set LLM_BASE_URL="https://api.deepseek.com" LLM_MODEL="deepseek-chat"
```

**Key safety (important):** the key is read only server-side inside `/ask`, never returned to the
browser, never logged, never baked into the image or repo — set it *only* as a Fly secret. `/ask`
is rate-limited per-IP (`ASK_RATE_PER_MIN`=6, `ASK_RATE_PER_DAY`=60) and globally
(`ASK_GLOBAL_DAILY`=1500/day) with a 1000-char input cap, so it can't be hammered into a big bill.
As belt-and-suspenders, **also set a hard spend limit on the DeepSeek account**. Geocoding uses
Nominatim (set `NOMINATIM_URL` to self-host if you expect real volume — the public server is
rate-limited).

## Render alternative

Create a **Web Service from this repo** (Docker), add a **Persistent Disk** mounted at `/data`
(≥15 GB), set env `STORE_URL`, `RECAL_URL`, `BIRDTRIP_DB=/data/birdtrip.sqlite`,
`BIRDTRIP_DUCK_MEM=3GB`, `BIRDTRIP_DUCK_TMP=/data/duckdb_tmp`, pick an instance with ≥4 GB RAM.
HTTPS and a URL are automatic; same boot-time download behavior.
