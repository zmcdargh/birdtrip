#!/bin/sh
# Ensure the precomputed store is present on the volume, then start the server.
# The store (and recal map) are fetched from object storage on first boot if missing, so the
# image stays small and the 7 GB artifact lives only on the persistent volume.
set -e

PARQ="${BIRDTRIP_DB%.sqlite}.parquet"          # e.g. /data/birdtrip.parquet
RECAL="${BIRDTRIP_DB%.sqlite}.recal.json"
GRID="${PARQ%.parquet}.grid.parquet"           # best_trips proxy sidecars (optional)
GRIDCEN="${PARQ%.parquet}.gridcen.parquet"
mkdir -p "$(dirname "$PARQ")" "${BIRDTRIP_DUCK_TMP:-/tmp}"

if [ ! -f "$PARQ" ]; then
  if [ -n "$STORE_URL" ]; then
    echo "[entrypoint] downloading store -> $PARQ"
    curl -fSL --retry 3 "$STORE_URL" -o "$PARQ.part" && mv "$PARQ.part" "$PARQ"
    if [ -n "$RECAL_URL" ]; then
      echo "[entrypoint] downloading recal map -> $RECAL"
      curl -fSL --retry 3 "$RECAL_URL" -o "$RECAL" || echo "[entrypoint] recal download failed (serving uncalibrated)"
    fi
  else
    echo "[entrypoint] WARNING: no store at $PARQ and STORE_URL unset — API will 503 until one is present." >&2
  fi
fi

# best_trips grid-proxy sidecars: derived from STORE_URL (…/birdtrip.parquet -> …/birdtrip.grid.parquet).
# Optional — if absent, best_trips falls back to scanning the store live. Non-fatal on failure.
if [ -n "$STORE_URL" ]; then
  if [ ! -f "$GRID" ]; then
    echo "[entrypoint] downloading best_trips grid proxy -> $GRID"
    curl -fSL --retry 3 "${STORE_URL%.parquet}.grid.parquet" -o "$GRID.part" && mv "$GRID.part" "$GRID" \
      || echo "[entrypoint] grid proxy not available (best_trips will scan live)"
  fi
  if [ ! -f "$GRIDCEN" ]; then
    curl -fSL --retry 3 "${STORE_URL%.parquet}.gridcen.parquet" -o "$GRIDCEN.part" && mv "$GRIDCEN.part" "$GRIDCEN" \
      || echo "[entrypoint] grid centroids not available"
  fi
fi

exec uvicorn birdtrip.api:app --host 0.0.0.0 --port "${PORT:-8080}"
