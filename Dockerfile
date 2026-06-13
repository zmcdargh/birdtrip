# birdtrip — production image (API + frontend). The 7 GB store is NOT baked in: it lives on a
# persistent volume and is fetched from object storage on first boot (see docker-entrypoint.sh /
# DEPLOY.md). The raw EBD never goes near the image.
#   docker build -t birdtrip .
#   docker run -p 8080:8080 -e STORE_URL=... -e RECAL_URL=... -v birdtrip_data:/data birdtrip
FROM python:3.11-slim

# curl for the boot-time store download; no build toolchain needed (wheels only)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY birdtrip ./birdtrip
RUN pip install --no-cache-dir -e ".[api]"

COPY frontend ./frontend
COPY data/taxonomy ./data/taxonomy          # small published taxonomy (safe to ship)
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x docker-entrypoint.sh

# the store path; the parquet sidecar (/data/birdtrip.parquet) is what actually gets served.
ENV BIRDTRIP_DB=/data/birdtrip.sqlite \
    PORT=8080 \
    BIRDTRIP_DUCK_MEM=3GB \
    BIRDTRIP_DUCK_TMP=/data/duckdb_tmp
EXPOSE 8080
ENTRYPOINT ["./docker-entrypoint.sh"]
