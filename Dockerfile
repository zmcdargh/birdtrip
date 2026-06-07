# birdtrip — production image. Serves the API + frontend over the precomputed SQLite store.
# The raw EBD never goes in the image; build the store first (locally or in CI):
#   python scripts/precompute_duckdb.py --ebd <ebd>.txt --out data/birdtrip.sqlite --current-year 2026
# then build:  docker build -t birdtrip .   and run:  docker run -p 8000:8000 birdtrip
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY birdtrip ./birdtrip
RUN pip install --no-cache-dir -e ".[api]"

COPY frontend ./frontend
# ship the prebuilt store + taxonomy (derived products only — no raw eBird data)
COPY data/birdtrip.sqlite ./data/birdtrip.sqlite
COPY data/taxonomy ./data/taxonomy

ENV BIRDTRIP_DB=/app/data/birdtrip.sqlite
EXPOSE 8000
CMD ["uvicorn", "birdtrip.api:app", "--host", "0.0.0.0", "--port", "8000"]
