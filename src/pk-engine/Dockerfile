# ── PK Engine (FastAPI) Dockerfile ────────────────────────
# Multi-stage build for smaller image
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
