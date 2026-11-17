# =============================================================================
# Providence Tower v2 — Production Docker Image
# =============================================================================
# Build context is the PROJECT ROOT (`.`), but all application code lives
# inside the `core/` sub-directory.
#
# RAG-time query embeddings and reranking run on Cloudflare Workers AI, so the
# API image installs only the runtime requirements. Local BGE corpus
# preparation uses core/requirements-embedding.txt outside this image.
# =============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies (required for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Layer 1 — Python dependencies
# ---------------------------------------------------------------------------
# Changes only when the runtime requirements change.
COPY core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Layer 2 — Application code
# ---------------------------------------------------------------------------
# Fastest-changing layer: copy the whole core/ package last.
COPY core/ ./core/

EXPOSE 8080

CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8080"]
