# =============================================================================
# Providence Tower v2 — Production Docker Image
# =============================================================================
# Build context is the PROJECT ROOT (`.`), but all application code lives
# inside the `core/` sub-directory.  The layered COPY/RUN order is carefully
# arranged to leverage Docker layer caching:
#
#   1. Copy dependency manifest & install → only rebuilds when deps change
#   2. Copy model downloader script       → only rebuilds when downloader changes
#   3. Download & cache model weights     → stable layer unless models change
#   4. Copy remaining application code    → rebuilt on every code change
# =============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Point sentence-transformers to the pre-downloaded cache folder so that
    # models are loaded from the baked-in cache rather than re-downloaded.
    SENTENCE_TRANSFORMERS_HOME=/app/cached_models

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
# a) Copy dependency manifest and install packages.
#    Changes only when requirements.txt changes.
COPY core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Layer 2 — Model download script
# ---------------------------------------------------------------------------
# b) Copy ONLY the download script (not the rest of core/).
COPY core/download_models.py .

# ---------------------------------------------------------------------------
# Layer 3 — Pre-download model weights (build-time cache)
# ---------------------------------------------------------------------------
# c) Execute download script to bake model weights into a dedicated layer.
#    This layer is only invalidated when download_models.py changes.
#    The weights are written into /app/cached_models/.
RUN python download_models.py

# ---------------------------------------------------------------------------
# Layer 4 — Application code
# ---------------------------------------------------------------------------
# d) Copy the remaining application files from core/ into the working directory.
#    This is the fastest-changing layer.
#    NOTE: download_models.py is re-copied here (from core/) — that's fine,
#    it's a small utility script that can remain in the final image.
COPY core/ .

EXPOSE 8080

CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8080"]