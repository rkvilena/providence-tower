"""
Production-only model downloader.

Downloads embedding and reranker models to a local cache directory.
This script is executed *during the Docker build* so that model weights
are baked into a dedicated image layer, avoiding re-downloads on every
code change.

Usage (inside Dockerfile):
    COPY core/download_models.py .
    RUN python download_models.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that "import core.*" works
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("download_models")

# ---------------------------------------------------------------------------
# Models & cache directory
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Write into a sub-directory relative to this script so the path is
# predictable at both build-time and run-time inside the container.
CACHE_DIR: Path = Path(__file__).resolve().parent / "cached_models"


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Model cache directory: %s", CACHE_DIR)

    # ------------------------------------------------------------------
    # Download embedding model
    # ------------------------------------------------------------------
    LOGGER.info("Downloading embedding model '%s' ...", EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(CACHE_DIR),
    )
    LOGGER.info("Embedding model cached successfully.")

    # ------------------------------------------------------------------
    # Download reranker (cross-encoder) model
    # ------------------------------------------------------------------
    LOGGER.info("Downloading reranker model '%s' ...", RERANKER_MODEL)

    # CrossEncoder does NOT support a cache_folder kwarg — rely on
    # HuggingFace environment variables instead so the weights land
    # under CACHE_DIR.
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
    os.environ["HF_HOME"] = str(CACHE_DIR)

    from sentence_transformers import CrossEncoder

    CrossEncoder(RERANKER_MODEL)
    LOGGER.info("Reranker model cached successfully.")

    LOGGER.info("All models downloaded and cached in %s", CACHE_DIR)


if __name__ == "__main__":
    main()
