"""Manual compatibility smoke check for local BGE vs Cloudflare BGE.

This is not a model-quality benchmark. It verifies that the corpus vectors
produced locally by ``core/embedding/embed.py``/``ptv2_embed.ipynb`` and the
query vectors produced by Cloudflare Workers AI share the same embedding
space before the mixed-provider retrieval path is relied on.

Requires local embedding dependencies and Cloudflare credentials
(``CLOUDFLARE_ACCOUNT_ID``/``CLOUDFLARE_AUTH_TOKEN``).

Usage, from the project root:

    python -m test.manual_bge_compatibility
"""

from __future__ import annotations

import sys

import numpy as np

from core.embedding.embedding_service import EmbeddingService
from core.inference.cloudflare_workers_ai import (
    EMBEDDING_DIMENSIONS,
    CloudflareWorkersAIClient,
)

SAMPLE_TEXT = "Who is Marco and which game does he appear in?"
# Same-model vectors should be strongly aligned; the Cloudflare endpoint may
# use a different pooling/quantization path, so require a clear directional
# match rather than bit-identical output.
MIN_COSINE_SIMILARITY = 0.90


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("Cannot compare zero-length embedding vectors.")
    return float(np.dot(left, right) / denominator)


def main() -> int:
    print(f"Sample text: {SAMPLE_TEXT!r}")
    local_vector = np.asarray(
        EmbeddingService().embed_query(SAMPLE_TEXT), dtype=np.float32
    )
    cloud_vector = np.asarray(
        CloudflareWorkersAIClient().embed_query(SAMPLE_TEXT), dtype=np.float32
    )

    print(f"Local BGE dimensions:      {local_vector.shape[0]}")
    print(f"Cloudflare BGE dimensions: {cloud_vector.shape[0]}")

    if local_vector.shape != (EMBEDDING_DIMENSIONS,):
        print(f"FAIL: local vector is not {EMBEDDING_DIMENSIONS}-dimensional.")
        return 1
    if cloud_vector.shape != (EMBEDDING_DIMENSIONS,):
        print(f"FAIL: Cloudflare vector is not {EMBEDDING_DIMENSIONS}-dimensional.")
        return 1

    similarity = cosine_similarity(local_vector, cloud_vector)
    print(f"Cosine similarity:         {similarity:.6f}")
    if similarity < MIN_COSINE_SIMILARITY:
        print(
            "FAIL: vectors are not directionally compatible "
            f"(expected >= {MIN_COSINE_SIMILARITY})."
        )
        return 1

    print("PASS: local and Cloudflare BGE vectors are dimensionally compatible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
