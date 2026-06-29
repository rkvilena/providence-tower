from __future__ import annotations

import logging
from typing import Any

from core.env import settings
from core.redis_client import get_binary_client
from core.vector_store.protocol import VectorStoreProtocol
from core.vector_store.local_redis import LocalRedisVectorStore
from core.vector_store.upstash_vector import UpstashVectorStore

LOGGER = logging.getLogger(__name__)


_PROVIDER_LOCAL = "redis_stack"
_PROVIDER_UPSTASH = "upstash"


def create_vector_store(
    *,
    index_name: str = "rag_chunks_idx",
    key_prefix: str = "rag:chunk:",
    distance_metric: str = "COSINE",
) -> VectorStoreProtocol:
    """Factory: resolve the active vector store provider from env vars."""
    provider = (settings.VECTOR_STORE_PROVIDER or _PROVIDER_LOCAL).strip().lower()

    # production: Upstash Vector (cloud)
    # otherwise use local Redis Stack (binary pool) for dev/test
    if provider == _PROVIDER_UPSTASH:
        url = settings.UPSTASH_VECTOR_URL
        token = settings.UPSTASH_VECTOR_TOKEN
        if not url or not token:
            raise RuntimeError(
                "VECTOR_STORE_PROVIDER=upstash requires UPSTASH_VECTOR_URL "
                "and UPSTASH_VECTOR_TOKEN to be set."
            )
        LOGGER.info("Vector store: Upstash Vector (%s)", url)
        return UpstashVectorStore(
            url=url,
            token=token,
            index_name=index_name,
        )

    client = get_binary_client()
    LOGGER.info("Vector store: Local Redis Stack (binary pool)")
    return LocalRedisVectorStore(
        client=client,
        index_name=index_name,
        key_prefix=key_prefix,
        distance_metric=distance_metric,
    )
