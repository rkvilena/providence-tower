"""Embedding package.

The local model stack (PyTorch/SentenceTransformers) is imported lazily so
vector-store and RAG runtime modules can use shared data models without
paying for—or requiring—the embedding dependencies.
"""

from __future__ import annotations

from core.embedding.schema import ChunkDocument

__all__ = [
    "ChunkDocument",
    "EmbeddingService",
    "RedisSearchResult",
    "RedisVectorStore",
]

_LAZY_IMPORTS = {
    "EmbeddingService": ("core.embedding.embedding_service", "EmbeddingService"),
    "RedisSearchResult": ("core.embedding.redis_store", "RedisSearchResult"),
    "RedisVectorStore": ("core.embedding.redis_store", "RedisVectorStore"),
}


def __getattr__(name: str):
    try:
        module_path, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    from importlib import import_module

    value = getattr(import_module(module_path), attribute)
    globals()[name] = value
    return value
