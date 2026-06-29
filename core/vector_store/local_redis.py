from __future__ import annotations

import logging
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField

try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ModuleNotFoundError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from core.embedding.embedding_service import ChunkDocument
from core.vector_store.search_result import VectorSearchResult
from core.vector_store.protocol import VectorStoreProtocol

LOGGER = logging.getLogger(__name__)


class LocalRedisVectorStore(VectorStoreProtocol):
    """Local Redis Stack vector store using RediSearch + HASH + HNSW.

    This is the reference implementation of ``VectorStoreProtocol`` that
    targets a local Redis Stack instance with the RediSearch module.
    """

    def __init__(
        self,
        client: redis.Redis,
        *,
        index_name: str = "rag_chunks_idx",
        key_prefix: str = "rag:chunk:",
        distance_metric: str = "COSINE",
    ) -> None:
        self._client = client
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.distance_metric = distance_metric.upper()

    # ---- VectorStoreProtocol --------------------------------------------------

    def ensure_index(self, vector_dim: int) -> None:
        if vector_dim < 1:
            raise ValueError("vector_dim must be >= 1")
        if self._index_exists():
            return

        fields = [
            TagField("chunk_id"),
            NumericField("page_id"),
            TextField("page_title"),
            TextField("section"),
            TextField("subsection"),
            TagField("source_file"),
            TextField("text"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": int(vector_dim),
                    "DISTANCE_METRIC": self.distance_metric,
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        ]
        definition = IndexDefinition(
            prefix=[self.key_prefix], index_type=IndexType.HASH
        )
        self._client.ft(self.index_name).create_index(fields, definition=definition)
        LOGGER.info("Created Redis index '%s' (dim=%s)", self.index_name, vector_dim)

    def search_similar(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 20,
        filter_entities: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        vec_bytes = np.asarray(query_vector, dtype=np.float32).tobytes()
        filter_str = self._build_filter_query(filter_entities)
        knn_query = f"{filter_str}=>[KNN {top_k} @embedding $vec AS score]"
        query = (
            Query(knn_query)
            .sort_by("score")
            .return_fields(
                "chunk_id",
                "page_id",
                "page_title",
                "section",
                "subsection",
                "source_file",
                "text",
                "score",
            )
            .paging(0, top_k)
            .dialect(2)
        )
        result = self._client.ft(self.index_name).search(
            query, query_params={"vec": vec_bytes}
        )
        return self._parse_ft_result(result)

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 20,
        filter_entities: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        # For RediSearch, hybrid is just KNN with a pre-filter.
        return self.search_similar(
            query_vector, top_k=top_k, filter_entities=filter_entities
        )

    # ---- ingest helper (used by CLI scripts) -----------------------------------

    def upsert_documents(
        self,
        documents: list[ChunkDocument],
        vectors: np.ndarray,
        *,
        batch_size: int = 1000,
    ) -> int:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors length mismatch")
        if not documents:
            return 0

        written = 0
        pipeline = self._client.pipeline(transaction=False)
        for idx, (doc, vec) in enumerate(zip(documents, vectors), start=1):
            key = f"{self.key_prefix}{doc.chunk_id}"
            mapping = {
                "chunk_id": doc.chunk_id,
                "page_id": doc.page_id,
                "page_title": doc.page_title,
                "section": doc.section,
                "subsection": doc.subsection,
                "source_file": doc.source_file,
                "text": doc.text,
                "embedding": np.asarray(vec, dtype=np.float32).tobytes(),
            }
            pipeline.hset(key, mapping=mapping)
            if idx % batch_size == 0:
                pipeline.execute()
                written += batch_size
                pipeline = self._client.pipeline(transaction=False)
        pipeline.execute()
        written += len(documents) % batch_size
        return written

    # ---- internals ------------------------------------------------------------

    def ping(self) -> bool:
        return bool(self._client.ping())

    def _index_exists(self) -> bool:
        try:
            self._client.ft(self.index_name).info()
            return True
        except redis.ResponseError:
            return False

    def _build_filter_query(self, entities: list[str] | None) -> str:
        if not entities:
            return "*"
        tokens = [self._escape_token(e) for e in entities if e.strip()]
        if not tokens:
            return "*"
        joined = "|".join(tokens)
        return f"@text:({joined})"

    @staticmethod
    def _escape_token(token: str) -> str:
        import re

        text = str(token).strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("\\", "\\\\").replace('"', '\\"')
        needs_quotes = any(ch.isspace() for ch in text) or any(
            ch in text for ch in "-:@()[]{}|"
        )
        return f'"{text}"' if needs_quotes else text

    def _parse_ft_result(self, result: Any) -> list[VectorSearchResult]:
        docs = list(getattr(result, "docs", []) or [])
        results: list[VectorSearchResult] = []
        for doc in docs:
            key = str(getattr(doc, "id", "") or "")

            def _get(name: str) -> Any:
                return getattr(doc, name) if hasattr(doc, name) else None

            def _to_str(val: Any) -> str:
                if val is None:
                    return ""
                if isinstance(val, bytes):
                    return val.decode("utf-8", errors="ignore")
                return str(val)

            score_raw = _get("score")
            try:
                score = float(score_raw)
            except Exception:
                score = 0.0

            text = _to_str(_get("text"))
            results.append(
                VectorSearchResult(
                    key=key,
                    score=score,
                    text=text,
                    metadata={
                        "chunk_id": _to_str(_get("chunk_id")) or key,
                        "page_id": _to_str(_get("page_id")),
                        "page_title": _to_str(_get("page_title")),
                        "section": _to_str(_get("section")) or None,
                        "subsection": _to_str(_get("subsection")) or None,
                        "source_file": _to_str(_get("source_file")) or None,
                    },
                )
            )
        return results
