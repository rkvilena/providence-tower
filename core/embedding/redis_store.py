from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import redis

from core.embedding.embedding_service import ChunkDocument


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RedisSearchResult:
    key: str
    score: float
    text: str
    metadata: dict[str, Any]


class RedisVectorStore:
    """
    Redis-backed vector store using RediSearch + HASH + HNSW.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        index_name: str = "rag_chunks_idx",
        key_prefix: str = "rag:chunk:",
        distance_metric: str = "COSINE",
    ) -> None:
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.distance_metric = distance_metric.upper()
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,
        )

    def ping(self) -> bool:
        return bool(self.client.ping())

    def ensure_index(self, vector_dim: int) -> None:
        if vector_dim < 1:
            raise ValueError("vector_dim must be >= 1")
        if self._index_exists():
            return

        create_command = [
            "FT.CREATE",
            self.index_name,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            self.key_prefix,
            "SCHEMA",
            "chunk_id",
            "TAG",
            "page_id",
            "NUMERIC",
            "page_title",
            "TEXT",
            "section",
            "TEXT",
            "subsection",
            "TEXT",
            "source_file",
            "TAG",
            "text",
            "TEXT",
            "embedding",
            "VECTOR",
            "HNSW",
            "10",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(vector_dim),
            "DISTANCE_METRIC",
            self.distance_metric,
            "M",
            "16",
            "EF_CONSTRUCTION",
            "200",
        ]
        self.client.execute_command(*create_command)
        LOGGER.info("Created Redis index '%s' (dim=%s)", self.index_name, vector_dim)

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
        pipeline = self.client.pipeline(transaction=False)
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
                pipeline = self.client.pipeline(transaction=False)
        pipeline.execute()
        written += len(documents) % batch_size
        return written

    def search_similar(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
        filter_query: str = "*",
    ) -> list[RedisSearchResult]:
        vec_bytes = np.asarray(query_vector, dtype=np.float32).tobytes()
        knn_query = f"{filter_query}=>[KNN {top_k} @embedding $vec AS score]"
        cmd = [
            "FT.SEARCH",
            self.index_name,
            knn_query,
            "PARAMS",
            "2",
            "vec",
            vec_bytes,
            "SORTBY",
            "score",
            "RETURN",
            "8",
            "chunk_id",
            "page_id",
            "page_title",
            "section",
            "subsection",
            "source_file",
            "text",
            "score",
            "DIALECT",
            "2",
        ]
        raw = self.client.execute_command(*cmd)
        return self._parse_search_results(raw)

    def _index_exists(self) -> bool:
        try:
            self.client.execute_command("FT.INFO", self.index_name)
            return True
        except redis.ResponseError:
            return False

    def _parse_search_results(self, raw: list[Any]) -> list[RedisSearchResult]:
        if not raw:
            return []
        results: list[RedisSearchResult] = []
        for i in range(1, len(raw), 2):
            key = raw[i].decode("utf-8") if isinstance(raw[i], bytes) else str(raw[i])
            flat_fields = raw[i + 1]
            fields: dict[str, Any] = {}
            for j in range(0, len(flat_fields), 2):
                field_key = flat_fields[j].decode("utf-8") if isinstance(flat_fields[j], bytes) else str(flat_fields[j])
                field_val = flat_fields[j + 1]
                if isinstance(field_val, bytes):
                    field_val = field_val.decode("utf-8", errors="ignore")
                fields[field_key] = field_val
            results.append(
                RedisSearchResult(
                    key=key,
                    score=float(fields.get("score", 0.0)),
                    text=str(fields.get("text", "")),
                    metadata={
                        "chunk_id": fields.get("chunk_id"),
                        "page_id": fields.get("page_id"),
                        "page_title": fields.get("page_title"),
                        "section": fields.get("section"),
                        "subsection": fields.get("subsection"),
                        "source_file": fields.get("source_file"),
                    },
                )
            )
        return results
