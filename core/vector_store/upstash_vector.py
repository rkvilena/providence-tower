from __future__ import annotations

import httpx
import logging

import numpy as np

from typing import Any
from core.embedding.embedding_service import ChunkDocument
from core.vector_store.search_result import VectorSearchResult
from core.vector_store.protocol import VectorStoreProtocol

LOGGER = logging.getLogger(__name__)


class UpstashVectorStore(VectorStoreProtocol):
    """Upstash Vector HTTP-based vector store driver.

    Handles the 4 critical mismatches between local Redis Stack and
    Upstash's REST API:

    1. Flat HASH → nested JSON metadata
    2. Float32 bytes → Python float list
    3. RediSearch tag syntax → SQL-like filter expressions
    4. Cosine distance (0=perfect, ASC) → Cosine similarity (1=perfect, DESC)
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        index_name: str = "rag_chunks_idx",
    ) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self.index_name = index_name
        self._client = httpx.Client(
            base_url=self._url, headers=self._headers(), timeout=30
        )

    # ---- VectorStoreProtocol --------------------------------------------------

    def ensure_index(self, vector_dim: int) -> None:
        """Upstash Vector index is auto-created on first upsert.

        We issue a lightweight info request to confirm the endpoint is
        reachable.  Index creation is deferred to the first upsert.
        """
        resp = self._client.get("/info", timeout=10)
        if resp.status_code == 200:
            LOGGER.info(
                "Upstash index '%s' reachable (dim=%s)", self.index_name, vector_dim
            )
        elif resp.status_code == 404:
            LOGGER.info(
                "Upstash index '%s' will be created on first upsert (dim=%s)",
                self.index_name,
                vector_dim,
            )
        else:
            LOGGER.warning("Upstash info returned %s: %s", resp.status_code, resp.text)

    def search_similar(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 20,
        filter_entities: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        vector_list = np.asarray(query_vector, dtype=np.float32).tolist()
        payload: dict[str, Any] = {
            "vector": vector_list,
            "topK": top_k,
            "includeMetadata": True,
            "includeVectors": False,
        }
        filter_str = self._build_upstash_filter(filter_entities)
        if filter_str:
            payload["filter"] = filter_str

        resp = self._client.post("/query", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return self._parse_upstash_results(data)

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 20,
        filter_entities: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        # Upstash Vector natively supports dense-only search.
        # Pass through to search_similar with optional filter.
        return self.search_similar(
            query_vector, top_k=top_k, filter_entities=filter_entities
        )

    # ---- ingest helper --------------------------------------------------------

    def upsert_documents(
        self,
        documents: list[ChunkDocument],
        vectors: np.ndarray,
        *,
        batch_size: int = 1000,
    ) -> int:
        """Upsert documents into Upstash Vector.

        Each document is converted to Upstash's expected shape:
        ``{"id": str, "vector": float[], "metadata": {...}}``.
        """

        if len(documents) != len(vectors):
            raise ValueError("documents and vectors length mismatch")
        if not documents:
            return 0

        written = 0
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            batch_vecs = vectors[start : start + batch_size]
            items: list[dict[str, Any]] = []
            for doc, vec in zip(batch_docs, batch_vecs):
                items.append(
                    {
                        "id": f"{doc.chunk_id}",
                        "vector": np.asarray(vec, dtype=np.float32).tolist(),
                        "metadata": {
                            "chunk_id": doc.chunk_id,
                            "page_id": doc.page_id,
                            "page_title": doc.page_title,
                            "section": doc.section or "",
                            "subsection": doc.subsection or "",
                            "source_file": doc.source_file,
                            "text": doc.text,
                        },
                    }
                )
            resp = self._client.post("/upsert", json=items, timeout=60)
            resp.raise_for_status()
            written += len(items)
            LOGGER.debug("Upserted %d vectors to Upstash", len(items))

        return written

    # ---- internals ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _build_upstash_filter(self, entities: list[str] | None) -> str | None:
        """Build an Upstash SQL-like filter expression from entity tokens.

        NOTE: Upstash Vector only supports SQL ``IN`` (exact match) on
        metadata fields, *not* full‑text containment like RediSearch's
        ``@text:()`` syntax.  Since the ``text`` metadata field contains
        full paragraphs (not short tags), using ``text IN (...)`` here
        would **always return zero results** — a paragraph never equals
        an entity token like ``"Edgar"`` exactly.

        To avoid silently killing all search results the entity pre‑filter
        is disabled for Upstash.  The dense vector similarity search alone
        provides adequate relevance.
        """
        return None

    def _parse_upstash_results(self, data: dict[str, Any]) -> list[VectorSearchResult]:
        """Parse Upstash query response into uniform ``VectorSearchResult``.

        Handles score inversion: Upstash returns cosine similarity
        (1 = perfect, 0 = worst, DESC) while the pipeline expects
        cosine distance (0 = perfect, higher = worse, ASC).
        We normalise: ``score = 1.0 - similarity``
        """
        results: list[VectorSearchResult] = []
        for item in data.get("result", []):
            metadata = item.get("metadata", {}) or {}
            upstash_score = float(item.get("score", 0.0))
            # Invert: similarity → distance (0 = perfect match)
            normalized_score = 1.0 - upstash_score

            results.append(
                VectorSearchResult(
                    key=str(item.get("id", "")),
                    score=max(0.0, normalized_score),  # clamp at 0
                    text=str(metadata.get("text", "")),
                    metadata={
                        "chunk_id": str(metadata.get("chunk_id", "")),
                        "page_id": str(metadata.get("page_id", "")),
                        "page_title": str(metadata.get("page_title", "")),
                        "section": str(metadata.get("section", "")) or None,
                        "subsection": str(metadata.get("subsection", "")) or None,
                        "source_file": str(metadata.get("source_file", "")) or None,
                    },
                )
            )
        return results
