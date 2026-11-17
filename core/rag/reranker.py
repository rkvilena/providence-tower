"""RAG reranking through Cloudflare Workers AI.

The local cross-encoder dependency is intentionally gone: reranking runs
against `@cf/baai/bge-reranker-base` at request time. Corpus preparation and
query embedding are handled elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.inference.cloudflare_workers_ai import (
    CloudflareWorkersAIClient,
    RerankScore,
)
from core.rag.schema import ChunkHit

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    """Ranked hits for one query with failure details for tracing."""

    hits: list[ChunkHit]
    top_score: float = 0.0
    failed: bool = False
    error: str = ""


class CloudflareReranker:
    """Rerank retrieval candidates with the Cloudflare BGE reranker."""

    def __init__(
        self,
        *,
        client: CloudflareWorkersAIClient | None = None,
    ) -> None:
        self.client = client or CloudflareWorkersAIClient()

    def rerank(self, query: str, hits: list[ChunkHit]) -> RerankResult:
        [result] = self.rerank_batch([(query, hits)])
        return result

    def rerank_batch(
        self, query_hits: list[tuple[str, list[ChunkHit]]]
    ) -> list[RerankResult]:
        """Rerank each non-empty query in its own remote request.

        Results are aligned with ``query_hits``. A failing query keeps its
        original vector-search order so callers can merge it safely.
        """
        return [self._rerank_one(query, hits) for query, hits in query_hits]

    def _rerank_one(self, query: str, hits: list[ChunkHit]) -> RerankResult:
        if not hits:
            return RerankResult(hits=[])

        try:
            scores = self.client.rerank(query, [hit.text for hit in hits])
        except Exception as exc:  # noqa: BLE001 - preserve retrieval on any failure
            LOGGER.warning("Cloudflare rerank failed: %s", exc)
            return RerankResult(
                hits=list(hits),
                failed=True,
                error=str(exc),
            )

        return RerankResult(
            hits=self._order_hits(hits, scores),
            top_score=max((score.score for score in scores), default=0.0),
        )

    @staticmethod
    def _order_hits(hits: list[ChunkHit], scores: list[RerankScore]) -> list[ChunkHit]:
        ranked = sorted(scores, key=lambda score: score.score, reverse=True)
        scored_indices = {score.index for score in ranked}
        return [hits[score.index] for score in ranked] + [
            hit for index, hit in enumerate(hits) if index not in scored_indices
        ]
