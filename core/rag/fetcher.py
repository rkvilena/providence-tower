from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from core.env import settings
from core.inference.cloudflare_workers_ai import (
    EMBEDDING_DIMENSIONS,
    CloudflareAPIError,
    CloudflareWorkersAIClient,
)
from core.rag.reranker import CloudflareReranker
from core.rag.schema import (
    ChunkHit,
    FetchTimingTotals,
    FetcherState,
    QueryFetchTiming,
    RagState,
)
from core.vector_store.protocol import VectorStoreProtocol

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankOutcome:
    """Timing/count stats for the rerank and merge phases."""

    rerank_ms: float = 0.0
    merge_ms: float = 0.0
    pair_count: int = 0
    model_calls: int = 0


NO_RERANK = RerankOutcome()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class FetcherNode:
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        *,
        cloudflare_client: CloudflareWorkersAIClient | None = None,
    ) -> None:
        self.store = vector_store
        # Only query embedding runs in the cloud; corpus preparation keeps
        # using the local BGE model.
        self.cloudflare = cloudflare_client or CloudflareWorkersAIClient()
        self.reranker = CloudflareReranker() if settings.RERANK_ENABLED else None

    def run(self, state: RagState) -> RagState:
        state.add_trace("Fetcher node started.")

        queries = [
            q.strip() for q in state.planner_state.planned_queries if str(q).strip()
        ]
        if not queries:
            state.add_trace("No planned queries found for fetching.")
            return state

        entities = self._normalize_entities(state.planner_state.entities)
        query_timings, hits_by_query = self._search_all_queries(
            state, queries, entities
        )

        # Preserve the pre-rerank candidate pool per query so benchmark runs
        # can tell whether comparisons used the same candidates.
        candidate_chunk_ids_by_query: dict[str, list[str]] = {}
        for query, query_hits in hits_by_query:
            candidate_chunk_ids_by_query.setdefault(query, []).extend(
                hit.chunk_id for hit in query_hits
            )

        # Best-ranked hit per chunk across all queries. Lower rank wins;
        # equal ranks fall back to the lower retrieval score.
        hits_by_chunk_id: dict[str, ChunkHit] = {}
        best_rank_by_chunk_id: dict[str, int] = {}

        outcome = (
            self._rerank_and_merge(
                state, hits_by_query, hits_by_chunk_id, best_rank_by_chunk_id
            )
            if self.reranker
            # Vector-search order merge when reranking is unavailable.
            else self._merge_search_order(
                hits_by_query, hits_by_chunk_id, best_rank_by_chunk_id
            )
        )

        sort_start = time.perf_counter()
        sorted_hits = sorted(
            hits_by_chunk_id.values(),
            key=lambda hit: (best_rank_by_chunk_id.get(hit.chunk_id, 10**6), hit.score),
        )
        final_sort_ms = _ms(sort_start)

        totals = self._summarize_timings(
            query_timings,
            final_sort_ms=final_sort_ms,
            unique_chunks=len(sorted_hits),
            rerank_ms=outcome.rerank_ms,
            merge_ms=outcome.merge_ms,
            rerank_pair_count=outcome.pair_count,
            rerank_model_calls=outcome.model_calls,
        )
        state.fetcher_state = FetcherState(
            chunks=sorted_hits,
            candidate_chunk_ids_by_query=candidate_chunk_ids_by_query,
            query_timings=query_timings,
            totals=totals,
        )

        state.add_trace(
            f"Fetcher node completed. Total unique chunks retrieved: {len(sorted_hits)} | "
            f"embed={totals.embedding_ms:.1f}ms search={totals.vector_search_ms:.1f}ms "
            f"convert={totals.hit_conversion_ms:.1f}ms rerank={totals.rerank_ms:.1f}ms "
            f"(pairs={totals.rerank_pair_count} calls={totals.rerank_model_calls}) "
            f"merge={totals.merge_ms:.1f}ms sort={totals.final_sort_ms:.1f}ms"
        )
        return state

    def _search_all_queries(
        self,
        state: RagState,
        queries: list[str],
        entities: list[str],
    ) -> tuple[list[QueryFetchTiming], list[tuple[str, list[ChunkHit]]]]:
        """Embed and search every planned query.

        A failing query is traced on the state and skipped; its (partial)
        timing is still recorded. Reranking is deliberately kept out of
        this loop so each query is reranked once after retrieval.
        """
        query_timings: list[QueryFetchTiming] = []
        hits_by_query: list[tuple[str, list[ChunkHit]]] = []

        for query in queries:
            state.add_trace(f"Fetching for query: '{query}'")
            timing = QueryFetchTiming(query=query)
            try:
                query_hits = self._search_query(query, entities, timing)
                hits_by_query.append((query, query_hits))
            except Exception as e:
                LOGGER.error("Failed to fetch for query '%s': %s", query, e)
                state.add_trace(f"Error fetching for query '{query}': {str(e)}")
            query_timings.append(timing)

        return query_timings, hits_by_query

    def _search_query(
        self, query: str, entities: list[str], timing: QueryFetchTiming
    ) -> list[ChunkHit]:
        """Embed one query, search the store, and convert the hits.

        Updates ``timing`` in place; raises on failure so the caller can
        record the error while keeping the partial timing.
        """
        embed_start = time.perf_counter()
        query_vector = self._embed_query(query)
        timing.embedding_ms = _ms(embed_start)

        search_start = time.perf_counter()
        results = self.store.search_hybrid(
            query_vector, top_k=settings.RERANK_TOP_K, filter_entities=entities
        )
        timing.vector_search_ms = _ms(search_start)

        convert_start = time.perf_counter()
        query_hits = self._to_hits(results)
        timing.hit_conversion_ms = _ms(convert_start)
        timing.result_count = len(query_hits)
        return query_hits

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed one query through Cloudflare and normalize the vector shape.

        Raises instead of falling back to local inference so mixed-provider
        drift cannot silently corrupt retrieval.
        """
        vector = np.asarray(self.cloudflare.embed_query(query), dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != EMBEDDING_DIMENSIONS:
            raise CloudflareAPIError(
                "Cloudflare query embedding returned an unexpected shape "
                f"{vector.shape}; expected ({EMBEDDING_DIMENSIONS},)."
            )
        if not np.all(np.isfinite(vector)):
            raise CloudflareAPIError(
                "Cloudflare query embedding returned non-finite values."
            )
        return vector

    def _rerank_and_merge(
        self,
        state: RagState,
        hits_by_query: list[tuple[str, list[ChunkHit]]],
        hits_by_chunk_id: dict[str, ChunkHit],
        best_rank_by_chunk_id: dict[str, int],
    ) -> RerankOutcome:
        """Rerank each query's candidates with Cloudflare, then merge."""
        if self.reranker is None or not any(hits for _, hits in hits_by_query):
            return NO_RERANK

        rerank_start = time.perf_counter()
        rerank_results = self.reranker.rerank_batch(hits_by_query)
        rerank_ms = _ms(rerank_start)

        merge_start = time.perf_counter()
        for (query, query_hits), rerank_result in zip(hits_by_query, rerank_results):
            if rerank_result.failed:
                state.add_trace(
                    f"Rerank failed for query '{query}': {rerank_result.error}. "
                    "Using vector-search order."
                )
            else:
                state.add_trace(
                    f"Reranked query '{query}' with top_score="
                    f"{rerank_result.top_score:.4f}"
                )
            self._merge_hits(
                rerank_result.hits, hits_by_chunk_id, best_rank_by_chunk_id
            )
        merge_ms = _ms(merge_start)

        return RerankOutcome(
            rerank_ms=rerank_ms,
            merge_ms=merge_ms,
            pair_count=sum(len(hits) for _, hits in hits_by_query),
            # One Cloudflare request per non-empty query.
            model_calls=sum(1 for _, hits in hits_by_query if hits),
        )

    def _merge_search_order(
        self,
        hits_by_query: list[tuple[str, list[ChunkHit]]],
        hits_by_chunk_id: dict[str, ChunkHit],
        best_rank_by_chunk_id: dict[str, int],
    ) -> RerankOutcome:
        """Merge hits in vector-search order when reranking is unavailable."""
        merge_start = time.perf_counter()
        for _, query_hits in hits_by_query:
            self._merge_hits(query_hits, hits_by_chunk_id, best_rank_by_chunk_id)
        return RerankOutcome(merge_ms=_ms(merge_start))

    def _merge_hits(
        self,
        ordered_hits: list[ChunkHit],
        hits_by_chunk_id: dict[str, ChunkHit],
        best_rank_by_chunk_id: dict[str, int],
    ) -> None:
        for rank, hit in enumerate(ordered_hits):
            chunk_id = hit.chunk_id
            current_rank = best_rank_by_chunk_id.get(chunk_id)
            if current_rank is None or rank < current_rank:
                best_rank_by_chunk_id[chunk_id] = rank
                hits_by_chunk_id[chunk_id] = hit
            elif rank == current_rank and hit.score < hits_by_chunk_id[chunk_id].score:
                hits_by_chunk_id[chunk_id] = hit

    def _summarize_timings(
        self,
        query_timings: list[QueryFetchTiming],
        *,
        final_sort_ms: float,
        unique_chunks: int,
        rerank_ms: float,
        merge_ms: float,
        rerank_pair_count: int,
        rerank_model_calls: int,
    ) -> FetchTimingTotals:
        return FetchTimingTotals(
            embedding_ms=round(sum(t.embedding_ms for t in query_timings), 2),
            vector_search_ms=round(sum(t.vector_search_ms for t in query_timings), 2),
            hit_conversion_ms=round(sum(t.hit_conversion_ms for t in query_timings), 2),
            rerank_ms=round(rerank_ms, 2),
            final_sort_ms=final_sort_ms,
            merge_ms=round(merge_ms, 2),
            query_count=len(query_timings),
            unique_chunks=unique_chunks,
            rerank_pair_count=rerank_pair_count,
            rerank_model_calls=rerank_model_calls,
        )

    def _to_hits(self, results) -> list[ChunkHit]:
        hits: list[ChunkHit] = []
        for result in results:
            metadata = result.metadata
            hits.append(
                ChunkHit(
                    chunk_id=str(metadata.get("chunk_id") or result.key),
                    page_id=str(metadata.get("page_id", "")),
                    page_title=str(metadata.get("page_title", "")),
                    score=result.score,
                    text=result.text,
                    section=metadata.get("section"),
                    subsection=metadata.get("subsection"),
                )
            )
        return hits

    def _normalize_entities(self, entities: list[str]) -> list[str]:
        """Strip, filter generic/short terms, dedupe, and cap at 8."""
        stop_terms = {"ys", "game", "games", "series"}
        cleaned: list[str] = []
        seen: set[str] = set()
        for entity in entities or []:
            text = str(entity).strip()
            if not text:
                continue
            key = text.lower()
            if key in stop_terms or len(key) < 3 or key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned[:8]
