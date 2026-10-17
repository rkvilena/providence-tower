from __future__ import annotations

import logging
import time

from core.env import settings
from core.embedding.embedding_service import EmbeddingService
from core.rag.reranker import CrossEncoderReranker
from core.rag.schema import (
    ChunkHit,
    FetchTimingTotals,
    FetcherState,
    QueryFetchTiming,
    RagState,
)
from core.vector_store.protocol import VectorStoreProtocol

LOGGER = logging.getLogger(__name__)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class FetcherNode:
    def __init__(
        self,
        vector_store: VectorStoreProtocol,
    ) -> None:
        self.store = vector_store
        self.embedder = EmbeddingService()
        self.reranker = (
            CrossEncoderReranker(
                model_name=settings.RERANK_MODEL,
                min_top_score=settings.RERANK_MIN_TOP_SCORE,
                batch_size=settings.RERANK_BATCH_SIZE,
            )
            if settings.RERANK_ENABLED
            else None
        )

    def run(self, state: RagState) -> RagState:
        state.add_trace("Fetcher node started.")

        queries = [
            q.strip() for q in state.planner_state.planned_queries if str(q).strip()
        ]
        if not queries:
            state.add_trace("No planned queries found for fetching.")
            return state

        entities = self._normalize_entities(state.planner_state.entities)

        all_hits: dict[str, ChunkHit] = {}
        rank_by_chunk_id: dict[str, int] = {}
        query_timings: list[QueryFetchTiming] = []
        hits_by_query: list[tuple[str, list[ChunkHit]]] = []

        # Phase 1: embed + search per query (no reranking yet).
        for query in queries:
            state.add_trace(f"Fetching for query: '{query}'")
            timing = QueryFetchTiming(query=query)
            try:
                embed_start = time.perf_counter()
                query_vector = self.embedder.embed_query(query)
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
                hits_by_query.append((query, query_hits))
            except Exception as e:
                LOGGER.error("Failed to fetch for query '%s': %s", query, e)
                state.add_trace(f"Error fetching for query '{query}': {str(e)}")
            query_timings.append(timing)

        # Phase 2: rerank all query/chunk pairs in one shared model call.
        rerank_ms = 0.0
        rerank_pairs = 0
        rerank_calls = 0
        merge_ms = 0.0
        if self.reranker and any(hits for _, hits in hits_by_query):
            rerank_start = time.perf_counter()
            rerank_results = self.reranker.rerank_batch(hits_by_query)
            rerank_ms = _ms(rerank_start)
            rerank_calls = 1
            rerank_pairs = sum(len(hits) for _, hits in hits_by_query)

            merge_start = time.perf_counter()
            for (query, query_hits), rerank_result in zip(
                hits_by_query, rerank_results
            ):
                state.add_trace(
                    f"Reranker top_score={rerank_result.top_score:.4f} threshold={settings.RERANK_MIN_TOP_SCORE:.2f} "
                    f"passed={rerank_result.passed_threshold}"
                )
                ordered_hits = (
                    rerank_result.hits if rerank_result.passed_threshold else query_hits
                )
                self._merge_hits(ordered_hits, all_hits, rank_by_chunk_id)
            merge_ms = _ms(merge_start)

        # Phase 3: merge results (vector-search order when reranking is unavailable).
        if not self.reranker:
            merge_start = time.perf_counter()
            for _, query_hits in hits_by_query:
                self._merge_hits(query_hits, all_hits, rank_by_chunk_id)
            merge_ms = _ms(merge_start)

        sort_start = time.perf_counter()
        sorted_hits = sorted(
            all_hits.values(),
            key=lambda hit: (rank_by_chunk_id.get(hit.chunk_id, 10**6), hit.score),
        )
        final_sort_ms = _ms(sort_start)

        totals = self._summarize_timings(
            query_timings,
            final_sort_ms=final_sort_ms,
            unique_chunks=len(sorted_hits),
            rerank_ms=rerank_ms,
            merge_ms=merge_ms,
            rerank_pair_count=rerank_pairs,
            rerank_model_calls=rerank_calls,
        )
        state.fetcher_state = FetcherState(
            chunks=sorted_hits, query_timings=query_timings, totals=totals
        )

        state.add_trace(
            f"Fetcher node completed. Total unique chunks retrieved: {len(sorted_hits)} | "
            f"embed={totals.embedding_ms:.1f}ms search={totals.vector_search_ms:.1f}ms "
            f"convert={totals.hit_conversion_ms:.1f}ms rerank={totals.rerank_ms:.1f}ms "
            f"(pairs={totals.rerank_pair_count} calls={totals.rerank_model_calls}) "
            f"merge={totals.merge_ms:.1f}ms sort={totals.final_sort_ms:.1f}ms"
        )
        return state

    def _merge_hits(
        self,
        ordered_hits: list[ChunkHit],
        all_hits: dict[str, ChunkHit],
        rank_by_chunk_id: dict[str, int],
    ) -> None:
        for rank, hit in enumerate(ordered_hits):
            chunk_id = hit.chunk_id
            current_rank = rank_by_chunk_id.get(chunk_id)
            if current_rank is None or rank < current_rank:
                rank_by_chunk_id[chunk_id] = rank
                all_hits[chunk_id] = hit
            elif rank == current_rank and hit.score < all_hits[chunk_id].score:
                all_hits[chunk_id] = hit

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
        for res in results:
            chunk_id = str(res.metadata.get("chunk_id") or res.key)
            hits.append(
                ChunkHit(
                    chunk_id=chunk_id,
                    page_id=str(res.metadata.get("page_id", "")),
                    page_title=str(res.metadata.get("page_title", "")),
                    score=res.score,
                    text=res.text,
                    section=res.metadata.get("section"),
                    subsection=res.metadata.get("subsection"),
                )
            )
        return hits

    def _normalize_entities(self, entities: list[str]) -> list[str]:
        stop = {"ys", "game", "games", "series"}
        cleaned: list[str] = []
        seen: set[str] = set()
        for e in entities or []:
            text = str(e).strip()
            if not text:
                continue
            low = text.lower()
            if low in stop:
                continue
            if len(low) < 3:
                continue
            if low in seen:
                continue
            seen.add(low)
            cleaned.append(text)
        return cleaned[:8]
