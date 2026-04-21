from __future__ import annotations

import logging

from core.embedding.embedding_service import EmbeddingService
from core.embedding.redis_store import RedisVectorStore
from core.rag.schema import ChunkHit, FetcherState, RagState

LOGGER = logging.getLogger(__name__)


class FetcherNode:
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
        index_name: str = "rag_chunks_idx",
    ) -> None:
        self.embedder = EmbeddingService(model_name=model_name)
        self.store = RedisVectorStore(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            index_name=index_name,
        )

    def run(self, state: RagState) -> RagState:
        state.add_trace("Fetcher node started.")

        queries = [q.strip() for q in state.planner_state.planned_queries if str(q).strip()]
        if not queries:
            state.add_trace("No planned queries found for fetching.")
            return state

        all_hits: dict[str, ChunkHit] = {}

        for query in queries:
            state.add_trace(f"Fetching for query: '{query}'")
            try:
                query_vector = self.embedder.embed_query(query)
                results = self.store.search_similar(query_vector)

                for res in results:
                    chunk_id = res.metadata.get("chunk_id") or res.key

                    hit = ChunkHit(
                        chunk_id=str(chunk_id),
                        page_id=str(res.metadata.get("page_id", "")),
                        page_title=str(res.metadata.get("page_title", "")),
                        score=res.score,
                        text=res.text,
                        section=res.metadata.get("section"),
                        subsection=res.metadata.get("subsection"),
                    )

                    if chunk_id not in all_hits or res.score < all_hits[chunk_id].score:
                        all_hits[chunk_id] = hit
            except Exception as e:
                LOGGER.error("Failed to fetch for query '%s': %s", query, e)
                state.add_trace(f"Error fetching for query '{query}': {str(e)}")

        # Sort combined results by score ascending (best relevance first)
        sorted_hits = sorted(all_hits.values(), key=lambda x: x.score)
        
        state.fetcher_state = FetcherState(chunks=sorted_hits)
        
        state.add_trace(f"Fetcher node completed. Total unique chunks retrieved: {len(sorted_hits)}")
        return state
