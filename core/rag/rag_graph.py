from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from core.rag.planner import PlannerNode
from core.rag.fetcher import FetcherNode
from core.rag.thinker import ThinkerNode
from core.rag.context import ContextNode
from core.rag.schema import ChunkHit, RagState, GraphState
from core.vector_store.protocol import VectorStoreProtocol


class RagGraph:
    def __init__(self, vector_store: VectorStoreProtocol) -> None:
        self.planner = PlannerNode()
        self.fetcher = FetcherNode(vector_store=vector_store)
        self.thinker = ThinkerNode()
        self.context = ContextNode()
        self.graph = self._build()

    def run(self, state: RagState) -> RagState:
        result = self.graph.invoke({"state": state})
        return result["state"]

    def warmup(self) -> None:
        _ = self.fetcher.embedder.embed_query("warmup")
        if self.fetcher.reranker is not None:
            _ = self.fetcher.reranker.rerank(
                "warmup",
                [
                    ChunkHit(
                        chunk_id="warmup",
                        page_id="0",
                        page_title="warmup",
                        score=0.0,
                        text="warmup",
                        section=None,
                        subsection=None,
                    )
                ],
            )

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("fetcher", self._fetcher_node)
        graph.add_node("thinker", self._thinker_node)
        graph.add_node("context", self._context_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "fetcher")
        graph.add_edge("fetcher", "thinker")
        graph.add_edge("thinker", "context")
        graph.add_edge("context", END)

        return graph.compile()

    def _planner_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        start = time.perf_counter()
        updated = self.planner.run(state)
        updated.node_latencies_ms["planner"] = round(
            (time.perf_counter() - start) * 1000, 2
        )
        return {"state": updated}

    def _fetcher_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        start = time.perf_counter()
        updated = self.fetcher.run(state)
        updated.node_latencies_ms["fetcher"] = round(
            (time.perf_counter() - start) * 1000, 2
        )
        return {"state": updated}

    def _thinker_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        start = time.perf_counter()
        updated = self.thinker.run(state)
        updated.node_latencies_ms["thinker"] = round(
            (time.perf_counter() - start) * 1000, 2
        )
        return {"state": updated}

    def _context_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        start = time.perf_counter()
        updated = self.context.run(state)
        updated.node_latencies_ms["context"] = round(
            (time.perf_counter() - start) * 1000, 2
        )
        return {"state": updated}
