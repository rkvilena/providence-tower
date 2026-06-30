from __future__ import annotations

import time
from typing import Callable

from langgraph.graph import END, START, StateGraph

from core.rag.planner import PlannerNode
from core.rag.fetcher import FetcherNode
from core.rag.thinker import ThinkerNode
from core.rag.context import ContextNode
from core.rag.schema import RagState, GraphState
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

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node(
            "planner", lambda gs: self._timed_node(gs, self.planner, "planner")
        )
        graph.add_node(
            "fetcher", lambda gs: self._timed_node(gs, self.fetcher, "fetcher")
        )
        graph.add_node(
            "thinker", lambda gs: self._timed_node(gs, self.thinker, "thinker")
        )
        graph.add_node(
            "context", lambda gs: self._timed_node(gs, self.context, "context")
        )

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "fetcher")
        graph.add_edge("fetcher", "thinker")
        graph.add_edge("thinker", "context")
        graph.add_edge("context", END)

        return graph.compile()

    def _timed_node(
        self, graph_state: GraphState, node: Callable[[RagState], RagState], name: str
    ) -> GraphState:
        state = graph_state["state"]
        start = time.perf_counter()
        updated = node.run(state)
        updated.node_latencies_ms[name] = round((time.perf_counter() - start) * 1000, 2)
        return {"state": updated}
