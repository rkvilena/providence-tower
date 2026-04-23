from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from core.rag.planner import PlannerNode
from core.rag.fetcher import FetcherNode
from core.rag.thinker import ThinkerNode
from core.rag.schema import RagState


class GraphState(TypedDict):
    state: RagState


class RagGraph:
    def __init__(self, *, phase: str) -> None:
        if phase not in ["planner", "fetcher", "thinker"]:
            raise ValueError(f"Unsupported phase: {phase}")
        self.phase = phase
        self.planner = PlannerNode() if phase == "planner" else None
        self.fetcher = FetcherNode() if phase == "fetcher" else None
        self.thinker = ThinkerNode() if phase == "thinker" else None
        self.graph = self._build()

    def run(self, state: RagState) -> RagState:
        start = time.perf_counter()
        result = self.graph.invoke({"state": state})
        updated = result["state"]
        elapsed = (time.perf_counter() - start) * 1000
        updated.node_latencies_ms[self.phase] = round(elapsed, 2)
        return updated

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("fetcher", self._fetcher_node)
        graph.add_node("thinker", self._thinker_node)

        if self.phase == "planner":
            graph.add_edge(START, "planner")
            graph.add_edge("planner", END)
        elif self.phase == "fetcher":
            graph.add_edge(START, "fetcher")
            graph.add_edge("fetcher", END)
        elif self.phase == "thinker":
            graph.add_edge(START, "thinker")
            graph.add_edge("thinker", END)

        return graph.compile()

    def _planner_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        if self.planner is None:
            raise RuntimeError("Planner node not initialized for current graph phase")
        updated = self.planner.run(state)
        return {"state": updated}

    def _fetcher_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        if self.fetcher is None:
            raise RuntimeError("Fetcher node not initialized for current graph phase")
        updated = self.fetcher.run(state)
        return {"state": updated}

    def _thinker_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        if self.thinker is None:
            raise RuntimeError("Thinker node not initialized for current graph phase")
        updated = self.thinker.run(state)
        return {"state": updated}
