from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from core.rag.planner import PlannerNode
from core.rag.schema import RagState


class GraphState(TypedDict):
    state: RagState


class RagGraph:
    def __init__(self, *, phase: str) -> None:
        if phase != "planner":
            raise ValueError(f"Unsupported phase: {phase}")
        self.phase = phase
        self.planner = PlannerNode()
        self.graph = self._build()

    def run(self, state: RagState) -> RagState:
        result = self.graph.invoke({"state": state})
        return result["state"]

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", END)
        return graph.compile()

    def _planner_node(self, graph_state: GraphState) -> GraphState:
        state = graph_state["state"]
        start = time.perf_counter()
        updated = self.planner.run(state)
        elapsed = (time.perf_counter() - start) * 1000
        updated.node_latencies_ms["planner"] = round(elapsed, 2)
        return {"state": updated}
