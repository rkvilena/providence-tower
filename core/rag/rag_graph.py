from __future__ import annotations

import time
from typing import Callable, Iterator

from langgraph.graph import END, START, StateGraph
from langgraph.config import get_config

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

    def stream(
        self, state: RagState, *, on_delta=None
    ) -> Iterator[tuple[str, RagState]]:
        """Yield the current state after each completed graph node.

        ``on_delta`` is forwarded to the Thinker node so answer-text deltas
        can be streamed to the caller while the graph runs.
        """
        current = state
        for update in self.graph.stream(
            {"state": state},
            stream_mode="updates",
            config={"configurable": {"on_delta": on_delta}},
        ):
            for node_name, node_update in update.items():
                candidate = (
                    node_update.get("state") if isinstance(node_update, dict) else None
                )
                if candidate is not None:
                    current = candidate
                yield node_name, current

    def _build(self):
        graph = StateGraph(GraphState)
        nodes = (
            ("planner", self.planner),
            ("fetcher", self.fetcher),
            ("thinker", self.thinker),
            ("context", self.context),
        )
        for name, node in nodes:
            graph.add_node(
                name,
                # Defaults bind the loop variables at definition time.
                lambda graph_state, node=node, name=name: self._timed_node(
                    graph_state, node, name
                ),
            )

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "fetcher")
        graph.add_edge("fetcher", "thinker")
        graph.add_edge("thinker", "context")
        graph.add_edge("context", END)

        return graph.compile()

    def _timed_node(
        self, graph_state: GraphState, node: Callable[..., RagState], name: str
    ) -> GraphState:
        state = graph_state["state"]
        start = time.perf_counter()
        on_delta = None
        config = get_config()
        if config is not None and config.get("configurable"):
            on_delta = config["configurable"].get("on_delta")
        if name == "thinker" and on_delta is not None:
            updated = node.run(state, on_delta=on_delta)
        else:
            updated = node.run(state)
        updated.node_latencies_ms[name] = round((time.perf_counter() - start) * 1000, 2)
        return {"state": updated}
