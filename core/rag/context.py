from __future__ import annotations

from core.rag.schema import RagState


class ContextNode:
    """No-op node — history persistence is handled by RagService."""

    def run(self, state: RagState) -> RagState:
        return state
