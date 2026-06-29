from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.vector_store.search_result import VectorSearchResult


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Structural subtyping interface for vector store drivers.

    Every concrete provider (local Redis Stack, Upstash Vector, …) must
    satisfy this contract so the RAG pipeline remains completely engine-
    agnostic.
    """

    def ensure_index(self, vector_dim: int) -> None:
        """Verify / build the vector index for the given dimension."""
        ...

    def search_similar(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 20,
        filter_entities: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Dense vector similarity search.

        Parameters
        ----------
        query_vector:
            Query embedding (normalised float32 NumPy array).
        top_k:
            Maximum number of results to return.
        filter_entities:
            Optional list of entity tokens to filter on.  Each driver
            translates these into its own native filter syntax.

        Returns
        -------
        A list of ``VectorSearchResult`` sorted by score (ascending
        distance — 0 = perfect match).
        """
        ...

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 20,
        filter_entities: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        """Hybrid dense / keyword search.

        The default implementation forwards to ``search_similar``, but
        specialised drivers may override this to combine keyword and
        vector signals.
        """
        ...
