from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VectorSearchResult:
    """Uniform domain model returned by every vector store driver.

    This is the single type that the RAG pipeline consumes — no
    database-specific result objects ever leak out of the driver classes.
    """

    key: str
    score: float
    text: str
    metadata: dict[str, Any]
