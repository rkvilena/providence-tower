from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from core.rag.schema import ChunkHit


@dataclass(frozen=True)
class RerankResult:
    hits: list[ChunkHit]
    top_score: float
    passed_threshold: bool


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_top_score: float = 0.7,
    ) -> None:
        self.model = CrossEncoder(model_name)
        self.min_top_score = min_top_score

    def rerank(self, query: str, hits: list[ChunkHit]) -> RerankResult:
        if not hits:
            return RerankResult(hits=[], top_score=0.0, passed_threshold=False)

        pairs = [(query, hit.text) for hit in hits]
        scores = self.model.predict(pairs).tolist()

        scored_hits = sorted(zip(hits, scores), key=lambda item: item[1], reverse=True)
        top_score = float(scored_hits[0][1]) if scored_hits else 0.0

        return RerankResult(
            hits=[hit for hit, _ in scored_hits],
            top_score=top_score,
            passed_threshold=top_score > self.min_top_score,
        )
