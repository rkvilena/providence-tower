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
        batch_size: int = 32,
    ) -> None:
        self.model = CrossEncoder(model_name)
        self.min_top_score = min_top_score
        self.batch_size = batch_size

    def rerank(self, query: str, hits: list[ChunkHit]) -> RerankResult:
        [result] = self.rerank_batch([(query, hits)])
        return result

    def rerank_batch(
        self, query_hits: list[tuple[str, list[ChunkHit]]]
    ) -> list[RerankResult]:
        """Score all query/chunk pairs in one model call, then rank per query.

        The returned list is aligned with ``query_hits``. Each result is ranked
        and thresholded independently per query; only the model invocation is
        shared.
        """
        results: list[RerankResult | None] = [None] * len(query_hits)
        candidates: list[tuple[int, ChunkHit]] = []  # (result_index, hit)

        for index, (_, hits) in enumerate(query_hits):
            if not hits:
                results[index] = RerankResult(hits=[], top_score=0.0, passed_threshold=False)
                continue
            candidates.extend((index, hit) for hit in hits)

        if candidates:
            pairs = [
                (query_hits[index][0], hit.text)
                for index, hit in candidates
            ]
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            ).tolist()

            scored_by_index: dict[int, list[tuple[ChunkHit, float]]] = {}
            for (index, hit), score in zip(candidates, scores):
                scored_by_index.setdefault(index, []).append((hit, float(score)))

            for index, group in scored_by_index.items():
                scored_hits = sorted(group, key=lambda item: item[1], reverse=True)
                top_score = float(scored_hits[0][1])
                results[index] = RerankResult(
                    hits=[hit for hit, _ in scored_hits],
                    top_score=top_score,
                    passed_threshold=top_score > self.min_top_score,
                )

        return [result for result in results if result is not None]
