from __future__ import annotations

from typing import Protocol, Sequence

from rag.contracts.retrieval import RetrievalResult


class Reranker(Protocol):
    def rerank(
        self,
        results: Sequence[RetrievalResult],
        limit: int | None = None,
    ) -> tuple[RetrievalResult, ...]: ...


class NoOpReranker:
    def rerank(
        self,
        results: Sequence[RetrievalResult],
        limit: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        ordered = tuple(
            sorted(
                results,
                key=lambda item: (
                    -item.score,
                    item.chunk.doc_id,
                    item.chunk.chunk_index,
                ),
            )
        )
        if limit is None:
            return ordered
        return ordered[:limit]
