from __future__ import annotations

from dataclasses import replace
from typing import Protocol, Sequence

from rag.contracts.retrieval import RetrievalCandidate, RetrievalQuery


class Reranker(Protocol):
    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]: ...


class NoOpReranker:
    def rerank(
        self,
        query: RetrievalQuery,
        candidates: Sequence[RetrievalCandidate],
        limit: int | None = None,
    ) -> tuple[RetrievalCandidate, ...]:
        _ = query
        selected = tuple(candidates) if limit is None else tuple(candidates[:limit])
        return tuple(
            replace(candidate, rank=rank)
            for rank, candidate in enumerate(selected, start=1)
        )
