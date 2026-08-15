from __future__ import annotations

from typing import Protocol, Sequence

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import (
    RetrievalCandidate,
    RetrievalQuery,
    RetrieverSource,
    ScoreProvenance,
)


class Retriever(Protocol):
    def retrieve(
        self, query: RetrievalQuery, limit: int = 5
    ) -> tuple[RetrievalCandidate, ...]: ...


class NoOpRetriever:
    def __init__(self, corpus: Sequence[Chunk]) -> None:
        self._corpus = tuple(sorted(corpus, key=lambda c: (c.doc_id, c.chunk_index)))

    def retrieve(
        self, query: RetrievalQuery, limit: int = 5
    ) -> tuple[RetrievalCandidate, ...]:
        _ = query
        selected = self._corpus[:limit]
        return tuple(
            RetrievalCandidate(
                chunk=chunk,
                scores=ScoreProvenance(dense_score=1.0 - (idx * 0.01)),
                sources=frozenset({RetrieverSource.DENSE}),
            )
            for idx, chunk in enumerate(selected)
        )
