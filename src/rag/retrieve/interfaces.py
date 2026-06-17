from __future__ import annotations

from typing import Protocol, Sequence

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import RetrievalResult


class Retriever(Protocol):
    def retrieve(self, query: str, limit: int = 5) -> tuple[RetrievalResult, ...]: ...


class NoOpRetriever:
    def __init__(self, corpus: Sequence[Chunk]) -> None:
        self._corpus = tuple(sorted(corpus, key=lambda c: (c.doc_id, c.chunk_index)))

    def retrieve(self, query: str, limit: int = 5) -> tuple[RetrievalResult, ...]:
        selected = self._corpus[:limit]
        return tuple(
            RetrievalResult(
                chunk=chunk,
                score=1.0 - (idx * 0.01),
                retrieval_method="noop",
            )
            for idx, chunk in enumerate(selected)
        )
