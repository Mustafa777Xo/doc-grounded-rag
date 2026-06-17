from __future__ import annotations

from typing import Protocol, Sequence

from rag.contracts.chunk import Chunk


class Indexer(Protocol):
    def index(self, chunks: Sequence[Chunk]) -> None: ...


class NoOpIndexer:
    def index(self, chunks: Sequence[Chunk]) -> None:
        _ = chunks
