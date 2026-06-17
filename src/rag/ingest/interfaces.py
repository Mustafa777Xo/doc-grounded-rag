from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from rag.contracts.document import Document


class Ingestor(Protocol):
    def ingest(self, sources: Sequence[Path]) -> tuple[Document, ...]: ...


class NoOpIngestor:
    def ingest(self, sources: Sequence[Path]) -> tuple[Document, ...]:
        ordered = tuple(sorted(sources, key=lambda p: str(p)))
        return tuple(
            Document(
                doc_id=f"doc-{i}",
                source_file=path.name,
                pages=(f"noop page text from {path.name}",),
            )
            for i, path in enumerate(ordered)
        )
