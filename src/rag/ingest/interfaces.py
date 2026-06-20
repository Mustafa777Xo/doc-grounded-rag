from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from rag.contracts.document import Document, ParsedPage


class Ingestor(Protocol):
    def ingest(self, sources: Sequence[Path]) -> tuple[Document, ...]: ...


class NoOpIngestor:
    def ingest(self, sources: Sequence[Path]) -> tuple[Document, ...]:
        ordered = tuple(sorted(sources, key=lambda p: str(p)))
        documents: list[Document] = []
        for i, path in enumerate(ordered):
            doc_id = f"doc-{i}"
            text = f"noop page text from {path.name}"
            documents.append(
                Document(
                    doc_id=doc_id,
                    source_file=path.name,
                    source_path=str(path),
                    total_pages=1,
                    pages=(
                        ParsedPage(
                            doc_id=doc_id,
                            source_file=path.name,
                            page_number=0,
                            text=text,
                        ),
                    ),
                )
            )
        return tuple(documents)
