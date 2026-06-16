from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int  # start with 0
    chunk_index: int  # position within the document, 0-indexed
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
