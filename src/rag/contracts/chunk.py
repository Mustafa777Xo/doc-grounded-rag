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
    char_start: int
    char_end: int
    text: str

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.doc_id:
            raise ValueError("doc_id cannot be empty")
        if not self.source_file:
            raise ValueError("source_file cannot be empty")
        if self.page < 0:
            raise ValueError("page cannot be negative")
        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")
        if self.char_start < 0:
            raise ValueError("char_start cannot be negative")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if not self.text:
            raise ValueError("text cannot be empty")
        if len(self.text) != self.char_end - self.char_start:
            raise ValueError("text length must match half-open character span")

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
