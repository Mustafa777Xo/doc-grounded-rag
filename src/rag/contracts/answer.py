from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    source_file: str
    page: int  # 0-indexed, same conventions as Chunk.page
    chunk_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class AnswerWithCitations:
    answer_text: str
    citations: tuple[Citation, ...]
    grounded: bool  # False if context was insufficient

    def to_dict(self) -> dict[str, object]:
        return {
            "answer_text": self.answer_text,
            "citations": [c.to_dict() for c in self.citations],
            "grounded": self.grounded,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
