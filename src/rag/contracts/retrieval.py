from __future__ import annotations

import json
from dataclasses import dataclass

from rag.contracts.chunk import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    retrieval_method: str  # eg. "semantic", "keyword", "hybrid"

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "retrieval_method": self.retrieval_method,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
