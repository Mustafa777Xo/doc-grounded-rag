from __future__ import annotations

from typing import Literal

from rag.logging import RagError

RetrievalStage = Literal["normalization", "dense", "keyword", "fusion"]


class RetrievalError(RagError):
    """Failure in a retrieval stage for one typed query."""

    def __init__(
        self,
        *,
        stage: RetrievalStage,
        query_id: str,
        message: str,
        hint: str | None = None,
    ) -> None:
        self.stage = stage
        self.query_id = query_id
        self.stage_message = message
        self.hint = hint
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        message = (
            f"Retrieval failed during {self.stage} "
            f"(stage={self.stage}, query_id={self.query_id}): {self.stage_message}"
        )
        if self.hint is None:
            return message
        return f"{message}. hint={self.hint}"


class RerankingError(RagError):
    """Failure while reranking candidates for one typed query."""

    stage: Literal["reranking"] = "reranking"

    def __init__(
        self,
        *,
        query_id: str,
        message: str,
        hint: str | None = None,
    ) -> None:
        self.query_id = query_id
        self.stage_message = message
        self.hint = hint
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        message = (
            "Reranking failed "
            f"(stage={self.stage}, query_id={self.query_id}): {self.stage_message}"
        )
        if self.hint is None:
            return message
        return f"{message}. hint={self.hint}"
