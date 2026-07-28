from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rag.config import Settings
from rag.contracts.indexing import EmbeddingRecord
from rag.embed.service import EmbeddingRequest, EmbeddingService


@dataclass(frozen=True)
class EmbeddingPreparationPolicy:
    max_chars: int = 4096
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

    @classmethod
    def from_settings(cls, settings: Settings) -> EmbeddingPreparationPolicy:
        return cls(
            max_chars=settings.embedding_max_chars,
            batch_size=settings.embedding_batch_size,
        )


@dataclass(frozen=True)
class PreparedEmbeddingRequest:
    chunk_id: str
    request: EmbeddingRequest
    original_text_length: int
    prepared_text_length: int
    truncated: bool
    reason: str | None = None


class EmbeddingTextPreparer:
    def __init__(self, policy: EmbeddingPreparationPolicy | None = None) -> None:
        self._policy = policy if policy is not None else EmbeddingPreparationPolicy()

    def prepare(
        self,
        *,
        chunk_id: str,
        text: str,
        content_hash: str,
    ) -> PreparedEmbeddingRequest:
        original_length = len(text)
        prepared_text = text
        truncated = False
        reason: str | None = None

        if original_length > self._policy.max_chars:
            prepared_text = text[: self._policy.max_chars]
            truncated = True
            reason = (
                "text exceeded embedding max_chars "
                f"({original_length}>{self._policy.max_chars}); truncated"
            )

        request = EmbeddingRequest(
            chunk_id=chunk_id,
            text=prepared_text,
            content_hash=content_hash,
        )
        return PreparedEmbeddingRequest(
            chunk_id=chunk_id,
            request=request,
            original_text_length=original_length,
            prepared_text_length=len(prepared_text),
            truncated=truncated,
            reason=reason,
        )

    def prepare_many(
        self,
        items: Sequence[tuple[str, str, str]],
    ) -> tuple[PreparedEmbeddingRequest, ...]:
        return tuple(
            self.prepare(chunk_id=chunk_id, text=text, content_hash=content_hash)
            for chunk_id, text, content_hash in items
        )


class EmbeddingBatcher:
    def __init__(
        self,
        *,
        service: EmbeddingService,
        policy: EmbeddingPreparationPolicy | None = None,
    ) -> None:
        self._service = service
        self._policy = policy if policy is not None else EmbeddingPreparationPolicy()

    def embed(
        self,
        prepared_requests: Sequence[PreparedEmbeddingRequest],
    ) -> tuple[EmbeddingRecord, ...]:
        records: list[EmbeddingRecord] = []
        for start in range(0, len(prepared_requests), self._policy.batch_size):
            batch = prepared_requests[start : start + self._policy.batch_size]
            records.extend(
                self._service.embed_batch(tuple(item.request for item in batch))
            )
        return tuple(records)
