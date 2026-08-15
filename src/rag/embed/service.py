from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rag.contracts.indexing import EMBEDDING_SCHEMA_VERSION, EmbeddingRecord
from rag.embed.model_client import EmbeddingBatchError, EmbeddingProvider


class EmbeddingServiceError(RuntimeError):
    def __init__(self, *, chunk_id: str, message: str) -> None:
        self.chunk_id = chunk_id
        self.message = message
        super().__init__(f"Embedding failed for chunk_id={chunk_id!r}: {message}")


@dataclass(frozen=True)
class EmbeddingRequest:
    chunk_id: str
    text: str
    content_hash: str

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.text:
            raise ValueError("text cannot be empty")
        if not self.content_hash:
            raise ValueError("content_hash cannot be empty")


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def embed_one(self, request: EmbeddingRequest) -> EmbeddingRecord:
        try:
            vector = self._provider.embed_text(request.text)
        except Exception as exc:
            raise EmbeddingServiceError(
                chunk_id=request.chunk_id,
                message=f"provider error: {exc}",
            ) from exc

        if len(vector) != self._provider.dim:
            raise EmbeddingServiceError(
                chunk_id=request.chunk_id,
                message=(
                    "provider returned vector dimension "
                    f"{len(vector)}, expected {self._provider.dim}"
                ),
            )

        try:
            return EmbeddingRecord(
                schema_version=EMBEDDING_SCHEMA_VERSION,
                chunk_id=request.chunk_id,
                vector=vector,
                dim=self._provider.dim,
                model_name=self._provider.model_name,
                model_version=self._provider.model_version,
                content_hash=request.content_hash,
            )
        except ValueError as exc:
            raise EmbeddingServiceError(
                chunk_id=request.chunk_id,
                message=f"invalid embedding record: {exc}",
            ) from exc

    def embed_batch(
        self,
        requests: Sequence[EmbeddingRequest],
    ) -> tuple[EmbeddingRecord, ...]:
        ordered_requests = tuple(requests)
        if not ordered_requests:
            return ()
        try:
            vectors = self._provider.embed_batch(
                tuple(request.text for request in ordered_requests)
            )
        except EmbeddingBatchError as exc:
            try:
                failed_chunk_id = ordered_requests[exc.failed_index].chunk_id
            except IndexError:
                failed_chunk_id = ",".join(
                    request.chunk_id for request in ordered_requests
                )
            raise EmbeddingServiceError(
                chunk_id=failed_chunk_id,
                message=f"provider batch error: {exc}",
            ) from exc
        except Exception as exc:
            chunk_ids = ",".join(request.chunk_id for request in ordered_requests)
            raise EmbeddingServiceError(
                chunk_id=chunk_ids,
                message=f"provider batch error: {exc}",
            ) from exc
        if len(vectors) != len(ordered_requests):
            chunk_ids = ",".join(request.chunk_id for request in ordered_requests)
            raise EmbeddingServiceError(
                chunk_id=chunk_ids,
                message=(
                    f"provider returned {len(vectors)} vectors, "
                    f"expected {len(ordered_requests)}"
                ),
            )
        return tuple(
            self._build_record(request=request, vector=vector)
            for request, vector in zip(ordered_requests, vectors, strict=True)
        )

    def _build_record(
        self,
        *,
        request: EmbeddingRequest,
        vector: tuple[float, ...],
    ) -> EmbeddingRecord:
        if len(vector) != self._provider.dim:
            raise EmbeddingServiceError(
                chunk_id=request.chunk_id,
                message=(
                    "provider returned vector dimension "
                    f"{len(vector)}, expected {self._provider.dim}"
                ),
            )
        try:
            return EmbeddingRecord(
                schema_version=EMBEDDING_SCHEMA_VERSION,
                chunk_id=request.chunk_id,
                vector=vector,
                dim=self._provider.dim,
                model_name=self._provider.model_name,
                model_version=self._provider.model_version,
                content_hash=request.content_hash,
            )
        except ValueError as exc:
            raise EmbeddingServiceError(
                chunk_id=request.chunk_id,
                message=f"invalid embedding record: {exc}",
            ) from exc
