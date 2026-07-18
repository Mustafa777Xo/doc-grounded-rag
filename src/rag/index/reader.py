from __future__ import annotations

from dataclasses import dataclass, field

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import RetrievalResult
from rag.embed import EmbeddingRequest, EmbeddingService, EmbeddingServiceError
from rag.index.sync import ContentHasher
from rag.index.vector_store import VectorQueryResult, VectorStore, VectorStoreError


class SemanticIndexReaderError(RuntimeError):
    def __init__(self, *, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"Semantic index reader failed during {stage}: {message}")


@dataclass(frozen=True)
class SemanticIndexReader:
    store: VectorStore
    embedding_service: EmbeddingService
    hasher: ContentHasher = field(default_factory=ContentHasher)

    def retrieve(self, query: str, limit: int = 5) -> tuple[RetrievalResult, ...]:
        if not query:
            raise SemanticIndexReaderError(
                stage="embed_query",
                message="query cannot be empty",
            )
        if limit <= 0:
            raise SemanticIndexReaderError(
                stage="query_index",
                message="limit must be greater than zero",
            )

        try:
            query_hash = self.hasher.hash_text(query)
            embedding = self.embedding_service.embed_one(
                EmbeddingRequest(
                    chunk_id="query",
                    text=query,
                    content_hash=query_hash,
                )
            )
        except (EmbeddingServiceError, ValueError) as exc:
            raise SemanticIndexReaderError(
                stage="embed_query",
                message=str(exc),
            ) from exc

        try:
            hits = self.store.query(embedding.vector, limit=limit)
        except VectorStoreError as exc:
            raise SemanticIndexReaderError(
                stage="query_index",
                message=str(exc),
            ) from exc

        return tuple(_to_retrieval_result(hit) for hit in hits)


def _to_retrieval_result(hit: VectorQueryResult) -> RetrievalResult:
    metadata = hit.row.metadata
    return RetrievalResult(
        chunk=Chunk(
            chunk_id=metadata.chunk_id,
            doc_id=metadata.doc_id,
            source_file=metadata.source_file,
            page=metadata.page,
            chunk_index=metadata.chunk_index,
            char_start=metadata.char_start,
            char_end=metadata.char_end,
            text=metadata.text,
        ),
        score=hit.score,
        retrieval_method="semantic",
    )
