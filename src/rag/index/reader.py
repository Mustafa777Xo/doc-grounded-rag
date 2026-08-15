from __future__ import annotations

from dataclasses import dataclass, field

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import (
    RetrievalCandidate,
    RetrievalQuery,
    RetrieverSource,
    ScoreProvenance,
)
from rag.embed import EmbeddingRequest, EmbeddingService, EmbeddingServiceError
from rag.errors import RetrievalError
from rag.index.sync import ContentHasher
from rag.index.vector_store import VectorQueryResult, VectorStore, VectorStoreError


@dataclass(frozen=True)
class SemanticIndexReader:
    store: VectorStore
    embedding_service: EmbeddingService
    hasher: ContentHasher = field(default_factory=ContentHasher)

    def retrieve(
        self, query: RetrievalQuery, limit: int = 5
    ) -> tuple[RetrievalCandidate, ...]:
        if limit <= 0:
            raise RetrievalError(
                stage="dense",
                query_id=query.query_id,
                message="limit must be greater than zero",
                hint="Set dense_top_k to a positive value.",
            )

        try:
            query_hash = self.hasher.hash_text(query.normalized_text)
            embedding = self.embedding_service.embed_one(
                EmbeddingRequest(
                    chunk_id=query.query_id,
                    text=query.normalized_text,
                    content_hash=query_hash,
                )
            )
        except (EmbeddingServiceError, ValueError) as exc:
            raise RetrievalError(
                stage="dense",
                query_id=query.query_id,
                message=str(exc),
                hint="Verify the embedding model and normalized query text.",
            ) from exc

        try:
            hits = self.store.query(embedding.vector, limit=limit)
        except VectorStoreError as exc:
            raise RetrievalError(
                stage="dense",
                query_id=query.query_id,
                message=str(exc),
                hint="Verify the vector index path, collection, and schema.",
            ) from exc

        return tuple(_to_retrieval_result(hit) for hit in hits)


def _to_retrieval_result(hit: VectorQueryResult) -> RetrievalCandidate:
    metadata = hit.row.metadata
    return RetrievalCandidate(
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
        scores=ScoreProvenance(dense_score=hit.score),
        sources=frozenset({RetrieverSource.DENSE}),
    )
