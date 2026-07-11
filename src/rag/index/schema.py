from __future__ import annotations

from dataclasses import dataclass

from rag.contracts.indexing import (
    CHUNK_SCHEMA_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    VECTOR_INDEX_SCHEMA_VERSION,
    VectorIndexRow,
)


@dataclass(frozen=True)
class VectorStoreSchema:
    collection_name: str
    vector_index_schema_version: str
    embedding_schema_version: str
    chunk_schema_version: str
    model_name: str
    model_version: str
    dim: int

    def __post_init__(self) -> None:
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")
        if self.vector_index_schema_version != VECTOR_INDEX_SCHEMA_VERSION:
            raise ValueError("vector_index_schema_version must be vector_index.v1")
        if self.embedding_schema_version != EMBEDDING_SCHEMA_VERSION:
            raise ValueError("embedding_schema_version must be embedding.v1")
        if self.chunk_schema_version != CHUNK_SCHEMA_VERSION:
            raise ValueError("chunk_schema_version must be chunk.v1")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.model_version:
            raise ValueError("model_version cannot be empty")
        if self.dim <= 0:
            raise ValueError("dim must be greater than zero")

    @classmethod
    def from_row(
        cls,
        *,
        collection_name: str,
        row: VectorIndexRow,
    ) -> VectorStoreSchema:
        return cls(
            collection_name=collection_name,
            vector_index_schema_version=row.schema_version,
            embedding_schema_version=row.embedding.schema_version,
            chunk_schema_version=row.metadata.chunk_schema_version,
            model_name=row.embedding.model_name,
            model_version=row.embedding.model_version,
            dim=row.embedding.dim,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "collection_name": self.collection_name,
            "vector_index_schema_version": self.vector_index_schema_version,
            "embedding_schema_version": self.embedding_schema_version,
            "chunk_schema_version": self.chunk_schema_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dim": self.dim,
        }
