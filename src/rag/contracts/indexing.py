from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Final

EMBEDDING_SCHEMA_VERSION: Final = "embedding.v1"
VECTOR_INDEX_SCHEMA_VERSION: Final = "vector_index.v1"
CHUNK_SCHEMA_VERSION: Final = "chunk.v1"


@dataclass(frozen=True)
class EmbeddingRecord:
    schema_version: str
    chunk_id: str
    vector: tuple[float, ...]
    dim: int
    model_name: str
    model_version: str
    content_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != EMBEDDING_SCHEMA_VERSION:
            raise ValueError("schema_version must be embedding.v1")
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.vector:
            raise ValueError("vector cannot be empty")
        if self.dim <= 0:
            raise ValueError("dim must be greater than zero")
        if self.dim != len(self.vector):
            raise ValueError("dim must match vector length")
        if any(not math.isfinite(value) for value in self.vector):
            raise ValueError("vector values must be finite")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.model_version:
            raise ValueError("model_version cannot be empty")
        if not self.content_hash:
            raise ValueError("content_hash cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "chunk_id": self.chunk_id,
            "vector": list(self.vector),
            "dim": self.dim,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "content_hash": self.content_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class VectorIndexMetadata:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    chunk_schema_version: str
    content_hash: str

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
        if self.chunk_schema_version != CHUNK_SCHEMA_VERSION:
            raise ValueError("chunk_schema_version must be chunk.v1")
        if not self.content_hash:
            raise ValueError("content_hash cannot be empty")

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
            "chunk_schema_version": self.chunk_schema_version,
            "content_hash": self.content_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class VectorIndexRow:
    schema_version: str
    row_id: str
    embedding: EmbeddingRecord
    metadata: VectorIndexMetadata
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if self.schema_version != VECTOR_INDEX_SCHEMA_VERSION:
            raise ValueError("schema_version must be vector_index.v1")
        if not self.row_id:
            raise ValueError("row_id cannot be empty")
        if self.row_id != self.embedding.chunk_id:
            raise ValueError("row_id must match embedding chunk_id")
        if self.row_id != self.metadata.chunk_id:
            raise ValueError("row_id must match metadata chunk_id")
        if self.embedding.content_hash != self.metadata.content_hash:
            raise ValueError("embedding and metadata content_hash must match")
        if not self.created_at:
            raise ValueError("created_at cannot be empty")
        if not self.updated_at:
            raise ValueError("updated_at cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "row_id": self.row_id,
            "embedding": self.embedding.to_dict(),
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
