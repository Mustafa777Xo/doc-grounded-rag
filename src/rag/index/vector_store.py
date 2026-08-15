from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, cast

from rag.contracts.indexing import (
    EmbeddingRecord,
    VectorIndexMetadata,
    VectorIndexRow,
)
from rag.contracts.retrieval import QueryFilters
from rag.index.schema import VectorStoreSchema
from rag.index.sync import IndexedChunkState


class VectorStoreError(RuntimeError):
    """Base error for vector store failures."""


class VectorStoreSchemaError(VectorStoreError):
    """Raised when a collection schema is missing or incompatible."""


@dataclass(frozen=True)
class VectorQueryResult:
    row: VectorIndexRow
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


class VectorStore(Protocol):
    def bootstrap_collection(self, schema: VectorStoreSchema) -> None: ...

    def upsert(self, rows: Sequence[VectorIndexRow]) -> int: ...

    def query(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> tuple[VectorQueryResult, ...]: ...

    def delete(self, chunk_ids: Sequence[str]) -> int: ...

    def list_rows(self) -> tuple[VectorIndexRow, ...]: ...

    def get_indexed_states(self) -> tuple[IndexedChunkState, ...]: ...


class SQLiteVectorStore:
    def __init__(self, path: Path, collection_name: str = "default") -> None:
        if not collection_name:
            raise ValueError("collection_name cannot be empty")
        self.path = path
        self.collection_name = collection_name

    def bootstrap_collection(self, schema: VectorStoreSchema) -> None:
        if schema.collection_name != self.collection_name:
            raise VectorStoreSchemaError(
                "schema collection_name must match store collection_name"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._create_tables(connection)
            existing = self._load_schema(connection)
            if existing is None:
                self._insert_schema(connection, schema)
                return
            if existing != schema:
                checked_fields = (
                    "vector_index_schema_version",
                    "embedding_schema_version",
                    "chunk_schema_version",
                    "model_name",
                    "model_version",
                    "dim",
                )
                mismatches = "; ".join(
                    f"{field_name}: existing={getattr(existing, field_name)!r}, "
                    f"requested={getattr(schema, field_name)!r}"
                    for field_name in checked_fields
                    if getattr(existing, field_name) != getattr(schema, field_name)
                )
                raise VectorStoreSchemaError(
                    "vector store schema mismatch for collection "
                    f"{self.collection_name!r}: {mismatches}. "
                    "Rebuild the collection before changing its embedding model, "
                    "revision, dimension, or schema version."
                )

    def upsert(self, rows: Sequence[VectorIndexRow]) -> int:
        schema = self._require_schema()
        with self._connect() as connection:
            for row in rows:
                self._validate_row_against_schema(row=row, schema=schema)
                connection.execute(
                    """
                    INSERT INTO vector_index_rows (
                        collection_name,
                        chunk_id,
                        doc_id,
                        source_file,
                        page,
                        chunk_index,
                        content_hash,
                        model_name,
                        model_version,
                        dim,
                        vector_json,
                        row_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection_name, chunk_id) DO UPDATE SET
                        doc_id=excluded.doc_id,
                        source_file=excluded.source_file,
                        page=excluded.page,
                        chunk_index=excluded.chunk_index,
                        content_hash=excluded.content_hash,
                        model_name=excluded.model_name,
                        model_version=excluded.model_version,
                        dim=excluded.dim,
                        vector_json=excluded.vector_json,
                        row_json=excluded.row_json
                    """,
                    (
                        self.collection_name,
                        row.row_id,
                        row.metadata.doc_id,
                        row.metadata.source_file,
                        row.metadata.page,
                        row.metadata.chunk_index,
                        row.metadata.content_hash,
                        row.embedding.model_name,
                        row.embedding.model_version,
                        row.embedding.dim,
                        json.dumps(list(row.embedding.vector), separators=(",", ":")),
                        row.to_json(),
                    ),
                )
        return len(rows)

    def query(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> tuple[VectorQueryResult, ...]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        schema = self._require_schema()
        query_vector = tuple(vector)
        if len(query_vector) != schema.dim:
            raise VectorStoreError(
                f"query vector dimension {len(query_vector)} does not match "
                f"collection dim {schema.dim}"
            )
        with self._connect() as connection:
            rows = self._load_row_payloads(connection, filters=filters)

        results = tuple(
            VectorQueryResult(
                row=row,
                score=_cosine_similarity(query_vector, row.embedding.vector),
            )
            for row in rows
        )
        return tuple(
            sorted(results, key=lambda result: (-result.score, result.row.row_id))[
                :limit
            ]
        )

    def delete(self, chunk_ids: Sequence[str]) -> int:
        self._require_schema()
        deleted = 0
        with self._connect() as connection:
            for chunk_id in chunk_ids:
                if not chunk_id:
                    raise ValueError("chunk_id cannot be empty")
                cursor = connection.execute(
                    """
                    DELETE FROM vector_index_rows
                    WHERE collection_name = ? AND chunk_id = ?
                    """,
                    (self.collection_name, chunk_id),
                )
                deleted += cursor.rowcount
        return deleted

    def list_rows(self) -> tuple[VectorIndexRow, ...]:
        self._require_schema()
        with self._connect() as connection:
            return self._load_row_payloads(connection)

    def get_indexed_states(self) -> tuple[IndexedChunkState, ...]:
        self._require_schema()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT chunk_id, content_hash
                FROM vector_index_rows
                WHERE collection_name = ?
                ORDER BY chunk_id
                """,
                (self.collection_name,),
            )
            return tuple(
                IndexedChunkState(chunk_id=row[0], content_hash=row[1])
                for row in cursor.fetchall()
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _create_tables(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_store_schema (
                collection_name TEXT PRIMARY KEY,
                vector_index_schema_version TEXT NOT NULL,
                embedding_schema_version TEXT NOT NULL,
                chunk_schema_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                dim INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_index_rows (
                collection_name TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                row_json TEXT NOT NULL,
                PRIMARY KEY (collection_name, chunk_id),
                FOREIGN KEY (collection_name)
                    REFERENCES vector_store_schema(collection_name)
                    ON DELETE CASCADE
            )
            """
        )

    def _load_schema(self, connection: sqlite3.Connection) -> VectorStoreSchema | None:
        cursor = connection.execute(
            """
            SELECT
                collection_name,
                vector_index_schema_version,
                embedding_schema_version,
                chunk_schema_version,
                model_name,
                model_version,
                dim
            FROM vector_store_schema
            WHERE collection_name = ?
            """,
            (self.collection_name,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return VectorStoreSchema(
            collection_name=row[0],
            vector_index_schema_version=row[1],
            embedding_schema_version=row[2],
            chunk_schema_version=row[3],
            model_name=row[4],
            model_version=row[5],
            dim=row[6],
        )

    def _insert_schema(
        self,
        connection: sqlite3.Connection,
        schema: VectorStoreSchema,
    ) -> None:
        connection.execute(
            """
            INSERT INTO vector_store_schema (
                collection_name,
                vector_index_schema_version,
                embedding_schema_version,
                chunk_schema_version,
                model_name,
                model_version,
                dim
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schema.collection_name,
                schema.vector_index_schema_version,
                schema.embedding_schema_version,
                schema.chunk_schema_version,
                schema.model_name,
                schema.model_version,
                schema.dim,
            ),
        )

    def _require_schema(self) -> VectorStoreSchema:
        if not self.path.exists():
            raise VectorStoreSchemaError(
                f"vector store has not been bootstrapped: {self.path}"
            )
        with self._connect() as connection:
            self._create_tables(connection)
            schema = self._load_schema(connection)
        if schema is None:
            raise VectorStoreSchemaError(
                f"collection has not been bootstrapped: {self.collection_name}"
            )
        return schema

    def _validate_row_against_schema(
        self,
        *,
        row: VectorIndexRow,
        schema: VectorStoreSchema,
    ) -> None:
        row_schema = VectorStoreSchema.from_row(
            collection_name=self.collection_name,
            row=row,
        )
        if row_schema != schema:
            raise VectorStoreSchemaError(
                f"row {row.row_id!r} does not match collection schema"
            )

    def _load_row_payloads(
        self,
        connection: sqlite3.Connection,
        *,
        filters: QueryFilters | None = None,
    ) -> tuple[VectorIndexRow, ...]:
        clauses = ["collection_name = ?"]
        parameters: list[str | int] = [self.collection_name]
        if filters is not None and filters.doc_ids:
            placeholders = ", ".join("?" for _ in filters.doc_ids)
            clauses.append(f"doc_id IN ({placeholders})")
            parameters.extend(sorted(filters.doc_ids))
        if filters is not None and filters.source_files:
            placeholders = ", ".join("?" for _ in filters.source_files)
            clauses.append(f"source_file IN ({placeholders})")
            parameters.extend(sorted(filters.source_files))
        if filters is not None and filters.pages:
            placeholders = ", ".join("?" for _ in filters.pages)
            clauses.append(f"page IN ({placeholders})")
            parameters.extend(sorted(filters.pages))
        query = (
            "SELECT row_json FROM vector_index_rows WHERE "
            + " AND ".join(clauses)
            + " ORDER BY chunk_id"
        )
        cursor = connection.execute(query, parameters)
        return tuple(_row_from_json(row[0]) for row in cursor.fetchall())


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _row_from_json(payload: str) -> VectorIndexRow:
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise VectorStoreError("stored vector row payload must be an object")
    return _row_from_dict(cast(dict[str, object], loaded))


def _row_from_dict(payload: dict[str, object]) -> VectorIndexRow:
    embedding_payload = _require_dict(payload, "embedding")
    metadata_payload = _require_dict(payload, "metadata")
    embedding = EmbeddingRecord(
        schema_version=_require_str(embedding_payload, "schema_version"),
        chunk_id=_require_str(embedding_payload, "chunk_id"),
        vector=_require_float_tuple(embedding_payload, "vector"),
        dim=_require_int(embedding_payload, "dim"),
        model_name=_require_str(embedding_payload, "model_name"),
        model_version=_require_str(embedding_payload, "model_version"),
        content_hash=_require_str(embedding_payload, "content_hash"),
    )
    metadata = VectorIndexMetadata(
        chunk_id=_require_str(metadata_payload, "chunk_id"),
        doc_id=_require_str(metadata_payload, "doc_id"),
        source_file=_require_str(metadata_payload, "source_file"),
        page=_require_int(metadata_payload, "page"),
        chunk_index=_require_int(metadata_payload, "chunk_index"),
        char_start=_require_int(metadata_payload, "char_start"),
        char_end=_require_int(metadata_payload, "char_end"),
        text=_require_str(metadata_payload, "text"),
        chunk_schema_version=_require_str(metadata_payload, "chunk_schema_version"),
        content_hash=_require_str(metadata_payload, "content_hash"),
    )
    return VectorIndexRow(
        schema_version=_require_str(payload, "schema_version"),
        row_id=_require_str(payload, "row_id"),
        embedding=embedding,
        metadata=metadata,
        created_at=_require_str(payload, "created_at"),
        updated_at=_require_str(payload, "updated_at"),
    )


def _require_dict(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise VectorStoreError(f"{key} must be an object")
    return cast(dict[str, object], value)


def _require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise VectorStoreError(f"{key} must be a non-empty string")
    return value


def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise VectorStoreError(f"{key} must be an integer")
    return value


def _require_float_tuple(payload: dict[str, object], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise VectorStoreError(f"{key} must be a list")
    vector: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise VectorStoreError(f"{key} values must be numeric")
        vector.append(float(item))
    return tuple(vector)
