from __future__ import annotations

from pathlib import Path

import pytest

from rag.contracts.indexing import (
    CHUNK_SCHEMA_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    VECTOR_INDEX_SCHEMA_VERSION,
    EmbeddingRecord,
    VectorIndexMetadata,
    VectorIndexRow,
)
from rag.index import SQLiteVectorStore, VectorStoreSchema, VectorStoreSchemaError


def _schema(
    *,
    collection_name: str = "default",
    model_name: str = "local-hash-embedder",
    model_version: str = "v1",
    dim: int = 2,
) -> VectorStoreSchema:
    return VectorStoreSchema(
        collection_name=collection_name,
        vector_index_schema_version=VECTOR_INDEX_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        model_name=model_name,
        model_version=model_version,
        dim=dim,
    )


def _row(
    chunk_id: str,
    vector: tuple[float, ...],
    *,
    text: str = "hello world",
    model_name: str = "local-hash-embedder",
    model_version: str = "v1",
    content_hash: str | None = None,
) -> VectorIndexRow:
    hash_value = content_hash if content_hash is not None else f"sha256:{chunk_id}"
    embedding = EmbeddingRecord(
        schema_version=EMBEDDING_SCHEMA_VERSION,
        chunk_id=chunk_id,
        vector=vector,
        dim=len(vector),
        model_name=model_name,
        model_version=model_version,
        content_hash=hash_value,
    )
    metadata = VectorIndexMetadata(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="policy.pdf",
        page=0,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        text=text,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        content_hash=hash_value,
    )
    return VectorIndexRow(
        schema_version=VECTOR_INDEX_SCHEMA_VERSION,
        row_id=chunk_id,
        embedding=embedding,
        metadata=metadata,
        created_at="2026-07-11T00:00:00Z",
        updated_at="2026-07-11T00:00:00Z",
    )


def test_sqlite_vector_store_bootstraps_empty_path(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "vector.sqlite"
    store = SQLiteVectorStore(path)

    store.bootstrap_collection(_schema())

    assert path.exists()
    assert store.list_rows() == ()


def test_sqlite_vector_store_bootstrap_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    schema = _schema()

    store.bootstrap_collection(schema)
    store.bootstrap_collection(schema)

    assert store.list_rows() == ()


def test_sqlite_vector_store_bootstrap_rejects_schema_mismatch(
    tmp_path: Path,
) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema(dim=2))

    with pytest.raises(VectorStoreSchemaError, match="schema mismatch"):
        store.bootstrap_collection(_schema(dim=3))


def test_sqlite_vector_store_rejects_collection_mismatch(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite", collection_name="a")

    with pytest.raises(VectorStoreSchemaError, match="collection_name"):
        store.bootstrap_collection(_schema(collection_name="b"))


def test_sqlite_vector_store_upsert_stores_metadata(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema())
    row = _row("chunk-1", (1.0, 0.0), text="alpha")

    assert store.upsert((row,)) == 1

    rows = store.list_rows()
    assert len(rows) == 1
    stored = rows[0]
    assert stored.row_id == "chunk-1"
    assert stored.metadata.doc_id == "doc-1"
    assert stored.metadata.source_file == "policy.pdf"
    assert stored.metadata.content_hash == "sha256:chunk-1"


def test_sqlite_vector_store_repeated_upsert_does_not_duplicate(
    tmp_path: Path,
) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema())

    store.upsert((_row("chunk-1", (1.0, 0.0), text="alpha"),))
    store.upsert((_row("chunk-1", (0.0, 1.0), text="bravo"),))

    rows = store.list_rows()
    assert len(rows) == 1
    assert rows[0].embedding.vector == (0.0, 1.0)
    assert rows[0].metadata.text == "bravo"


def test_sqlite_vector_store_query_returns_relevant_top_k(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema())
    store.upsert(
        (
            _row("chunk-a", (1.0, 0.0), text="alpha"),
            _row("chunk-b", (0.0, 1.0), text="bravo"),
        )
    )

    results = store.query((1.0, 0.0), limit=2)

    assert tuple(result.row.row_id for result in results) == ("chunk-a", "chunk-b")
    assert results[0].score > results[1].score
    assert results[0].row.metadata.source_file == "policy.pdf"


def test_sqlite_vector_store_query_tie_order_is_deterministic(
    tmp_path: Path,
) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema())
    store.upsert(
        (
            _row("chunk-b", (1.0, 0.0)),
            _row("chunk-a", (1.0, 0.0)),
        )
    )

    results = store.query((1.0, 0.0), limit=2)

    assert tuple(result.row.row_id for result in results) == ("chunk-a", "chunk-b")


def test_sqlite_vector_store_query_rejects_wrong_dimension(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema(dim=2))

    with pytest.raises(Exception, match="dimension"):
        store.query((1.0, 0.0, 0.0), limit=1)


def test_sqlite_vector_store_delete_removes_vectors(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema())
    store.upsert(
        (
            _row("chunk-a", (1.0, 0.0)),
            _row("chunk-b", (0.0, 1.0)),
        )
    )

    deleted = store.delete(("chunk-a", "missing"))

    assert deleted == 1
    assert tuple(row.row_id for row in store.list_rows()) == ("chunk-b",)


def test_sqlite_vector_store_rejects_row_schema_mismatch(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema(model_name="expected", dim=2))

    with pytest.raises(VectorStoreSchemaError, match="collection schema"):
        store.upsert((_row("chunk-1", (1.0, 0.0), model_name="other"),))
    with pytest.raises(VectorStoreSchemaError, match="collection schema"):
        store.upsert((_row("chunk-2", (1.0, 0.0, 0.0), model_name="expected"),))


def test_sqlite_vector_store_reports_schema_mismatch_details(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema(model_name="old-model", dim=2))

    with pytest.raises(VectorStoreSchemaError) as exc_info:
        store.bootstrap_collection(_schema(model_name="new-model", dim=3))

    message = str(exc_info.value)
    assert "model_name: existing='old-model', requested='new-model'" in message
    assert "dim: existing=2, requested=3" in message
    assert "Rebuild the collection" in message


def test_sqlite_vector_store_get_indexed_states(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vector.sqlite")
    store.bootstrap_collection(_schema())
    store.upsert(
        (
            _row("chunk-b", (0.0, 1.0), content_hash="sha256:b"),
            _row("chunk-a", (1.0, 0.0), content_hash="sha256:a"),
        )
    )

    states = store.get_indexed_states()

    assert tuple(state.chunk_id for state in states) == ("chunk-a", "chunk-b")
    assert tuple(state.content_hash for state in states) == ("sha256:a", "sha256:b")
