from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from rag.contracts.chunk import Chunk
from rag.contracts.indexing import VectorIndexRow
from rag.contracts.retrieval import QueryFilters
from rag.embed import (
    EmbeddingBatcher,
    EmbeddingPreparationPolicy,
    EmbeddingService,
    EmbeddingTextPreparer,
    MockEmbeddingProvider,
)
from rag.index import (
    ContentHasher,
    IndexedChunkState,
    IndexSyncError,
    SQLiteVectorStore,
    VectorIndexWriter,
    VectorQueryResult,
    VectorStoreSchema,
)
from rag.index.sync import ChunkChangeDetector


def _chunk(
    chunk_id: str = "doc-1-p0-s0-e5-a1b2c3d4e5f6",
    *,
    chunk_index: int = 0,
    page: int = 0,
    text: str = "alpha",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="policy.pdf",
        page=page,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        text=text,
    )


def _schema(dim: int = 3) -> VectorStoreSchema:
    return VectorStoreSchema(
        collection_name="default",
        vector_index_schema_version="vector_index.v1",
        embedding_schema_version="embedding.v1",
        chunk_schema_version="chunk.v1",
        model_name="mock-embedder",
        model_version="test",
        dim=dim,
    )


def _writer(
    tmp_path: Path,
    *,
    provider: MockEmbeddingProvider | None = None,
    store: SQLiteVectorStore | None = None,
    schema: VectorStoreSchema | None = None,
) -> tuple[VectorIndexWriter, SQLiteVectorStore]:
    vector_store = store if store is not None else SQLiteVectorStore(tmp_path / "v.db")
    embedding_provider = provider if provider is not None else MockEmbeddingProvider()
    service = EmbeddingService(provider=embedding_provider)
    writer = VectorIndexWriter(
        store=vector_store,
        schema=schema if schema is not None else _schema(dim=embedding_provider.dim),
        preparer=EmbeddingTextPreparer(
            policy=EmbeddingPreparationPolicy(max_chars=100, batch_size=2)
        ),
        batcher=EmbeddingBatcher(
            service=service,
            policy=EmbeddingPreparationPolicy(max_chars=100, batch_size=2),
        ),
        detector=ChunkChangeDetector(hasher=ContentHasher()),
        clock=lambda: "2026-07-11T00:00:00Z",
    )
    return writer, vector_store


def test_index_writer_first_sync_writes_new_chunks(tmp_path: Path) -> None:
    writer, store = _writer(tmp_path)
    chunks = (
        _chunk("chunk-a", text="alpha"),
        _chunk("chunk-b", chunk_index=1, page=1, text="bravo"),
    )

    summary = writer.sync(chunks)

    assert summary.new_count == 2
    assert summary.changed_count == 0
    assert summary.unchanged_count == 0
    assert summary.deleted_count == 0
    assert summary.written_count == 2
    assert summary.removed_count == 0
    assert summary.error_count == 0
    rows = store.list_rows()
    assert tuple(row.row_id for row in rows) == ("chunk-a", "chunk-b")
    assert rows[0].metadata.source_file == "policy.pdf"
    assert rows[0].created_at == "2026-07-11T00:00:00Z"


def test_index_writer_rerun_unchanged_data_writes_zero_rows(tmp_path: Path) -> None:
    writer, store = _writer(tmp_path)
    chunks = (_chunk("chunk-a", text="alpha"),)

    first = writer.sync(chunks)
    second = writer.sync(chunks)

    assert first.written_count == 1
    assert second.new_count == 0
    assert second.changed_count == 0
    assert second.unchanged_count == 1
    assert second.deleted_count == 0
    assert second.written_count == 0
    assert len(store.list_rows()) == 1


def test_index_writer_changed_chunk_is_upserted_without_duplicate(
    tmp_path: Path,
) -> None:
    writer, store = _writer(tmp_path)

    writer.sync((_chunk("chunk-a", text="alpha"),))
    summary = writer.sync((_chunk("chunk-a", text="bravo"),))

    rows = store.list_rows()
    assert summary.changed_count == 1
    assert summary.written_count == 1
    assert len(rows) == 1
    assert rows[0].row_id == "chunk-a"
    assert rows[0].metadata.text == "bravo"
    assert rows[0].metadata.content_hash == rows[0].embedding.content_hash


def test_index_writer_deletes_indexed_rows_missing_from_current_chunks(
    tmp_path: Path,
) -> None:
    writer, store = _writer(tmp_path)
    writer.sync(
        (
            _chunk("chunk-a", text="alpha"),
            _chunk("chunk-b", chunk_index=1, text="bravo"),
        )
    )

    summary = writer.sync((_chunk("chunk-a", text="alpha"),))

    assert summary.unchanged_count == 1
    assert summary.deleted_count == 1
    assert summary.written_count == 0
    assert summary.removed_count == 1
    assert tuple(row.row_id for row in store.list_rows()) == ("chunk-a",)


def test_index_writer_provider_failure_has_chunk_context(tmp_path: Path) -> None:
    writer, store = _writer(
        tmp_path,
        provider=MockEmbeddingProvider(fail_on_text="fail"),
    )

    with pytest.raises(IndexSyncError, match="chunk-fail") as exc_info:
        writer.sync(
            (
                _chunk("chunk-ok", text="ok"),
                _chunk("chunk-fail", chunk_index=1, text="fail"),
            )
        )

    assert exc_info.value.stage == "embed"
    assert exc_info.value.chunk_id == "chunk-fail"
    assert exc_info.value.partial_summary is not None
    assert exc_info.value.partial_summary.new_count == 2
    assert exc_info.value.partial_summary.error_count == 1
    assert store.list_rows() == ()


def test_index_writer_bootstrap_schema_mismatch_has_stage_context(
    tmp_path: Path,
) -> None:
    store = SQLiteVectorStore(tmp_path / "v.db")
    store.bootstrap_collection(_schema(dim=2))
    writer, _ = _writer(tmp_path, store=store, schema=_schema(dim=3))

    with pytest.raises(IndexSyncError, match="bootstrap") as exc_info:
        writer.sync((_chunk("chunk-a"),))

    assert exc_info.value.stage == "bootstrap"
    assert exc_info.value.partial_summary is not None
    assert exc_info.value.partial_summary.error_count == 1


def test_index_writer_upsert_failure_preserves_partial_summary() -> None:
    store = _FailingUpsertStore()
    writer = VectorIndexWriter(
        store=store,
        schema=_schema(),
        preparer=EmbeddingTextPreparer(),
        batcher=EmbeddingBatcher(
            service=EmbeddingService(provider=MockEmbeddingProvider()),
        ),
        clock=lambda: "2026-07-11T00:00:00Z",
    )

    with pytest.raises(IndexSyncError, match="upsert") as exc_info:
        writer.sync((_chunk("chunk-a"),))

    assert exc_info.value.stage == "upsert"
    assert exc_info.value.partial_summary is not None
    assert exc_info.value.partial_summary.new_count == 1
    assert exc_info.value.partial_summary.error_count == 1


class _FailingUpsertStore:
    def bootstrap_collection(self, schema: VectorStoreSchema) -> None:
        _ = schema

    def upsert(self, rows: Sequence[VectorIndexRow]) -> int:
        _ = rows
        raise RuntimeError("store unavailable")

    def query(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> tuple[VectorQueryResult, ...]:
        _ = filters
        return ()

    def delete(self, chunk_ids: Sequence[str]) -> int:
        return 0

    def list_rows(self) -> tuple[VectorIndexRow, ...]:
        return ()

    def get_indexed_states(self) -> tuple[IndexedChunkState, ...]:
        return ()
