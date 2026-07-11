from __future__ import annotations

import pytest

from rag.contracts.chunk import Chunk
from rag.contracts.indexing import (
    CHUNK_SCHEMA_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    VECTOR_INDEX_SCHEMA_VERSION,
    EmbeddingRecord,
    VectorIndexMetadata,
    VectorIndexRow,
)
from rag.index import (
    ChunkChangeDetectionError,
    ChunkChangeDetector,
    ContentHasher,
    IndexedChunkState,
)


def _chunk(
    chunk_id: str = "chunk-1",
    *,
    doc_id: str = "doc-1",
    source_file: str = "policy.pdf",
    page: int = 0,
    chunk_index: int = 0,
    char_start: int = 0,
    text: str = "hello world",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=source_file,
        page=page,
        chunk_index=chunk_index,
        char_start=char_start,
        char_end=char_start + len(text),
        text=text,
    )


def _index_row(chunk: Chunk, content_hash: str) -> VectorIndexRow:
    embedding = EmbeddingRecord(
        schema_version=EMBEDDING_SCHEMA_VERSION,
        chunk_id=chunk.chunk_id,
        vector=(0.1, 0.2, 0.3),
        dim=3,
        model_name="local-hash-embedder",
        model_version="v1",
        content_hash=content_hash,
    )
    metadata = VectorIndexMetadata(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        source_file=chunk.source_file,
        page=chunk.page,
        chunk_index=chunk.chunk_index,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        text=chunk.text,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        content_hash=content_hash,
    )
    return VectorIndexRow(
        schema_version=VECTOR_INDEX_SCHEMA_VERSION,
        row_id=chunk.chunk_id,
        embedding=embedding,
        metadata=metadata,
        created_at="2026-07-11T00:00:00Z",
        updated_at="2026-07-11T00:00:00Z",
    )


def test_content_hash_is_stable_for_same_text_and_schema() -> None:
    hasher = ContentHasher()

    first = hasher.hash_text("hello world")
    second = hasher.hash_text("hello world")

    assert first == second
    assert first.startswith("sha256:")


def test_content_hash_changes_when_text_changes() -> None:
    hasher = ContentHasher()

    assert hasher.hash_text("hello world") != hasher.hash_text("hello there")


def test_content_hash_changes_when_schema_version_changes() -> None:
    first = ContentHasher(chunk_schema_version="chunk.v1").hash_text("hello world")
    second = ContentHasher(chunk_schema_version="chunk.v2").hash_text("hello world")

    assert first != second


def test_content_hash_is_independent_of_source_metadata() -> None:
    hasher = ContentHasher()
    first = _chunk(doc_id="doc-1", source_file="a.pdf", page=0)
    second = _chunk(doc_id="doc-2", source_file="b.pdf", page=3)

    assert hasher.hash_chunk(first) == hasher.hash_chunk(second)


def test_content_hasher_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError, match="chunk_schema_version"):
        ContentHasher(chunk_schema_version="")
    with pytest.raises(ValueError, match="text"):
        ContentHasher().hash_text("")


def test_change_detector_marks_new_changed_unchanged_and_deleted() -> None:
    hasher = ContentHasher()
    unchanged = _chunk("unchanged", text="same")
    changed = _chunk("changed", text="current")
    new = _chunk("new", text="new text")
    existing_states = (
        IndexedChunkState(
            chunk_id="unchanged",
            content_hash=hasher.hash_chunk(unchanged),
        ),
        IndexedChunkState(
            chunk_id="changed",
            content_hash=hasher.hash_text("previous"),
        ),
        IndexedChunkState(
            chunk_id="deleted",
            content_hash=hasher.hash_text("deleted"),
        ),
    )

    summary = ChunkChangeDetector(hasher=hasher).detect(
        current_chunks=(unchanged, changed, new),
        existing_states=existing_states,
    )

    assert tuple(change.state for change in summary.changes) == (
        "unchanged",
        "changed",
        "new",
        "deleted",
    )
    assert summary.unchanged_count == 1
    assert summary.changed_count == 1
    assert summary.new_count == 1
    assert summary.deleted_count == 1


def test_change_detector_preserves_deterministic_ordering() -> None:
    hasher = ContentHasher()
    first = _chunk("first", text="first")
    second = _chunk("second", text="second")
    existing_states = (
        IndexedChunkState(chunk_id="deleted-a", content_hash=hasher.hash_text("a")),
        IndexedChunkState(chunk_id="deleted-b", content_hash=hasher.hash_text("b")),
    )

    summary = ChunkChangeDetector(hasher=hasher).detect(
        current_chunks=(first, second),
        existing_states=existing_states,
    )

    assert tuple(change.chunk_id for change in summary.changes) == (
        "first",
        "second",
        "deleted-a",
        "deleted-b",
    )


def test_change_detector_rejects_duplicate_current_chunk_ids() -> None:
    chunk = _chunk("duplicate")

    with pytest.raises(ChunkChangeDetectionError, match="duplicate current"):
        ChunkChangeDetector().detect(
            current_chunks=(chunk, chunk),
            existing_states=(),
        )


def test_change_detector_rejects_duplicate_existing_chunk_ids() -> None:
    state = IndexedChunkState(chunk_id="duplicate", content_hash="sha256:abc")

    with pytest.raises(ChunkChangeDetectionError, match="duplicate existing"):
        ChunkChangeDetector().detect(
            current_chunks=(),
            existing_states=(state, state),
        )


def test_change_detector_can_consume_vector_index_rows() -> None:
    hasher = ContentHasher()
    chunk = _chunk("chunk-1", text="hello world")
    row = _index_row(chunk, hasher.hash_chunk(chunk))

    summary = ChunkChangeDetector(hasher=hasher).detect_from_index_rows(
        current_chunks=(chunk,),
        existing_rows=(row,),
    )

    assert summary.unchanged_count == 1
    assert summary.changes[0].state == "unchanged"
    assert summary.changes[0].existing_hash == hasher.hash_chunk(chunk)


def test_indexed_chunk_state_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        IndexedChunkState(chunk_id="", content_hash="sha256:abc")
    with pytest.raises(ValueError, match="content_hash"):
        IndexedChunkState(chunk_id="chunk-1", content_hash="")
