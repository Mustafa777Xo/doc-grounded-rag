from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import Settings
from rag.contracts.chunk import Chunk
from rag.storage import (
    CHUNK_RECORD_SCHEMA_VERSION,
    ChunkRecord,
    ChunkStoreDuplicateError,
    ChunkStoreValidationError,
    JsonlChunkWriter,
)


def _chunk(
    chunk_id: str = "doc-1-p0-s0-e5-a1b2c3d4e5f6",
    *,
    chunk_index: int = 0,
    page: int = 0,
    char_start: int = 0,
    text: str = "alpha",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="policy.pdf",
        page=page,
        chunk_index=chunk_index,
        char_start=char_start,
        char_end=char_start + len(text),
        text=text,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_overwrite_writes_schema_valid_chunk_records(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "chunks.jsonl"
    chunks = (
        _chunk(),
        _chunk(
            "doc-1-p0-s6-e10-f6e5d4c3b2a1",
            chunk_index=1,
            char_start=6,
            text="beta",
        ),
    )

    result = JsonlChunkWriter(path).write(chunks)

    assert result.records_written == 2
    assert result.total_records == 2
    payloads = _read_jsonl(path)
    assert [payload["chunk_id"] for payload in payloads] == [
        "doc-1-p0-s0-e5-a1b2c3d4e5f6",
        "doc-1-p0-s6-e10-f6e5d4c3b2a1",
    ]
    assert all(
        payload["schema_version"] == CHUNK_RECORD_SCHEMA_VERSION for payload in payloads
    )
    written_chunks = [ChunkRecord.from_dict(payload).chunk for payload in payloads]
    assert written_chunks == list(chunks)


def test_overwrite_repeated_runs_replace_artifact_without_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    writer = JsonlChunkWriter(path, write_mode="overwrite")
    chunks = (_chunk(),)

    writer.write(chunks)
    first = path.read_text(encoding="utf-8")
    writer.write(chunks)

    assert path.read_text(encoding="utf-8") == first
    assert len(_read_jsonl(path)) == 1


def test_safe_append_appends_new_unique_chunks(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    writer = JsonlChunkWriter(path, write_mode="safe_append")

    first_result = writer.write((_chunk(),))
    second_result = writer.write(
        (
            _chunk(
                "doc-1-p1-s0-e5-bbbbbbbbbbbb",
                chunk_index=1,
                page=1,
            ),
        )
    )

    assert first_result.total_records == 1
    assert second_result.records_written == 1
    assert second_result.total_records == 2
    assert len(_read_jsonl(path)) == 2


def test_safe_append_rejects_repeated_chunk_id(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    writer = JsonlChunkWriter(path, write_mode="safe_append")
    chunks = (_chunk(),)

    writer.write(chunks)

    with pytest.raises(ChunkStoreDuplicateError, match="safe_append"):
        writer.write(chunks)

    assert len(_read_jsonl(path)) == 1


def test_writer_rejects_duplicate_chunk_ids_in_batch(tmp_path: Path) -> None:
    writer = JsonlChunkWriter(tmp_path / "chunks.jsonl")

    with pytest.raises(ChunkStoreDuplicateError, match="duplicate chunk_id"):
        writer.write(
            (
                _chunk(),
                _chunk(chunk_index=1),
            )
        )


def test_writer_loads_output_path_and_mode_from_settings(tmp_path: Path) -> None:
    path = tmp_path / "configured.jsonl"
    settings = Settings(
        docs_dir=Path("data/pdfs"),
        chunk_output_path=path,
        chunk_write_mode="safe_append",
    )

    result = JsonlChunkWriter.from_settings(settings).write((_chunk(),))

    assert result.path == path
    assert result.mode == "safe_append"
    assert path.exists()


def test_safe_append_validates_existing_jsonl_before_writing(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ChunkStoreValidationError, match="invalid JSONL record"):
        JsonlChunkWriter(path, write_mode="safe_append").write((_chunk(),))


def test_chunk_record_rejects_invalid_schema_version() -> None:
    payload = {
        "schema_version": "chunk.v0",
        **_chunk().to_dict(),
    }

    with pytest.raises(ChunkStoreValidationError, match="schema_version"):
        ChunkRecord.from_dict(payload)
