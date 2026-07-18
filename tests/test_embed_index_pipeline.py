from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

from rag.contracts.chunk import Chunk
from rag.embed import HashEmbeddingProvider
from rag.index import SQLiteVectorStore
from rag.logging import get_logger
from rag.pipeline.embed_index_pipeline import (
    EmbedIndexPipelineError,
    build_embed_index_pipeline,
    load_chunk_records,
    main,
)
from rag.storage import ChunkRecord


def _chunk(
    chunk_id: str = "chunk-a",
    *,
    chunk_index: int = 0,
    text: str = "Policy coverage applies.",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="policy.pdf",
        page=0,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        text=text,
    )


def _write_chunks(path: Path, chunks: tuple[Chunk, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(ChunkRecord.from_chunk(chunk).to_json_line() for chunk in chunks),
        encoding="utf-8",
    )


def _parse_json_lines(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_embed_index_pipeline_builds_index_from_chunk_artifact(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    _write_chunks(
        chunk_path,
        (
            _chunk("chunk-a", text="alpha"),
            _chunk("chunk-b", chunk_index=1, text="bravo"),
        ),
    )
    pipeline = build_embed_index_pipeline(
        index_path=index_path,
        collection_name="default",
        provider=HashEmbeddingProvider(dim=4),
    )

    summary = pipeline.run(chunk_path)

    assert summary.succeeded is True
    assert summary.chunks_loaded == 2
    assert summary.sync.new_count == 2
    assert summary.sync.written_count == 2
    rows = SQLiteVectorStore(index_path).list_rows()
    assert tuple(row.row_id for row in rows) == ("chunk-a", "chunk-b")
    assert rows[0].metadata.source_file == "policy.pdf"


def test_embed_index_pipeline_rerun_reports_idempotent_sync(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    _write_chunks(chunk_path, (_chunk("chunk-a", text="alpha"),))
    pipeline = build_embed_index_pipeline(
        index_path=index_path,
        collection_name="default",
        provider=HashEmbeddingProvider(dim=4),
    )

    first = pipeline.run(chunk_path)
    second = pipeline.run(chunk_path)

    assert first.sync.written_count == 1
    assert second.sync.new_count == 0
    assert second.sync.unchanged_count == 1
    assert second.sync.written_count == 0
    assert len(SQLiteVectorStore(index_path).list_rows()) == 1


def test_embed_index_main_emits_summary_and_exit_code(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    _write_chunks(chunk_path, (_chunk("chunk-a"),))
    stdout = io.StringIO()

    exit_code = main(
        (
            "--chunks",
            str(chunk_path),
            "--index",
            str(index_path),
            "--collection",
            "default",
        ),
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["succeeded"] is True
    assert payload["chunks_loaded"] == 1
    assert payload["sync"]["written"] == 1
    assert index_path.exists()


def test_embed_index_main_returns_nonzero_for_malformed_chunks(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    chunk_path.write_text("{not-json}\n", encoding="utf-8")
    stdout = io.StringIO()

    exit_code = main(
        (
            "--chunks",
            str(chunk_path),
            "--index",
            str(index_path),
            "--collection",
            "default",
        ),
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 1
    assert payload["succeeded"] is False
    assert "stage=load_chunks" in payload["error"]


def test_embed_index_pipeline_schema_mismatch_fails_with_sync_stage(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    _write_chunks(chunk_path, (_chunk("chunk-a"),))
    first = build_embed_index_pipeline(
        index_path=index_path,
        collection_name="default",
        provider=HashEmbeddingProvider(dim=4),
    )
    second = build_embed_index_pipeline(
        index_path=index_path,
        collection_name="default",
        provider=HashEmbeddingProvider(dim=5),
    )

    first.run(chunk_path)

    try:
        second.run(chunk_path)
    except Exception as exc:
        assert "stage=sync_index" in str(exc)
    else:
        raise AssertionError("expected sync_index schema mismatch failure")


def test_embed_index_pipeline_emits_structured_stage_logs(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    _write_chunks(chunk_path, (_chunk("chunk-a"),))
    stream = io.StringIO()
    logger = get_logger(
        name=f"rag.embed_index.test.{uuid.uuid4().hex}",
        stream=stream,
    )
    pipeline = build_embed_index_pipeline(
        index_path=tmp_path / "vector.sqlite",
        collection_name="default",
        provider=HashEmbeddingProvider(dim=4),
        logger=logger,
    )

    pipeline.run(chunk_path)

    logs = _parse_json_lines(stream)
    for stage in ("load_chunks", "sync_index"):
        assert any(
            entry["event"] == "stage_start" and entry["stage"] == stage
            for entry in logs
        )
        finishes = [
            entry
            for entry in logs
            if entry["event"] == "stage_finish" and entry["stage"] == stage
        ]
        assert len(finishes) == 1
        assert isinstance(finishes[0]["duration_ms"], int)


def test_load_chunk_records_rejects_duplicate_ids(tmp_path: Path) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunk_path,
        (
            _chunk("chunk-a", text="alpha"),
            _chunk("chunk-a", chunk_index=1, text="bravo"),
        ),
    )

    try:
        load_chunk_records(chunk_path)
    except EmbedIndexPipelineError as exc:
        assert "duplicate chunk_id" in str(exc)
    else:
        raise AssertionError("expected duplicate chunk_id failure")
