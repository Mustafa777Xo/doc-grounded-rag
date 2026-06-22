from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

import pytest

from rag.logging import get_logger
from rag.pipeline.ingest import (
    IngestionPipeline,
    IngestionPipelineError,
    discover_pdf_inputs,
    main,
)
from rag.storage import JsonlChunkWriter

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pdfs"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _parse_json_lines(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def test_ingestion_pipeline_processes_one_pdf_end_to_end(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    pipeline = IngestionPipeline(
        writer=JsonlChunkWriter(output_path),
        correlation_id_factory=lambda: "ingest-test",
    )

    summary = pipeline.run(FIXTURES_DIR / "sample_policy.pdf")

    assert summary.succeeded is True
    assert summary.documents_discovered == 1
    assert summary.documents_succeeded == 1
    assert summary.documents_failed == 0
    assert summary.pages_processed == 2
    assert summary.chunks_produced > 0
    assert summary.documents[0].status == "succeeded"
    records = _read_jsonl(output_path)
    assert len(records) == summary.chunks_produced
    assert all(record["schema_version"] == "chunk.v1" for record in records)
    assert all(record["source_file"] == "sample_policy.pdf" for record in records)


def test_ingestion_pipeline_output_matches_deterministic_snapshot(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "chunks.jsonl"
    pipeline = IngestionPipeline(
        writer=JsonlChunkWriter(output_path),
        correlation_id_factory=lambda: "ingest-snapshot-test",
    )

    summary = pipeline.run(FIXTURES_DIR / "sample_policy.pdf")

    assert summary.succeeded is True
    assert _read_jsonl(output_path) == [
        {
            "char_end": 47,
            "char_start": 0,
            "chunk_id": "12300f777421a63b-p0-s0-e47-42e1bbda7461",
            "chunk_index": 0,
            "doc_id": "12300f777421a63b",
            "page": 0,
            "schema_version": "chunk.v1",
            "source_file": "sample_policy.pdf",
            "text": "Policy coverage applies to full-time employees.",
        },
        {
            "char_end": 36,
            "char_start": 0,
            "chunk_id": "12300f777421a63b-p1-s0-e36-a980f854f11d",
            "chunk_index": 1,
            "doc_id": "12300f777421a63b",
            "page": 1,
            "schema_version": "chunk.v1",
            "source_file": "sample_policy.pdf",
            "text": "Claims must be filed within 30 days.",
        },
    ]


def test_ingestion_pipeline_overwrite_is_byte_stable_across_repeated_runs(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "chunks.jsonl"
    pipeline = IngestionPipeline(
        writer=JsonlChunkWriter(output_path, write_mode="overwrite"),
        correlation_id_factory=lambda: "ingest-repeat-test",
    )

    first_summary = pipeline.run(FIXTURES_DIR / "sample_policy.pdf")
    first_output = output_path.read_text(encoding="utf-8")
    second_summary = pipeline.run(FIXTURES_DIR / "sample_policy.pdf")

    assert first_summary.to_dict() == second_summary.to_dict()
    assert output_path.read_text(encoding="utf-8") == first_output


def test_ingestion_pipeline_emits_structured_stage_logs(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = get_logger(name=f"rag.ingest.test.{uuid.uuid4().hex}", stream=stream)
    pipeline = IngestionPipeline(
        writer=JsonlChunkWriter(tmp_path / "chunks.jsonl"),
        logger=logger,
        correlation_id_factory=lambda: "ingest-log-test",
    )

    pipeline.run(FIXTURES_DIR / "sample_policy.pdf")

    logs = _parse_json_lines(stream)
    for stage in ("parse", "normalize", "chunk", "persist"):
        assert any(
            entry["event"] == "stage_start" and entry["stage"] == stage
            for entry in logs
        )
        finish_logs = [
            entry
            for entry in logs
            if entry["event"] == "stage_finish" and entry["stage"] == stage
        ]
        assert len(finish_logs) == 1
        assert isinstance(finish_logs[0]["duration_ms"], int)
        assert finish_logs[0]["correlation_id"] == "ingest-log-test"


def test_directory_ingestion_continues_after_corrupt_pdf_then_reports_failure(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "chunks.jsonl"
    pipeline = IngestionPipeline(
        writer=JsonlChunkWriter(output_path),
        correlation_id_factory=lambda: "ingest-dir-test",
    )

    summary = pipeline.run(FIXTURES_DIR)

    assert summary.succeeded is False
    assert summary.documents_discovered == 3
    assert summary.documents_succeeded == 2
    assert summary.documents_failed == 1
    failures = [
        document for document in summary.documents if document.status == "failed"
    ]
    assert len(failures) == 1
    assert "sample_corrupt.pdf" in failures[0].source_path
    assert failures[0].error is not None
    assert "stage=parse" in failures[0].error
    records = _read_jsonl(output_path)
    assert len(records) == summary.chunks_produced
    assert {record["source_file"] for record in records} == {
        "sample_empty_page.pdf",
        "sample_policy.pdf",
    }


def test_discover_pdf_inputs_rejects_missing_and_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(IngestionPipelineError, match="does not exist"):
        discover_pdf_inputs(tmp_path / "missing.pdf")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(IngestionPipelineError, match="no PDFs"):
        discover_pdf_inputs(empty_dir)


def test_ingest_main_emits_summary_and_exit_code(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"
    stdout = io.StringIO()

    exit_code = main(
        (
            "--input",
            str(FIXTURES_DIR / "sample_policy.pdf"),
            "--output",
            str(output_path),
            "--mode",
            "overwrite",
        ),
        stdout=stdout,
    )

    payload = json.loads(stdout.getvalue())
    assert exit_code == 0
    assert payload["succeeded"] is True
    assert payload["documents_succeeded"] == 1
    assert output_path.exists()


def test_ingest_main_returns_nonzero_when_safe_append_duplicates(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "chunks.jsonl"
    args = (
        "--input",
        str(FIXTURES_DIR / "sample_policy.pdf"),
        "--output",
        str(output_path),
        "--mode",
        "safe_append",
    )

    assert main(args, stdout=io.StringIO()) == 0
    second_stdout = io.StringIO()
    exit_code = main(args, stdout=second_stdout)

    payload = json.loads(second_stdout.getvalue())
    assert exit_code == 1
    assert payload["succeeded"] is False
    assert "stage=persist" in payload["error"]
