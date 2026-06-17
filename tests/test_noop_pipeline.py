from __future__ import annotations

import io
import json
import logging
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import RetrievalResult
from rag.generate.interfaces import NoOpGenerator
from rag.index.interfaces import NoOpIndexer
from rag.ingest.interfaces import NoOpIngestor
from rag.logging import StageError, get_logger
from rag.pipeline.orchestrator import NoOpPipeline, documents_to_noop_chunks
from rag.rerank.interfaces import NoOpReranker
from rag.retrieve.interfaces import NoOpRetriever, Retriever

RetrieverFactory = Callable[[Sequence[Chunk]], Retriever]


def _parse_json_lines(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def _build_pipeline(
    *,
    retriever_factory: RetrieverFactory = NoOpRetriever,
    logger: logging.Logger | None = None,
) -> NoOpPipeline:
    return NoOpPipeline(
        ingestor=NoOpIngestor(),
        indexer=NoOpIndexer(),
        retriever_factory=retriever_factory,
        reranker=NoOpReranker(),
        generator=NoOpGenerator(),
        correlation_id_factory=lambda: "test-correlation-id",
        logger=logger
        if logger is not None
        else get_logger(name=f"rag.pipeline.test.{uuid.uuid4().hex}"),
    )


def test_documents_to_noop_chunks_preserves_traceability() -> None:
    documents = NoOpIngestor().ingest((Path("b.pdf"), Path("a.pdf")))
    chunks = documents_to_noop_chunks(documents)
    assert tuple(chunk.chunk_id for chunk in chunks) == (
        "doc-0-chunk-0",
        "doc-1-chunk-0",
    )
    assert tuple(chunk.source_file for chunk in chunks) == ("a.pdf", "b.pdf")
    assert all(chunk.page == 0 for chunk in chunks)


def test_noop_pipeline_runs_from_mock_input_to_mock_cited_answer() -> None:
    pipeline = _build_pipeline()
    answer = pipeline.run(
        question="What does the mock document say?",
        sources=(Path("mock-company-handbook.pdf"),),
    )
    assert answer.grounded is True
    assert answer.citations
    assert answer.citations[0].source_file == "mock-company-handbook.pdf"
    assert "No-op grounded answer" in answer.answer_text


def test_noop_pipeline_logs_each_stage_start_finish_and_duration() -> None:
    stream = io.StringIO()
    logger = get_logger(name=f"rag.pipeline.test.{uuid.uuid4().hex}", stream=stream)
    pipeline = _build_pipeline(logger=logger)
    pipeline.run(
        question="What does the mock document say?",
        sources=(Path("mock-company-handbook.pdf"),),
    )
    logs = _parse_json_lines(stream)
    stages = ("ingest", "chunk", "index", "retrieve", "rerank", "generate")
    for stage in stages:
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


class FailingRetriever:
    def __init__(self, _: Sequence[Chunk]) -> None:
        return None

    def retrieve(self, query: str, limit: int = 5) -> tuple[RetrievalResult, ...]:
        _ = query
        _ = limit
        raise RuntimeError("mock retrieval failed")


def test_noop_pipeline_bubbles_stage_failures_with_context() -> None:
    pipeline = _build_pipeline(retriever_factory=FailingRetriever)
    with pytest.raises(StageError, match="stage=retrieve") as exc_info:
        pipeline.run(
            question="What does the mock document say?",
            sources=(Path("mock-company-handbook.pdf"),),
        )
    assert "mock retrieval failed" in str(exc_info.value)
    assert exc_info.value.correlation_id == "test-correlation-id"
