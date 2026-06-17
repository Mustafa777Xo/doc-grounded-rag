from __future__ import annotations

import io
import json
import logging
import uuid

import pytest

from rag.logging import StageError, get_logger, stage_transition


def _parse_json_lines(buffer: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in buffer.getvalue().splitlines() if line.strip()]


def _run_noop_pipeline(*, logger: logging.Logger, correlation_id: str) -> None:
    for stage in ("ingest", "chunk", "index", "retrieve", "rerank", "generate"):
        with stage_transition(
            logger=logger, stage=stage, correlation_id=correlation_id
        ):
            pass


def test_noop_pipeline_emits_structured_logs_for_stage_transitions() -> None:
    stream = io.StringIO()
    logger = get_logger(name=f"rag.test.{uuid.uuid4().hex}", stream=stream)
    correlation_id = "req-123"
    _run_noop_pipeline(logger=logger, correlation_id=correlation_id)
    logs = _parse_json_lines(stream)
    assert len(logs) == 12  # start + finish for 6 stages
    assert all(log["correlation_id"] == correlation_id for log in logs)
    assert any(
        log["event"] == "stage_start" and log["stage"] == "ingest" for log in logs
    )
    assert any(
        log["event"] == "stage_finish" and log["stage"] == "generate" for log in logs
    )


def test_error_path_emits_actionable_stage_aware_message() -> None:
    stream = io.StringIO()
    logger = get_logger(name=f"rag.test.{uuid.uuid4().hex}", stream=stream)
    correlation_id = "req-err-1"
    with pytest.raises(StageError, match="stage=retrieve"):
        with stage_transition(
            logger=logger, stage="retrieve", correlation_id=correlation_id
        ):
            raise RuntimeError("vector index unavailable")
    logs = _parse_json_lines(stream)
    error_logs = [entry for entry in logs if entry.get("event") == "stage_error"]
    assert len(error_logs) == 1
    error_log = error_logs[0]
    assert error_log["stage"] == "retrieve"
    assert "vector index unavailable" in str(error_log["error_message"])
    assert "Check stage inputs and upstream outputs." in str(error_log["error_message"])
