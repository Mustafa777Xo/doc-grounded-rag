from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator, TextIO


class RagError(Exception):
    """Base error for the rag package."""


class PipelineError(RagError):
    """Base error for pipeline-related failures."""


class StageError(PipelineError):
    """Error that always carries stage + correlation context."""

    def __init__(
        self,
        *,
        stage: str,
        correlation_id: str,
        message: str,
        hint: str | None = None,
    ) -> None:
        self.stage = stage
        self.correlation_id = correlation_id
        self.stage_message = message
        self.hint = hint
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        base = (
            f"{self.stage_message} "
            f"(stage={self.stage}, correlation_id={self.correlation_id})"
        )
        if self.hint is None:
            return base
        return f"{base}. hint={self.hint}"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
            "stage": getattr(record, "stage", None),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        for key in ("duration_ms", "error_type", "error_message"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, separators=(",", ":"), default=str)


def get_logger(
    *,
    name: str = "rag",
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def _base_extra(*, stage: str, correlation_id: str, event: str) -> dict[str, object]:
    return {"stage": stage, "correlation_id": correlation_id, "event": event}


def log_stage_start(*, logger: logging.Logger, stage: str, correlation_id: str) -> None:
    logger.info(
        "Stage started",
        extra=_base_extra(
            stage=stage, correlation_id=correlation_id, event="stage_start"
        ),
    )


def log_stage_finish(
    *,
    logger: logging.Logger,
    stage: str,
    correlation_id: str,
    duration_ms: int,
) -> None:
    extra = _base_extra(
        stage=stage, correlation_id=correlation_id, event="stage_finish"
    )
    extra["duration_ms"] = duration_ms
    logger.info("Stage finished", extra=extra)


def log_stage_error(
    *,
    logger: logging.Logger,
    stage: str,
    correlation_id: str,
    error: Exception,
) -> None:
    extra = _base_extra(stage=stage, correlation_id=correlation_id, event="stage_error")
    extra["error_type"] = type(error).__name__
    extra["error_message"] = str(error)
    logger.error("Stage failed", extra=extra)


@contextmanager
def stage_transition(
    *,
    logger: logging.Logger,
    stage: str,
    correlation_id: str,
) -> Iterator[None]:
    started = time.perf_counter()
    log_stage_start(logger=logger, stage=stage, correlation_id=correlation_id)
    try:
        yield
    except StageError as err:
        log_stage_error(
            logger=logger, stage=stage, correlation_id=correlation_id, error=err
        )
        raise
    except Exception as err:
        wrapped = StageError(
            stage=stage,
            correlation_id=correlation_id,
            message=f"Unhandled exception in stage '{stage}': {err}",
            hint="Check stage inputs and upstream outputs.",
        )
        log_stage_error(
            logger=logger, stage=stage, correlation_id=correlation_id, error=wrapped
        )
        raise wrapped from err
    else:
        duration_ms = int((time.perf_counter() - started) * 1000)
        log_stage_finish(
            logger=logger,
            stage=stage,
            correlation_id=correlation_id,
            duration_ms=duration_ms,
        )
