from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Sequence

from rag.chunking.ids import ChunkIdCollisionError, ensure_unique_chunk_ids
from rag.config import Settings
from rag.contracts.chunk import Chunk

CHUNK_RECORD_SCHEMA_VERSION: Final = "chunk.v1"
WriteMode = Literal["overwrite", "safe_append"]


class ChunkStoreError(RuntimeError):
    """Base error for chunk artifact storage failures."""


class ChunkStoreValidationError(ChunkStoreError):
    """Raised when a stored chunk record does not match the storage schema."""


class ChunkStoreDuplicateError(ChunkStoreError):
    def __init__(
        self,
        message: str,
        *,
        chunk_id: str,
        path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.chunk_id = chunk_id
        self.path = path


@dataclass(frozen=True)
class ChunkRecord:
    schema_version: str
    chunk: Chunk

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkRecord:
        return cls(schema_version=CHUNK_RECORD_SCHEMA_VERSION, chunk=chunk)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ChunkRecord:
        schema_version = _require_str(payload, "schema_version")
        if schema_version != CHUNK_RECORD_SCHEMA_VERSION:
            raise ChunkStoreValidationError(
                "unsupported chunk record schema_version: "
                f"{schema_version!r}; expected {CHUNK_RECORD_SCHEMA_VERSION!r}"
            )

        try:
            chunk = Chunk(
                chunk_id=_require_str(payload, "chunk_id"),
                doc_id=_require_str(payload, "doc_id"),
                source_file=_require_str(payload, "source_file"),
                page=_require_int(payload, "page"),
                chunk_index=_require_int(payload, "chunk_index"),
                char_start=_require_int(payload, "char_start"),
                char_end=_require_int(payload, "char_end"),
                text=_require_str(payload, "text"),
            )
        except ValueError as exc:
            raise ChunkStoreValidationError(
                f"invalid chunk record payload: {exc}"
            ) from exc

        return cls(schema_version=schema_version, chunk=chunk)

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": self.schema_version, **self.chunk.to_dict()}

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True) + "\n"


@dataclass(frozen=True)
class ChunkWriteResult:
    path: Path
    mode: WriteMode
    records_written: int
    total_records: int


class JsonlChunkWriter:
    def __init__(self, output_path: Path, write_mode: WriteMode = "overwrite") -> None:
        self.output_path = output_path
        self.write_mode = write_mode

    @classmethod
    def from_settings(cls, settings: Settings) -> JsonlChunkWriter:
        return cls(
            output_path=settings.chunk_output_path,
            write_mode=settings.chunk_write_mode,
        )

    def write(self, chunks: Sequence[Chunk]) -> ChunkWriteResult:
        records = [ChunkRecord.from_chunk(chunk) for chunk in chunks]
        self._ensure_unique_batch(records)

        if self.write_mode == "overwrite":
            return self._overwrite(records)
        if self.write_mode == "safe_append":
            return self._safe_append(records)

        raise ChunkStoreError(f"unsupported chunk write mode: {self.write_mode!r}")

    def _overwrite(self, records: Sequence[ChunkRecord]) -> ChunkWriteResult:
        self._ensure_writable_path()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.with_name(f".{self.output_path.name}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(record.to_json_line())
        temp_path.replace(self.output_path)
        return ChunkWriteResult(
            path=self.output_path,
            mode="overwrite",
            records_written=len(records),
            total_records=len(records),
        )

    def _safe_append(self, records: Sequence[ChunkRecord]) -> ChunkWriteResult:
        self._ensure_writable_path()
        existing_records = self._read_existing_records()
        existing_ids = {record.chunk.chunk_id for record in existing_records}

        for record in records:
            if record.chunk.chunk_id in existing_ids:
                raise ChunkStoreDuplicateError(
                    "safe_append would duplicate chunk_id "
                    f"{record.chunk.chunk_id!r} in {self.output_path}",
                    chunk_id=record.chunk.chunk_id,
                    path=self.output_path,
                )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(record.to_json_line())

        return ChunkWriteResult(
            path=self.output_path,
            mode="safe_append",
            records_written=len(records),
            total_records=len(existing_records) + len(records),
        )

    def _read_existing_records(self) -> list[ChunkRecord]:
        if not self.output_path.exists():
            return []

        records: list[ChunkRecord] = []
        seen: dict[str, int] = {}
        with self.output_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ChunkStoreValidationError(
                        f"invalid JSONL record at {self.output_path}:{line_number}: "
                        f"{exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ChunkStoreValidationError(
                        f"chunk record at {self.output_path}:{line_number} "
                        "must be a JSON object"
                    )

                record = ChunkRecord.from_dict(payload)
                chunk_id = record.chunk.chunk_id
                if chunk_id in seen:
                    raise ChunkStoreDuplicateError(
                        "existing chunk artifact contains duplicate chunk_id "
                        f"{chunk_id!r} at lines {seen[chunk_id]} and {line_number}",
                        chunk_id=chunk_id,
                        path=self.output_path,
                    )
                seen[chunk_id] = line_number
                records.append(record)

        return records

    def _ensure_unique_batch(self, records: Sequence[ChunkRecord]) -> None:
        try:
            ensure_unique_chunk_ids(tuple(record.chunk for record in records))
        except ChunkIdCollisionError as exc:
            raise ChunkStoreDuplicateError(
                "chunk batch contains duplicate chunk_id "
                f"{exc.chunk_id!r} at indexes {exc.first_index} "
                f"and {exc.duplicate_index}",
                chunk_id=exc.chunk_id,
                path=self.output_path,
            ) from exc

    def _ensure_writable_path(self) -> None:
        if self.output_path.exists() and self.output_path.is_dir():
            raise ChunkStoreError(
                f"chunk output path must be a file, got directory: {self.output_path}"
            )


def _require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value == "":
        raise ChunkStoreValidationError(f"{key} must be a non-empty string")
    return value


def _require_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int:
        raise ChunkStoreValidationError(f"{key} must be an integer")
    return value
