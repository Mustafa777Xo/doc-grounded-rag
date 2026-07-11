from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Sequence

from rag.contracts.chunk import Chunk
from rag.contracts.indexing import CHUNK_SCHEMA_VERSION, VectorIndexRow

ChunkChangeState = Literal["new", "changed", "unchanged", "deleted"]


class ChunkChangeDetectionError(ValueError):
    """Raised when change detection inputs cannot be compared safely."""


@dataclass(frozen=True)
class ContentHasher:
    chunk_schema_version: str = CHUNK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.chunk_schema_version:
            raise ValueError("chunk_schema_version cannot be empty")

    def hash_text(self, text: str) -> str:
        if not text:
            raise ValueError("text cannot be empty")
        payload = {
            "chunk_schema_version": self.chunk_schema_version,
            "text": text,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def hash_chunk(self, chunk: Chunk) -> str:
        return self.hash_text(chunk.text)


@dataclass(frozen=True)
class IndexedChunkState:
    chunk_id: str
    content_hash: str

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if not self.content_hash:
            raise ValueError("content_hash cannot be empty")

    @classmethod
    def from_index_row(cls, row: VectorIndexRow) -> IndexedChunkState:
        return cls(
            chunk_id=row.metadata.chunk_id,
            content_hash=row.metadata.content_hash,
        )


@dataclass(frozen=True)
class ChunkChange:
    state: ChunkChangeState
    chunk_id: str
    current_chunk: Chunk | None = None
    existing_hash: str | None = None
    current_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")
        if self.state in ("new", "changed", "unchanged"):
            if self.current_chunk is None:
                raise ValueError("current_chunk is required for current chunk states")
            if self.current_hash is None:
                raise ValueError("current_hash is required for current chunk states")
        if self.state == "deleted" and self.existing_hash is None:
            raise ValueError("existing_hash is required for deleted changes")


@dataclass(frozen=True)
class ChangeDetectionSummary:
    changes: tuple[ChunkChange, ...]

    @property
    def new_count(self) -> int:
        return self._count("new")

    @property
    def changed_count(self) -> int:
        return self._count("changed")

    @property
    def unchanged_count(self) -> int:
        return self._count("unchanged")

    @property
    def deleted_count(self) -> int:
        return self._count("deleted")

    def _count(self, state: ChunkChangeState) -> int:
        return sum(1 for change in self.changes if change.state == state)

    def by_state(self, state: ChunkChangeState) -> tuple[ChunkChange, ...]:
        return tuple(change for change in self.changes if change.state == state)


class ChunkChangeDetector:
    def __init__(self, hasher: ContentHasher | None = None) -> None:
        self._hasher = hasher if hasher is not None else ContentHasher()

    def detect(
        self,
        *,
        current_chunks: Sequence[Chunk],
        existing_states: Sequence[IndexedChunkState],
    ) -> ChangeDetectionSummary:
        current_by_id = self._ensure_unique_current(current_chunks)
        existing_by_id = self._ensure_unique_existing(existing_states)

        changes: list[ChunkChange] = []
        for chunk in current_chunks:
            current_hash = self._hasher.hash_chunk(chunk)
            existing = existing_by_id.get(chunk.chunk_id)
            if existing is None:
                changes.append(
                    ChunkChange(
                        state="new",
                        chunk_id=chunk.chunk_id,
                        current_chunk=chunk,
                        current_hash=current_hash,
                    )
                )
                continue
            state: ChunkChangeState = (
                "unchanged" if existing.content_hash == current_hash else "changed"
            )
            changes.append(
                ChunkChange(
                    state=state,
                    chunk_id=chunk.chunk_id,
                    current_chunk=chunk,
                    existing_hash=existing.content_hash,
                    current_hash=current_hash,
                )
            )

        for existing in existing_states:
            if existing.chunk_id not in current_by_id:
                changes.append(
                    ChunkChange(
                        state="deleted",
                        chunk_id=existing.chunk_id,
                        existing_hash=existing.content_hash,
                    )
                )

        return ChangeDetectionSummary(changes=tuple(changes))

    def detect_from_index_rows(
        self,
        *,
        current_chunks: Sequence[Chunk],
        existing_rows: Sequence[VectorIndexRow],
    ) -> ChangeDetectionSummary:
        return self.detect(
            current_chunks=current_chunks,
            existing_states=tuple(
                IndexedChunkState.from_index_row(row) for row in existing_rows
            ),
        )

    def _ensure_unique_current(
        self,
        current_chunks: Sequence[Chunk],
    ) -> dict[str, Chunk]:
        seen: dict[str, Chunk] = {}
        for chunk in current_chunks:
            if chunk.chunk_id in seen:
                raise ChunkChangeDetectionError(
                    f"duplicate current chunk_id: {chunk.chunk_id}"
                )
            seen[chunk.chunk_id] = chunk
        return seen

    def _ensure_unique_existing(
        self,
        existing_states: Sequence[IndexedChunkState],
    ) -> dict[str, IndexedChunkState]:
        seen: dict[str, IndexedChunkState] = {}
        for state in existing_states:
            if state.chunk_id in seen:
                raise ChunkChangeDetectionError(
                    f"duplicate existing chunk_id: {state.chunk_id}"
                )
            seen[state.chunk_id] = state
        return seen
