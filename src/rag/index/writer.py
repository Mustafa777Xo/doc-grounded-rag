from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from rag.contracts.chunk import Chunk
from rag.contracts.indexing import (
    CHUNK_SCHEMA_VERSION,
    VECTOR_INDEX_SCHEMA_VERSION,
    EmbeddingRecord,
    VectorIndexMetadata,
    VectorIndexRow,
)
from rag.embed import EmbeddingBatcher, EmbeddingServiceError, EmbeddingTextPreparer
from rag.index.schema import VectorStoreSchema
from rag.index.sync import ChangeDetectionSummary, ChunkChangeDetector
from rag.index.vector_store import VectorStore


class IndexSyncError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        message: str,
        chunk_id: str | None = None,
        partial_summary: IndexSyncSummary | None = None,
    ) -> None:
        self.stage = stage
        self.chunk_id = chunk_id
        self.partial_summary = partial_summary
        context = f" during {stage}"
        if chunk_id is not None:
            context += f" for chunk_id={chunk_id!r}"
        super().__init__(f"Index sync failed{context}: {message}")


@dataclass(frozen=True)
class IndexSyncSummary:
    new_count: int
    changed_count: int
    unchanged_count: int
    deleted_count: int
    written_count: int = 0
    removed_count: int = 0
    error_count: int = 0

    @classmethod
    def empty(cls) -> IndexSyncSummary:
        return cls(
            new_count=0,
            changed_count=0,
            unchanged_count=0,
            deleted_count=0,
        )

    @classmethod
    def from_changes(cls, changes: ChangeDetectionSummary) -> IndexSyncSummary:
        return cls(
            new_count=changes.new_count,
            changed_count=changes.changed_count,
            unchanged_count=changes.unchanged_count,
            deleted_count=changes.deleted_count,
        )


class VectorIndexWriter:
    def __init__(
        self,
        *,
        store: VectorStore,
        schema: VectorStoreSchema,
        batcher: EmbeddingBatcher,
        preparer: EmbeddingTextPreparer | None = None,
        detector: ChunkChangeDetector | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._store = store
        self._schema = schema
        self._batcher = batcher
        self._preparer = preparer if preparer is not None else EmbeddingTextPreparer()
        self._detector = detector if detector is not None else ChunkChangeDetector()
        self._clock = clock if clock is not None else _utc_now

    def sync(self, chunks: Sequence[Chunk]) -> IndexSyncSummary:
        summary = IndexSyncSummary.empty()
        try:
            self._store.bootstrap_collection(self._schema)
        except Exception as exc:
            raise _sync_error(
                stage="bootstrap",
                exc=exc,
                summary=summary,
            ) from exc

        try:
            existing_states = self._store.get_indexed_states()
            changes = self._detector.detect(
                current_chunks=chunks,
                existing_states=existing_states,
            )
        except Exception as exc:
            raise _sync_error(
                stage="detect",
                exc=exc,
                summary=summary,
            ) from exc

        summary = IndexSyncSummary.from_changes(changes)
        write_changes = changes.by_state("new") + changes.by_state("changed")
        rows: tuple[VectorIndexRow, ...] = ()
        if write_changes:
            try:
                prepared = self._preparer.prepare_many(
                    tuple(
                        (
                            change.chunk_id,
                            _require_current_chunk(change).text,
                            _require_current_hash(change),
                        )
                        for change in write_changes
                    )
                )
                embeddings = self._batcher.embed(prepared)
            except Exception as exc:
                raise _sync_error(
                    stage="embed",
                    exc=exc,
                    summary=summary,
                    chunk_id=_chunk_id_from_exception(exc),
                ) from exc

            if len(embeddings) != len(write_changes):
                raise _sync_error(
                    stage="embed",
                    exc=ValueError(
                        "embedding result count does not match chunks selected "
                        "for writing"
                    ),
                    summary=summary,
                )

            try:
                rows = tuple(
                    build_vector_index_row(
                        chunk=_require_current_chunk(change),
                        embedding=embedding,
                        content_hash=_require_current_hash(change),
                        timestamp=self._clock(),
                    )
                    for change, embedding in zip(write_changes, embeddings)
                )
            except Exception as exc:
                raise _sync_error(
                    stage="build_rows",
                    exc=exc,
                    summary=summary,
                    chunk_id=_chunk_id_from_exception(exc),
                ) from exc

        try:
            written_count = self._store.upsert(rows) if rows else 0
        except Exception as exc:
            raise _sync_error(
                stage="upsert",
                exc=exc,
                summary=summary,
            ) from exc

        deleted_ids = tuple(change.chunk_id for change in changes.by_state("deleted"))
        try:
            removed_count = self._store.delete(deleted_ids) if deleted_ids else 0
        except Exception as exc:
            raise _sync_error(
                stage="delete",
                exc=exc,
                summary=replace(summary, written_count=written_count),
            ) from exc

        return replace(
            summary,
            written_count=written_count,
            removed_count=removed_count,
        )


def build_vector_index_row(
    *,
    chunk: Chunk,
    embedding: EmbeddingRecord,
    content_hash: str,
    timestamp: str,
) -> VectorIndexRow:
    if embedding.chunk_id != chunk.chunk_id:
        raise ValueError(
            "embedding chunk_id must match chunk chunk_id: "
            f"{embedding.chunk_id!r} != {chunk.chunk_id!r}"
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
        created_at=timestamp,
        updated_at=timestamp,
    )


def _sync_error(
    *,
    stage: str,
    exc: Exception,
    summary: IndexSyncSummary,
    chunk_id: str | None = None,
) -> IndexSyncError:
    return IndexSyncError(
        stage=stage,
        message=str(exc),
        chunk_id=chunk_id,
        partial_summary=replace(summary, error_count=summary.error_count + 1),
    )


def _chunk_id_from_exception(exc: Exception) -> str | None:
    if isinstance(exc, EmbeddingServiceError):
        return exc.chunk_id
    return None


def _require_current_chunk(change: object) -> Chunk:
    current_chunk = getattr(change, "current_chunk", None)
    chunk_id = getattr(change, "chunk_id", "<unknown>")
    if not isinstance(current_chunk, Chunk):
        raise ValueError(f"current_chunk is required for chunk_id={chunk_id!r}")
    return current_chunk


def _require_current_hash(change: object) -> str:
    current_hash = getattr(change, "current_hash", None)
    chunk_id = getattr(change, "chunk_id", "<unknown>")
    if not isinstance(current_hash, str) or not current_hash:
        raise ValueError(f"current_hash is required for chunk_id={chunk_id!r}")
    return current_hash


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        )
    )
