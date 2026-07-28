from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from rag.chunking.ids import ChunkIdCollisionError, ensure_unique_chunk_ids
from rag.config import Settings
from rag.contracts.chunk import Chunk
from rag.contracts.indexing import (
    CHUNK_SCHEMA_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    VECTOR_INDEX_SCHEMA_VERSION,
)
from rag.embed import (
    EmbeddingBatcher,
    EmbeddingPreparationPolicy,
    EmbeddingProvider,
    EmbeddingService,
    EmbeddingTextPreparer,
    HashEmbeddingProvider,
)
from rag.index import IndexSyncSummary, SQLiteVectorStore, VectorIndexWriter
from rag.index.schema import VectorStoreSchema
from rag.logging import get_logger, new_correlation_id, stage_transition
from rag.storage import ChunkRecord, ChunkStoreValidationError


class EmbedIndexPipelineError(RuntimeError):
    """Raised when the embed+index pipeline cannot complete."""


@dataclass(frozen=True)
class EmbedIndexRunSummary:
    chunk_path: str
    index_path: str
    collection_name: str
    model_name: str
    model_version: str
    embedding_dim: int
    chunks_loaded: int
    sync: IndexSyncSummary

    @property
    def succeeded(self) -> bool:
        return self.sync.error_count == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_path": self.chunk_path,
            "index_path": self.index_path,
            "collection_name": self.collection_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_dim": self.embedding_dim,
            "chunks_loaded": self.chunks_loaded,
            "succeeded": self.succeeded,
            "sync": {
                "new": self.sync.new_count,
                "changed": self.sync.changed_count,
                "unchanged": self.sync.unchanged_count,
                "deleted": self.sync.deleted_count,
                "written": self.sync.written_count,
                "removed": self.sync.removed_count,
                "errors": self.sync.error_count,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class EmbedIndexPipeline:
    def __init__(
        self,
        *,
        writer: VectorIndexWriter,
        index_path: Path,
        collection_name: str,
        provider: EmbeddingProvider,
        logger: logging.Logger | None = None,
        correlation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._writer = writer
        self._index_path = index_path
        self._collection_name = collection_name
        self._provider = provider
        self._logger = (
            logger if logger is not None else get_logger(name="rag.embed_index")
        )
        self._correlation_id_factory = (
            correlation_id_factory
            if correlation_id_factory is not None
            else new_correlation_id
        )

    def run(self, chunk_path: Path) -> EmbedIndexRunSummary:
        correlation_id = str(self._correlation_id_factory())
        with stage_transition(
            logger=self._logger,
            stage="load_chunks",
            correlation_id=correlation_id,
        ):
            chunks = load_chunk_records(chunk_path)

        with stage_transition(
            logger=self._logger,
            stage="sync_index",
            correlation_id=correlation_id,
        ):
            sync_summary = self._writer.sync(chunks)

        return EmbedIndexRunSummary(
            chunk_path=str(chunk_path),
            index_path=str(self._index_path),
            collection_name=self._collection_name,
            model_name=self._provider.model_name,
            model_version=self._provider.model_version,
            embedding_dim=self._provider.dim,
            chunks_loaded=len(chunks),
            sync=sync_summary,
        )


def load_chunk_records(path: Path) -> tuple[Chunk, ...]:
    if not path.exists():
        raise EmbedIndexPipelineError(f"chunk artifact does not exist: {path}")
    if not path.is_file():
        raise EmbedIndexPipelineError(f"chunk artifact must be a file: {path}")

    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EmbedIndexPipelineError(
                    f"invalid JSONL record at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise EmbedIndexPipelineError(
                    f"chunk record at {path}:{line_number} must be a JSON object"
                )
            try:
                chunks.append(
                    ChunkRecord.from_dict(cast(dict[str, object], payload)).chunk
                )
            except ChunkStoreValidationError as exc:
                raise EmbedIndexPipelineError(
                    f"invalid chunk record at {path}:{line_number}: {exc}"
                ) from exc

    if not chunks:
        raise EmbedIndexPipelineError(f"chunk artifact has no chunk records: {path}")

    try:
        ensure_unique_chunk_ids(tuple(chunks))
    except ChunkIdCollisionError as exc:
        raise EmbedIndexPipelineError(
            "chunk artifact contains duplicate chunk_id "
            f"{exc.chunk_id!r} at indexes {exc.first_index} "
            f"and {exc.duplicate_index}"
        ) from exc
    return tuple(chunks)


def build_embed_index_pipeline(
    *,
    index_path: Path,
    collection_name: str,
    provider: EmbeddingProvider | None = None,
    settings: Settings | None = None,
    logger: logging.Logger | None = None,
) -> EmbedIndexPipeline:
    resolved_settings = (
        settings if settings is not None else Settings(docs_dir=Path("."))
    )
    resolved_provider = provider if provider is not None else HashEmbeddingProvider()
    policy = EmbeddingPreparationPolicy.from_settings(resolved_settings)
    service = EmbeddingService(provider=resolved_provider)
    schema = VectorStoreSchema(
        collection_name=collection_name,
        vector_index_schema_version=VECTOR_INDEX_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        model_name=resolved_provider.model_name,
        model_version=resolved_provider.model_version,
        dim=resolved_provider.dim,
    )
    writer = VectorIndexWriter(
        store=SQLiteVectorStore(index_path, collection_name=collection_name),
        schema=schema,
        preparer=EmbeddingTextPreparer(policy=policy),
        batcher=EmbeddingBatcher(service=service, policy=policy),
    )
    return EmbedIndexPipeline(
        writer=writer,
        index_path=index_path,
        collection_name=collection_name,
        provider=resolved_provider,
        logger=logger,
    )


def _default_settings() -> Settings:
    return Settings(docs_dir=Path("."))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    settings = _default_settings()
    parser = argparse.ArgumentParser(
        description="Embed Sprint 1 chunk artifacts and sync a local vector index."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=settings.chunk_output_path,
        help="Sprint 1 chunk JSONL artifact path",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=settings.vector_index_path,
        help="SQLite vector index path",
    )
    parser.add_argument(
        "--collection",
        default=settings.vector_collection_name,
        help="Vector index collection name",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = _parse_args(argv)
    output = stdout if stdout is not None else sys.stdout
    logger = get_logger(name="rag.embed_index", stream=sys.stderr)
    try:
        pipeline = build_embed_index_pipeline(
            index_path=args.index,
            collection_name=args.collection,
            logger=logger,
        )
        summary = pipeline.run(args.chunks)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "succeeded": False,
                    "error": str(exc),
                    "chunk_path": str(args.chunks),
                    "index_path": str(args.index),
                    "collection_name": args.collection,
                },
                sort_keys=True,
            ),
            file=output,
        )
        return 1

    print(summary.to_json(), file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
