from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from rag.chunking import ChunkingPolicy, ChunkSplitter
from rag.config import Settings
from rag.contracts.chunk import Chunk
from rag.ingest.normalize import TextNormalizer
from rag.ingest.parser import PdfParser
from rag.logging import StageError, get_logger, new_correlation_id, stage_transition
from rag.storage import JsonlChunkWriter, WriteMode

DocumentStatus = Literal["succeeded", "failed"]


class IngestionPipelineError(RuntimeError):
    """Raised when the ingestion pipeline cannot complete successfully."""


@dataclass(frozen=True)
class DocumentIngestionSummary:
    source_path: str
    doc_id: str | None
    pages_processed: int
    chunks_produced: int
    status: DocumentStatus
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "doc_id": self.doc_id,
            "pages_processed": self.pages_processed,
            "chunks_produced": self.chunks_produced,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class IngestionRunSummary:
    input_path: str
    output_path: str
    write_mode: WriteMode
    documents_discovered: int
    documents_succeeded: int
    documents_failed: int
    pages_processed: int
    chunks_produced: int
    documents: tuple[DocumentIngestionSummary, ...]

    @property
    def succeeded(self) -> bool:
        return self.documents_failed == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "write_mode": self.write_mode,
            "documents_discovered": self.documents_discovered,
            "documents_succeeded": self.documents_succeeded,
            "documents_failed": self.documents_failed,
            "pages_processed": self.pages_processed,
            "chunks_produced": self.chunks_produced,
            "succeeded": self.succeeded,
            "documents": [document.to_dict() for document in self.documents],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class IngestionPipeline:
    def __init__(
        self,
        *,
        parser: PdfParser | None = None,
        normalizer: TextNormalizer | None = None,
        splitter: ChunkSplitter | None = None,
        writer: JsonlChunkWriter,
        logger: logging.Logger | None = None,
        correlation_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._parser = parser if parser is not None else PdfParser()
        self._normalizer = normalizer if normalizer is not None else TextNormalizer()
        self._splitter = splitter if splitter is not None else ChunkSplitter()
        self._writer = writer
        self._logger = logger if logger is not None else get_logger(name="rag.ingest")
        self._correlation_id_factory = (
            correlation_id_factory
            if correlation_id_factory is not None
            else new_correlation_id
        )

    def run(self, input_path: Path) -> IngestionRunSummary:
        source_paths = discover_pdf_inputs(input_path)
        correlation_id = self._new_correlation_id()

        all_chunks: list[Chunk] = []
        document_summaries: list[DocumentIngestionSummary] = []
        for source_path in source_paths:
            summary, chunks = self._process_document(
                source_path=source_path,
                correlation_id=correlation_id,
            )
            document_summaries.append(summary)
            all_chunks.extend(chunks)

        if all_chunks:
            with stage_transition(
                logger=self._logger,
                stage="persist",
                correlation_id=correlation_id,
            ):
                self._writer.write(tuple(all_chunks))

        documents_succeeded = sum(
            1 for summary in document_summaries if summary.status == "succeeded"
        )
        documents_failed = len(document_summaries) - documents_succeeded
        return IngestionRunSummary(
            input_path=str(input_path),
            output_path=str(self._writer.output_path),
            write_mode=self._writer.write_mode,
            documents_discovered=len(source_paths),
            documents_succeeded=documents_succeeded,
            documents_failed=documents_failed,
            pages_processed=sum(
                summary.pages_processed for summary in document_summaries
            ),
            chunks_produced=sum(
                summary.chunks_produced for summary in document_summaries
            ),
            documents=tuple(document_summaries),
        )

    def _process_document(
        self,
        *,
        source_path: Path,
        correlation_id: str,
    ) -> tuple[DocumentIngestionSummary, tuple[Chunk, ...]]:
        doc_id: str | None = None
        pages_processed = 0
        try:
            with stage_transition(
                logger=self._logger,
                stage="parse",
                correlation_id=correlation_id,
            ):
                document = self._parser.parse(source_path)
                doc_id = document.doc_id
                pages_processed = document.total_pages

            with stage_transition(
                logger=self._logger,
                stage="normalize",
                correlation_id=correlation_id,
            ):
                normalized = self._normalizer.normalize_document(document)

            with stage_transition(
                logger=self._logger,
                stage="chunk",
                correlation_id=correlation_id,
            ):
                chunks = self._splitter.split_document(normalized)

        except StageError as exc:
            return (
                DocumentIngestionSummary(
                    source_path=str(source_path),
                    doc_id=doc_id,
                    pages_processed=pages_processed,
                    chunks_produced=0,
                    status="failed",
                    error=str(exc),
                ),
                (),
            )

        return (
            DocumentIngestionSummary(
                source_path=str(source_path),
                doc_id=doc_id,
                pages_processed=pages_processed,
                chunks_produced=len(chunks),
                status="succeeded",
            ),
            chunks,
        )

    def _new_correlation_id(self) -> str:
        return str(self._correlation_id_factory())


def discover_pdf_inputs(input_path: Path) -> tuple[Path, ...]:
    if not input_path.exists():
        raise IngestionPipelineError(f"input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise IngestionPipelineError(f"input file must be a PDF: {input_path}")
        return (input_path,)
    if input_path.is_dir():
        pdfs = tuple(
            sorted(
                (
                    path
                    for path in input_path.iterdir()
                    if path.is_file() and path.suffix.lower() == ".pdf"
                ),
                key=lambda path: path.name,
            )
        )
        if not pdfs:
            raise IngestionPipelineError(f"input directory has no PDFs: {input_path}")
        return pdfs
    raise IngestionPipelineError(
        f"input path must be a PDF file or directory: {input_path}"
    )


def build_ingestion_pipeline(
    *,
    output_path: Path,
    write_mode: WriteMode,
    logger: logging.Logger | None = None,
) -> IngestionPipeline:
    settings = Settings(
        docs_dir=Path("."),
        chunk_output_path=output_path,
        chunk_write_mode=write_mode,
    )
    return IngestionPipeline(
        splitter=ChunkSplitter(policy=ChunkingPolicy.from_settings(settings)),
        writer=JsonlChunkWriter.from_settings(settings),
        logger=logger,
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse PDFs, normalize text, chunk content, and persist JSONL chunks."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="PDF file or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Settings(docs_dir=Path(".")).chunk_output_path,
        help="JSONL output path",
    )
    parser.add_argument(
        "--mode",
        choices=("overwrite", "safe_append"),
        default=Settings(docs_dir=Path(".")).chunk_write_mode,
        help="Chunk artifact write mode",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    args = _parse_args(argv)
    output = stdout if stdout is not None else sys.stdout
    logger = get_logger(name="rag.ingest", stream=sys.stderr)
    try:
        pipeline = build_ingestion_pipeline(
            output_path=args.output,
            write_mode=args.mode,
            logger=logger,
        )
        summary = pipeline.run(args.input)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "succeeded": False,
                    "error": str(exc),
                    "input_path": str(args.input),
                    "output_path": str(args.output),
                    "write_mode": args.mode,
                },
                sort_keys=True,
            ),
            file=output,
        )
        return 1

    print(summary.to_json(), file=output)
    return 0 if summary.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
