from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from rag.contracts.answer import AnswerWithCitations
from rag.contracts.chunk import Chunk
from rag.contracts.document import Document
from rag.generate.interfaces import Generator, NoOpGenerator
from rag.index.interfaces import Indexer, NoOpIndexer
from rag.ingest.interfaces import Ingestor, NoOpIngestor
from rag.logging import get_logger, new_correlation_id, stage_transition
from rag.rerank.interfaces import NoOpReranker, Reranker
from rag.retrieve.interfaces import NoOpRetriever, Retriever

T = TypeVar("T")

DEFAULT_NOOP_QUESTION = "What does the mock document say?"
DEFAULT_NOOP_SOURCES = (Path("mock-company-handbook.pdf"),)


RetrieverFactory = Callable[[Sequence[Chunk]], Retriever]


@dataclass(frozen=True)
class NoOpPipeline:
    ingestor: Ingestor
    indexer: Indexer
    retriever_factory: RetrieverFactory
    reranker: Reranker
    generator: Generator
    retrieve_limit: int = 5
    rerank_limit: int | None = None
    correlation_id_factory: Callable[[], str] = new_correlation_id
    logger: logging.Logger = field(
        default_factory=lambda: get_logger(name="rag.pipeline")
    )

    def run(
        self,
        *,
        question: str,
        sources: Sequence[Path],
    ) -> AnswerWithCitations:
        correlation_id = self.correlation_id_factory()

        documents = self._run_stage(
            stage="ingest",
            correlation_id=correlation_id,
            action=lambda: self.ingestor.ingest(sources),
        )
        chunks = self._run_stage(
            stage="chunk",
            correlation_id=correlation_id,
            action=lambda: documents_to_noop_chunks(documents),
        )
        self._run_stage(
            stage="index",
            correlation_id=correlation_id,
            action=lambda: self.indexer.index(chunks),
        )
        retriever = self.retriever_factory(chunks)
        retrieved = self._run_stage(
            stage="retrieve",
            correlation_id=correlation_id,
            action=lambda: retriever.retrieve(question, limit=self.retrieve_limit),
        )
        reranked = self._run_stage(
            stage="rerank",
            correlation_id=correlation_id,
            action=lambda: self.reranker.rerank(retrieved, limit=self.rerank_limit),
        )
        return self._run_stage(
            stage="generate",
            correlation_id=correlation_id,
            action=lambda: self.generator.generate(question, reranked),
        )

    def _run_stage(
        self,
        *,
        stage: str,
        correlation_id: str,
        action: Callable[[], T],
    ) -> T:
        with stage_transition(
            logger=self.logger,
            stage=stage,
            correlation_id=correlation_id,
        ):
            return action()


def documents_to_noop_chunks(documents: Sequence[Document]) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    for document in documents:
        for page in document.pages:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}-chunk-{page.page_number}",
                    doc_id=document.doc_id,
                    source_file=document.source_file,
                    page=page.page_number,
                    chunk_index=page.page_number,
                    char_start=0,
                    char_end=len(page.text),
                    text=page.text,
                )
            )
    return tuple(chunks)


def build_noop_pipeline() -> NoOpPipeline:
    return NoOpPipeline(
        ingestor=NoOpIngestor(),
        indexer=NoOpIndexer(),
        retriever_factory=NoOpRetriever,
        reranker=NoOpReranker(),
        generator=NoOpGenerator(),
    )


def run_noop() -> AnswerWithCitations:
    pipeline = build_noop_pipeline()
    return pipeline.run(
        question=DEFAULT_NOOP_QUESTION,
        sources=DEFAULT_NOOP_SOURCES,
    )


def main() -> None:
    answer = run_noop()
    print(answer.to_json())


if __name__ == "__main__":
    main()
