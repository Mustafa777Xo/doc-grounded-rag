from __future__ import annotations

from typing import Protocol, Sequence

from rag.contracts.answer import AnswerWithCitations, Citation
from rag.contracts.retrieval import RetrievalResult


class Generator(Protocol):
    def generate(
        self, query: str, context: Sequence[RetrievalResult]
    ) -> AnswerWithCitations: ...


class NoOpGenerator:
    def generate(
        self, query: str, context: Sequence[RetrievalResult]
    ) -> AnswerWithCitations:
        if not context:
            return AnswerWithCitations(
                answer_text="The answer is not available in the provided documents.",
                citations=(),
                grounded=False,
            )
        top = context[0].chunk
        citation = Citation(
            source_file=top.source_file,
            page=top.page,
            chunk_index=top.chunk_index,
        )
        return AnswerWithCitations(
            answer_text=f"No-op grounded answer for query: {query}",
            citations=(citation,),
            grounded=True,
        )
