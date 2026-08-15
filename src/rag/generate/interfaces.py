from __future__ import annotations

from typing import Protocol, Sequence

from rag.contracts.answer import AnswerWithCitations, Citation
from rag.contracts.retrieval import RetrievalCandidate, RetrievalQuery


class Generator(Protocol):
    def generate(
        self, query: RetrievalQuery, context: Sequence[RetrievalCandidate]
    ) -> AnswerWithCitations: ...


class NoOpGenerator:
    def generate(
        self, query: RetrievalQuery, context: Sequence[RetrievalCandidate]
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
            answer_text=f"No-op grounded answer for query: {query.original_text}",
            citations=(citation,),
            grounded=True,
        )
