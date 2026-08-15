from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from rag.contracts.retrieval import RetrievalQuery


@dataclass(frozen=True)
class EvalSummary:
    total_questions: int
    answered_questions: int
    grounded_answers: int


class Evaluator(Protocol):
    def evaluate(self, queries: Sequence[RetrievalQuery]) -> EvalSummary: ...


class NoOpEvaluator:
    def evaluate(self, queries: Sequence[RetrievalQuery]) -> EvalSummary:
        total_questions = len(tuple(queries))
        return EvalSummary(
            total_questions=total_questions,
            answered_questions=0,
            grounded_answers=0,
        )
