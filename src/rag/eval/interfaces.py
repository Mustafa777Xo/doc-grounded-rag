from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class EvalSummary:
    total_questions: int
    answered_questions: int
    grounded_answers: int


class Evaluator(Protocol):
    def evaluate(self, questions: Sequence[str]) -> EvalSummary: ...


class NoOpEvaluator:
    def evaluate(self, questions: Sequence[str]) -> EvalSummary:
        total_questions = len(tuple(questions))
        return EvalSummary(
            total_questions=total_questions,
            answered_questions=0,
            grounded_answers=0,
        )
