from __future__ import annotations

import pytest

from rag.errors import RerankingError, RetrievalError, RetrievalStage
from rag.logging import RagError


@pytest.mark.parametrize(
    "stage",
    ["normalization", "dense", "keyword", "fusion"],
)
def test_retrieval_error_preserves_stage_and_query(
    stage: RetrievalStage,
) -> None:
    error = RetrievalError(
        stage=stage,
        query_id="query-1",
        message="stage failed",
        hint="check stage input",
    )

    assert isinstance(error, RagError)
    assert error.stage == stage
    assert error.query_id == "query-1"
    assert error.stage_message == "stage failed"
    assert error.hint == "check stage input"
    assert f"stage={stage}" in str(error)
    assert "query_id=query-1" in str(error)
    assert "hint=check stage input" in str(error)


def test_reranking_error_preserves_stage_and_query() -> None:
    error = RerankingError(
        query_id="query-2",
        message="model unavailable",
        hint="prefetch the pinned model",
    )

    assert isinstance(error, RagError)
    assert error.stage == "reranking"
    assert error.query_id == "query-2"
    assert error.stage_message == "model unavailable"
    assert "stage=reranking" in str(error)
    assert "hint=prefetch the pinned model" in str(error)
