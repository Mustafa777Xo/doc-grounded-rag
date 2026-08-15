from __future__ import annotations

import pytest

from rag.contracts.retrieval import QueryFilters
from rag.errors import RetrievalError
from rag.retrieve import QueryNormalizer


def test_query_normalizer_preserves_original_and_collapses_unicode_whitespace() -> None:
    original = "  POLICY\tcoverage\n\u00a0applies  "

    query = QueryNormalizer().normalize(
        query_id="query-1",
        original_text=original,
        filters=QueryFilters(doc_ids=frozenset({"doc-1"})),
    )

    assert query.original_text == original
    assert query.normalized_text == "policy coverage applies"
    assert query.filters.doc_ids == frozenset({"doc-1"})


def test_query_normalizer_applies_nfkc_and_unicode_casefold() -> None:
    normalizer = QueryNormalizer()

    compatibility = normalizer.normalize(
        query_id="query-1",
        original_text="ＦＵＬＬ－ＴＩＭＥ Straße",
    )
    composed = normalizer.normalize(
        query_id="query-2",
        original_text="CAFÉ",
    )
    decomposed = normalizer.normalize(
        query_id="query-3",
        original_text="Cafe\u0301",
    )

    assert compatibility.normalized_text == "full-time strasse"
    assert composed.normalized_text == decomposed.normalized_text == "café"


def test_query_normalizer_preserves_punctuation_without_expansion() -> None:
    query = QueryNormalizer().normalize(
        query_id="query-1",
        original_text="  WHAT?! -- Coverage...  ",
    )

    assert query.normalized_text == "what?! -- coverage..."


def test_query_normalizer_is_deterministic() -> None:
    normalizer = QueryNormalizer()

    first = normalizer.normalize(query_id="query-1", original_text="  Mixed CASE  ")
    second = normalizer.normalize(query_id="query-1", original_text="  Mixed CASE  ")

    assert first == second
    assert first.to_json() == second.to_json()


@pytest.mark.parametrize("text", ["", "   ", "\t\n\u00a0"])
def test_query_normalizer_rejects_empty_normalized_query(text: str) -> None:
    with pytest.raises(RetrievalError, match="empty after normalization") as exc_info:
        QueryNormalizer().normalize(query_id="query-1", original_text=text)

    assert exc_info.value.stage == "normalization"
    assert exc_info.value.query_id == "query-1"
