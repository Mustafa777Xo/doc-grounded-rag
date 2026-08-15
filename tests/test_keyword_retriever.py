from __future__ import annotations

import math

import pytest

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import QueryFilters, RetrievalQuery, RetrieverSource
from rag.errors import RetrievalError
from rag.retrieve import BM25KeywordRetriever, KeywordIndexError, QueryNormalizer
from rag.retrieve.keyword import KeywordScoreEngine, tokenize_lexical_text


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "doc-1",
    source_file: str = "policy.pdf",
    page: int = 0,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=source_file,
        page=page,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        text=text,
    )


def _query(
    text: str,
    *,
    filters: QueryFilters | None = None,
) -> RetrievalQuery:
    return QueryNormalizer().normalize(
        query_id="query-1",
        original_text=text,
        filters=filters,
    )


class _FakeScoreEngine:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores
        self.queries: list[tuple[str, ...]] = []

    def score(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]:
        self.queries.append(query_tokens)
        return self.scores


def test_lexical_tokenization_policy_covers_unicode_acronyms_and_punctuation() -> None:
    assert tokenize_lexical_text(" ＦＭＬＡ 401(k) C++ A-B Straße ") == (
        "fmla",
        "401",
        "k",
        "c",
        "a",
        "b",
        "strasse",
    )


def test_keyword_retriever_enforces_filters_top_k_ties_and_diagnostics() -> None:
    chunks = (
        _chunk("chunk-c", "PTO excluded", doc_id="other", page=2),
        _chunk(
            "chunk-b",
            "PTO enrollment",
            doc_id="allowed",
            source_file="benefits.pdf",
            page=2,
            chunk_index=2,
        ),
        _chunk(
            "chunk-a",
            "PTO policy details",
            doc_id="allowed",
            source_file="benefits.pdf",
            page=2,
            chunk_index=1,
        ),
    )
    engine = _FakeScoreEngine((0.8, 0.8, 0.99))
    indexed_tokens: list[tuple[tuple[str, ...], ...]] = []

    def factory(
        corpus_tokens: tuple[tuple[str, ...], ...],
    ) -> KeywordScoreEngine:
        indexed_tokens.append(corpus_tokens)
        return engine

    retriever = BM25KeywordRetriever(chunks, score_engine_factory=factory)
    filters = QueryFilters(
        doc_ids=frozenset({"allowed", "second-allowed"}),
        source_files=frozenset({"benefits.pdf"}),
        pages=frozenset({2}),
    )

    results = retriever.retrieve(_query("pto POLICY", filters=filters), limit=2)

    assert indexed_tokens == [
        (
            ("pto", "policy", "details"),
            ("pto", "enrollment"),
            ("pto", "excluded"),
        )
    ]
    assert engine.queries == [("pto", "policy")]
    assert tuple(result.chunk.chunk_id for result in results) == (
        "chunk-a",
        "chunk-b",
    )
    assert tuple(result.rank for result in results) == (1, 2)
    assert results[0].chunk.to_dict() == chunks[2].to_dict()
    assert results[0].scores.keyword_score == pytest.approx(0.8)
    assert results[0].sources == frozenset({RetrieverSource.KEYWORD})
    assert results[0].keyword_diagnostics is not None
    assert results[0].keyword_diagnostics.source_rank == 1
    assert results[0].keyword_diagnostics.matched_terms == ("policy", "pto")
    assert results[1].keyword_diagnostics is not None
    assert results[1].keyword_diagnostics.matched_terms == ("pto",)


@pytest.mark.parametrize("query_text", ["!!! -- ...", "unknown-term"])
def test_keyword_retriever_returns_empty_for_tokenless_or_oov_queries(
    query_text: str,
) -> None:
    retriever = BM25KeywordRetriever((_chunk("chunk-a", "policy coverage"),))

    assert retriever.retrieve(_query(query_text), limit=5) == ()


def test_keyword_retriever_returns_empty_for_empty_corpus() -> None:
    retriever = BM25KeywordRetriever(())

    assert retriever.retrieve(_query("policy"), limit=5) == ()


def test_keyword_retriever_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(KeywordIndexError, match="duplicate"):
        BM25KeywordRetriever(
            (
                _chunk("duplicate", "first"),
                _chunk("duplicate", "second"),
            )
        )


def test_keyword_retriever_wraps_invalid_limit_and_backend_failures() -> None:
    class FailingEngine:
        def score(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]:
            _ = query_tokens
            raise RuntimeError("scorer unavailable")

    retriever = BM25KeywordRetriever(
        (_chunk("chunk-a", "policy"),),
        score_engine_factory=lambda _tokens: FailingEngine(),
    )

    with pytest.raises(RetrievalError, match="limit") as limit_error:
        retriever.retrieve(_query("policy"), limit=0)
    assert limit_error.value.stage == "keyword"

    with pytest.raises(RetrievalError, match="scorer unavailable") as score_error:
        retriever.retrieve(_query("policy"), limit=1)
    assert score_error.value.stage == "keyword"
    assert score_error.value.query_id == "query-1"


@pytest.mark.parametrize("scores", [(), (math.nan,)])
def test_keyword_retriever_rejects_invalid_backend_scores(
    scores: tuple[float, ...],
) -> None:
    retriever = BM25KeywordRetriever(
        (_chunk("chunk-a", "policy"),),
        score_engine_factory=lambda _tokens: _FakeScoreEngine(scores),
    )

    with pytest.raises(RetrievalError):
        retriever.retrieve(_query("policy"), limit=1)


def test_real_bm25_ranks_exact_terms_and_acronyms_reproducibly() -> None:
    chunks = (
        _chunk(
            "leave-policy",
            "Employees may request FMLA leave for qualifying family events.",
        ),
        _chunk(
            "vacation-policy",
            "Employees accrue PTO for vacations and personal days.",
        ),
        _chunk(
            "claims-policy",
            "Insurance claims must be filed within thirty days.",
        ),
    )
    first_retriever = BM25KeywordRetriever(chunks)
    second_retriever = BM25KeywordRetriever(tuple(reversed(chunks)))

    first = first_retriever.retrieve(_query("fmla"), limit=3)
    second = second_retriever.retrieve(_query("FMLA"), limit=3)

    assert first == second
    assert tuple(result.chunk.chunk_id for result in first) == ("leave-policy",)
    assert first[0].scores.keyword_score is not None
    assert first[0].scores.keyword_score > 0.0
    assert first[0].keyword_diagnostics is not None
    assert first[0].keyword_diagnostics.matched_terms == ("fmla",)


def test_rebuilding_from_snapshot_removes_changed_and_deleted_terms() -> None:
    original = BM25KeywordRetriever(
        (
            _chunk("changed", "legacyterm policy"),
            _chunk("deleted", "legacyterm archive"),
        )
    )
    rebuilt = BM25KeywordRetriever(
        (
            _chunk(
                "changed",
                "replacementterm policy",
                source_file="updated.pdf",
                page=4,
            ),
        )
    )

    assert {
        result.chunk.chunk_id for result in original.retrieve(_query("legacyterm"))
    } == {
        "changed",
        "deleted",
    }
    assert rebuilt.retrieve(_query("legacyterm")) == ()
    replacement = rebuilt.retrieve(_query("replacementterm"))
    assert tuple(result.chunk.chunk_id for result in replacement) == ("changed",)
    assert replacement[0].chunk.source_file == "updated.pdf"
    assert replacement[0].chunk.page == 4
