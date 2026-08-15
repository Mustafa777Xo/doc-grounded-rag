from __future__ import annotations

from pathlib import Path

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import (
    KeywordDiagnostics,
    RetrievalCandidate,
    RetrievalQuery,
    RetrieverSource,
    ScoreProvenance,
)
from rag.eval.interfaces import Evaluator, NoOpEvaluator
from rag.generate.interfaces import Generator, NoOpGenerator
from rag.index.interfaces import Indexer, NoOpIndexer
from rag.ingest.interfaces import Ingestor, NoOpIngestor
from rag.rerank.interfaces import NoOpReranker, Reranker
from rag.retrieve.interfaces import NoOpRetriever, Retriever


def _accept_ingestor(_: Ingestor) -> None:
    return None


def _accept_indexer(_: Indexer) -> None:
    return None


def _accept_retriever(_: Retriever) -> None:
    return None


def _accept_reranker(_: Reranker) -> None:
    return None


def _accept_generator(_: Generator) -> None:
    return None


def _accept_evaluator(_: Evaluator) -> None:
    return None


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-1",
        original_text="policy coverage",
        normalized_text="policy coverage",
    )


def _make_chunks() -> tuple[Chunk, ...]:
    return (
        Chunk(
            chunk_id="c-1",
            doc_id="d-1",
            source_file="a.pdf",
            page=0,
            chunk_index=0,
            char_start=0,
            char_end=5,
            text="alpha",
        ),
        Chunk(
            chunk_id="c-2",
            doc_id="d-2",
            source_file="b.pdf",
            page=1,
            chunk_index=0,
            char_start=0,
            char_end=4,
            text="beta",
        ),
        Chunk(
            chunk_id="c-3",
            doc_id="d-3",
            source_file="c.pdf",
            page=2,
            chunk_index=0,
            char_start=0,
            char_end=5,
            text="gamma",
        ),
    )


def test_noop_adapters_satisfy_protocol_types() -> None:
    _accept_ingestor(NoOpIngestor())
    _accept_indexer(NoOpIndexer())
    _accept_retriever(NoOpRetriever(corpus=_make_chunks()))
    _accept_reranker(NoOpReranker())
    _accept_generator(NoOpGenerator())
    _accept_evaluator(NoOpEvaluator())


def test_noop_ingestor_is_deterministic() -> None:
    ingestor = NoOpIngestor()
    sources = (Path("b.pdf"), Path("a.pdf"))
    first = ingestor.ingest(sources)
    second = ingestor.ingest(sources)
    assert first == second
    assert tuple(doc.source_file for doc in first) == ("a.pdf", "b.pdf")


def test_noop_indexer_accepts_chunks_without_side_effects() -> None:
    indexer = NoOpIndexer()
    chunks = _make_chunks()
    before = chunks
    indexer.index(chunks)
    assert chunks == before


def test_noop_retriever_is_deterministic_and_respects_limit() -> None:
    retriever = NoOpRetriever(corpus=_make_chunks())
    first = retriever.retrieve(query=_query(), limit=2)
    second = retriever.retrieve(query=_query(), limit=2)
    assert first == second
    assert len(first) == 2
    assert all(isinstance(item, RetrievalCandidate) for item in first)
    assert all(item.sources == frozenset({RetrieverSource.DENSE}) for item in first)


def test_noop_reranker_is_deterministic() -> None:
    reranker = NoOpReranker()
    chunk_a, chunk_b, _ = _make_chunks()
    input_results = (
        RetrievalCandidate(
            chunk=chunk_a,
            scores=ScoreProvenance(dense_score=0.2),
            sources=frozenset({RetrieverSource.DENSE}),
        ),
        RetrievalCandidate(
            chunk=chunk_b,
            scores=ScoreProvenance(keyword_score=0.9),
            sources=frozenset({RetrieverSource.KEYWORD}),
            keyword_diagnostics=KeywordDiagnostics(
                source_rank=1,
                matched_terms=("policy",),
            ),
        ),
    )
    first = reranker.rerank(_query(), input_results)
    second = reranker.rerank(_query(), input_results)
    assert first == second
    assert len(first) == len(input_results)
    assert tuple(result.rank for result in first) == (1, 2)
    assert {r.chunk.chunk_id for r in first} == {"c-1", "c-2"}


def test_noop_generator_no_evidence_response() -> None:
    generator = NoOpGenerator()
    answer = generator.generate(query=_query(), context=())
    assert answer.grounded is False
    assert answer.citations == ()
    assert "not available" in answer.answer_text.lower()


def test_noop_generator_grounded_response_is_deterministic() -> None:
    generator = NoOpGenerator()
    chunk_a, _, _ = _make_chunks()
    context = (
        RetrievalCandidate(
            chunk=chunk_a,
            scores=ScoreProvenance(dense_score=1.0),
            sources=frozenset({RetrieverSource.DENSE}),
        ),
    )
    first = generator.generate(query=_query(), context=context)
    second = generator.generate(query=_query(), context=context)
    assert first == second
    assert first.grounded is True
    assert len(first.citations) == 1
    assert first.citations[0].source_file == "a.pdf"
    assert first.citations[0].page == 0
    assert first.citations[0].chunk_index == 0


def test_noop_evaluator_accepts_typed_queries() -> None:
    summary = NoOpEvaluator().evaluate((_query(),))
    assert summary.total_questions == 1
