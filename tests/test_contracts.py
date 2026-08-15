# file: tests/test_contracts.py
from __future__ import annotations

import json
import math

import pytest

from rag.contracts.answer import AnswerWithCitations, Citation
from rag.contracts.chunk import Chunk
from rag.contracts.document import Document, ParsedPage
from rag.contracts.indexing import (
    CHUNK_SCHEMA_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    VECTOR_INDEX_SCHEMA_VERSION,
    EmbeddingRecord,
    VectorIndexMetadata,
    VectorIndexRow,
)
from rag.contracts.retrieval import (
    RETRIEVAL_CANDIDATE_SCHEMA_VERSION,
    RETRIEVAL_QUERY_SCHEMA_VERSION,
    KeywordDiagnostics,
    QueryFilters,
    RetrievalCandidate,
    RetrievalQuery,
    RetrieverSource,
    ScoreProvenance,
)

# ── ParsedPage ────────────────────────────────────────────────────────────────


def _make_page(
    *,
    doc_id: str = "d1",
    source_file: str = "a.pdf",
    page_number: int = 0,
    text: str = "hello",
) -> ParsedPage:
    return ParsedPage(
        doc_id=doc_id,
        source_file=source_file,
        page_number=page_number,
        text=text,
    )


def test_parsed_page_valid() -> None:
    page = _make_page()
    assert page.doc_id == "d1"
    assert page.source_file == "a.pdf"
    assert page.page_number == 0
    assert page.text == "hello"


def test_parsed_page_allows_empty_text() -> None:
    page = _make_page(text="")
    assert page.text == ""


def test_parsed_page_rejects_empty_doc_id() -> None:
    with pytest.raises(ValueError, match="doc_id"):
        _make_page(doc_id="")


def test_parsed_page_rejects_empty_source_file() -> None:
    with pytest.raises(ValueError, match="source_file"):
        _make_page(source_file="")


def test_parsed_page_rejects_negative_page_number() -> None:
    with pytest.raises(ValueError, match="page_number"):
        _make_page(page_number=-1)


def test_parsed_page_to_json_round_trip() -> None:
    parsed = json.loads(_make_page().to_json())
    assert parsed["doc_id"] == "d1"
    assert parsed["source_file"] == "a.pdf"
    assert parsed["page_number"] == 0
    assert parsed["text"] == "hello"


# ── Document ──────────────────────────────────────────────────────────────────


def test_document_valid() -> None:
    doc = Document(
        doc_id="d1",
        source_file="a.pdf",
        source_path="data/a.pdf",
        total_pages=1,
        pages=(_make_page(),),
    )
    assert doc.doc_id == "d1"
    assert doc.source_file == "a.pdf"
    assert doc.source_path == "data/a.pdf"
    assert doc.total_pages == 1
    assert doc.pages == (_make_page(),)


def test_document_rejects_empty_doc_id() -> None:
    with pytest.raises(ValueError, match="doc_id"):
        Document(
            doc_id="",
            source_file="a.pdf",
            source_path="data/a.pdf",
            total_pages=1,
            pages=(_make_page(),),
        )


def test_document_rejects_empty_source_file() -> None:
    with pytest.raises(ValueError, match="source_file"):
        Document(
            doc_id="d1",
            source_file="",
            source_path="data/a.pdf",
            total_pages=1,
            pages=(_make_page(),),
        )


def test_document_rejects_empty_source_path() -> None:
    with pytest.raises(ValueError, match="source_path"):
        Document(
            doc_id="d1",
            source_file="a.pdf",
            source_path="",
            total_pages=1,
            pages=(_make_page(),),
        )


def test_document_rejects_invalid_total_pages() -> None:
    with pytest.raises(ValueError, match="total_pages"):
        Document(
            doc_id="d1",
            source_file="a.pdf",
            source_path="data/a.pdf",
            total_pages=0,
            pages=(_make_page(),),
        )


def test_document_rejects_empty_pages() -> None:
    with pytest.raises(ValueError, match="pages"):
        Document(
            doc_id="d1",
            source_file="a.pdf",
            source_path="data/a.pdf",
            total_pages=1,
            pages=(),
        )


def test_document_rejects_total_pages_mismatch() -> None:
    with pytest.raises(ValueError, match="total_pages"):
        Document(
            doc_id="d1",
            source_file="a.pdf",
            source_path="data/a.pdf",
            total_pages=2,
            pages=(_make_page(),),
        )


def test_document_rejects_mismatched_page_doc_id() -> None:
    with pytest.raises(ValueError, match="doc_id"):
        Document(
            doc_id="d1",
            source_file="a.pdf",
            source_path="data/a.pdf",
            total_pages=1,
            pages=(_make_page(doc_id="other"),),
        )


def test_document_rejects_mismatched_page_source_file() -> None:
    with pytest.raises(ValueError, match="source_file"):
        Document(
            doc_id="d1",
            source_file="a.pdf",
            source_path="data/a.pdf",
            total_pages=1,
            pages=(_make_page(source_file="other.pdf"),),
        )


def test_document_rejects_non_contiguous_pages() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        Document(
            doc_id="d1",
            source_file="a.pdf",
            source_path="data/a.pdf",
            total_pages=1,
            pages=(_make_page(page_number=1),),
        )


def test_document_to_dict() -> None:
    doc = Document(
        doc_id="d1",
        source_file="a.pdf",
        source_path="data/a.pdf",
        total_pages=2,
        pages=(
            _make_page(page_number=0, text="page one"),
            _make_page(page_number=1, text="page two"),
        ),
    )
    result = doc.to_dict()
    assert result["doc_id"] == "d1"
    assert result["source_file"] == "a.pdf"
    assert result["source_path"] == "data/a.pdf"
    assert result["total_pages"] == 2
    assert result["pages"] == [
        {
            "doc_id": "d1",
            "source_file": "a.pdf",
            "page_number": 0,
            "text": "page one",
        },
        {
            "doc_id": "d1",
            "source_file": "a.pdf",
            "page_number": 1,
            "text": "page two",
        },
    ]


def test_document_to_json_round_trip() -> None:
    doc = Document(
        doc_id="d1",
        source_file="a.pdf",
        source_path="data/a.pdf",
        total_pages=1,
        pages=(_make_page(),),
    )
    parsed = json.loads(doc.to_json())
    assert parsed["doc_id"] == "d1"
    assert parsed["source_path"] == "data/a.pdf"
    assert parsed["total_pages"] == 1
    assert parsed["pages"][0]["text"] == "hello"


def test_document_is_immutable() -> None:
    doc = Document(
        doc_id="d1",
        source_file="a.pdf",
        source_path="data/a.pdf",
        total_pages=1,
        pages=(_make_page(),),
    )
    with pytest.raises(Exception):
        doc.doc_id = "other"  # type: ignore[misc]


# ── Chunk ─────────────────────────────────────────────────────────────────────


def test_chunk_valid() -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_file="a.pdf",
        page=0,
        chunk_index=0,
        char_start=0,
        char_end=9,
        text="some text",
    )
    assert chunk.chunk_id == "c1"
    assert chunk.chunk_index == 0
    assert chunk.char_start == 0
    assert chunk.char_end == 9


def test_chunk_is_hashable() -> None:
    # frozen=True dataclasses must be hashable — required for sets and dict keys
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_file="a.pdf",
        page=0,
        chunk_index=0,
        char_start=0,
        char_end=9,
        text="some text",
    )
    assert hash(chunk) is not None
    chunk_set = {chunk}
    assert len(chunk_set) == 1


def test_chunk_is_immutable() -> None:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_file="a.pdf",
        page=0,
        chunk_index=0,
        char_start=0,
        char_end=9,
        text="some text",
    )
    with pytest.raises(Exception):
        chunk.text = "other"  # type: ignore[misc]


# ── Retrieval contracts ──────────────────────────────────────────────────────


def _make_chunk() -> Chunk:
    return Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_file="a.pdf",
        page=0,
        chunk_index=0,
        char_start=0,
        char_end=9,
        text="some text",
    )


def test_chunk_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        Chunk(
            chunk_id="",
            doc_id="d1",
            source_file="a.pdf",
            page=0,
            chunk_index=0,
            char_start=0,
            char_end=4,
            text="text",
        )


def test_chunk_rejects_negative_page() -> None:
    with pytest.raises(ValueError, match="page"):
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            source_file="a.pdf",
            page=-1,
            chunk_index=0,
            char_start=0,
            char_end=4,
            text="text",
        )


def test_chunk_rejects_negative_chunk_index() -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            source_file="a.pdf",
            page=0,
            chunk_index=-1,
            char_start=0,
            char_end=4,
            text="text",
        )


def test_chunk_rejects_invalid_span() -> None:
    with pytest.raises(ValueError, match="char_end"):
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            source_file="a.pdf",
            page=0,
            chunk_index=0,
            char_start=4,
            char_end=4,
            text="text",
        )


def test_chunk_rejects_span_text_length_mismatch() -> None:
    with pytest.raises(ValueError, match="text length"):
        Chunk(
            chunk_id="c1",
            doc_id="d1",
            source_file="a.pdf",
            page=0,
            chunk_index=0,
            char_start=2,
            char_end=10,
            text="text",
        )


def test_retrieval_candidate_holds_complete_chunk() -> None:
    chunk = _make_chunk()
    candidate = RetrievalCandidate(
        chunk=chunk,
        scores=ScoreProvenance(dense_score=0.87),
        sources=frozenset({RetrieverSource.DENSE}),
    )
    assert candidate.chunk is chunk
    assert candidate.scores.dense_score == 0.87
    assert candidate.sources == frozenset({RetrieverSource.DENSE})


def test_retrieval_candidate_is_immutable() -> None:
    candidate = RetrievalCandidate(
        chunk=_make_chunk(),
        scores=ScoreProvenance(keyword_score=0.5),
        sources=frozenset({RetrieverSource.KEYWORD}),
        keyword_diagnostics=KeywordDiagnostics(
            source_rank=1,
            matched_terms=("policy",),
        ),
    )
    with pytest.raises(Exception):
        candidate.rank = 1  # type: ignore[misc]


def test_retrieval_query_rejects_blank_fields() -> None:
    with pytest.raises(ValueError, match="query_id"):
        RetrievalQuery(query_id="", original_text="query", normalized_text="query")
    with pytest.raises(ValueError, match="original_text"):
        RetrievalQuery(query_id="q-1", original_text=" ", normalized_text="query")
    with pytest.raises(ValueError, match="normalized_text"):
        RetrievalQuery(query_id="q-1", original_text="query", normalized_text=" ")


def test_query_filters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="doc_ids"):
        QueryFilters(doc_ids=frozenset({""}))
    with pytest.raises(ValueError, match="source_files"):
        QueryFilters(source_files=frozenset({""}))
    with pytest.raises(ValueError, match="pages"):
        QueryFilters(pages=frozenset({-1}))


@pytest.mark.parametrize("invalid_score", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("score_name", ["dense", "keyword", "fusion", "rerank"])
def test_score_provenance_rejects_non_finite_scores(
    score_name: str,
    invalid_score: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _score_provenance_with(score_name, invalid_score)


def _score_provenance_with(score_name: str, value: float) -> ScoreProvenance:
    if score_name == "dense":
        return ScoreProvenance(dense_score=value)
    if score_name == "keyword":
        return ScoreProvenance(keyword_score=value)
    if score_name == "fusion":
        return ScoreProvenance(dense_score=0.5, fusion_score=value)
    return ScoreProvenance(dense_score=0.5, rerank_score=value)


def test_score_provenance_requires_retriever_score() -> None:
    with pytest.raises(ValueError, match="retriever score"):
        ScoreProvenance(fusion_score=0.1)


def test_candidate_accepts_scores_from_both_retrievers() -> None:
    candidate = RetrievalCandidate(
        chunk=_make_chunk(),
        scores=ScoreProvenance(
            dense_score=0.72,
            keyword_score=3.4,
            fusion_score=0.03,
        ),
        sources=frozenset({RetrieverSource.DENSE, RetrieverSource.KEYWORD}),
        keyword_diagnostics=KeywordDiagnostics(
            source_rank=2,
            matched_terms=("coverage", "policy"),
        ),
    )
    assert candidate.scores.dense_score == 0.72
    assert candidate.scores.keyword_score == 3.4


def test_candidate_rejects_inconsistent_sources_and_scores() -> None:
    with pytest.raises(ValueError, match="dense source"):
        RetrievalCandidate(
            chunk=_make_chunk(),
            scores=ScoreProvenance(dense_score=0.5),
            sources=frozenset({RetrieverSource.KEYWORD}),
        )


def test_candidate_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        RetrievalCandidate(
            chunk=_make_chunk(),
            scores=ScoreProvenance(dense_score=0.5),
            sources=frozenset({RetrieverSource.DENSE}),
            rank=0,
        )


def test_keyword_diagnostics_require_canonical_terms_and_positive_rank() -> None:
    with pytest.raises(ValueError, match="source_rank"):
        KeywordDiagnostics(source_rank=0, matched_terms=("policy",))
    with pytest.raises(ValueError, match="cannot be empty"):
        KeywordDiagnostics(source_rank=1, matched_terms=())
    with pytest.raises(ValueError, match="sorted and unique"):
        KeywordDiagnostics(source_rank=1, matched_terms=("policy", "coverage"))


def test_candidate_requires_keyword_diagnostics_with_keyword_source() -> None:
    with pytest.raises(ValueError, match="keyword_diagnostics"):
        RetrievalCandidate(
            chunk=_make_chunk(),
            scores=ScoreProvenance(keyword_score=1.0),
            sources=frozenset({RetrieverSource.KEYWORD}),
        )


# ── AnswerWithCitations ───────────────────────────────────────────────────────


def test_answer_with_citations_grounded() -> None:
    citation = Citation(source_file="a.pdf", page=0, chunk_index=2)
    answer = AnswerWithCitations(
        answer_text="The answer is X.",
        citations=(citation,),
        grounded=True,
    )
    assert answer.grounded is True
    assert len(answer.citations) == 1
    assert answer.citations[0].source_file == "a.pdf"


def test_answer_no_evidence_response() -> None:
    # When context is insufficient, grounded=False and answer signals unavailability
    answer = AnswerWithCitations(
        answer_text="The answer is not available in the provided documents.",
        citations=(),
        grounded=False,
    )
    assert answer.grounded is False
    assert answer.citations == ()


def test_citation_fields() -> None:
    citation = Citation(source_file="report.pdf", page=3, chunk_index=7)
    assert citation.source_file == "report.pdf"
    assert citation.page == 3
    assert citation.chunk_index == 7


def test_answer_to_json_round_trip() -> None:
    citation = Citation(source_file="a.pdf", page=1, chunk_index=3)
    answer = AnswerWithCitations(
        answer_text="The answer is X.",
        citations=(citation,),
        grounded=True,
    )
    parsed = json.loads(answer.to_json())
    assert parsed["answer_text"] == "The answer is X."
    assert parsed["grounded"] is True
    assert parsed["citations"][0]["source_file"] == "a.pdf"


# ── Chunk serialization ───────────────────────────────────────────────────────


def test_chunk_to_dict() -> None:
    chunk = _make_chunk()
    result = chunk.to_dict()
    assert result["chunk_id"] == "c1"
    assert result["doc_id"] == "d1"
    assert result["source_file"] == "a.pdf"
    assert result["page"] == 0
    assert result["chunk_index"] == 0
    assert result["char_start"] == 0
    assert result["char_end"] == 9
    assert result["text"] == "some text"


def test_chunk_to_json_round_trip() -> None:
    chunk = _make_chunk()
    parsed = json.loads(chunk.to_json())
    assert parsed["chunk_id"] == "c1"
    assert parsed["char_start"] == 0
    assert parsed["char_end"] == 9
    assert parsed["text"] == "some text"


# ── Retrieval serialization ──────────────────────────────────────────────────


def test_retrieval_query_serialization_is_canonical() -> None:
    query = RetrievalQuery(
        query_id="q-1",
        original_text="  Policy coverage? ",
        normalized_text="Policy coverage?",
        filters=QueryFilters(
            doc_ids=frozenset({"doc-b", "doc-a"}),
            source_files=frozenset({"b.pdf", "a.pdf"}),
            pages=frozenset({2, 0}),
        ),
    )
    assert query.to_json() == query.to_json()
    parsed = json.loads(query.to_json())
    assert parsed["schema_version"] == RETRIEVAL_QUERY_SCHEMA_VERSION
    assert parsed["filters"] == {
        "doc_ids": ["doc-a", "doc-b"],
        "pages": [0, 2],
        "source_files": ["a.pdf", "b.pdf"],
    }


def test_reranked_candidate_serialization_is_canonical() -> None:
    candidate = RetrievalCandidate(
        chunk=_make_chunk(),
        scores=ScoreProvenance(
            dense_score=0.91,
            keyword_score=2.4,
            fusion_score=0.04,
            rerank_score=7.3,
        ),
        sources=frozenset({RetrieverSource.KEYWORD, RetrieverSource.DENSE}),
        keyword_diagnostics=KeywordDiagnostics(
            source_rank=3,
            matched_terms=("coverage", "policy"),
        ),
        rank=1,
    )
    assert candidate.to_json() == candidate.to_json()
    parsed = json.loads(candidate.to_json())
    assert parsed["schema_version"] == RETRIEVAL_CANDIDATE_SCHEMA_VERSION
    assert parsed["scores"] == {
        "dense_score": 0.91,
        "fusion_score": 0.04,
        "keyword_score": 2.4,
        "rerank_score": 7.3,
    }
    assert parsed["sources"] == ["dense", "keyword"]
    assert parsed["keyword_diagnostics"] == {
        "matched_terms": ["coverage", "policy"],
        "source_rank": 3,
    }
    assert parsed["rank"] == 1
    assert parsed["chunk"]["chunk_id"] == "c1"


# ── Embedding and index contracts ────────────────────────────────────────────


def _make_embedding_record(
    *,
    schema_version: str = EMBEDDING_SCHEMA_VERSION,
    chunk_id: str = "chunk-1",
    vector: tuple[float, ...] = (0.1, 0.2, 0.3),
    dim: int = 3,
    model_name: str = "local-hash-embedder",
    model_version: str = "v1",
    content_hash: str = "sha256:abc123",
) -> EmbeddingRecord:
    return EmbeddingRecord(
        schema_version=schema_version,
        chunk_id=chunk_id,
        vector=vector,
        dim=dim,
        model_name=model_name,
        model_version=model_version,
        content_hash=content_hash,
    )


def _make_vector_metadata(
    *,
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    source_file: str = "policy.pdf",
    page: int = 0,
    chunk_index: int = 0,
    char_start: int = 0,
    char_end: int = 11,
    text: str = "hello world",
    chunk_schema_version: str = CHUNK_SCHEMA_VERSION,
    content_hash: str = "sha256:abc123",
) -> VectorIndexMetadata:
    return VectorIndexMetadata(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_file=source_file,
        page=page,
        chunk_index=chunk_index,
        char_start=char_start,
        char_end=char_end,
        text=text,
        chunk_schema_version=chunk_schema_version,
        content_hash=content_hash,
    )


def _make_vector_row(
    *,
    schema_version: str = VECTOR_INDEX_SCHEMA_VERSION,
    row_id: str = "chunk-1",
    embedding: EmbeddingRecord | None = None,
    metadata: VectorIndexMetadata | None = None,
    created_at: str = "2026-07-11T00:00:00Z",
    updated_at: str = "2026-07-11T00:00:00Z",
) -> VectorIndexRow:
    return VectorIndexRow(
        schema_version=schema_version,
        row_id=row_id,
        embedding=embedding if embedding is not None else _make_embedding_record(),
        metadata=metadata if metadata is not None else _make_vector_metadata(),
        created_at=created_at,
        updated_at=updated_at,
    )


def test_embedding_record_valid() -> None:
    record = _make_embedding_record()
    assert record.schema_version == EMBEDDING_SCHEMA_VERSION
    assert record.chunk_id == "chunk-1"
    assert record.vector == (0.1, 0.2, 0.3)
    assert record.dim == 3


def test_embedding_record_rejects_invalid_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _make_embedding_record(schema_version="embedding.v0")


def test_embedding_record_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        _make_embedding_record(chunk_id="")
    with pytest.raises(ValueError, match="model_name"):
        _make_embedding_record(model_name="")
    with pytest.raises(ValueError, match="model_version"):
        _make_embedding_record(model_version="")
    with pytest.raises(ValueError, match="content_hash"):
        _make_embedding_record(content_hash="")


def test_embedding_record_rejects_invalid_vector_and_dim() -> None:
    with pytest.raises(ValueError, match="vector"):
        _make_embedding_record(vector=(), dim=0)
    with pytest.raises(ValueError, match="dim"):
        _make_embedding_record(dim=0)
    with pytest.raises(ValueError, match="dim"):
        _make_embedding_record(vector=(0.1, 0.2), dim=3)
    with pytest.raises(ValueError, match="finite"):
        _make_embedding_record(vector=(0.1, math.inf), dim=2)


def test_embedding_record_to_json_round_trip() -> None:
    parsed = json.loads(_make_embedding_record().to_json())
    assert parsed["schema_version"] == EMBEDDING_SCHEMA_VERSION
    assert parsed["chunk_id"] == "chunk-1"
    assert parsed["vector"] == [0.1, 0.2, 0.3]
    assert parsed["dim"] == 3


def test_vector_index_metadata_valid() -> None:
    metadata = _make_vector_metadata()
    assert metadata.chunk_schema_version == CHUNK_SCHEMA_VERSION
    assert metadata.source_file == "policy.pdf"
    assert metadata.text == "hello world"


def test_vector_index_metadata_rejects_empty_required_fields() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        _make_vector_metadata(chunk_id="")
    with pytest.raises(ValueError, match="doc_id"):
        _make_vector_metadata(doc_id="")
    with pytest.raises(ValueError, match="source_file"):
        _make_vector_metadata(source_file="")
    with pytest.raises(ValueError, match="text"):
        _make_vector_metadata(text="")
    with pytest.raises(ValueError, match="content_hash"):
        _make_vector_metadata(content_hash="")


def test_vector_index_metadata_rejects_invalid_location_and_span() -> None:
    with pytest.raises(ValueError, match="page"):
        _make_vector_metadata(page=-1)
    with pytest.raises(ValueError, match="chunk_index"):
        _make_vector_metadata(chunk_index=-1)
    with pytest.raises(ValueError, match="char_start"):
        _make_vector_metadata(char_start=-1)
    with pytest.raises(ValueError, match="char_end"):
        _make_vector_metadata(char_start=4, char_end=4, text="same")
    with pytest.raises(ValueError, match="text length"):
        _make_vector_metadata(char_start=0, char_end=20, text="short")


def test_vector_index_metadata_rejects_invalid_chunk_schema() -> None:
    with pytest.raises(ValueError, match="chunk_schema_version"):
        _make_vector_metadata(chunk_schema_version="chunk.v0")


def test_vector_index_metadata_to_json_round_trip() -> None:
    parsed = json.loads(_make_vector_metadata().to_json())
    assert parsed["chunk_id"] == "chunk-1"
    assert parsed["doc_id"] == "doc-1"
    assert parsed["chunk_schema_version"] == CHUNK_SCHEMA_VERSION


def test_vector_index_row_valid() -> None:
    row = _make_vector_row()
    assert row.schema_version == VECTOR_INDEX_SCHEMA_VERSION
    assert row.row_id == "chunk-1"
    assert row.embedding.chunk_id == "chunk-1"
    assert row.metadata.chunk_id == "chunk-1"


def test_vector_index_row_rejects_invalid_schema_and_empty_fields() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _make_vector_row(schema_version="vector_index.v0")
    with pytest.raises(ValueError, match="row_id"):
        _make_vector_row(row_id="")
    with pytest.raises(ValueError, match="created_at"):
        _make_vector_row(created_at="")
    with pytest.raises(ValueError, match="updated_at"):
        _make_vector_row(updated_at="")


def test_vector_index_row_rejects_mismatched_ids_and_hashes() -> None:
    with pytest.raises(ValueError, match="embedding chunk_id"):
        _make_vector_row(row_id="other")
    with pytest.raises(ValueError, match="metadata chunk_id"):
        _make_vector_row(metadata=_make_vector_metadata(chunk_id="other"))
    with pytest.raises(ValueError, match="content_hash"):
        _make_vector_row(
            metadata=_make_vector_metadata(content_hash="sha256:different")
        )


def test_vector_index_row_to_json_round_trip() -> None:
    parsed = json.loads(_make_vector_row().to_json())
    assert parsed["schema_version"] == VECTOR_INDEX_SCHEMA_VERSION
    assert parsed["row_id"] == "chunk-1"
    assert parsed["embedding"]["schema_version"] == EMBEDDING_SCHEMA_VERSION
    assert parsed["metadata"]["chunk_schema_version"] == CHUNK_SCHEMA_VERSION
