# file: tests/test_contracts.py
from __future__ import annotations

import json

import pytest

from rag.contracts.answer import AnswerWithCitations, Citation
from rag.contracts.chunk import Chunk
from rag.contracts.document import Document, ParsedPage
from rag.contracts.retrieval import RetrievalResult

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


# ── RetrievalResult ───────────────────────────────────────────────────────────


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


def test_retrieval_result_holds_chunk() -> None:
    chunk = _make_chunk()
    result = RetrievalResult(chunk=chunk, score=0.87, retrieval_method="semantic")
    assert result.chunk is chunk
    assert result.score == 0.87
    assert result.retrieval_method == "semantic"


def test_retrieval_result_is_immutable() -> None:
    result = RetrievalResult(chunk=_make_chunk(), score=0.5, retrieval_method="keyword")
    with pytest.raises(Exception):
        result.score = 0.9  # type: ignore[misc]


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


# ── RetrievalResult serialization ────────────────────────────────────────────


def test_retrieval_result_to_json_round_trip() -> None:
    result = RetrievalResult(chunk=_make_chunk(), score=0.91, retrieval_method="hybrid")
    parsed = json.loads(result.to_json())
    assert parsed["score"] == 0.91
    assert parsed["retrieval_method"] == "hybrid"
    assert parsed["chunk"]["chunk_id"] == "c1"
