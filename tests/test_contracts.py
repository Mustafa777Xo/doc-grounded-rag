# file: tests/test_contracts.py
from __future__ import annotations

import json

import pytest

from rag.contracts.answer import AnswerWithCitations, Citation
from rag.contracts.chunk import Chunk
from rag.contracts.document import Document
from rag.contracts.retrieval import RetrievalResult

# ── Document ──────────────────────────────────────────────────────────────────


def test_document_valid() -> None:
    doc = Document(doc_id="d1", source_file="a.pdf", pages=("hello",))
    assert doc.doc_id == "d1"
    assert doc.source_file == "a.pdf"
    assert doc.pages == ("hello",)


def test_document_rejects_empty_doc_id() -> None:
    with pytest.raises(ValueError, match="doc_id"):
        Document(doc_id="", source_file="a.pdf", pages=("text",))


def test_document_rejects_empty_source_file() -> None:
    with pytest.raises(ValueError, match="source_file"):
        Document(doc_id="d1", source_file="", pages=("text",))


def test_document_rejects_empty_pages() -> None:
    with pytest.raises(ValueError, match="pages"):
        Document(doc_id="d1", source_file="a.pdf", pages=())


def test_document_to_dict() -> None:
    doc = Document(doc_id="d1", source_file="a.pdf", pages=("page one", "page two"))
    result = doc.to_dict()
    assert result["doc_id"] == "d1"
    assert result["source_file"] == "a.pdf"
    assert result["pages"] == ["page one", "page two"]


def test_document_to_json_round_trip() -> None:
    doc = Document(doc_id="d1", source_file="a.pdf", pages=("hello",))
    parsed = json.loads(doc.to_json())
    assert parsed["doc_id"] == "d1"
    assert parsed["pages"] == ["hello"]


def test_document_is_immutable() -> None:
    doc = Document(doc_id="d1", source_file="a.pdf", pages=("text",))
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
        text="some text",
    )
    assert chunk.chunk_id == "c1"
    assert chunk.chunk_index == 0


def test_chunk_is_hashable() -> None:
    # frozen=True dataclasses must be hashable — required for sets and dict keys
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        source_file="a.pdf",
        page=0,
        chunk_index=0,
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
        text="some text",
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
    assert result["text"] == "some text"


def test_chunk_to_json_round_trip() -> None:
    chunk = _make_chunk()
    parsed = json.loads(chunk.to_json())
    assert parsed["chunk_id"] == "c1"
    assert parsed["text"] == "some text"


# ── RetrievalResult serialization ────────────────────────────────────────────


def test_retrieval_result_to_json_round_trip() -> None:
    result = RetrievalResult(chunk=_make_chunk(), score=0.91, retrieval_method="hybrid")
    parsed = json.loads(result.to_json())
    assert parsed["score"] == 0.91
    assert parsed["retrieval_method"] == "hybrid"
    assert parsed["chunk"]["chunk_id"] == "c1"
