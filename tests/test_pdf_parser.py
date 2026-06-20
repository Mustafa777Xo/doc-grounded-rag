from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag.ingest.parser import PdfParseError, PdfParser

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "pdfs"


def _expected_doc_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def test_pdf_parser_returns_page_ordered_document_metadata() -> None:
    source_path = FIXTURES_DIR / "sample_policy.pdf"
    document = PdfParser().parse(source_path)

    assert document.doc_id == _expected_doc_id(source_path)
    assert document.source_file == "sample_policy.pdf"
    assert document.source_path == str(source_path)
    assert document.total_pages == 2
    assert tuple(page.page_number for page in document.pages) == (0, 1)
    assert all(page.doc_id == document.doc_id for page in document.pages)
    assert all(page.source_file == document.source_file for page in document.pages)
    assert "Policy coverage applies" in document.pages[0].text
    assert "Claims must be filed" in document.pages[1].text


def test_pdf_parser_preserves_empty_pages() -> None:
    source_path = FIXTURES_DIR / "sample_empty_page.pdf"
    document = PdfParser().parse(source_path)

    assert document.total_pages == 3
    assert tuple(page.page_number for page in document.pages) == (0, 1, 2)
    assert "Visible text before" in document.pages[0].text
    assert document.pages[1].text == ""
    assert "Visible text after" in document.pages[2].text


def test_pdf_parser_rejects_corrupt_pdf_with_context() -> None:
    source_path = FIXTURES_DIR / "sample_corrupt.pdf"

    with pytest.raises(PdfParseError, match="sample_corrupt.pdf") as exc_info:
        PdfParser().parse(source_path)

    assert exc_info.value.source_path == source_path
    assert "could not be read" in exc_info.value.message


def test_pdf_parser_rejects_missing_path_with_context(tmp_path: Path) -> None:
    source_path = tmp_path / "missing.pdf"

    with pytest.raises(PdfParseError, match="file does not exist") as exc_info:
        PdfParser().parse(source_path)

    assert exc_info.value.source_path == source_path


def test_pdf_parser_rejects_directory_path_with_context(tmp_path: Path) -> None:
    with pytest.raises(PdfParseError, match="path is not a file") as exc_info:
        PdfParser().parse(tmp_path)

    assert exc_info.value.source_path == tmp_path
