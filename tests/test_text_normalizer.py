from __future__ import annotations

from rag.contracts.document import Document, ParsedPage
from rag.ingest.normalize import NormalizationPolicy, TextNormalizer


def _make_document(*, page_texts: tuple[str, ...]) -> Document:
    doc_id = "doc-1"
    source_file = "policy.pdf"
    return Document(
        doc_id=doc_id,
        source_file=source_file,
        source_path="data/raw/policy.pdf",
        total_pages=len(page_texts),
        pages=tuple(
            ParsedPage(
                doc_id=doc_id,
                source_file=source_file,
                page_number=page_number,
                text=text,
            )
            for page_number, text in enumerate(page_texts)
        ),
    )


def test_normalize_text_is_deterministic() -> None:
    normalizer = TextNormalizer()
    text = "Eligibility   applies\r\nto full-time employees."

    assert normalizer.normalize_text(text) == normalizer.normalize_text(text)


def test_normalize_text_normalizes_line_breaks_and_whitespace() -> None:
    text = "Alpha\t\tbeta\r\nGamma   delta\rOmega"

    assert TextNormalizer().normalize_text(text) == "Alpha beta Gamma delta Omega"


def test_normalize_noisy_extracted_text_matches_regression_output() -> None:
    text = (
        "BENEFITS\r\n"
        "Coverage   applies\r\n"
        "to full-time employees.\f\n\n"
        "-----\n"
        "1. Submit   claim\r\n"
        "2. Wait\tfor approval\n\n\n"
        "Questions:\rContact HR"
    )

    assert (
        TextNormalizer().normalize_text(text) == "BENEFITS\n"
        "Coverage applies to full-time employees.\n\n"
        "---\n"
        "1. Submit claim\n"
        "2. Wait for approval\n\n"
        "Questions:\n"
        "Contact HR"
    )


def test_normalize_text_preserves_paragraph_breaks() -> None:
    text = "First paragraph\ncontinues here.\n\n\nSecond paragraph\ncontinues."

    assert (
        TextNormalizer().normalize_text(text)
        == "First paragraph continues here.\n\nSecond paragraph continues."
    )


def test_normalize_text_preserves_headings() -> None:
    text = "BENEFITS\nCoverage applies\nto employees.\n\nEligibility:\nFull time only."

    assert (
        TextNormalizer().normalize_text(text)
        == "BENEFITS\nCoverage applies to employees.\n\nEligibility:\nFull time only."
    )


def test_normalize_text_preserves_list_items() -> None:
    text = "Requirements\n- Submit form\n- Provide ID\n1. Wait for approval"

    assert (
        TextNormalizer().normalize_text(text)
        == "Requirements\n- Submit form\n- Provide ID\n1. Wait for approval"
    )


def test_normalize_text_collapses_repeated_separators() -> None:
    text = "Section\n----------\n==========\nNext"

    assert TextNormalizer().normalize_text(text) == "Section\n---\nNext"


def test_normalize_text_empty_page_stays_empty() -> None:
    assert TextNormalizer().normalize_text("") == ""
    assert TextNormalizer().normalize_text(" \n\t ") == ""


def test_normalize_text_preserves_low_quality_non_empty_pages() -> None:
    assert TextNormalizer().normalize_text("\f  ???  \f") == "???"


def test_normalization_policy_can_disable_wrapped_line_joining() -> None:
    normalizer = TextNormalizer(policy=NormalizationPolicy(join_wrapped_lines=False))

    assert normalizer.normalize_text("Alpha\nbeta") == "Alpha\nbeta"


def test_normalize_document_preserves_metadata_and_page_order() -> None:
    document = _make_document(
        page_texts=(
            "First   page\ncontinues.",
            "",
            "Third\rpage",
        )
    )

    normalized = TextNormalizer().normalize_document(document)

    assert normalized is not document
    assert normalized.doc_id == document.doc_id
    assert normalized.source_file == document.source_file
    assert normalized.source_path == document.source_path
    assert normalized.total_pages == document.total_pages
    assert tuple(page.page_number for page in normalized.pages) == (0, 1, 2)
    assert normalized.pages[0].text == "First page continues."
    assert normalized.pages[1].text == ""
    assert normalized.pages[2].text == "Third page"
    assert document.pages[0].text == "First   page\ncontinues."
