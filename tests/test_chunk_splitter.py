from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunking.ids import ChunkIdCollisionError
from rag.chunking.splitter import ChunkingPolicy, ChunkSplitter
from rag.config import Settings
from rag.contracts.document import Document, ParsedPage


def _make_document(page_texts: tuple[str, ...]) -> Document:
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


def test_split_document_is_deterministic() -> None:
    document = _make_document(("Alpha beta gamma.\n\nDelta epsilon zeta.",))
    splitter = ChunkSplitter(
        policy=ChunkingPolicy(target_size=20, overlap=5, hard_max_size=30)
    )

    assert splitter.split_document(document) == splitter.split_document(document)


def test_split_short_page_returns_one_chunk_with_exact_metadata() -> None:
    text = "Short page."
    document = _make_document((text,))
    chunks = ChunkSplitter().split_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id.startswith("doc-1-p0-s0-e11-")
    assert chunk.doc_id == "doc-1"
    assert chunk.source_file == "policy.pdf"
    assert chunk.page == 0
    assert chunk.chunk_index == 0
    assert chunk.char_start == 0
    assert chunk.char_end == len(text)
    assert chunk.text == text


def test_split_document_skips_empty_and_whitespace_pages() -> None:
    chunks = ChunkSplitter().split_document(_make_document(("", " \n\t ")))

    assert chunks == ()


def test_split_long_paragraph_respects_hard_max_size() -> None:
    words = [f"word{i:02d}" for i in range(30)]
    text = " ".join(words)
    splitter = ChunkSplitter(
        policy=ChunkingPolicy(target_size=40, overlap=8, hard_max_size=50)
    )

    chunks = splitter.split_document(_make_document((text,)))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert all(
        chunk.text == text[chunk.char_start : chunk.char_end] for chunk in chunks
    )


def test_split_document_applies_character_overlap() -> None:
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
    splitter = ChunkSplitter(
        policy=ChunkingPolicy(target_size=24, overlap=6, hard_max_size=32)
    )

    chunks = splitter.split_document(_make_document((text,)))

    assert len(chunks) > 1
    assert chunks[1].char_start <= chunks[0].char_end
    assert chunks[0].char_end - chunks[1].char_start <= 6
    assert chunks[1].text == text[chunks[1].char_start : chunks[1].char_end]


def test_split_document_prefers_paragraph_boundaries() -> None:
    text = "First paragraph fits.\n\nSecond paragraph fits."
    splitter = ChunkSplitter(
        policy=ChunkingPolicy(target_size=25, overlap=0, hard_max_size=40)
    )

    chunks = splitter.split_document(_make_document((text,)))

    assert tuple(chunk.text for chunk in chunks) == (
        "First paragraph fits.",
        "Second paragraph fits.",
    )


def test_chunk_ids_and_indexes_are_stable_global_and_ordered() -> None:
    document = _make_document(("Page one text.", "Page two text."))
    splitter = ChunkSplitter()
    chunks = splitter.split_document(document)

    assert chunks == splitter.split_document(document)
    assert chunks[0].chunk_id.startswith("doc-1-p0-s0-e14-")
    assert chunks[1].chunk_id.startswith("doc-1-p1-s0-e14-")
    assert chunks[0].chunk_id != chunks[1].chunk_id
    assert tuple(chunk.chunk_index for chunk in chunks) == (0, 1)
    assert tuple(chunk.page for chunk in chunks) == (0, 1)


class CollidingChunkIdGenerator:
    def generate(
        self,
        *,
        doc_id: str,
        page: int,
        char_start: int,
        char_end: int,
        text: str,
    ) -> str:
        _ = doc_id
        _ = page
        _ = char_start
        _ = char_end
        _ = text
        return "duplicate"


def test_splitter_surfaces_chunk_id_collisions() -> None:
    splitter = ChunkSplitter(
        policy=ChunkingPolicy(target_size=8, overlap=0, hard_max_size=12),
        id_generator=CollidingChunkIdGenerator(),
    )

    with pytest.raises(ChunkIdCollisionError, match="duplicate"):
        splitter.split_document(_make_document(("alpha beta gamma delta",)))


def test_chunking_policy_from_settings() -> None:
    settings = Settings(
        docs_dir=Path("data/pdfs"),
        chunk_size=300,
        chunk_overlap=30,
        chunk_hard_max=450,
    )

    policy = ChunkingPolicy.from_settings(settings)

    assert policy.target_size == 300
    assert policy.overlap == 30
    assert policy.hard_max_size == 450


def test_chunking_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="target_size"):
        ChunkingPolicy(target_size=0)
    with pytest.raises(ValueError, match="overlap"):
        ChunkingPolicy(target_size=10, overlap=10)
    with pytest.raises(ValueError, match="hard_max_size"):
        ChunkingPolicy(hard_max_size=0)
    with pytest.raises(ValueError, match="target_size"):
        ChunkingPolicy(target_size=20, overlap=5, hard_max_size=10)
