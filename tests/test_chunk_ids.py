from __future__ import annotations

import re

import pytest

from rag.chunking.ids import (
    ChunkIdCollisionError,
    ChunkIdGenerator,
    ensure_unique_chunk_ids,
)
from rag.contracts.chunk import Chunk


def _make_chunk(*, chunk_id: str, chunk_index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="policy.pdf",
        page=0,
        chunk_index=chunk_index,
        char_start=0,
        char_end=5,
        text="alpha",
    )


def test_chunk_id_generation_is_stable_for_same_inputs() -> None:
    generator = ChunkIdGenerator()
    first = generator.generate(
        doc_id="doc-1",
        page=0,
        char_start=0,
        char_end=5,
        text="alpha",
    )
    second = generator.generate(
        doc_id="doc-1",
        page=0,
        char_start=0,
        char_end=5,
        text="alpha",
    )

    assert first == second


def test_chunk_id_generation_matches_regression_value() -> None:
    assert (
        ChunkIdGenerator().generate(
            doc_id="doc-1",
            page=0,
            char_start=0,
            char_end=47,
            text="Policy coverage applies to full-time employees.",
        )
        == "doc-1-p0-s0-e47-69b2f7eeb3b2"
    )


def test_chunk_id_changes_when_text_changes() -> None:
    generator = ChunkIdGenerator()
    first = generator.generate(
        doc_id="doc-1",
        page=0,
        char_start=0,
        char_end=5,
        text="alpha",
    )
    second = generator.generate(
        doc_id="doc-1",
        page=0,
        char_start=0,
        char_end=4,
        text="beta",
    )

    assert first != second


def test_chunk_id_changes_when_metadata_changes() -> None:
    generator = ChunkIdGenerator()
    base = generator.generate(
        doc_id="doc-1",
        page=0,
        char_start=0,
        char_end=5,
        text="alpha",
    )

    assert base != generator.generate(
        doc_id="doc-2",
        page=0,
        char_start=0,
        char_end=5,
        text="alpha",
    )
    assert base != generator.generate(
        doc_id="doc-1",
        page=1,
        char_start=0,
        char_end=5,
        text="alpha",
    )
    assert base != generator.generate(
        doc_id="doc-1",
        page=0,
        char_start=1,
        char_end=6,
        text="alpha",
    )


def test_chunk_id_format_is_readable_with_hash_suffix() -> None:
    chunk_id = ChunkIdGenerator().generate(
        doc_id="doc-1",
        page=2,
        char_start=10,
        char_end=42,
        text="stable chunk text",
    )

    assert re.fullmatch(r"doc-1-p2-s10-e42-[0-9a-f]{12}", chunk_id) is not None


def test_chunk_id_generator_rejects_invalid_inputs() -> None:
    generator = ChunkIdGenerator()
    with pytest.raises(ValueError, match="doc_id"):
        generator.generate(doc_id="", page=0, char_start=0, char_end=1, text="a")
    with pytest.raises(ValueError, match="page"):
        generator.generate(doc_id="doc-1", page=-1, char_start=0, char_end=1, text="a")
    with pytest.raises(ValueError, match="char_start"):
        generator.generate(doc_id="doc-1", page=0, char_start=-1, char_end=1, text="a")
    with pytest.raises(ValueError, match="char_end"):
        generator.generate(doc_id="doc-1", page=0, char_start=1, char_end=1, text="a")
    with pytest.raises(ValueError, match="text"):
        generator.generate(doc_id="doc-1", page=0, char_start=0, char_end=1, text="")


def test_chunk_id_generator_rejects_invalid_hash_length() -> None:
    with pytest.raises(ValueError, match="hash_length"):
        ChunkIdGenerator(hash_length=0)
    with pytest.raises(ValueError, match="hash_length"):
        ChunkIdGenerator(hash_length=65)


def test_ensure_unique_chunk_ids_passes_for_unique_ids() -> None:
    ensure_unique_chunk_ids(
        (
            _make_chunk(chunk_id="chunk-1", chunk_index=0),
            _make_chunk(chunk_id="chunk-2", chunk_index=1),
        )
    )


def test_ensure_unique_chunk_ids_raises_for_duplicates() -> None:
    with pytest.raises(ChunkIdCollisionError, match="duplicate_index=1") as exc_info:
        ensure_unique_chunk_ids(
            (
                _make_chunk(chunk_id="duplicate", chunk_index=0),
                _make_chunk(chunk_id="duplicate", chunk_index=1),
            )
        )

    assert exc_info.value.chunk_id == "duplicate"
    assert exc_info.value.first_index == 0
    assert exc_info.value.duplicate_index == 1
