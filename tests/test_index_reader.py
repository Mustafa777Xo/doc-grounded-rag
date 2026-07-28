from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.contracts.chunk import Chunk
from rag.embed import EmbeddingService, HashEmbeddingProvider
from rag.index import SemanticIndexReader, SemanticIndexReaderError, SQLiteVectorStore
from rag.pipeline.embed_index_pipeline import build_embed_index_pipeline
from rag.storage import ChunkRecord


def _chunk(
    chunk_id: str,
    *,
    chunk_index: int,
    page: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        source_file="benefits.pdf",
        page=page,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        text=text,
    )


def _write_chunks(path: Path, chunks: tuple[Chunk, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(ChunkRecord.from_chunk(chunk).to_json_line() for chunk in chunks),
        encoding="utf-8",
    )


def _build_reader(
    *,
    index_path: Path,
    collection_name: str = "default",
    dim: int = 8,
) -> SemanticIndexReader:
    provider = HashEmbeddingProvider(dim=dim)
    return SemanticIndexReader(
        store=SQLiteVectorStore(index_path, collection_name=collection_name),
        embedding_service=EmbeddingService(provider=provider),
    )


def test_semantic_index_reader_smoke_query_returns_relevant_hit(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    target_text = "Policy coverage applies to full-time employees."
    _write_chunks(
        chunk_path,
        (
            _chunk("benefits-policy", chunk_index=0, page=0, text=target_text),
            _chunk(
                "claims-policy",
                chunk_index=1,
                page=1,
                text="Claims must be filed within 30 days.",
            ),
        ),
    )
    pipeline = build_embed_index_pipeline(
        index_path=index_path,
        collection_name="default",
        provider=provider,
    )
    pipeline.run(chunk_path)
    reader = SemanticIndexReader(
        store=SQLiteVectorStore(index_path),
        embedding_service=EmbeddingService(provider=provider),
    )

    results = reader.retrieve(target_text, limit=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "benefits-policy"
    assert results[0].retrieval_method == "semantic"
    assert results[0].score > results[1].score


def test_semantic_index_reader_preserves_citation_metadata(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    text = "Claims must be filed within 30 days."
    _write_chunks(
        chunk_path,
        (_chunk("claims-policy", chunk_index=7, page=3, text=text),),
    )
    build_embed_index_pipeline(
        index_path=index_path,
        collection_name="default",
        provider=provider,
    ).run(chunk_path)

    result = SemanticIndexReader(
        store=SQLiteVectorStore(index_path),
        embedding_service=EmbeddingService(provider=provider),
    ).retrieve(text, limit=1)[0]

    assert result.chunk.to_dict() == {
        "chunk_id": "claims-policy",
        "doc_id": "doc-1",
        "source_file": "benefits.pdf",
        "page": 3,
        "chunk_index": 7,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
    }


def test_semantic_index_reader_result_contract_shape_is_stable(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    text = "Eligibility rules apply to full-time employees."
    _write_chunks(
        chunk_path,
        (_chunk("eligibility", chunk_index=0, page=0, text=text),),
    )
    build_embed_index_pipeline(
        index_path=index_path,
        collection_name="default",
        provider=provider,
    ).run(chunk_path)
    reader = SemanticIndexReader(
        store=SQLiteVectorStore(index_path),
        embedding_service=EmbeddingService(provider=provider),
    )

    payload = json.loads(reader.retrieve(text, limit=1)[0].to_json())

    assert payload["chunk"] == {
        "chunk_id": "eligibility",
        "doc_id": "doc-1",
        "source_file": "benefits.pdf",
        "page": 0,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": len(text),
        "text": text,
    }
    assert payload["score"] == pytest.approx(1.0)
    assert payload["retrieval_method"] == "semantic"


def test_semantic_index_reader_rejects_invalid_limit(tmp_path: Path) -> None:
    reader = _build_reader(index_path=tmp_path / "missing.sqlite")

    with pytest.raises(SemanticIndexReaderError, match="limit"):
        reader.retrieve("query", limit=0)


def test_semantic_index_reader_wraps_missing_store_with_query_context(
    tmp_path: Path,
) -> None:
    reader = _build_reader(index_path=tmp_path / "missing.sqlite")

    with pytest.raises(SemanticIndexReaderError, match="query_index") as exc_info:
        reader.retrieve("query", limit=1)

    assert exc_info.value.stage == "query_index"


def test_semantic_index_reader_wraps_empty_query_with_embed_context(
    tmp_path: Path,
) -> None:
    reader = _build_reader(index_path=tmp_path / "missing.sqlite")

    with pytest.raises(SemanticIndexReaderError, match="embed_query") as exc_info:
        reader.retrieve("", limit=1)

    assert exc_info.value.stage == "embed_query"
