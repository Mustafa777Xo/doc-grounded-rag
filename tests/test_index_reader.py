from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from rag.contracts.chunk import Chunk
from rag.contracts.indexing import (
    CHUNK_SCHEMA_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    VECTOR_INDEX_SCHEMA_VERSION,
    EmbeddingRecord,
    VectorIndexMetadata,
    VectorIndexRow,
)
from rag.contracts.retrieval import QueryFilters, RetrievalQuery, RetrieverSource
from rag.embed import EmbeddingService, HashEmbeddingProvider
from rag.errors import RetrievalError
from rag.index import (
    SemanticIndexReader,
    SQLiteVectorStore,
    VectorQueryResult,
    VectorStore,
    VectorStoreError,
    VectorStoreSchema,
)
from rag.pipeline.embed_index_pipeline import build_embed_index_pipeline
from rag.storage import ChunkRecord


def _chunk(
    chunk_id: str,
    *,
    chunk_index: int,
    page: int,
    text: str,
    doc_id: str = "doc-1",
    source_file: str = "benefits.pdf",
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


def _query(text: str) -> RetrievalQuery:
    return RetrievalQuery(
        query_id="query-1",
        original_text=f"  {text}  ",
        normalized_text=text,
    )


def _hit(chunk: Chunk, *, score: float) -> VectorQueryResult:
    content_hash = f"sha256:{chunk.chunk_id}"
    return VectorQueryResult(
        row=VectorIndexRow(
            schema_version=VECTOR_INDEX_SCHEMA_VERSION,
            row_id=chunk.chunk_id,
            embedding=EmbeddingRecord(
                schema_version=EMBEDDING_SCHEMA_VERSION,
                chunk_id=chunk.chunk_id,
                vector=(1.0, 0.0),
                dim=2,
                model_name="fake-embedder",
                model_version="test",
                content_hash=content_hash,
            ),
            metadata=VectorIndexMetadata(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_file=chunk.source_file,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
                chunk_schema_version=CHUNK_SCHEMA_VERSION,
                content_hash=content_hash,
            ),
            created_at="2026-08-15T00:00:00Z",
            updated_at="2026-08-15T00:00:00Z",
        ),
        score=score,
    )


class _RecordingProvider:
    dim = 2
    model_name = "fake-embedder"
    model_version = "test"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_text(self, text: str) -> tuple[float, ...]:
        self.texts.append(text)
        return (1.0, 0.0)

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed_text(text) for text in texts)


class _RecordingStore:
    def __init__(self, hits: tuple[VectorQueryResult, ...]) -> None:
        self.hits = hits
        self.calls: list[tuple[tuple[float, ...], int, QueryFilters | None]] = []

    def query(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        filters: QueryFilters | None = None,
    ) -> tuple[VectorQueryResult, ...]:
        self.calls.append((tuple(vector), limit, filters))
        return self.hits


def test_semantic_index_reader_uses_normalized_text_filters_and_top_k() -> None:
    provider = _RecordingProvider()
    filters = QueryFilters(
        doc_ids=frozenset({"doc-a", "doc-b"}),
        source_files=frozenset({"benefits.pdf"}),
        pages=frozenset({2}),
    )
    matching_a = _chunk(
        "chunk-a",
        chunk_index=7,
        page=2,
        text="Alpha coverage.",
        doc_id="doc-a",
    )
    matching_b = _chunk(
        "chunk-b",
        chunk_index=8,
        page=2,
        text="Beta coverage.",
        doc_id="doc-b",
    )
    store = _RecordingStore(
        (
            _hit(
                _chunk(
                    "excluded",
                    chunk_index=0,
                    page=2,
                    text="Excluded.",
                    doc_id="other-doc",
                ),
                score=0.99,
            ),
            _hit(matching_b, score=0.8),
            _hit(matching_a, score=0.8),
            _hit(
                _chunk(
                    "chunk-c",
                    chunk_index=9,
                    page=2,
                    text="Gamma coverage.",
                    doc_id="doc-a",
                ),
                score=0.7,
            ),
        )
    )
    reader = SemanticIndexReader(
        store=cast(VectorStore, store),
        embedding_service=EmbeddingService(provider=provider),
    )
    query = RetrievalQuery(
        query_id="query-1",
        original_text="  POLICY Coverage  ",
        normalized_text="policy coverage",
        filters=filters,
    )

    results = reader.retrieve(query, limit=2)

    assert provider.texts == ["policy coverage"]
    assert store.calls == [((1.0, 0.0), 2, filters)]
    assert tuple(result.chunk.chunk_id for result in results) == (
        "chunk-a",
        "chunk-b",
    )
    assert results[0].chunk.to_dict() == matching_a.to_dict()
    assert results[0].scores.dense_score == pytest.approx(0.8)
    assert results[0].sources == frozenset({RetrieverSource.DENSE})


def test_semantic_index_reader_returns_empty_for_empty_or_filtered_hits() -> None:
    provider = _RecordingProvider()
    filters = QueryFilters(doc_ids=frozenset({"allowed"}))
    excluded = _hit(
        _chunk(
            "excluded",
            chunk_index=0,
            page=0,
            text="Excluded.",
            doc_id="other",
        ),
        score=1.0,
    )
    query = RetrievalQuery(
        query_id="query-1",
        original_text="query",
        normalized_text="query",
        filters=filters,
    )

    for hits in ((), (excluded,)):
        reader = SemanticIndexReader(
            store=cast(VectorStore, _RecordingStore(hits)),
            embedding_service=EmbeddingService(provider=provider),
        )
        assert reader.retrieve(query, limit=5) == ()


def test_semantic_index_reader_returns_empty_for_bootstrapped_empty_index(
    tmp_path: Path,
) -> None:
    provider = HashEmbeddingProvider(dim=8)
    store = SQLiteVectorStore(tmp_path / "empty.sqlite")
    store.bootstrap_collection(
        VectorStoreSchema(
            collection_name="default",
            vector_index_schema_version=VECTOR_INDEX_SCHEMA_VERSION,
            embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
            chunk_schema_version=CHUNK_SCHEMA_VERSION,
            model_name=provider.model_name,
            model_version=provider.model_version,
            dim=provider.dim,
        )
    )
    reader = SemanticIndexReader(
        store=store,
        embedding_service=EmbeddingService(provider=provider),
    )

    assert reader.retrieve(_query("query"), limit=5) == ()


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

    results = reader.retrieve(_query(target_text), limit=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "benefits-policy"
    assert results[0].sources == frozenset({RetrieverSource.DENSE})
    first_score = results[0].scores.dense_score
    second_score = results[1].scores.dense_score
    assert first_score is not None
    assert second_score is not None
    assert first_score > second_score


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
    ).retrieve(_query(text), limit=1)[0]

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

    payload = json.loads(reader.retrieve(_query(text), limit=1)[0].to_json())

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
    assert payload["scores"] == {
        "dense_score": pytest.approx(1.0),
        "fusion_score": None,
        "keyword_score": None,
        "rerank_score": None,
    }
    assert payload["sources"] == ["dense"]
    assert payload["rank"] is None


def test_semantic_index_reader_rejects_invalid_limit(tmp_path: Path) -> None:
    reader = _build_reader(index_path=tmp_path / "missing.sqlite")

    with pytest.raises(RetrievalError, match="limit") as exc_info:
        reader.retrieve(_query("query"), limit=0)

    assert exc_info.value.stage == "dense"
    assert exc_info.value.query_id == "query-1"


def test_semantic_index_reader_wraps_missing_store_with_query_context(
    tmp_path: Path,
) -> None:
    reader = _build_reader(index_path=tmp_path / "missing.sqlite")

    with pytest.raises(RetrievalError, match="vector store") as exc_info:
        reader.retrieve(_query("query"), limit=1)

    assert exc_info.value.stage == "dense"
    assert exc_info.value.query_id == "query-1"
    assert isinstance(exc_info.value.__cause__, VectorStoreError)
