from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import QueryFilters
from rag.embed import EmbeddingService
from rag.embed.sentence_transformer import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    SentenceTransformerEmbeddingProvider,
)
from rag.index import SemanticIndexReader, SQLiteVectorStore
from rag.pipeline.embed_index_pipeline import build_embed_index_pipeline
from rag.retrieve import QueryNormalizer
from rag.storage import ChunkRecord

pytestmark = pytest.mark.integration


def test_cached_local_provider_returns_deterministic_semantic_vectors(
    tmp_path: Path,
) -> None:
    if os.environ.get("RUN_LOCAL_MODEL_TESTS") != "1":
        pytest.skip("set RUN_LOCAL_MODEL_TESTS=1 to exercise cached model weights")

    provider = SentenceTransformerEmbeddingProvider()
    same_a, same_b, different = provider.embed_batch(
        (
            "Employees receive health insurance benefits.",
            "Employees receive health insurance benefits.",
            "A lunar eclipse occurs when Earth blocks sunlight.",
        )
    )

    assert provider.model_name == DEFAULT_EMBEDDING_MODEL
    assert provider.model_version == DEFAULT_EMBEDDING_MODEL_REVISION
    assert provider.dim == DEFAULT_EMBEDDING_DIMENSION
    assert len(same_a) == DEFAULT_EMBEDDING_DIMENSION
    assert same_a == pytest.approx(same_b)
    assert math.sqrt(sum(value * value for value in same_a)) == pytest.approx(1.0)
    similarity = sum(
        left * right for left, right in zip(same_a, different, strict=True)
    )
    assert similarity < 0.9

    chunks = (
        Chunk(
            chunk_id="benefits-p0-c0",
            doc_id="benefits",
            source_file="benefits.pdf",
            page=0,
            chunk_index=0,
            char_start=0,
            char_end=len("Employees receive health insurance benefits."),
            text="Employees receive health insurance benefits.",
        ),
        Chunk(
            chunk_id="astronomy-p0-c0",
            doc_id="astronomy",
            source_file="astronomy.pdf",
            page=0,
            chunk_index=0,
            char_start=0,
            char_end=len("A lunar eclipse occurs when Earth blocks sunlight."),
            text="A lunar eclipse occurs when Earth blocks sunlight.",
        ),
    )
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    chunk_path.write_text(
        "".join(ChunkRecord.from_chunk(chunk).to_json_line() for chunk in chunks),
        encoding="utf-8",
    )

    summary = build_embed_index_pipeline(
        index_path=index_path,
        collection_name="semantic-integration",
        provider=provider,
    ).run(chunk_path)

    store = SQLiteVectorStore(index_path, collection_name="semantic-integration")
    rows = store.list_rows()
    assert summary.succeeded is True
    assert summary.model_name == DEFAULT_EMBEDDING_MODEL
    assert summary.embedding_dim == DEFAULT_EMBEDDING_DIMENSION
    assert len(rows) == 2
    assert len(rows[0].embedding.vector) == DEFAULT_EMBEDDING_DIMENSION

    reader = SemanticIndexReader(
        store=store,
        embedding_service=EmbeddingService(provider=provider),
    )
    query = QueryNormalizer().normalize(
        query_id="semantic-query",
        original_text="What medical coverage do workers receive?",
    )
    results = reader.retrieve(query, limit=2)

    assert tuple(result.chunk.chunk_id for result in results) == (
        "benefits-p0-c0",
        "astronomy-p0-c0",
    )
    assert results[0].scores.dense_score is not None
    assert results[1].scores.dense_score is not None
    assert results[0].scores.dense_score > results[1].scores.dense_score
    assert results[0].chunk.to_dict() == chunks[0].to_dict()

    filtered_query = QueryNormalizer().normalize(
        query_id="filtered-query",
        original_text="What medical coverage do workers receive?",
        filters=QueryFilters(source_files=frozenset({"benefits.pdf"})),
    )
    filtered = reader.retrieve(filtered_query, limit=2)

    assert tuple(result.chunk.chunk_id for result in filtered) == ("benefits-p0-c0",)
