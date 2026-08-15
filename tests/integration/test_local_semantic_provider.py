from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from rag.contracts.chunk import Chunk
from rag.embed.sentence_transformer import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_MODEL_REVISION,
    SentenceTransformerEmbeddingProvider,
)
from rag.index import SQLiteVectorStore
from rag.pipeline.embed_index_pipeline import build_embed_index_pipeline
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

    chunk = Chunk(
        chunk_id="benefits-p0-c0",
        doc_id="benefits",
        source_file="benefits.pdf",
        page=0,
        chunk_index=0,
        char_start=0,
        char_end=44,
        text="Employees receive health insurance benefits.",
    )
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    chunk_path.write_text(
        ChunkRecord.from_chunk(chunk).to_json_line(), encoding="utf-8"
    )

    summary = build_embed_index_pipeline(
        index_path=index_path,
        collection_name="semantic-integration",
        provider=provider,
    ).run(chunk_path)

    rows = SQLiteVectorStore(
        index_path, collection_name="semantic-integration"
    ).list_rows()
    assert summary.succeeded is True
    assert summary.model_name == DEFAULT_EMBEDDING_MODEL
    assert summary.embedding_dim == DEFAULT_EMBEDDING_DIMENSION
    assert len(rows) == 1
    assert len(rows[0].embedding.vector) == DEFAULT_EMBEDDING_DIMENSION
