from __future__ import annotations

from pathlib import Path

import pytest

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import RetrievalQuery
from rag.embed import (
    EmbeddingBatcher,
    EmbeddingPreparationPolicy,
    EmbeddingService,
    EmbeddingTextPreparer,
    HashEmbeddingProvider,
)
from rag.index import ContentHasher, SemanticIndexReader, SQLiteVectorStore
from rag.pipeline.embed_index_pipeline import (
    EmbedIndexPipeline,
    build_embed_index_pipeline,
)
from rag.storage import ChunkRecord

POLICY_TEXT = "Policy coverage applies to full-time employees."
CLAIMS_TEXT = "Claims must be filed within 30 days."
UPDATED_CLAIMS_TEXT = "Claims must be filed within 45 days."
POLICY_HASH = "sha256:9df27e665f55fc517d6734cb3faa6c6d93449052a44bd208b71dc675e228c0f5"
CLAIMS_HASH = "sha256:cb5f8c7a7d6c0383a933dfa36d04bc4b6ebd7d50864a24e04112f7702db5bcbd"
UPDATED_CLAIMS_HASH = (
    "sha256:6c8b571a2a634ed434bf78b31018468e1bb3a646034b0ee25918b5a348642a01"
)


def _chunk(
    chunk_id: str,
    *,
    chunk_index: int,
    page: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="benefits-handbook",
        source_file="benefits.pdf",
        page=page,
        chunk_index=chunk_index,
        char_start=0,
        char_end=len(text),
        text=text,
    )


def _policy_chunk() -> Chunk:
    return _chunk("benefits-policy", chunk_index=0, page=0, text=POLICY_TEXT)


def _claims_chunk(text: str = CLAIMS_TEXT) -> Chunk:
    return _chunk("claims-policy", chunk_index=1, page=1, text=text)


def _write_chunks(path: Path, chunks: tuple[Chunk, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(ChunkRecord.from_chunk(chunk).to_json_line() for chunk in chunks),
        encoding="utf-8",
    )


def _build_pipeline(
    *,
    index_path: Path,
    provider: HashEmbeddingProvider,
    collection_name: str = "regression",
) -> EmbedIndexPipeline:
    return build_embed_index_pipeline(
        index_path=index_path,
        collection_name=collection_name,
        provider=provider,
    )


def _stored_rows(index_path: Path) -> tuple[dict[str, object], ...]:
    rows = SQLiteVectorStore(index_path, collection_name="regression").list_rows()
    return tuple(row.to_dict() for row in rows)


def test_content_hash_known_values_are_stable() -> None:
    hasher = ContentHasher()

    assert hasher.hash_text(POLICY_TEXT) == POLICY_HASH
    assert hasher.hash_text(CLAIMS_TEXT) == CLAIMS_HASH
    assert hasher.hash_text(UPDATED_CLAIMS_TEXT) == UPDATED_CLAIMS_HASH


def test_embedding_batch_regression_preserves_order_and_model_fields() -> None:
    provider = HashEmbeddingProvider(dim=8)
    preparer = EmbeddingTextPreparer(
        policy=EmbeddingPreparationPolicy(max_chars=100, batch_size=2)
    )
    prepared = preparer.prepare_many(
        (
            ("benefits-policy", POLICY_TEXT, POLICY_HASH),
            ("claims-policy", CLAIMS_TEXT, CLAIMS_HASH),
            ("claims-policy-updated", UPDATED_CLAIMS_TEXT, UPDATED_CLAIMS_HASH),
        )
    )
    batcher = EmbeddingBatcher(
        service=EmbeddingService(provider=provider),
        policy=EmbeddingPreparationPolicy(max_chars=100, batch_size=2),
    )

    records = batcher.embed(prepared)

    assert tuple(record.chunk_id for record in records) == (
        "benefits-policy",
        "claims-policy",
        "claims-policy-updated",
    )
    assert tuple(record.content_hash for record in records) == (
        POLICY_HASH,
        CLAIMS_HASH,
        UPDATED_CLAIMS_HASH,
    )
    assert {record.model_name for record in records} == {"local-hash-embedder"}
    assert {record.model_version for record in records} == {"v1"}
    assert all(record.dim == 8 for record in records)


def test_full_embed_index_output_has_stable_contract_snapshot(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    _write_chunks(chunk_path, (_policy_chunk(), _claims_chunk()))

    summary = _build_pipeline(index_path=index_path, provider=provider).run(chunk_path)
    rows = SQLiteVectorStore(index_path, collection_name="regression").list_rows()

    assert summary.to_dict()["sync"] == {
        "new": 2,
        "changed": 0,
        "unchanged": 0,
        "deleted": 0,
        "written": 2,
        "removed": 0,
        "errors": 0,
    }
    assert tuple(row.row_id for row in rows) == ("benefits-policy", "claims-policy")
    assert tuple(row.embedding.content_hash for row in rows) == (
        POLICY_HASH,
        CLAIMS_HASH,
    )
    assert tuple(row.metadata.content_hash for row in rows) == (
        POLICY_HASH,
        CLAIMS_HASH,
    )
    assert all(row.embedding.dim == 8 for row in rows)
    assert all(len(row.embedding.vector) == 8 for row in rows)
    assert rows[0].metadata.to_dict() == {
        "chunk_id": "benefits-policy",
        "doc_id": "benefits-handbook",
        "source_file": "benefits.pdf",
        "page": 0,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": len(POLICY_TEXT),
        "text": POLICY_TEXT,
        "chunk_schema_version": "chunk.v1",
        "content_hash": POLICY_HASH,
    }


def test_repeated_full_reindex_does_not_duplicate_or_rewrite_rows(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    pipeline = _build_pipeline(index_path=index_path, provider=provider)
    _write_chunks(chunk_path, (_policy_chunk(), _claims_chunk()))

    first = pipeline.run(chunk_path)
    first_rows = _stored_rows(index_path)
    second = pipeline.run(chunk_path)
    second_rows = _stored_rows(index_path)
    third = pipeline.run(chunk_path)
    third_rows = _stored_rows(index_path)

    assert first.sync.written_count == 2
    assert second.sync.unchanged_count == 2
    assert second.sync.written_count == 0
    assert third.sync.unchanged_count == 2
    assert third.sync.written_count == 0
    assert len(third_rows) == 2
    assert second_rows == first_rows
    assert third_rows == first_rows


def test_changed_and_deleted_chunks_reindex_without_duplicate_rows(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    pipeline = _build_pipeline(index_path=index_path, provider=provider)
    _write_chunks(chunk_path, (_policy_chunk(), _claims_chunk()))
    pipeline.run(chunk_path)

    _write_chunks(chunk_path, (_claims_chunk(UPDATED_CLAIMS_TEXT),))
    summary = pipeline.run(chunk_path)
    rows = SQLiteVectorStore(index_path, collection_name="regression").list_rows()

    assert summary.sync.changed_count == 1
    assert summary.sync.deleted_count == 1
    assert summary.sync.written_count == 1
    assert summary.sync.removed_count == 1
    assert tuple(row.row_id for row in rows) == ("claims-policy",)
    assert rows[0].metadata.text == UPDATED_CLAIMS_TEXT
    assert rows[0].metadata.content_hash == UPDATED_CLAIMS_HASH


def test_malformed_chunk_artifact_does_not_mutate_existing_index(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    pipeline = _build_pipeline(index_path=index_path, provider=provider)
    _write_chunks(chunk_path, (_policy_chunk(), _claims_chunk()))
    pipeline.run(chunk_path)
    before = _stored_rows(index_path)

    chunk_path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(Exception, match="stage=load_chunks"):
        pipeline.run(chunk_path)

    assert _stored_rows(index_path) == before


def test_duplicate_chunk_artifact_does_not_mutate_existing_index(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    pipeline = _build_pipeline(index_path=index_path, provider=provider)
    _write_chunks(chunk_path, (_policy_chunk(), _claims_chunk()))
    pipeline.run(chunk_path)
    before = _stored_rows(index_path)

    _write_chunks(chunk_path, (_policy_chunk(), _policy_chunk()))
    with pytest.raises(Exception, match="stage=load_chunks"):
        pipeline.run(chunk_path)

    assert _stored_rows(index_path) == before


def test_schema_drift_failure_preserves_existing_index_rows(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    _write_chunks(chunk_path, (_policy_chunk(), _claims_chunk()))
    _build_pipeline(
        index_path=index_path,
        provider=HashEmbeddingProvider(dim=8),
    ).run(chunk_path)
    before = _stored_rows(index_path)

    drifted_pipeline = _build_pipeline(
        index_path=index_path,
        provider=HashEmbeddingProvider(dim=9),
    )
    with pytest.raises(Exception, match="stage=sync_index"):
        drifted_pipeline.run(chunk_path)

    assert _stored_rows(index_path) == before


def test_retrieval_contract_after_reindex_preserves_required_metadata(
    tmp_path: Path,
) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.sqlite"
    provider = HashEmbeddingProvider(dim=8)
    _write_chunks(chunk_path, (_policy_chunk(), _claims_chunk()))
    _build_pipeline(index_path=index_path, provider=provider).run(chunk_path)
    reader = SemanticIndexReader(
        store=SQLiteVectorStore(index_path, collection_name="regression"),
        embedding_service=EmbeddingService(provider=provider),
    )

    query = RetrievalQuery(
        query_id="regression-query",
        original_text=POLICY_TEXT,
        normalized_text=POLICY_TEXT,
    )
    payload = reader.retrieve(query, limit=1)[0].to_dict()

    assert set(payload) == {"schema_version", "chunk", "scores", "sources", "rank"}
    assert payload["sources"] == ["dense"]
    scores = payload["scores"]
    assert isinstance(scores, dict)
    assert isinstance(scores["dense_score"], float)
    assert payload["chunk"] == {
        "chunk_id": "benefits-policy",
        "doc_id": "benefits-handbook",
        "source_file": "benefits.pdf",
        "page": 0,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": len(POLICY_TEXT),
        "text": POLICY_TEXT,
    }
