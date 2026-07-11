from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Settings
from rag.embed import (
    EmbeddingBatcher,
    EmbeddingPreparationPolicy,
    EmbeddingService,
    EmbeddingServiceError,
    EmbeddingTextPreparer,
    MockEmbeddingProvider,
)


def test_embedding_preparation_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        EmbeddingPreparationPolicy(max_chars=0)
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingPreparationPolicy(batch_size=0)


def test_embedding_preparation_policy_loads_from_settings() -> None:
    settings = Settings(
        docs_dir=Path("data/pdfs"),
        embedding_max_chars=12,
        embedding_batch_size=4,
    )

    policy = EmbeddingPreparationPolicy.from_settings(settings)

    assert policy.max_chars == 12
    assert policy.batch_size == 4


def test_text_preparer_leaves_under_budget_text_unchanged() -> None:
    preparer = EmbeddingTextPreparer(policy=EmbeddingPreparationPolicy(max_chars=10))

    prepared = preparer.prepare(
        chunk_id="chunk-1",
        text="12345",
        content_hash="sha256:one",
    )

    assert prepared.request.text == "12345"
    assert prepared.original_text_length == 5
    assert prepared.prepared_text_length == 5
    assert prepared.truncated is False
    assert prepared.reason is None


def test_text_preparer_truncates_oversized_text_with_reason() -> None:
    preparer = EmbeddingTextPreparer(policy=EmbeddingPreparationPolicy(max_chars=5))

    prepared = preparer.prepare(
        chunk_id="chunk-1",
        text="123456",
        content_hash="sha256:one",
    )

    assert prepared.request.text == "12345"
    assert prepared.request.content_hash == "sha256:one"
    assert prepared.original_text_length == 6
    assert prepared.prepared_text_length == 5
    assert prepared.truncated is True
    assert prepared.reason is not None
    assert "truncated" in prepared.reason


def test_text_preparer_boundary_behavior() -> None:
    preparer = EmbeddingTextPreparer(policy=EmbeddingPreparationPolicy(max_chars=5))

    exact = preparer.prepare(
        chunk_id="exact",
        text="12345",
        content_hash="sha256:exact",
    )
    over = preparer.prepare(
        chunk_id="over",
        text="123456",
        content_hash="sha256:over",
    )

    assert exact.truncated is False
    assert exact.request.text == "12345"
    assert over.truncated is True
    assert over.request.text == "12345"


def test_text_preparer_rejects_empty_fields() -> None:
    preparer = EmbeddingTextPreparer()

    with pytest.raises(ValueError, match="chunk_id"):
        preparer.prepare(chunk_id="", text="text", content_hash="sha256:one")
    with pytest.raises(ValueError, match="text"):
        preparer.prepare(chunk_id="chunk-1", text="", content_hash="sha256:one")
    with pytest.raises(ValueError, match="content_hash"):
        preparer.prepare(chunk_id="chunk-1", text="text", content_hash="")


def test_text_preparer_prepare_many_preserves_order() -> None:
    preparer = EmbeddingTextPreparer()

    prepared = preparer.prepare_many(
        (
            ("chunk-1", "first", "sha256:first"),
            ("chunk-2", "second", "sha256:second"),
        )
    )

    assert tuple(item.chunk_id for item in prepared) == ("chunk-1", "chunk-2")
    assert tuple(item.request.text for item in prepared) == ("first", "second")


def test_embedding_batcher_preserves_output_order_across_batches() -> None:
    preparer = EmbeddingTextPreparer()
    prepared = preparer.prepare_many(
        (
            ("chunk-1", "first", "sha256:first"),
            ("chunk-2", "second", "sha256:second"),
            ("chunk-3", "third", "sha256:third"),
        )
    )
    batcher = EmbeddingBatcher(
        service=EmbeddingService(provider=MockEmbeddingProvider()),
        policy=EmbeddingPreparationPolicy(batch_size=2),
    )

    records = batcher.embed(prepared)

    assert tuple(record.chunk_id for record in records) == (
        "chunk-1",
        "chunk-2",
        "chunk-3",
    )
    assert tuple(record.content_hash for record in records) == (
        "sha256:first",
        "sha256:second",
        "sha256:third",
    )


def test_embedding_batcher_empty_input_returns_empty_output() -> None:
    batcher = EmbeddingBatcher(
        service=EmbeddingService(provider=MockEmbeddingProvider())
    )

    assert batcher.embed(()) == ()


def test_embedding_batcher_bubbles_service_failure_with_chunk_context() -> None:
    preparer = EmbeddingTextPreparer()
    prepared = preparer.prepare_many(
        (
            ("chunk-1", "ok", "sha256:ok"),
            ("chunk-2", "fail", "sha256:fail"),
        )
    )
    batcher = EmbeddingBatcher(
        service=EmbeddingService(provider=MockEmbeddingProvider(fail_on_text="fail")),
        policy=EmbeddingPreparationPolicy(batch_size=1),
    )

    with pytest.raises(EmbeddingServiceError, match="chunk-2"):
        batcher.embed(prepared)
