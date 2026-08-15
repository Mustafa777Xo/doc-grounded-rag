from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from rag.contracts.indexing import EMBEDDING_SCHEMA_VERSION
from rag.embed import (
    EmbeddingRequest,
    EmbeddingService,
    EmbeddingServiceError,
    HashEmbeddingProvider,
    MockEmbeddingProvider,
)


def _request(
    *,
    chunk_id: str = "chunk-1",
    text: str = "Policy coverage applies.",
    content_hash: str = "sha256:abc123",
) -> EmbeddingRequest:
    return EmbeddingRequest(
        chunk_id=chunk_id,
        text=text,
        content_hash=content_hash,
    )


class BatchTrackingProvider:
    dim = 2
    model_name = "batch-tracker"
    model_version = "test"

    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def embed_text(self, text: str) -> tuple[float, ...]:
        raise AssertionError("embed_text must not be called for a batch")

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        batch = tuple(texts)
        self.batches.append(batch)
        return tuple((1.0, 0.0) for _ in batch)


def test_embedding_service_embeds_one_text_as_record() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider())

    record = service.embed_one(_request())

    assert record.schema_version == EMBEDDING_SCHEMA_VERSION
    assert record.chunk_id == "chunk-1"
    assert record.vector == (1.0 / 3.0, 2.0 / 3.0, 1.0)
    assert record.dim == 3
    assert record.model_name == "mock-embedder"
    assert record.model_version == "test"
    assert record.content_hash == "sha256:abc123"


def test_embedding_service_embeds_batch_in_input_order() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider())
    requests = (
        _request(chunk_id="chunk-1", text="first", content_hash="sha256:first"),
        _request(chunk_id="chunk-2", text="second", content_hash="sha256:second"),
    )

    records = service.embed_batch(requests)

    assert tuple(record.chunk_id for record in records) == ("chunk-1", "chunk-2")
    assert tuple(record.content_hash for record in records) == (
        "sha256:first",
        "sha256:second",
    )


def test_embedding_service_sends_each_prepared_batch_in_one_provider_call() -> None:
    provider = BatchTrackingProvider()
    service = EmbeddingService(provider=provider)
    requests = (
        _request(chunk_id="chunk-1", text="first"),
        _request(chunk_id="chunk-2", text="second"),
    )

    records = service.embed_batch(requests)

    assert provider.batches == [("first", "second")]
    assert tuple(record.chunk_id for record in records) == ("chunk-1", "chunk-2")


def test_embedding_service_hides_provider_response_shape() -> None:
    service = EmbeddingService(
        provider=MockEmbeddingProvider(
            dim=2,
            model_name="custom-provider",
            model_version="v9",
        )
    )

    record = service.embed_one(_request())

    assert record.to_dict() == {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "chunk_id": "chunk-1",
        "vector": [0.5, 1.0],
        "dim": 2,
        "model_name": "custom-provider",
        "model_version": "v9",
        "content_hash": "sha256:abc123",
    }


def test_embedding_request_rejects_empty_fields() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        _request(chunk_id="")
    with pytest.raises(ValueError, match="text"):
        _request(text="")
    with pytest.raises(ValueError, match="content_hash"):
        _request(content_hash="")


def test_embedding_service_wraps_provider_failure_with_chunk_context() -> None:
    service = EmbeddingService(
        provider=MockEmbeddingProvider(fail_on_text="provider unavailable")
    )

    with pytest.raises(EmbeddingServiceError, match="chunk-err") as exc_info:
        service.embed_one(_request(chunk_id="chunk-err", text="provider unavailable"))

    assert exc_info.value.chunk_id == "chunk-err"
    assert "provider error" in str(exc_info.value)


def test_embedding_service_rejects_wrong_provider_dimension() -> None:
    service = EmbeddingService(provider=MockEmbeddingProvider(force_wrong_dim=True))

    with pytest.raises(EmbeddingServiceError, match="dimension"):
        service.embed_one(_request(chunk_id="chunk-dim"))


def test_hash_embedding_provider_is_deterministic_for_same_text() -> None:
    provider = HashEmbeddingProvider(dim=8)

    first = provider.embed_text("same text")
    second = provider.embed_text("same text")

    assert first == second


def test_hash_embedding_provider_changes_for_different_text() -> None:
    provider = HashEmbeddingProvider(dim=8)

    assert provider.embed_text("first text") != provider.embed_text("second text")


def test_hash_embedding_provider_returns_finite_expected_dim() -> None:
    provider = HashEmbeddingProvider(dim=16)

    vector = provider.embed_text("Policy coverage applies.")

    assert len(vector) == 16
    assert all(math.isfinite(value) for value in vector)
    assert all(-1.0 <= value <= 1.0 for value in vector)


def test_hash_embedding_provider_rejects_invalid_config_and_empty_text() -> None:
    with pytest.raises(ValueError, match="dim"):
        HashEmbeddingProvider(dim=0)
    with pytest.raises(ValueError, match="model_name"):
        HashEmbeddingProvider(model_name="")
    with pytest.raises(ValueError, match="model_version"):
        HashEmbeddingProvider(model_version="")
    with pytest.raises(Exception, match="text"):
        HashEmbeddingProvider().embed_text("")


def test_mock_embedding_provider_is_deterministic_and_configurable() -> None:
    provider = MockEmbeddingProvider(dim=4)

    assert provider.embed_text("alpha") == provider.embed_text("beta")
    assert provider.embed_text("alpha") == (0.25, 0.5, 0.75, 1.0)
