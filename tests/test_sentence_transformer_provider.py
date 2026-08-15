from __future__ import annotations

from typing import Any

import pytest

from rag.embed import EmbeddingProviderError, SentenceTransformerEmbeddingProvider


class FakeSentenceTransformer:
    def __init__(self, *, dim: int = 3) -> None:
        self.dim = dim
        self.max_seq_length = 0
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []

    def get_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.encode_calls.append((sentences, kwargs))
        return [[1.0] + [0.0] * (self.dim - 1) for _ in sentences]


def test_provider_loads_pinned_model_from_local_cache_only() -> None:
    model = FakeSentenceTransformer()
    load_calls: list[tuple[str, dict[str, object]]] = []

    def loader(model_name: str, **kwargs: object) -> FakeSentenceTransformer:
        load_calls.append((model_name, kwargs))
        return model

    provider = SentenceTransformerEmbeddingProvider(
        model_name="sentence-transformers/test-model",
        model_version="revision-1",
        dim=3,
        max_sequence_length=128,
        model_loader=loader,
    )

    assert load_calls == [
        (
            "sentence-transformers/test-model",
            {
                "revision": "revision-1",
                "device": "cpu",
                "local_files_only": True,
            },
        )
    ]
    assert model.max_seq_length == 128
    assert provider.model_name == "sentence-transformers/test-model"
    assert provider.model_version == "revision-1"


def test_provider_encodes_one_true_batch_in_order_and_normalizes() -> None:
    model = FakeSentenceTransformer()
    provider = SentenceTransformerEmbeddingProvider(
        dim=3,
        model_loader=lambda _name, **_kwargs: model,
    )

    vectors = provider.embed_batch(("first", "second"))

    assert vectors == ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert model.encode_calls == [
        (
            ["first", "second"],
            {
                "batch_size": 2,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[1.0, 0.0]], "dimension"),
        ([[2.0, 0.0, 0.0]], "non-normalized"),
        ([[float("nan"), 0.0, 0.0]], "non-finite"),
    ],
)
def test_provider_rejects_invalid_model_output(
    vectors: list[list[float]], message: str
) -> None:
    model = FakeSentenceTransformer()
    model.encode = lambda _sentences, **_kwargs: vectors  # type: ignore[method-assign]
    provider = SentenceTransformerEmbeddingProvider(
        dim=3,
        model_loader=lambda _name, **_kwargs: model,
    )

    with pytest.raises(EmbeddingProviderError, match=message):
        provider.embed_batch(("text",))


def test_provider_reports_cache_miss_with_prefetch_command() -> None:
    def loader(_model_name: str, **_kwargs: object) -> Any:
        raise OSError("cache miss")

    with pytest.raises(EmbeddingProviderError, match="prefetch"):
        SentenceTransformerEmbeddingProvider(model_loader=loader)


def test_provider_rejects_unexpected_loaded_dimension() -> None:
    model = FakeSentenceTransformer(dim=2)

    with pytest.raises(EmbeddingProviderError, match="dimension 2, expected 3"):
        SentenceTransformerEmbeddingProvider(
            dim=3,
            model_loader=lambda _name, **_kwargs: model,
        )
