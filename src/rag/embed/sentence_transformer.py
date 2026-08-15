from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from rag.embed.model_client import EmbeddingProviderError

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_MAX_SEQUENCE_LENGTH = 256


class SentenceTransformerModel(Protocol):
    max_seq_length: int

    def get_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: list[str], **kwargs: object) -> Any: ...


ModelLoader = Callable[..., SentenceTransformerModel]


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        model_version: str = DEFAULT_EMBEDDING_MODEL_REVISION,
        dim: int = DEFAULT_EMBEDDING_DIMENSION,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name cannot be empty")
        if not model_version:
            raise ValueError("model_version cannot be empty")
        if dim <= 0:
            raise ValueError("dim must be greater than zero")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be greater than zero")

        self._model_name = model_name
        self._model_version = model_version
        self._dim = dim
        loader = model_loader if model_loader is not None else _load_model
        try:
            self._model = loader(
                model_name,
                revision=model_version,
                device="cpu",
                local_files_only=True,
            )
        except Exception as exc:
            raise EmbeddingProviderError(
                "failed to load cached embedding model "
                f"{model_name!r} at revision {model_version!r}: {exc}. "
                "Run `python scripts/benchmark_retrieval_stack.py prefetch` "
                "before offline use."
            ) from exc

        actual_dim = self._model.get_embedding_dimension()
        if actual_dim != dim:
            raise EmbeddingProviderError(
                f"embedding model dimension {actual_dim}, expected {dim}"
            )
        self._model.max_seq_length = max_sequence_length

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> tuple[float, ...]:
        return self.embed_batch((text,))[0]

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        ordered_texts = tuple(texts)
        if not ordered_texts:
            return ()
        if any(not text for text in ordered_texts):
            raise EmbeddingProviderError("text cannot be empty")

        try:
            encoded = self._model.encode(
                list(ordered_texts),
                batch_size=len(ordered_texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingProviderError(
                f"embedding model encode failed: {exc}"
            ) from exc

        vectors = tuple(tuple(float(value) for value in vector) for vector in encoded)
        if len(vectors) != len(ordered_texts):
            raise EmbeddingProviderError(
                f"embedding model returned {len(vectors)} vectors, "
                f"expected {len(ordered_texts)}"
            )
        for vector in vectors:
            self._validate_vector(vector)
        return vectors

    def _validate_vector(self, vector: tuple[float, ...]) -> None:
        if len(vector) != self.dim:
            raise EmbeddingProviderError(
                f"embedding model returned dimension {len(vector)}, expected {self.dim}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingProviderError("embedding model returned non-finite values")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise EmbeddingProviderError(
                f"embedding model returned non-normalized vector with norm {norm}"
            )


def _load_model(model_name: str, **kwargs: object) -> SentenceTransformerModel:
    try:
        module = importlib.import_module("sentence_transformers")
    except ImportError as exc:
        raise EmbeddingProviderError(
            "sentence-transformers is not installed; run `make install`"
        ) from exc
    model_class = cast(Callable[..., object], module.SentenceTransformer)
    return cast(SentenceTransformerModel, model_class(model_name, **kwargs))
