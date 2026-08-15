from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence


class EmbeddingProviderError(RuntimeError):
    """Raised by embedding providers when a text cannot be embedded."""


class EmbeddingBatchError(EmbeddingProviderError):
    def __init__(self, message: str, *, failed_index: int) -> None:
        if failed_index < 0:
            raise ValueError("failed_index cannot be negative")
        super().__init__(message)
        self.failed_index = failed_index


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_text(self, text: str) -> tuple[float, ...]: ...

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True)
class HashEmbeddingProvider:
    dim: int = 16
    model_name: str = "local-hash-embedder"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be greater than zero")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.model_version:
            raise ValueError("model_version cannot be empty")

    def embed_text(self, text: str) -> tuple[float, ...]:
        if text == "":
            raise EmbeddingProviderError("text cannot be empty")

        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        return tuple(values[: self.dim])

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for index, text in enumerate(texts):
            try:
                vectors.append(self.embed_text(text))
            except EmbeddingProviderError as exc:
                raise EmbeddingBatchError(str(exc), failed_index=index) from exc
        return tuple(vectors)


@dataclass(frozen=True)
class MockEmbeddingProvider:
    dim: int = 3
    model_name: str = "mock-embedder"
    model_version: str = "test"
    fail_on_text: str | None = None
    force_wrong_dim: bool = False

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be greater than zero")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.model_version:
            raise ValueError("model_version cannot be empty")

    def embed_text(self, text: str) -> tuple[float, ...]:
        if self.fail_on_text is not None and text == self.fail_on_text:
            raise EmbeddingProviderError(f"mock provider failed for text: {text}")

        vector = tuple(float(index + 1) / float(self.dim) for index in range(self.dim))
        if self.force_wrong_dim:
            return vector + (1.0,)
        return vector

    def embed_batch(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for index, text in enumerate(texts):
            try:
                vectors.append(self.embed_text(text))
            except EmbeddingProviderError as exc:
                raise EmbeddingBatchError(str(exc), failed_index=index) from exc
        return tuple(vectors)
