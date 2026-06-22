from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag.contracts.chunk import Chunk


class ChunkIdFactory(Protocol):
    def generate(
        self,
        *,
        doc_id: str,
        page: int,
        char_start: int,
        char_end: int,
        text: str,
    ) -> str: ...


class ChunkIdCollisionError(ValueError):
    def __init__(
        self,
        *,
        chunk_id: str,
        first_index: int,
        duplicate_index: int,
    ) -> None:
        self.chunk_id = chunk_id
        self.first_index = first_index
        self.duplicate_index = duplicate_index
        super().__init__(
            "Duplicate chunk_id detected "
            f"(chunk_id={chunk_id}, first_index={first_index}, "
            f"duplicate_index={duplicate_index})"
        )


@dataclass(frozen=True)
class ChunkIdGenerator:
    hash_length: int = 12

    def __post_init__(self) -> None:
        if self.hash_length <= 0:
            raise ValueError("hash_length must be greater than zero")
        if self.hash_length > 64:
            raise ValueError("hash_length cannot exceed SHA-256 hex length")

    def generate(
        self,
        *,
        doc_id: str,
        page: int,
        char_start: int,
        char_end: int,
        text: str,
    ) -> str:
        if not doc_id:
            raise ValueError("doc_id cannot be empty")
        if page < 0:
            raise ValueError("page cannot be negative")
        if char_start < 0:
            raise ValueError("char_start cannot be negative")
        if char_end <= char_start:
            raise ValueError("char_end must be greater than char_start")
        if not text:
            raise ValueError("text cannot be empty")

        payload = {
            "doc_id": doc_id,
            "page": page,
            "char_start": char_start,
            "char_end": char_end,
            "text": text,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()[: self.hash_length]
        return f"{doc_id}-p{page}-s{char_start}-e{char_end}-{fingerprint}"


def ensure_unique_chunk_ids(chunks: Sequence[Chunk]) -> None:
    seen: dict[str, int] = {}
    for index, chunk in enumerate(chunks):
        first_index = seen.get(chunk.chunk_id)
        if first_index is not None:
            raise ChunkIdCollisionError(
                chunk_id=chunk.chunk_id,
                first_index=first_index,
                duplicate_index=index,
            )
        seen[chunk.chunk_id] = index
