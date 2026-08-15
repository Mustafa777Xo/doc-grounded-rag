from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum

from rag.contracts.chunk import Chunk

RETRIEVAL_QUERY_SCHEMA_VERSION = "retrieval_query.v1"
RETRIEVAL_CANDIDATE_SCHEMA_VERSION = "retrieval_candidate.v1"


class RetrieverSource(StrEnum):
    DENSE = "dense"
    KEYWORD = "keyword"


@dataclass(frozen=True)
class QueryFilters:
    doc_ids: frozenset[str] = frozenset()
    source_files: frozenset[str] = frozenset()
    pages: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if any(not doc_id for doc_id in self.doc_ids):
            raise ValueError("doc_ids cannot contain empty values")
        if any(not source_file for source_file in self.source_files):
            raise ValueError("source_files cannot contain empty values")
        if any(page < 0 for page in self.pages):
            raise ValueError("pages cannot contain negative values")

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_ids": sorted(self.doc_ids),
            "source_files": sorted(self.source_files),
            "pages": sorted(self.pages),
        }


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    original_text: str
    normalized_text: str
    filters: QueryFilters = field(default_factory=QueryFilters)

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("query_id cannot be empty")
        if not self.original_text.strip():
            raise ValueError("original_text cannot be blank")
        if not self.normalized_text.strip():
            raise ValueError("normalized_text cannot be blank")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RETRIEVAL_QUERY_SCHEMA_VERSION,
            "query_id": self.query_id,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "filters": self.filters.to_dict(),
        }

    def to_json(self) -> str:
        return _to_canonical_json(self.to_dict())


@dataclass(frozen=True)
class ScoreProvenance:
    dense_score: float | None = None
    keyword_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        scores = {
            "dense_score": self.dense_score,
            "keyword_score": self.keyword_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
        }
        for name, score in scores.items():
            if score is not None and not math.isfinite(score):
                raise ValueError(f"{name} must be finite")
        if self.dense_score is None and self.keyword_score is None:
            raise ValueError("at least one retriever score is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "dense_score": self.dense_score,
            "keyword_score": self.keyword_score,
            "fusion_score": self.fusion_score,
            "rerank_score": self.rerank_score,
        }


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: Chunk
    scores: ScoreProvenance
    sources: frozenset[RetrieverSource]
    rank: int | None = None

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("sources cannot be empty")
        has_dense_source = RetrieverSource.DENSE in self.sources
        has_keyword_source = RetrieverSource.KEYWORD in self.sources
        if has_dense_source != (self.scores.dense_score is not None):
            raise ValueError("dense source and dense_score must be present together")
        if has_keyword_source != (self.scores.keyword_score is not None):
            raise ValueError(
                "keyword source and keyword_score must be present together"
            )
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be at least 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RETRIEVAL_CANDIDATE_SCHEMA_VERSION,
            "chunk": self.chunk.to_dict(),
            "scores": self.scores.to_dict(),
            "sources": sorted(source.value for source in self.sources),
            "rank": self.rank,
        }

    def to_json(self) -> str:
        return _to_canonical_json(self.to_dict())


def _to_canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
