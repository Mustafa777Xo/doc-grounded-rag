from __future__ import annotations

import importlib
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from rag.contracts.chunk import Chunk
from rag.contracts.retrieval import (
    KeywordDiagnostics,
    RetrievalCandidate,
    RetrievalQuery,
    RetrieverSource,
    ScoreProvenance,
)
from rag.errors import RetrievalError
from rag.retrieve.filters import matches_query_filters
from rag.retrieve.normalization import normalize_retrieval_text

BM25_METHOD = "lucene"
BM25_K1 = 1.5
BM25_B = 0.75
LEXICAL_TOKEN_PATTERN = r"(?u)\b\w+\b"
_TOKEN_RE = re.compile(LEXICAL_TOKEN_PATTERN)


class KeywordIndexError(RuntimeError):
    """Raised when an immutable keyword index cannot be built."""


class KeywordScoreEngine(Protocol):
    def score(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]: ...


KeywordScoreEngineFactory = Callable[[tuple[tuple[str, ...], ...]], KeywordScoreEngine]


class BM25KeywordRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        score_engine_factory: KeywordScoreEngineFactory | None = None,
    ) -> None:
        ordered_chunks = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        chunk_ids = tuple(chunk.chunk_id for chunk in ordered_chunks)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise KeywordIndexError("keyword corpus contains duplicate chunk IDs")

        self._chunks = ordered_chunks
        self._corpus_tokens = tuple(
            tokenize_lexical_text(chunk.text) for chunk in ordered_chunks
        )
        self._engine: KeywordScoreEngine | None = None
        if ordered_chunks and any(self._corpus_tokens):
            factory = (
                score_engine_factory
                if score_engine_factory is not None
                else _build_bm25_score_engine
            )
            try:
                self._engine = factory(self._corpus_tokens)
            except Exception as exc:
                raise KeywordIndexError(f"failed to build BM25 index: {exc}") from exc

    def retrieve(
        self,
        query: RetrievalQuery,
        limit: int = 5,
    ) -> tuple[RetrievalCandidate, ...]:
        if limit <= 0:
            raise RetrievalError(
                stage="keyword",
                query_id=query.query_id,
                message="limit must be greater than zero",
                hint="Set keyword_top_k to a positive value.",
            )

        query_tokens = tokenize_lexical_text(query.normalized_text)
        if not query_tokens or self._engine is None:
            return ()
        try:
            scores = self._engine.score(query_tokens)
        except Exception as exc:
            raise RetrievalError(
                stage="keyword",
                query_id=query.query_id,
                message=f"BM25 scoring failed: {exc}",
                hint="Rebuild the keyword retriever from the current chunk snapshot.",
            ) from exc
        if len(scores) != len(self._chunks):
            raise RetrievalError(
                stage="keyword",
                query_id=query.query_id,
                message=(
                    f"BM25 returned {len(scores)} scores for {len(self._chunks)} chunks"
                ),
                hint="Rebuild the keyword retriever from the current chunk snapshot.",
            )

        query_term_set = set(query_tokens)
        matches: list[tuple[float, Chunk, tuple[str, ...]]] = []
        for chunk, chunk_tokens, score in zip(
            self._chunks,
            self._corpus_tokens,
            scores,
            strict=True,
        ):
            if not math.isfinite(score):
                raise RetrievalError(
                    stage="keyword",
                    query_id=query.query_id,
                    message=f"BM25 returned a non-finite score for {chunk.chunk_id!r}",
                    hint="Rebuild the keyword index and verify the BM25S runtime.",
                )
            if score <= 0.0 or not matches_query_filters(chunk, query.filters):
                continue
            matched_terms = tuple(sorted(query_term_set.intersection(chunk_tokens)))
            if matched_terms:
                matches.append((score, chunk, matched_terms))

        ordered_matches = sorted(
            matches,
            key=lambda match: (-match[0], match[1].chunk_id),
        )[:limit]
        return tuple(
            RetrievalCandidate(
                chunk=chunk,
                scores=ScoreProvenance(keyword_score=score),
                sources=frozenset({RetrieverSource.KEYWORD}),
                keyword_diagnostics=KeywordDiagnostics(
                    source_rank=rank,
                    matched_terms=matched_terms,
                ),
                rank=rank,
            )
            for rank, (score, chunk, matched_terms) in enumerate(
                ordered_matches,
                start=1,
            )
        )


def tokenize_lexical_text(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_retrieval_text(text)))


class _BM25ScoreEngine:
    def __init__(
        self,
        model: Any,
    ) -> None:
        self._model = model

    def score(self, query_tokens: tuple[str, ...]) -> tuple[float, ...]:
        raw_scores = self._model.get_scores(list(query_tokens))
        return tuple(float(score) for score in raw_scores)


def _build_bm25_score_engine(
    corpus_tokens: tuple[tuple[str, ...], ...],
) -> KeywordScoreEngine:
    try:
        module = importlib.import_module("bm25s")
    except ImportError as exc:
        raise KeywordIndexError("bm25s is not installed; run `make install`") from exc

    vocabulary = sorted({token for tokens in corpus_tokens for token in tokens})
    vocab = {token: index for index, token in enumerate(vocabulary)}
    corpus_ids = [[vocab[token] for token in tokens] for tokens in corpus_tokens]
    model_class = cast(Callable[..., Any], module.BM25)
    model = model_class(
        method=BM25_METHOD,
        k1=BM25_K1,
        b=BM25_B,
        dtype="float32",
        backend="numpy",
        csc_backend="numpy",
    )
    model.index(
        (corpus_ids, vocab),
        create_empty_token=False,
        show_progress=False,
    )
    return _BM25ScoreEngine(model)
