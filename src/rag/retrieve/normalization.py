from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from rag.contracts.retrieval import QueryFilters, RetrievalQuery
from rag.errors import RetrievalError


@dataclass(frozen=True)
class QueryNormalizer:
    def normalize(
        self,
        *,
        query_id: str,
        original_text: str,
        filters: QueryFilters | None = None,
    ) -> RetrievalQuery:
        normalized_text = unicodedata.normalize("NFKC", original_text)
        normalized_text = normalized_text.casefold()
        normalized_text = unicodedata.normalize("NFKC", normalized_text)
        normalized_text = " ".join(normalized_text.split())
        if not normalized_text:
            raise RetrievalError(
                stage="normalization",
                query_id=query_id,
                message="query is empty after normalization",
                hint="Provide a query containing non-whitespace text.",
            )
        try:
            return RetrievalQuery(
                query_id=query_id,
                original_text=original_text,
                normalized_text=normalized_text,
                filters=filters if filters is not None else QueryFilters(),
            )
        except ValueError as exc:
            raise RetrievalError(
                stage="normalization",
                query_id=query_id,
                message=str(exc),
                hint="Provide a non-empty query ID and query text.",
            ) from exc
