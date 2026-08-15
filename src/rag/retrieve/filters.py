from __future__ import annotations

from typing import Protocol

from rag.contracts.retrieval import QueryFilters


class FilterableMetadata(Protocol):
    @property
    def doc_id(self) -> str: ...

    @property
    def source_file(self) -> str: ...

    @property
    def page(self) -> int: ...


def matches_query_filters(
    metadata: FilterableMetadata,
    filters: QueryFilters,
) -> bool:
    return (
        (not filters.doc_ids or metadata.doc_id in filters.doc_ids)
        and (not filters.source_files or metadata.source_file in filters.source_files)
        and (not filters.pages or metadata.page in filters.pages)
    )
