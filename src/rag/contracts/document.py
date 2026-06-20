from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedPage:
    doc_id: str
    source_file: str
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id cannot be empty")
        if not self.source_file:
            raise ValueError("source_file cannot be empty")
        if self.page_number < 0:
            raise ValueError("page_number cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "text": self.text,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass(frozen=True)
class Document:
    doc_id: str
    source_file: str
    pages: tuple[ParsedPage, ...]

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id cannot be empty")
        if not self.source_file:
            raise ValueError("source_file cannot be empty")
        if not self.pages:
            raise ValueError("pages cannot be empty")
        for expected_page_number, page in enumerate(self.pages):
            if page.doc_id != self.doc_id:
                raise ValueError("page doc_id must match document doc_id")
            if page.source_file != self.source_file:
                raise ValueError("page source_file must match document source_file")
            if page.page_number != expected_page_number:
                raise ValueError("pages must be zero-based and contiguous")

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
