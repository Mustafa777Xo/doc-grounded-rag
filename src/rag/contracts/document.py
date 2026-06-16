from __future__ import annotations

import json
from dataclasses import dataclass

@dataclass(frozen=True)
class Document:
    doc_id: str
    source_file: str
    pages: tuple[str, ...] # raw text per page, 0-indexed (pages[0] = page 1)
    
    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id cannot be empty")
        if not self.source_file:
            raise ValueError("source_file cannot be empty")
        if not self.pages:
            raise ValueError("pages cannot be empty")
    
    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id":self.doc_id,
            "source_file": self.source_file,
            "pages": list(self.pages)
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    