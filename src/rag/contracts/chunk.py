from __future__ import annotations

import json
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int # start with 0
    chunk_id: int # position within the document, 0-indexed
    text: str