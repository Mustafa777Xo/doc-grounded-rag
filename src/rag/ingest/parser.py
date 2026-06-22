from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

from pypdf import PdfReader

from rag.contracts.document import Document, ParsedPage


class PdfParseError(Exception):
    def __init__(self, *, source_path: Path, message: str) -> None:
        self.source_path = source_path
        self.message = message
        super().__init__(f"Failed to parse PDF '{source_path}': {message}")


class PdfParser:
    def parse(self, source_path: Path) -> Document:
        if not source_path.exists():
            raise PdfParseError(source_path=source_path, message="file does not exist")
        if not source_path.is_file():
            raise PdfParseError(source_path=source_path, message="path is not a file")

        try:
            pdf_bytes = source_path.read_bytes()
        except OSError as exc:
            raise PdfParseError(
                source_path=source_path,
                message=f"file is unreadable: {exc}",
            ) from exc

        doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]

        try:
            reader = PdfReader(source_path)
            total_pages = len(reader.pages)
            if total_pages == 0:
                raise PdfParseError(source_path=source_path, message="PDF has no pages")

            pages = tuple(
                ParsedPage(
                    doc_id=doc_id,
                    source_file=source_path.name,
                    page_number=page_number,
                    text=cast(str | None, page.extract_text()) or "",
                )
                for page_number, page in enumerate(reader.pages)
            )
        except PdfParseError:
            raise
        except Exception as exc:
            raise PdfParseError(
                source_path=source_path,
                message=f"PDF could not be read: {exc}",
            ) from exc

        return Document(
            doc_id=doc_id,
            source_file=source_path.name,
            source_path=str(source_path),
            total_pages=total_pages,
            pages=pages,
        )
