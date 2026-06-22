from __future__ import annotations

import re
from dataclasses import dataclass

from rag.contracts.document import Document, ParsedPage

_LIST_ITEM_RE = re.compile(r"^(?:[-*]\s+|\d+[.)]\s+)")
_REPEATED_SPACE_RE = re.compile(r"[ \t]+")
_REPEATED_SEPARATOR_RUN_RE = re.compile(r"([=_*\-])\1{3,}")
_SEPARATOR_LINE_RE = re.compile(r"^([=_*\-])\1{2,}$")


@dataclass(frozen=True)
class NormalizationPolicy:
    collapse_whitespace: bool = True
    collapse_blank_lines: bool = True
    collapse_separator_runs: bool = True
    join_wrapped_lines: bool = True
    preserve_non_empty_pages: bool = True


class TextNormalizer:
    def __init__(self, policy: NormalizationPolicy | None = None) -> None:
        self._policy = policy if policy is not None else NormalizationPolicy()

    def normalize_document(self, document: Document) -> Document:
        return Document(
            doc_id=document.doc_id,
            source_file=document.source_file,
            source_path=document.source_path,
            total_pages=document.total_pages,
            pages=tuple(
                ParsedPage(
                    doc_id=page.doc_id,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    text=self.normalize_text(page.text),
                )
                for page in document.pages
            ),
        )

    def normalize_text(self, text: str) -> str:
        if text == "":
            return ""

        trimmed_original = text.strip()
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
        lines = [self._normalize_line(line) for line in normalized.split("\n")]
        if self._policy.collapse_blank_lines:
            lines = self._collapse_blank_lines(lines)
        if self._policy.join_wrapped_lines:
            normalized = self._join_wrapped_lines(lines)
        else:
            normalized = "\n".join(lines).strip()

        if normalized == "" and self._policy.preserve_non_empty_pages:
            return trimmed_original
        return normalized

    def _normalize_line(self, line: str) -> str:
        normalized = line.strip()
        if self._policy.collapse_whitespace:
            normalized = _REPEATED_SPACE_RE.sub(" ", normalized)
        if self._policy.collapse_separator_runs:
            normalized = _REPEATED_SEPARATOR_RUN_RE.sub(r"\1\1\1", normalized)
        return normalized

    def _collapse_blank_lines(self, lines: list[str]) -> list[str]:
        collapsed: list[str] = []
        previous_blank = False
        for line in lines:
            is_blank = line == ""
            if is_blank and previous_blank:
                continue
            collapsed.append(line)
            previous_blank = is_blank
        return collapsed

    def _join_wrapped_lines(self, lines: list[str]) -> str:
        output: list[str] = []
        paragraph: list[str] = []
        previous_was_separator = False

        for line in lines:
            if line == "":
                self._flush_paragraph(output, paragraph)
                if output and output[-1] != "":
                    output.append("")
                previous_was_separator = False
                continue

            if self._is_separator_line(line):
                self._flush_paragraph(output, paragraph)
                if not previous_was_separator:
                    output.append(line)
                previous_was_separator = True
                continue

            previous_was_separator = False
            if self._is_structural_line(line):
                self._flush_paragraph(output, paragraph)
                output.append(line)
                continue

            paragraph.append(line)

        self._flush_paragraph(output, paragraph)
        return "\n".join(output).strip()

    def _is_structural_line(self, line: str) -> bool:
        return _LIST_ITEM_RE.match(line) is not None or self._is_heading(line)

    def _is_heading(self, line: str) -> bool:
        if line.endswith(":"):
            return True
        letters = [char for char in line if char.isalpha()]
        return bool(letters) and line.upper() == line and len(line) <= 80

    def _is_separator_line(self, line: str) -> bool:
        return _SEPARATOR_LINE_RE.match(line) is not None

    def _flush_paragraph(self, output: list[str], paragraph: list[str]) -> None:
        if not paragraph:
            return
        output.append(" ".join(paragraph))
        paragraph.clear()
