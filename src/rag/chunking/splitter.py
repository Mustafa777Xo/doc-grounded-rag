from __future__ import annotations

from dataclasses import dataclass

from rag.chunking.ids import ChunkIdFactory, ChunkIdGenerator, ensure_unique_chunk_ids
from rag.config import Settings
from rag.contracts.chunk import Chunk
from rag.contracts.document import Document, ParsedPage


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int


@dataclass(frozen=True)
class ChunkingPolicy:
    target_size: int = 512
    overlap: int = 64
    hard_max_size: int = 768

    def __post_init__(self) -> None:
        if self.target_size <= 0:
            raise ValueError("target_size must be greater than zero")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
        if self.hard_max_size <= 0:
            raise ValueError("hard_max_size must be greater than zero")
        if self.overlap >= self.target_size:
            raise ValueError("overlap must be less than target_size")
        if self.target_size > self.hard_max_size:
            raise ValueError("target_size must be less than or equal to hard_max_size")

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkingPolicy:
        return cls(
            target_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            hard_max_size=settings.chunk_hard_max,
        )


class ChunkSplitter:
    def __init__(
        self,
        policy: ChunkingPolicy | None = None,
        id_generator: ChunkIdFactory | None = None,
    ) -> None:
        self._policy = policy if policy is not None else ChunkingPolicy()
        self._id_generator = (
            id_generator if id_generator is not None else ChunkIdGenerator()
        )

    def split_document(self, document: Document) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        for page in document.pages:
            page_spans = self._split_page(page)
            for span in page_spans:
                chunk_index = len(chunks)
                text = page.text[span.start : span.end]
                chunks.append(
                    Chunk(
                        chunk_id=self._id_generator.generate(
                            doc_id=document.doc_id,
                            page=page.page_number,
                            char_start=span.start,
                            char_end=span.end,
                            text=text,
                        ),
                        doc_id=document.doc_id,
                        source_file=document.source_file,
                        page=page.page_number,
                        chunk_index=chunk_index,
                        char_start=span.start,
                        char_end=span.end,
                        text=text,
                    )
                )
        result = tuple(chunks)
        ensure_unique_chunk_ids(result)
        return result

    def _split_page(self, page: ParsedPage) -> tuple[TextSpan, ...]:
        if page.text.strip() == "":
            return ()

        base_spans = self._build_base_spans(page.text)
        if not base_spans:
            return ()

        return tuple(self._apply_overlap(base_spans, page.text))

    def _build_base_spans(self, text: str) -> list[TextSpan]:
        spans: list[TextSpan] = []
        current_start: int | None = None
        current_end: int | None = None

        for block in self._paragraph_blocks(text):
            for span in self._split_block_if_needed(block, text):
                if current_start is None or current_end is None:
                    current_start = span.start
                    current_end = span.end
                    continue

                proposed_end = span.end
                proposed_len = proposed_end - current_start
                current_len = current_end - current_start
                if (
                    proposed_len <= self._policy.hard_max_size
                    and current_len < self._policy.target_size
                    and proposed_len <= self._policy.target_size
                ):
                    current_end = proposed_end
                else:
                    spans.append(TextSpan(start=current_start, end=current_end))
                    current_start = span.start
                    current_end = span.end

        if current_start is not None and current_end is not None:
            spans.append(TextSpan(start=current_start, end=current_end))
        return spans

    def _paragraph_blocks(self, text: str) -> list[TextSpan]:
        spans: list[TextSpan] = []
        cursor = 0
        for part in text.split("\n\n"):
            start = cursor
            end = start + len(part)
            stripped_start, stripped_end = self._trim_span(text, start, end)
            if stripped_start < stripped_end:
                spans.append(TextSpan(start=stripped_start, end=stripped_end))
            cursor = end + 2
        return spans

    def _split_block_if_needed(self, span: TextSpan, text: str) -> list[TextSpan]:
        if span.end - span.start <= self._policy.hard_max_size:
            return [span]

        spans: list[TextSpan] = []
        start = span.start
        while start < span.end:
            end = min(start + self._policy.hard_max_size, span.end)
            if end < span.end:
                boundary = self._find_split_boundary(text, start, end)
                if boundary > start:
                    end = boundary
            trimmed_start, trimmed_end = self._trim_span(text, start, end)
            if trimmed_start < trimmed_end:
                spans.append(TextSpan(start=trimmed_start, end=trimmed_end))
            start = self._next_non_whitespace(text, end, span.end)
        return spans

    def _find_split_boundary(self, text: str, start: int, end: int) -> int:
        lower_bound = start + max(1, self._policy.target_size)
        for index in range(end, lower_bound, -1):
            if text[index - 1].isspace():
                return index - 1
        return end

    def _apply_overlap(self, spans: list[TextSpan], text: str) -> list[TextSpan]:
        if self._policy.overlap == 0:
            return spans

        overlapped: list[TextSpan] = []
        previous: TextSpan | None = None
        for span in spans:
            if previous is None:
                overlapped.append(span)
                previous = span
                continue
            start = max(
                0,
                previous.end - self._policy.overlap,
                span.end - self._policy.hard_max_size,
            )
            start = self._next_non_whitespace(text, start, span.end)
            overlapped_span = TextSpan(start=start, end=span.end)
            overlapped.append(overlapped_span)
            previous = overlapped_span
        return overlapped

    def _trim_span(self, text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def _next_non_whitespace(self, text: str, start: int, end: int) -> int:
        while start < end and text[start].isspace():
            start += 1
        return start
