# Ingestion and Chunking

Sprint 1 defines the contracts used to hand PDF text from ingestion into
chunking, storage, indexing, retrieval, and citations.

## Contract Rules

- Page numbers are zero-based.
- Chunk spans use half-open Python slicing semantics: `[char_start, char_end)`.
- Parsed pages may contain empty text so ingestion can preserve PDF page count.
- Chunks must contain non-empty text.
- Every parsed page and chunk must preserve `doc_id` and `source_file`.
- Parsed documents also preserve `source_path` and `total_pages`.

## PDF Parser

The Sprint 1 parser uses `pypdf` for text-based PDFs. It does not perform OCR,
normalization, chunking, or table reconstruction.

Parser behavior:

- `doc_id` is the first 16 hex characters of the PDF file's SHA-256 hash.
- `source_file` is the PDF filename.
- `source_path` is the input path serialized as a string.
- `total_pages` is the number of PDF pages reported by the reader.
- Empty pages are preserved with `text == ""`.
- Missing, unreadable, directory, or corrupt inputs raise an explicit parser
  error.

## Text Normalization

Normalization runs after parsing and before chunking. It is deterministic and
returns a new document with the same metadata and page order.

Default normalization behavior:

- normalizes line endings
- collapses repeated spaces and tabs
- collapses repeated blank lines
- joins wrapped paragraph lines
- preserves headings, list items, and paragraph breaks
- collapses repeated separator runs
- preserves non-empty low-quality pages instead of reducing them to empty text

Normalization does not perform OCR, semantic cleanup, chunking, or table
reconstruction.

## Parsed Document Example

```json
{
  "doc_id": "policy-2026",
  "source_file": "policy.pdf",
  "source_path": "data/raw/policy.pdf",
  "total_pages": 2,
  "pages": [
    {
      "doc_id": "policy-2026",
      "source_file": "policy.pdf",
      "page_number": 0,
      "text": "Eligibility rules apply to full-time employees."
    },
    {
      "doc_id": "policy-2026",
      "source_file": "policy.pdf",
      "page_number": 1,
      "text": "Coverage begins after the waiting period."
    }
  ]
}
```

## Chunk Examples

```json
{
  "chunk_id": "policy-2026-p0-c0",
  "doc_id": "policy-2026",
  "source_file": "policy.pdf",
  "page": 0,
  "chunk_index": 0,
  "char_start": 0,
  "char_end": 47,
  "text": "Eligibility rules apply to full-time employees."
}
```

```json
{
  "chunk_id": "policy-2026-p1-c1",
  "doc_id": "policy-2026",
  "source_file": "policy.pdf",
  "page": 1,
  "chunk_index": 1,
  "char_start": 0,
  "char_end": 41,
  "text": "Coverage begins after the waiting period."
}
```

These fields are the minimum traceability data required for downstream citation
metadata. Later ingestion and chunking tickets will implement parsing,
normalization, deterministic chunk boundaries, stable IDs, and persistence.
