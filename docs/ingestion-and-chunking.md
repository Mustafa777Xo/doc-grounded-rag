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

## Chunking

Chunking runs after normalization and converts parsed pages into page-local
chunks. It is deterministic for the same normalized document and policy.

Default chunking policy:

- target size: `512` characters
- overlap: `64` characters
- hard max size: `768` characters

Chunking behavior:

- processes pages in document order
- skips empty and whitespace-only pages
- keeps chunks within one source page
- preserves `doc_id`, `source_file`, page number, and character spans
- prefers paragraph boundaries when they fit the policy
- splits very long paragraphs with deterministic character windows
- applies overlap only within the same page
- generates stable content-aware chunk IDs

Chunk ID behavior:

- format: `{doc_id}-p{page}-s{char_start}-e{char_end}-{hash12}`
- hash input includes `doc_id`, page, span, and exact chunk text
- collisions within one splitter output fail explicitly

## Chunk Storage

Chunk storage writes one schema-valid chunk record per JSONL line. The initial
artifact format is DB-compatible and intentionally flat so each line can map to
one future index or database row.

Stored record behavior:

- schema version: `chunk.v1`
- output path default: `data/processed/chunks/chunks.jsonl`
- write mode default: `overwrite`
- supported write modes: `overwrite`, `safe_append`
- `overwrite` replaces the artifact with the current validated batch
- `safe_append` validates existing records and fails if any `chunk_id` would be
  duplicated
- malformed existing JSONL files fail explicitly before appending

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
  "schema_version": "chunk.v1",
  "chunk_id": "policy-2026-p0-s0-e47-a1b2c3d4e5f6",
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
  "schema_version": "chunk.v1",
  "chunk_id": "policy-2026-p1-s0-e41-f6e5d4c3b2a1",
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
metadata and Sprint 2 indexing.
