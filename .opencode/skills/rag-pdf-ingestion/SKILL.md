---
name: rag-pdf-ingestion
description: Use for PDF discovery, PyPDF extraction, page normalization, chunking, empty-page handling, ingestion failures, or citation metadata changes in this doc-grounded-rag project.
---

# RAG PDF Ingestion

Keep ingestion deterministic, inspectable, and separate from downstream stages.

## Workflow

1. Read `docs/ingestion-and-chunking.md` and the contracts used by the affected path.
2. Keep file discovery, PDF parsing, normalization, chunking, and persistence as separate concerns.
3. Support text-based PDFs only. Do not introduce OCR unless the user explicitly changes v1 scope.
4. Preserve `doc_id`, `source_file`, page number, `chunk_id`, `chunk_index`, character spans, and raw text through every handoff.
5. Treat empty pages and extraction failures explicitly; do not silently discard evidence.
6. Ensure identical input and configuration produce identical normalized text, chunks, and identifiers.
7. Add focused tests for the changed boundary, including empty and malformed inputs where applicable.
8. Verify with focused pytest tests, then use `make check` for completion.

Do not combine parsing, chunking, embedding, and indexing into one function.
Prefer dependency injection for filesystem and parser boundaries.
