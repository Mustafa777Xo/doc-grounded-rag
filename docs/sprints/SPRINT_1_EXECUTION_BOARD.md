# Sprint 1 Execution Board

Goal: implement reliable PDF ingestion and deterministic chunking with full traceability for downstream retrieval and citations.

Duration: 2-3 days

Suggested GitHub labels: `sprint-1`, `ingestion`, `chunking`, `testing`, `data-quality`

## Scope Boundaries

- In scope: PDF parsing pipeline, metadata extraction, deterministic chunking, chunk persistence, validation, and tests.
- Out of scope: embeddings, vector DB integration, hybrid retrieval, reranking, and answer generation.

## Assumptions and Risks

- Assumption: PDFs are primarily text-based (OCR-heavy scanned files are handled as degraded quality in v1).
- Assumption: chunk outputs from Sprint 1 become canonical inputs for Sprint 2 indexing.
- Risk: parser differences across PDFs can create unstable chunk boundaries.
- Risk: missing page/file metadata will break citation integrity later.
- Risk: non-deterministic chunk IDs will cause indexing duplication.

## Proposed File Structure Additions (on top of Sprint 0)

```text
doc-grounded-rag/
  data/
    raw/
      .gitkeep
    processed/
      chunks/
        .gitkeep
  src/
    rag/
      ingest/
        parser.py
        normalize.py
        service.py
      chunking/
        __init__.py
        splitter.py
        policies.py
        ids.py
        service.py
      storage/
        __init__.py
        chunk_store.py
      pipeline/
        ingest_chunk_pipeline.py
  tests/
    fixtures/
      pdfs/
        sample_policy.pdf
        sample_short.pdf
        sample_empty_page.pdf
    test_pdf_parser.py
    test_chunk_splitter.py
    test_chunk_ids.py
    test_ingest_chunk_pipeline.py
  docs/
    ingestion-and-chunking.md
```

## Ticket Backlog (Task-by-Task)

### S1-01 - Define ingestion and chunking contracts

- Type: `architecture`
- Dependencies: Sprint 0 contracts
- Tasks:
  - finalize input/output contract for parsed pages (`doc_id`, `source_file`, `page_number`, `text`)
  - finalize chunk contract fields (`chunk_id`, `chunk_index`, `char_start`, `char_end`, metadata)
  - define explicit validation rules (required fields, text length bounds)
- Definition of Done:
  - contract schemas are updated and type-checked
  - invalid payload tests fail with explicit validation errors
  - docs include contract examples for one document and two chunks

### S1-02 - Implement PDF parser with page-level metadata

- Type: `backend`
- Dependencies: S1-01
- Tasks:
  - implement parser service for extracting page text from PDFs
  - capture file metadata (file path/name, doc_id, total pages)
  - include robust error handling for unreadable/corrupt files
- Definition of Done:
  - parser returns deterministic page-ordered output
  - each page record contains required metadata
  - tests cover normal, empty-page, and corrupt-file cases

### S1-03 - Implement text normalization pipeline

- Type: `backend`
- Dependencies: S1-02
- Tasks:
  - normalize whitespace, line breaks, and repeated separators
  - preserve meaningful structure where possible (headings/lists)
  - make normalization rules configurable and testable
- Definition of Done:
  - normalization is deterministic for the same input
  - tests validate behavior on noisy text samples
  - normalization does not drop all content on low-quality pages

### S1-04 - Build deterministic chunk splitter

- Type: `backend`
- Dependencies: S1-01, S1-03
- Tasks:
  - implement chunking policy (target size + overlap + hard max)
  - support page-aware chunk boundaries with metadata propagation
  - expose chunking config through typed settings
- Definition of Done:
  - same input always yields identical chunk boundaries
  - overlap behavior is verified in tests
  - edge cases handled (very short page, very long paragraph, empty content)

### S1-05 - Implement stable chunk ID strategy

- Type: `backend`
- Dependencies: S1-04
- Tasks:
  - generate deterministic `chunk_id` (based on doc/page/span/content fingerprint)
  - enforce uniqueness within document scope
  - define duplicate handling policy
- Definition of Done:
  - rerunning ingest on same file produces identical `chunk_id`s
  - collisions are detected and surfaced explicitly
  - tests validate stability across repeated runs

### S1-06 - Persist chunk artifacts to storage

- Type: `backend`
- Dependencies: S1-04, S1-05
- Tasks:
  - implement chunk store writer (JSONL first, DB-compatible shape)
  - include schema version in stored records
  - support idempotent write mode (overwrite/safe append policy)
- Definition of Done:
  - stored artifacts pass schema validation
  - repeated runs do not create silent duplicate records
  - output path and write mode are configurable

### S1-07 - Build ingest+chunk pipeline command

- Type: `backend`
- Dependencies: S1-02, S1-03, S1-04, S1-05, S1-06
- Tasks:
  - orchestrate parse -> normalize -> chunk -> persist
  - add CLI entrypoint for processing one file or directory
  - emit structured logs and stage timings
- Definition of Done:
  - one command processes sample PDFs end-to-end
  - per-document summary emitted (pages, chunks, failures)
  - failures are explicit and do not silently pass

### S1-08 - Add quality and regression tests

- Type: `testing`
- Dependencies: S1-07
- Tasks:
  - add unit tests for parser, normalizer, splitter, ID generator
  - add integration test for full ingest+chunk pipeline
  - add snapshot-style test for deterministic chunk outputs
- Definition of Done:
  - happy path and failure path tests both pass
  - deterministic output test catches boundary regressions
  - tests run through existing quality gate command

### S1-09 - Document usage, constraints, and known gaps

- Type: `docs`
- Dependencies: S1-07, S1-08
- Tasks:
  - write `docs/ingestion-and-chunking.md` with flow and contracts
  - update `README.md` with ingestion/chunking commands
  - document known limitations (OCR quality, tables, multi-column PDFs)
- Definition of Done:
  - a new contributor can run ingestion on a sample PDF quickly
  - limitations and expected failure modes are clearly stated
  - docs and commands are validated locally

## Execution Order

1. S1-01
2. S1-02
3. S1-03
4. S1-04
5. S1-05
6. S1-06
7. S1-07
8. S1-08
9. S1-09

## Sprint 1 Exit Criteria

- PDF ingestion is reliable on representative sample files.
- Chunking is deterministic and traceable to source pages/spans.
- Persisted chunk artifacts are schema-valid and idempotent.
- Tests cover critical edge cases and regression risks.
- Documentation is sufficient for another engineer to run the pipeline.

## Suggested Ticket Template

```md
Title: [S1-XX] <short task title>

Context
- Why this task exists
- Which module(s) it affects

Implementation Notes
- Files to create/update
- Key design constraints

Definition of Done
- [ ] DoD item 1
- [ ] DoD item 2
- [ ] DoD item 3

Validation
- Commands run:
  - `make lint`
  - `make test`
  - `<ingest command>`
```
