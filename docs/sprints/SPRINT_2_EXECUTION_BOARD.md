# Sprint 2 Execution Board

Goal: implement embedding generation and indexing so chunk artifacts become queryable with reliable, idempotent vector storage.

Duration: 2-3 days

Suggested GitHub labels: `sprint-2`, `embeddings`, `indexing`, `vector-db`, `testing`

## Scope Boundaries

- In scope: embedding service abstraction, embedding job flow, vector index write/read integration, incremental reindexing, index metadata/versioning, and smoke tests.
- Out of scope: hybrid retrieval fusion logic and reranking (Sprint 3), grounded answer generation (Sprint 4), advanced infra scaling.

## Assumptions and Risks

- Assumption: Sprint 1 outputs canonical chunk artifacts with stable `chunk_id` and citation metadata.
- Assumption: there is a local-first vector store option suitable for offline development.
- Risk: embedding model/version drift can silently invalidate existing index rows.
- Risk: non-idempotent writes can duplicate vectors and degrade retrieval quality.
- Risk: oversized chunk text can exceed embedding model input limits.
- Risk: metadata mismatch between chunk store and vector index breaks traceability.

## Proposed File Structure Additions (on top of previous sprints)

```text
doc-grounded-rag/
  src/
    rag/
      embed/
        __init__.py
        model_client.py
        service.py
        batching.py
        validation.py
      index/
        vector_store.py
        writer.py
        reader.py
        sync.py
        schema.py
      pipeline/
        embed_index_pipeline.py
  tests/
    fixtures/
      chunks/
        sample_chunks.jsonl
    test_embedding_service.py
    test_embedding_batching.py
    test_vector_store_writer.py
    test_index_sync.py
    test_embed_index_pipeline.py
  docs/
    embeddings-and-indexing.md
```

## Ticket Backlog (Task-by-Task)

### S2-01 - Define embedding and index contracts

- Type: `architecture`
- Dependencies: Sprint 1 contracts
- Tasks:
  - define `EmbeddingRecord` contract (`chunk_id`, `vector`, `dim`, `model_name`, `model_version`, `content_hash`)
  - define vector index row contract (ids, vector, metadata payload, timestamps)
  - define index schema version and compatibility rules
- Definition of Done:
  - contracts are typed and validated
  - invalid dimensions/metadata trigger explicit errors
  - docs include one canonical record example

### S2-02 - Implement embedding service abstraction

- Type: `backend`
- Dependencies: S2-01
- Tasks:
  - implement provider-agnostic embedding interface
  - add at least one concrete local provider adapter
  - include deterministic mock provider for tests
- Definition of Done:
  - service can embed single text and batch text
  - interface hides provider-specific response shapes
  - tests validate success and provider-failure paths

### S2-03 - Add text preparation and batching for embeddings

- Type: `backend`
- Dependencies: S2-02
- Tasks:
  - enforce max token/char budget before embed call
  - implement batching policy with configurable batch size
  - preserve ordering from input chunks to embedding results
- Definition of Done:
  - oversized chunk behavior is explicit (truncate or skip + reason)
  - batch execution is deterministic and memory-safe for MVP scale
  - tests validate ordering and boundary behavior

### S2-04 - Implement content hashing and change detection

- Type: `backend`
- Dependencies: S2-01
- Tasks:
  - compute stable `content_hash` from normalized chunk text + schema version
  - compare current hash against indexed hash for incremental updates
  - mark states: `new`, `changed`, `unchanged`, `deleted` (if source removed)
- Definition of Done:
  - unchanged chunks are skipped during reindex
  - changed chunks are re-embedded and upserted
  - tests validate hash stability and state transitions

### S2-05 - Implement vector store adapter and schema management

- Type: `backend`
- Dependencies: S2-01
- Tasks:
  - implement vector store interface (create collection, upsert, query, delete)
  - store required metadata (`doc_id`, `page`, `chunk_id`, `source_file`, `content_hash`, `model_version`)
  - add index/collection bootstrap routine with schema checks
- Definition of Done:
  - vector store can be initialized from empty state
  - schema mismatch is detected and surfaced clearly
  - adapter tests validate CRUD-like lifecycle for vectors

### S2-06 - Build idempotent index writer and sync flow

- Type: `backend`
- Dependencies: S2-03, S2-04, S2-05
- Tasks:
  - implement upsert-based writer keyed by `chunk_id`
  - support incremental sync from chunk artifacts
  - produce sync summary report (counts by state)
- Definition of Done:
  - rerunning sync on unchanged data produces zero writes
  - changed/new chunks are correctly written
  - sync report includes `new/changed/unchanged/deleted/error` counts

### S2-07 - Build embed+index pipeline command

- Type: `backend`
- Dependencies: S2-02, S2-03, S2-06
- Tasks:
  - orchestrate load chunks -> detect changes -> embed -> write index
  - add CLI entrypoint for one file set and full corpus sync
  - emit structured logs and stage timing metrics
- Definition of Done:
  - one command builds index from Sprint 1 chunk outputs
  - rerun shows idempotent behavior in logs/report
  - failure paths (provider down, malformed chunk, store unavailable) are explicit

### S2-08 - Add retrieval smoke checks from index

- Type: `testing`
- Dependencies: S2-07
- Tasks:
  - implement minimal semantic query smoke test against built index
  - verify returned rows preserve citation metadata integrity
  - validate index read API contract for Sprint 3 consumers
- Definition of Done:
  - smoke test proves index is queryable with relevant hit(s)
  - retrieval response includes required metadata fields
  - contract tests protect against shape regressions

### S2-09 - Add quality gates and regression tests

- Type: `testing`
- Dependencies: S2-07, S2-08
- Tasks:
  - add unit tests for embedding service, batching, hashing, sync
  - add integration test for full embed+index pipeline
  - add deterministic test for idempotent reindex behavior
- Definition of Done:
  - happy and failure path coverage exists for critical components
  - reindex regression test fails on duplicate-write regressions
  - all tests pass through existing quality commands

### S2-10 - Document embedding/index design and operations

- Type: `docs`
- Dependencies: S2-07, S2-09
- Tasks:
  - write `docs/embeddings-and-indexing.md`
  - update `README.md` with build/reindex/query-smoke commands
  - document operational limits (model dims, index size, rebuild strategy)
- Definition of Done:
  - another engineer can build and reindex from docs alone
  - model/versioning and reindex policy are clearly explained
  - all documented commands are validated locally

## Execution Order

1. S2-01
2. S2-02
3. S2-03 + S2-04 (parallel)
4. S2-05
5. S2-06
6. S2-07
7. S2-08
8. S2-09
9. S2-10

## Sprint 2 Exit Criteria

- Embeddings are generated through a stable, testable service abstraction.
- Vector index build and incremental reindex are idempotent.
- Indexed rows preserve full citation metadata required for downstream stages.
- Index is queryable with smoke-tested semantic retrieval.
- Documentation supports reproducible local setup and operations.

## Suggested Ticket Template

```md
Title: [S2-XX] <short task title>

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
  - `<embed/index command>`
  - `<semantic smoke command>`
```
