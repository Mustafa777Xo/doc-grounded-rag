# ADR 0001: System Boundaries

## Status

Accepted

## Context

doc-grounded-rag is a local-first RAG system for answering questions against a
single-domain PDF corpus. The project needs stable boundaries before real PDF
parsing, embeddings, indexing, retrieval, reranking, generation, and evaluation
logic are added.

The main risk is coupling early modules together too tightly. If ingestion,
chunking, indexing, retrieval, reranking, generation, and evaluation share hidden
state or untyped payloads, later changes will be difficult to test and debug.

## Decision

The system is split into small pipeline stages:

- `contracts`: shared typed data models
- `config`: runtime settings
- `logging`: structured logs and stage-aware errors
- `ingest`: source discovery and document extraction
- `index`: storing chunks and embeddings
- `retrieve`: returning candidate chunks for a query
- `rerank`: reordering retrieved results
- `generate`: producing grounded answers with citations
- `eval`: measuring retrieval and answer quality
- `pipeline`: orchestration only

Stages communicate through typed contracts instead of dictionaries or implicit
shared state. The pipeline layer owns stage ordering, logging transitions, and
surfacing stage-aware errors.

Sprint 0 uses no-op adapters to validate handoffs without implementing real RAG
logic.

## Consequences

Benefits:

- Each stage can be developed and tested independently.
- Pipeline failures include the stage and correlation context.
- Real implementations can replace no-op adapters without changing contracts.
- Citation metadata remains visible across handoffs.

Tradeoffs:

- Some early interfaces are intentionally simple and may evolve as real
  implementations arrive.
- The no-op pipeline includes lightweight mock chunk creation until a dedicated
  chunking module is implemented.
- Strict boundaries add small upfront ceremony, but reduce later integration
  risk.

## Merge Requirements

Changes that affect pipeline behavior must pass the local gate:

```sh
make check
```

Pipeline wiring changes should also verify:

```sh
make run-noop
```

Contract changes must preserve source traceability fields needed for citations.
