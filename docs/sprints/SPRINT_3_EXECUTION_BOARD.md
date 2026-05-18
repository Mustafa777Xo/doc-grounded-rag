# Sprint 3 Execution Board

Goal: implement hybrid retrieval (dense + keyword) and cross-encoder reranking with measurable retrieval quality improvements.

Duration: 3-4 days

Suggested GitHub labels: `sprint-3`, `retrieval`, `reranking`, `evaluation`, `performance`

## Scope Boundaries

- In scope: query processing, dense retrieval integration, keyword retrieval integration, score fusion, candidate assembly, reranking, retrieval metrics.
- Out of scope: grounded generation logic (Sprint 4), full answer correctness evaluation (Sprint 5), production scaling optimizations.

## Assumptions and Risks

- Assumption: Sprint 2 already provides indexed chunk embeddings and a queryable vector store.
- Assumption: Sprint 1 chunk metadata is stable and citation-ready.
- Risk: poor fusion strategy can degrade quality versus single retriever.
- Risk: cross-encoder latency can exceed acceptable query budget.
- Risk: score calibration mismatch between dense and BM25 channels.
- Risk: retrieval returns topically relevant but citation-weak chunks.

## Proposed File Structure Additions (on top of previous sprints)

```text
doc-grounded-rag/
  src/
    rag/
      retrieve/
        query.py
        dense_retriever.py
        keyword_retriever.py
        fusion.py
        candidate_builder.py
        service.py
      rerank/
        cross_encoder.py
        calibrator.py
        service.py
      eval/
        retrieval_metrics.py
        retrieval_runner.py
      pipeline/
        retrieve_rerank_pipeline.py
  tests/
    fixtures/
      queries/
        retrieval_eval_set.jsonl
    test_dense_retriever.py
    test_keyword_retriever.py
    test_fusion.py
    test_reranker.py
    test_retrieve_rerank_pipeline.py
    test_retrieval_metrics.py
  docs/
    retrieval-and-reranking.md
```

## Ticket Backlog (Task-by-Task)

### S3-01 - Define retrieval and rerank contracts

- Type: `architecture`
- Dependencies: Sprint 1 and 2 contracts
- Tasks:
  - finalize query contract (`query_id`, `text`, optional filters)
  - finalize retrieval candidate contract (`chunk_id`, `scores`, `retriever_source`, metadata)
  - finalize reranked result contract (`rank`, `rerank_score`, evidence fields)
- Definition of Done:
  - schemas are explicit and type-checked
  - contract tests include invalid score/metadata cases
  - docs include one example for pre-rerank and post-rerank payloads

### S3-02 - Implement query preprocessing and normalization

- Type: `backend`
- Dependencies: S3-01
- Tasks:
  - implement query cleaner (whitespace, punctuation normalization)
  - add optional stopword and query expansion hooks (off by default)
  - ensure preprocessing is deterministic and logged
- Definition of Done:
  - identical input query produces identical normalized query
  - preprocessing decisions are visible in debug logs
  - tests cover empty/short/noisy query edge cases

### S3-03 - Implement dense retriever integration

- Type: `backend`
- Dependencies: S3-01, Sprint 2 index
- Tasks:
  - implement vector retrieval adapter using existing embedding/index layer
  - support configurable `top_k_dense` and metadata filters
  - return calibrated dense scores and provenance metadata
- Definition of Done:
  - retriever returns deterministic ranked candidates for fixed index/query
  - empty-index and no-hit scenarios handled explicitly
  - tests validate ranking stability and filter behavior

### S3-04 - Implement keyword retriever integration

- Type: `backend`
- Dependencies: S3-01
- Tasks:
  - implement BM25 or equivalent keyword retriever over chunk text
  - support configurable `top_k_keyword` and same metadata filters
  - expose lexical match diagnostics (matched terms, raw score)
- Definition of Done:
  - keyword retriever returns reproducible ranking for fixed corpus/query
  - no-hit cases are explicit and non-fatal
  - tests cover term mismatch, casing, and punctuation behavior

### S3-05 - Implement fusion and candidate assembly

- Type: `backend`
- Dependencies: S3-03, S3-04
- Tasks:
  - implement fusion strategy (RRF recommended for MVP)
  - deduplicate by `chunk_id` and preserve source score components
  - implement candidate budget (`top_n_for_rerank`) with explainable tie-breakers
- Definition of Done:
  - fused output contains traceable score breakdown per candidate
  - deduplication is correct across retriever sources
  - tests verify fusion math and tie-break determinism

### S3-06 - Implement cross-encoder reranker

- Type: `backend`
- Dependencies: S3-05
- Tasks:
  - implement reranker adapter for query-candidate pair scoring
  - support batch scoring and configurable max sequence length
  - include graceful fallback when reranker is unavailable
- Definition of Done:
  - reranker produces stable ranking for fixed model/version
  - latency and batch stats are logged per query
  - fallback path is tested and clearly surfaced in logs/results

### S3-07 - Build retrieve+rereank pipeline command

- Type: `backend`
- Dependencies: S3-02, S3-03, S3-04, S3-05, S3-06
- Tasks:
  - orchestrate normalize -> dense+keyword -> fuse -> rerank
  - expose CLI command for single query and batch query file
  - emit stage-wise timings and candidate counts
- Definition of Done:
  - one command returns reranked evidence set for query input
  - output includes enough metadata for downstream citation usage
  - failure modes are explicit (index unavailable, reranker unavailable, empty results)

### S3-08 - Implement retrieval evaluation harness

- Type: `evaluation`
- Dependencies: S3-07
- Tasks:
  - implement metrics: `Recall@k`, `Precision@k`, `MRR` (optional `nDCG@k`)
  - define eval dataset format for query-to-relevant-chunk mapping
  - add runner to compare dense-only, keyword-only, and hybrid+rereank modes
- Definition of Done:
  - evaluation runner outputs per-query and aggregate metrics
  - comparison table shows whether hybrid+rereank beats baselines
  - tests validate metric calculations on small synthetic fixtures

### S3-09 - Tune retrieval defaults and quality gates

- Type: `evaluation`
- Dependencies: S3-08
- Tasks:
  - tune key parameters (`top_k`, fusion weights or RRF k, rerank cutoff)
  - set minimum acceptance thresholds for Sprint 3 metrics
  - document configuration and tradeoffs
- Definition of Done:
  - baseline config is committed with rationale
  - thresholds are encoded in eval checks or CI gate
  - tuning notes capture winning config and known failure queries

### S3-10 - Document retrieval/reranking design and known limits

- Type: `docs`
- Dependencies: S3-07, S3-08, S3-09
- Tasks:
  - write `docs/retrieval-and-reranking.md`
  - update `README.md` with query/eval commands
  - document known limits (acronyms, table-heavy content, ambiguous queries)
- Definition of Done:
  - another engineer can run retrieval eval end-to-end from docs
  - design choices and tradeoffs are clearly justified
  - all documented commands are validated locally

## Execution Order

1. S3-01
2. S3-02
3. S3-03 + S3-04 (parallel)
4. S3-05
5. S3-06
6. S3-07
7. S3-08
8. S3-09
9. S3-10

## Sprint 3 Exit Criteria

- Hybrid retrieval and reranking pipeline is fully runnable for single and batch queries.
- Retrieval outputs are traceable and compatible with citation requirements.
- Offline metrics are reproducible and demonstrate quality gains over baseline modes.
- Latency and failure behaviors are observable and explicitly handled.
- Documentation is complete for handoff and future tuning.

## Suggested Ticket Template

```md
Title: [S3-XX] <short task title>

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
  - `<retrieve command>`
  - `<retrieval eval command>`
```
