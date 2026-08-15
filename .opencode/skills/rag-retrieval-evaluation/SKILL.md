---
name: rag-retrieval-evaluation
description: Use for retrieval benchmarks, embedding or reranker comparisons, evaluation datasets, Precision@k, Recall@k, MRR, nDCG, or retrieval regression decisions in this project.
---

# RAG Retrieval Evaluation

Make retrieval decisions from reproducible evidence rather than anecdotal queries.

## Workflow

1. Define the decision being made, candidate systems, corpus version, query set, relevance labels, and success threshold.
2. Keep the evaluation dataset fixed and versioned. Include expected document, page, and chunk identifiers where available.
3. Separate semantic retrieval, keyword retrieval, hybrid score combination, reranking, and final context selection in results.
4. Report at least Recall@k and MRR for retrieval coverage and ranking quality. Add Precision@k or nDCG when the labels support them.
5. Record model name, revision, vector dimensions, normalization, chunking configuration, top-k values, hardware, and timing.
6. Compare candidates on identical inputs and distinguish quality, latency, memory, and index-size tradeoffs.
7. Store machine-readable benchmark output under `docs/benchmarks/` when the result informs a project decision.
8. Add a regression test or explicit threshold for adopted behavior where practical.

Do not use answer quality as a substitute for retrieval metrics. If labels are
incomplete, state that limitation rather than treating unjudged results as irrelevant.
