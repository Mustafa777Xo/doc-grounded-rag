# Retrieval and Reranking

Sprint 3 passes one typed query and one candidate shape through dense retrieval,
keyword retrieval, fusion, reranking, evaluation, and grounded generation.

## Query Contract

`RetrievalQuery` preserves both the user's input and the deterministic text used
for retrieval. `query_id` connects results and evaluation records across stages.
Filters are immutable sets of document IDs, source filenames, and zero-based
page numbers. An empty set means that dimension is unrestricted.

```json
{
  "schema_version": "retrieval_query.v1",
  "query_id": "query-001",
  "original_text": "  Who is eligible? ",
  "normalized_text": "Who is eligible?",
  "filters": {
    "doc_ids": ["benefits-handbook"],
    "pages": [],
    "source_files": ["benefits.pdf"]
  }
}
```

## Candidate Contract

`RetrievalCandidate` always contains the complete `Chunk`. Its `sources` identify
which retrievers found the chunk, while `scores` retain every available stage
score without treating scores from different systems as interchangeable.

- `dense_score` is vector similarity from dense retrieval.
- `keyword_score` is the raw lexical score from keyword retrieval.
- `fusion_score` is the score assigned during candidate fusion.
- `rerank_score` is the cross-encoder score.
- `rank` is one-based when assigned and `null` before ranking.

All present scores must be finite. A candidate must have a dense or keyword
score, and each retriever score must agree with its corresponding source.

### Dense-Only Candidate

```json
{
  "schema_version": "retrieval_candidate.v1",
  "chunk": {
    "chunk_id": "benefits-p0-c0",
    "doc_id": "benefits-handbook",
    "source_file": "benefits.pdf",
    "page": 0,
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 48,
    "text": "Eligibility applies to full-time employees only."
  },
  "scores": {
    "dense_score": 0.82,
    "keyword_score": null,
    "fusion_score": null,
    "rerank_score": null
  },
  "sources": ["dense"],
  "rank": null
}
```

### Fused Candidate

```json
{
  "schema_version": "retrieval_candidate.v1",
  "chunk": {
    "chunk_id": "benefits-p0-c0",
    "doc_id": "benefits-handbook",
    "source_file": "benefits.pdf",
    "page": 0,
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 48,
    "text": "Eligibility applies to full-time employees only."
  },
  "scores": {
    "dense_score": 0.82,
    "keyword_score": 3.41,
    "fusion_score": 0.0325,
    "rerank_score": null
  },
  "sources": ["dense", "keyword"],
  "rank": null
}
```

### Reranked Candidate

```json
{
  "schema_version": "retrieval_candidate.v1",
  "chunk": {
    "chunk_id": "benefits-p0-c0",
    "doc_id": "benefits-handbook",
    "source_file": "benefits.pdf",
    "page": 0,
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 48,
    "text": "Eligibility applies to full-time employees only."
  },
  "scores": {
    "dense_score": 0.82,
    "keyword_score": 3.41,
    "fusion_score": 0.0325,
    "rerank_score": 7.18
  },
  "sources": ["dense", "keyword"],
  "rank": 1
}
```

## Deterministic Serialization

`to_json()` sorts object keys, uses compact separators, rejects non-finite JSON
numbers, and sorts filter and source sets. Equivalent contracts therefore emit
the same bytes regardless of set iteration order.
