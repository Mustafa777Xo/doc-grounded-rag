# Retrieval and Reranking

Sprint 3 passes one typed query and one candidate shape through dense retrieval,
keyword retrieval, fusion, reranking, evaluation, and grounded generation.

## Query Contract

`RetrievalQuery` preserves both the user's input and the deterministic text used
for retrieval. `query_id` connects results and evaluation records across stages.
Filters are immutable sets of document IDs, source filenames, and zero-based
page numbers. An empty set means that dimension is unrestricted.

Query normalization applies NFKC Unicode normalization, Unicode `casefold()`, a
second NFKC pass, and Unicode whitespace trimming and collapse. Punctuation is
preserved. Query expansion and stopword removal remain disabled. Empty results
after normalization are rejected with a `normalization`-stage retrieval error,
while the original text remains available for display and diagnostics.

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

## Dense Retrieval

Dense retrieval embeds only `normalized_text` and records cosine similarity as
`dense_score`. The requested `limit` is the dense top-k budget. Results sort by
descending score and then ascending `chunk_id`, making equal-score order stable.

Metadata filters are applied before scoring and top-k selection. Values within
one dimension are ORed, while populated dimensions are ANDed. For example, two
document IDs and one page match either document only on that page. Empty filter
sets impose no restriction.

A bootstrapped empty collection and a filter with no matching rows both return
an empty candidate tuple. A missing or unbootstrapped collection remains an
actionable `dense`-stage error. Dense retrieval does not apply a score threshold;
a non-empty eligible collection returns its nearest neighbors up to top-k.

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

## Retrieval Configuration

The balanced CPU defaults are centralized in `rag.config.Settings` and can be
overridden with environment variables.

| Environment variable | Default | Purpose |
| --- | ---: | --- |
| `DENSE_TOP_K` | `20` | Dense candidates requested before fusion |
| `KEYWORD_TOP_K` | `20` | Keyword candidates requested before fusion |
| `FUSION_RRF_K` | `60` | Reciprocal Rank Fusion ranking constant |
| `RERANK_CANDIDATE_COUNT` | `20` | Fused candidates sent to the reranker |
| `FINAL_RESULT_COUNT` | `5` | Reranked candidates returned downstream |
| `RERANKER_BATCH_SIZE` | `16` | Query-passage pairs scored per model batch |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense model repository |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder repository |

Startup validation requires positive values, `FINAL_RESULT_COUNT` no greater
than `RERANK_CANDIDATE_COUNT`, and the rerank count no greater than the combined
dense and keyword candidate budgets.

## Stage-Aware Errors

Retrieval failures preserve the typed query ID and one internal stage:
`normalization`, `dense`, `keyword`, or `fusion`. Reranking failures use the
`reranking` stage. These domain errors retain actionable hints and remain the
cause when an outer pipeline boundary adds correlation context.
