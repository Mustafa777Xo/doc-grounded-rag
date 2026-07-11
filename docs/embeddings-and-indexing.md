# Embeddings and Indexing

Sprint 2 defines the contracts that hand Sprint 1 chunk artifacts into
embedding generation and vector indexing.

## Contract Rules

- Embedding records use schema version `embedding.v1`.
- Vector index rows use schema version `vector_index.v1`.
- Indexed chunk metadata must reference Sprint 1 chunk schema version `chunk.v1`.
- `dim` must match the vector length.
- Vector values must be finite numbers.
- Vector rows are keyed by `chunk_id`; `row_id`, embedding `chunk_id`, and
  metadata `chunk_id` must match.
- Embedding and metadata `content_hash` values must match.
- Indexed metadata includes chunk text and citation metadata so retrieval can
  return complete evidence without a second lookup.

## Embedding Record Example

```json
{
  "schema_version": "embedding.v1",
  "chunk_id": "policy-2026-p0-s0-e47-a1b2c3d4e5f6",
  "vector": [0.12, -0.03, 0.91],
  "dim": 3,
  "model_name": "local-hash-embedder",
  "model_version": "v1",
  "content_hash": "sha256:9f86d081884c7d659a2feaa0c55ad015"
}
```

## Embedding Service

The embedding service is provider-agnostic. Callers pass an embedding request
with `chunk_id`, chunk `text`, and `content_hash`. The service calls a provider,
validates the returned vector shape, and returns an `EmbeddingRecord`.

Service behavior:

- `embed_one` embeds one text and returns one `EmbeddingRecord`
- `embed_batch` embeds requests in input order
- provider-specific response shapes are hidden from callers
- provider failures are wrapped with chunk context
- provider `model_name`, `model_version`, and `dim` are copied into each record

Sprint 2 starts with a deterministic local hash provider. It is dependency-free
and suitable for testing index writes, idempotency, and schema compatibility. It
is not the final semantic-quality target for retrieval.

Future provider adapters must preserve:

- stable `model_name`
- stable `model_version`
- explicit embedding dimension
- finite numeric vector values

## Vector Index Row Example

```json
{
  "schema_version": "vector_index.v1",
  "row_id": "policy-2026-p0-s0-e47-a1b2c3d4e5f6",
  "embedding": {
    "schema_version": "embedding.v1",
    "chunk_id": "policy-2026-p0-s0-e47-a1b2c3d4e5f6",
    "vector": [0.12, -0.03, 0.91],
    "dim": 3,
    "model_name": "local-hash-embedder",
    "model_version": "v1",
    "content_hash": "sha256:9f86d081884c7d659a2feaa0c55ad015"
  },
  "metadata": {
    "chunk_id": "policy-2026-p0-s0-e47-a1b2c3d4e5f6",
    "doc_id": "policy-2026",
    "source_file": "policy.pdf",
    "page": 0,
    "chunk_index": 0,
    "char_start": 0,
    "char_end": 47,
    "text": "Eligibility rules apply to full-time employees.",
    "chunk_schema_version": "chunk.v1",
    "content_hash": "sha256:9f86d081884c7d659a2feaa0c55ad015"
  },
  "created_at": "2026-07-11T00:00:00Z",
  "updated_at": "2026-07-11T00:00:00Z"
}
```

## Compatibility And Reindexing

Rebuild or reindex vectors when any of these values change:

- embedding model name
- embedding model version
- embedding dimension
- embedding schema version
- vector index schema version
- chunk schema version
- chunk text or citation metadata that changes the content hash

Content hashing is implemented in a later Sprint 2 ticket. These contracts only
require that the hash is present and consistent across the embedding record and
index metadata.
