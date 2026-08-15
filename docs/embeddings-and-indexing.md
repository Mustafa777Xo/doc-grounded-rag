# Embeddings and Indexing

Sprint 2 defines the contracts that hand Sprint 1 chunk artifacts into
embedding generation and vector indexing.

## Local Operations

Use these commands to build and reindex a local SQLite vector index from a fresh
checkout. The examples use `/private/tmp` so the smoke run does not create
working data inside the repository.

Install dependencies first:

```sh
make install
```

Build a chunk artifact from the sample PDF:

```sh
make ingest INPUT=tests/fixtures/pdfs/sample_policy.pdf OUTPUT=/private/tmp/doc-grounded-rag-s2-docs-chunks.jsonl MODE=overwrite
```

Build the vector index:

```sh
make embed-index CHUNKS=/private/tmp/doc-grounded-rag-s2-docs-chunks.jsonl INDEX=/private/tmp/doc-grounded-rag-s2-docs-vector.sqlite COLLECTION=sprint2_docs
```

Expected summary fields from the first index build:

```json
{
  "succeeded": true,
  "sync": {
    "new": 2,
    "changed": 0,
    "unchanged": 0,
    "deleted": 0,
    "written": 2,
    "removed": 0,
    "errors": 0
  }
}
```

Reindex the same chunks by rerunning the same command:

```sh
make embed-index CHUNKS=/private/tmp/doc-grounded-rag-s2-docs-chunks.jsonl INDEX=/private/tmp/doc-grounded-rag-s2-docs-vector.sqlite COLLECTION=sprint2_docs
```

Expected reindex behavior:

```json
{
  "succeeded": true,
  "sync": {
    "new": 0,
    "changed": 0,
    "unchanged": 2,
    "deleted": 0,
    "written": 0,
    "removed": 0,
    "errors": 0
  }
}
```

Run the query smoke checks:

```sh
PYTHONPATH=src pytest tests/test_index_reader.py -q
```

Run the full quality gate before merging:

```sh
make check
```

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

Current local provider policy:

- provider: `HashEmbeddingProvider`
- model name: `local-hash-embedder`
- model version: `v1`
- default embedding dimension: `16`
- purpose: local smoke tests, deterministic indexing tests, and offline
  development

Future provider adapters must preserve:

- stable `model_name`
- stable `model_version`
- explicit embedding dimension
- finite numeric vector values

## Text Preparation And Batching

Text preparation runs before embedding. It enforces a character budget for the
embedding input and records whether the input was changed before calling the
provider.

Default embedding preparation policy:

- max chars: `4096`
- batch size: `32`
- oversized behavior: truncate to `max_chars`

Truncation is explicit in the prepared request metadata. The `content_hash`
remains tied to the canonical chunk text, not the truncated embedding input, so
incremental reindexing can still detect source chunk changes consistently.

Batching behavior:

- preserves input order
- executes fixed-size batches using the configured batch size
- returns `EmbeddingRecord` outputs in the same order as the prepared inputs
- lets provider and service failures bubble with chunk context

## Content Hashing And Change Detection

The `content_hash` tracks canonical chunk text for incremental indexing:

```json
{
  "chunk_schema_version": "chunk.v1",
  "text": "Eligibility rules apply to full-time employees."
}
```

The payload is serialized with sorted JSON keys and hashed with SHA-256. Stored
hashes use the format `sha256:<hex>`.

Change detection compares current chunks with indexed chunk state by `chunk_id`:

- `new`: chunk exists now but was not indexed before
- `changed`: chunk exists now and its current hash differs from the indexed hash
- `unchanged`: chunk exists now and its current hash matches the indexed hash
- `deleted`: indexed chunk is missing from the current chunk set

The hash intentionally excludes citation metadata and embedding model metadata.
Metadata validation is handled by vector index row contracts and later sync
checks. Model name, model version, and embedding dimension changes are handled
by schema/index compatibility checks rather than this text hash.

## Local Vector Store

Sprint 2 uses a SQLite-backed local vector store. It is dependency-free,
persistent across runs, and intended for MVP-scale development and smoke tests.
The store scans vectors in Python, so it is not intended for large production
corpora.

Collection schema includes:

- collection name
- vector index schema version
- embedding schema version
- chunk schema version
- model name
- model version
- embedding dimension

Bootstrap behavior:

- creates parent directories and database tables from empty state
- repeated bootstrap with the same schema is idempotent
- incompatible schema changes fail clearly

Stored rows are keyed by `chunk_id`. Upsert replaces the existing row for the
same chunk instead of creating duplicates. Delete removes rows by chunk ID.

Query behavior:

- validates query vector dimension against the collection dimension
- computes cosine similarity in Python over stored vectors
- returns deterministic top-k results sorted by score descending, then
  `chunk_id`
- preserves the full `VectorIndexRow` so retrieval can access chunk text and
  citation metadata

Schema, model, or dimension changes require a new collection or a rebuild of the
existing collection.

## Idempotent Index Sync

The index writer syncs canonical Sprint 1 `Chunk` objects into the vector store.
It is a library flow; the command-line pipeline is added separately.

Sync behavior:

- bootstraps the target collection schema before reading or writing rows
- compares current chunk hashes against indexed chunk state by `chunk_id`
- embeds and upserts only `new` and `changed` chunks
- skips `unchanged` chunks so repeated runs produce zero writes
- deletes indexed rows whose chunks are no longer present
- returns a summary with `new`, `changed`, `unchanged`, `deleted`, `written`,
  `removed`, and `error` counts

Every written row preserves the full chunk text and citation metadata. Failures
raise an index sync error with stage context such as `bootstrap`, `detect`,
`embed`, `build_rows`, `upsert`, or `delete`. When available, failures also
include the affected `chunk_id` and a partial summary with `error_count`
incremented.

## Embed And Index Command

Build or re-sync the local vector index from Sprint 1 chunk artifacts:

Prefetch the revision-pinned model once while network access is available:

```sh
python scripts/benchmark_retrieval_stack.py prefetch
```

Normal indexing loads `sentence-transformers/all-MiniLM-L6-v2` on CPU from the
Hugging Face cache only. It never downloads weights at runtime and does not fall
back to hash embeddings. The cache is read from `HF_HOME` when set, otherwise
from the default Hugging Face user cache. Model files must not be committed.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  make embed-index CHUNKS=data/processed/chunks/chunks.jsonl INDEX=data/index/vector_store.sqlite COLLECTION=default
```

The command loads and validates the JSONL chunk artifact, embeds only new or
changed chunks, upserts vector rows, deletes stale rows, prints one JSON summary
to stdout, and emits structured `load_chunks` and `sync_index` stage logs to
stderr.

Rerunning the same command against unchanged chunks should report zero writes
and unchanged rows in the sync summary.

Command inputs:

- `CHUNKS`: Sprint 1 JSONL chunk artifact
- `INDEX`: SQLite vector index path
- `COLLECTION`: logical collection name inside the index

The command writes one JSON summary to stdout and structured logs to stderr.
Pipeline stages are `load_chunks` and `sync_index`.

## Semantic Read Smoke

The minimal semantic reader embeds a query with the same provider family used for
indexing, queries the vector store, and returns `RetrievalCandidate` contracts
for Sprint 3 consumers.

Reader behavior:

- records `dense` as the retriever source and preserves vector similarity as
  `dense_score`
- uses vector similarity scores from the store
- reconstructs `Chunk` metadata from stored vector row metadata
- preserves citation fields: `source_file`, `page`, `chunk_id`, and
  `chunk_index`
- fails clearly for invalid limits, query embedding failures, or unavailable
  indexes

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
- embedding model revision (`EMBEDDING_MODEL_REVISION`)
- Sentence Transformers, Transformers, or PyTorch version
- embedding dimension
- pooling, L2-normalization, or model truncation policy
- embedding schema version
- vector index schema version
- chunk schema version
- chunk text changes that change the content hash

Use the same collection when chunk text changes but the model and schema stay
compatible. The sync flow will update changed chunks, insert new chunks, skip
unchanged chunks, and delete stale chunks.

Use a new collection name or rebuild the existing index when any stored
collection schema value changes:

- model name
- model version
- embedding dimension
- embedding schema version
- vector index schema version
- chunk schema version

Schema mismatches fail clearly during index bootstrap. They should not be worked
around by appending vectors with a different dimension or model version into the
same collection.

To rebuild in place, remove the SQLite index only after preserving anything you
need from it, then rerun `make embed-index`. To retain an old index, use a new
`INDEX` path or `COLLECTION` name. Never mix hash and MiniLM vectors in one
collection.

## Operational Limits And Troubleshooting

Default embedding preparation limits:

- max chars per embedded text: `4096`
- batch size: `32`
- oversized text behavior: truncate before embedding

The `content_hash` always tracks canonical chunk text, not truncated embedding
input. This keeps incremental reindexing stable even when long chunks are
shortened for embedding.

Common failures:

- Missing chunk artifact: rerun `make ingest` or pass the correct `CHUNKS` path.
- Malformed JSONL: regenerate the chunk artifact from Sprint 1 ingestion.
- Duplicate `chunk_id`: fix chunk generation before indexing; duplicates are
  rejected to prevent duplicate vectors.
- Missing or incompatible index schema: rebuild the index or use a new
  `COLLECTION`.
- Missing cached model: run the explicit prefetch command once with network
  access, then retry with the offline environment flags.
- Query smoke failure from a missing index: run `make embed-index` before the
  reader smoke test.
