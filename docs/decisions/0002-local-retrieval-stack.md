# ADR 0002: Local Retrieval Model Stack

## Status

Accepted

## Context

Sprint 2 uses a deterministic hash embedding provider to prove index writes,
idempotency, and query plumbing. That provider is intentionally not suitable for
semantic retrieval quality. Sprint 3 needs a local dense embedding model, a
keyword retriever, and a cross-encoder reranker before retrieval contracts are
expanded.

The selected stack must:

- run without inference APIs after one setup-time download
- support the English-only v1 corpus
- run on Python 3.11 and 3.12
- remain small enough for local CPU development
- expose exact model identity for reproducible indexing and evaluation
- use permissive licenses

## Decision

### Dense embeddings

Use:

- package: `sentence-transformers==5.6.0`
- array runtime: `numpy==2.2.6`
- model framework: `transformers==4.57.6`
- tensor framework: `torch==2.7.1`
- model: `sentence-transformers/all-MiniLM-L6-v2`
- revision: `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`
- license: Apache-2.0
- output: 384-dimensional, L2-normalized vectors
- input limit: 256 word pieces; longer input is truncated by the model
- baseline device: CPU

The production provider sets `normalize_embeddings=True`, loads on CPU with
`local_files_only=True`, and records the full model revision as `model_version`.
The existing hash provider remains the deterministic test double.

### Keyword retrieval

Use:

- package: `bm25s==0.3.10`
- license: MIT
- backend: NumPy sparse scoring
- optional extras: none for the first implementation

The first keyword retriever will not use stemming. Tokenization, stopword, and
case-normalization decisions belong to the keyword-retriever ticket because they
affect lexical index compatibility.

### Cross-encoder reranking

Use:

- package: `sentence-transformers==5.6.0`
- model framework: `transformers==4.57.6`
- tensor framework: `torch==2.7.1`
- model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- revision: `c5ee24cb16019beea0893ab7796b1df96625c6b8`
- license: Apache-2.0
- output: one finite relevance score per query-passage pair
- model size: 6 transformer layers with hidden size 384
- weight artifact: `pytorch_model.bin` loaded with `use_safetensors=False`
- baseline device: CPU

Changing the cross-encoder does not invalidate stored dense vectors, but it does
invalidate reranking latency and quality baselines.

## Dependency Policy

This decision records exact direct package versions. Transformers is pinned
explicitly because Sentence Transformers 5.6.0 permits Transformers 5.x, while
the selected cross-encoder returned non-finite scores with Transformers 5.14.1
during the compatibility spike. The reranker's safetensors-backed parameter
storage also produced non-finite linear outputs on the Apple CPU baseline, even
though every stored weight was finite. Loading the repository's equivalent
`pytorch_model.bin` artifact with Transformers 4.57.6 and PyTorch 2.7.1 produced
finite, correctly ranked scores. The real reranker adapter must preserve that
loading policy until a later dependency upgrade is revalidated.

The embedding stack is declared as runtime dependencies by the real embedding
provider ticket. NumPy is pinned to the S3-00 verified version because newer
NumPy 2.5 stubs require a Python 3.12 type grammar while this project type-checks
against Python 3.11. The BM25S declaration remains deferred until the keyword
retriever is implemented.

Benchmark results record the complete verified stack, including resolved NumPy
and SciPy versions. Those results describe the verified environment; they are
not a replacement for a future project lockfile.

## Cache And Offline Policy

Setup may download only the required configuration, tokenizer, and selected
weight artifacts from the two pinned Hugging Face revisions once. Runtime smoke
checks and future local retrieval must work with:

```sh
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

Model constructors must also use `local_files_only=True`. The cache location is
`HF_HOME` when configured and otherwise the Hugging Face user cache. Model files
must not be committed to the repository.

## Compatibility And Benchmark Policy

Compatibility is checked in clean environments for Python 3.11 and 3.12.
This workstation verified Python 3.11.13 and 3.12.6 because its installed pyenv
definitions did not include the newer security-only source releases.
The committed latency baseline is measured on Apple arm64 using CPU inference.
It records model load time plus warm p50 and p95 timings. Sprint 3 uses this as a
characterization baseline, not a hard performance gate.

The benchmark procedure and machine-readable result are implemented by:

```sh
python scripts/benchmark_retrieval_stack.py prefetch
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/benchmark_retrieval_stack.py benchmark \
  --chunks /private/tmp/doc-grounded-rag-s3-chunks.jsonl \
  --compatibility-result /private/tmp/doc-grounded-rag-s3-py311.json \
  --output docs/benchmarks/s3-model-selection-apple-arm64.json
```

## Rebuild Policy

Create a new vector collection and fully re-embed the corpus when any of these
change:

- embedding model repository or revision
- Sentence Transformers version
- Transformers or PyTorch version
- vector dimension
- pooling or L2-normalization policy
- model input truncation policy
- embedding schema version

Never mix hash embeddings and MiniLM embeddings in one collection. The current
collection schema checks must continue to reject incompatible model identity or
dimensions.

Rebuild only the lexical index when the BM25S version, tokenization, stopwords,
stemming policy, or source chunks change.

Changing the cross-encoder requires reranking benchmark and evaluation
rebaselining, but not dense reindexing.

## Alternatives Considered

- In-repository BM25: rejected because BM25S provides a focused, tested
  implementation without a service dependency.
- `rank-bm25`: rejected in favor of BM25S's sparse index and faster query path.
- Larger embedding and reranking models: deferred until evaluation shows that the
  MiniLM quality/latency tradeoff is insufficient.
- MPS inference: excluded from the baseline because the current goal is a stable,
  portable CPU measurement.
- Runtime model downloads: rejected because normal local querying must not depend
  on network availability.

## Consequences

- Real semantic retrieval adds a PyTorch-based runtime footprint when its provider
  ticket is implemented.
- English-only MiniLM models do not satisfy future multilingual requirements.
- Embedding identity becomes an operational index boundary rather than a
  configuration detail.
- Local setup needs enough disk space for the pinned model snapshots and Python
  environments.
