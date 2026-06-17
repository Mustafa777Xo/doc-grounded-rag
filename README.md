# doc-grounded-rag

A modular retrieval-augmented generation (RAG) system for querying company
knowledge stored in PDF documents.

The system ingests PDFs, extracts and chunks content with metadata, indexes it
using embeddings, retrieves and reranks relevant context, and generates answers
grounded only in retrieved evidence. Answers include citations to source
documents and pages.

## Scope

- Single-domain document corpus
- PDF input only for v1
- Local-first development
- No external APIs required for Sprint 0
- Retrieval quality and grounded generation are the main product goals

## Architecture

Sprint 0 establishes the project foundation and no-op pipeline. The architecture
details are documented here:

- [Architecture](docs/architecture.md)
- [Sprint 0 execution board](docs/sprints/SPRINT_0_EXECUTION_BOARD.md)
- [ADR 0001: System boundaries](docs/decisions/0001-system-boundaries.md)

The target pipeline is:

1. Ingest PDF documents
2. Extract text and attach page-level metadata
3. Split text into traceable chunks
4. Index chunks and embeddings
5. Retrieve relevant chunks
6. Rerank retrieved chunks
7. Generate a grounded answer
8. Return the answer with citations

Sprint 0 currently provides typed contracts, interfaces, no-op adapters,
structured logging, config loading, and a no-op end-to-end pipeline.

## Quick Start

Prerequisites:

- Python 3.11 or 3.12
- `pip`
- `make`

From a fresh clone:

```sh
python -m venv .venv
source .venv/bin/activate
make install
make run-noop
make check
```

Expected result:

- `make run-noop` emits JSON stage logs and prints a mock cited answer.
- `make check` runs linting, formatting checks, strict type checking, tests, and
  coverage.

This should take less than 15 minutes on a normal connection.

## Configuration

Copy the example environment file when you start working with real local data:

```sh
test -f .env || cp .env.example .env
```

Then set at least:

```sh
DOCS_DIR=data/pdfs
PROFILE=dev
CHUNK_SIZE=512
CHUNK_OVERLAP=64
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
```

The no-op pipeline does not require real PDFs or a populated `.env` file.

## Commands

```sh
make install
```

Install the package in editable mode with development dependencies.

```sh
make format
```

Format Python files and sort imports with Ruff.

```sh
make lint
```

Run Ruff linting and formatting checks without modifying files.

```sh
make typecheck
```

Run strict mypy checks over `src/` and `tests/`.

```sh
make test
```

Run the pytest suite.

```sh
make coverage
```

Run tests with coverage and enforce the current minimum threshold.

```sh
make check
```

Run the local quality gate: lint, typecheck, tests, and coverage.

```sh
make run-noop
```

Run the Sprint 0 no-op pipeline from mock input to mock cited answer.

```sh
make clean
```

Remove local tool caches.

## Coding Standards

- Use the `src/` package layout and import the package as `rag`.
- Keep shared data contracts in `src/rag/contracts`.
- Preserve citation metadata through pipeline handoffs.
- Keep stages independent: ingest, index, retrieve, rerank, generate, eval, and
  pipeline should communicate through typed contracts.
- Use dependency injection for components that touch files, models, databases,
  or indexes.
- Prefer explicit, typed Python compatible with strict mypy.
- Log pipeline stage start, finish, duration, and failures with stage context.
- Do not generate answers from unsupported context.

## PR Checklist

Before merging, confirm:

- Contract changes preserve required traceability fields such as `doc_id`,
  `source_file`, `page`, `chunk_id`, and `chunk_index`.
- New or changed behavior has focused tests.
- Pipeline-facing code logs stage transitions and raises clear stage-aware
  errors.
- `make check` passes locally.
- `make run-noop` still runs from mock input to mock cited answer when pipeline
  wiring changes.

## Evaluation

The planned evaluation pipeline will measure retrieval quality, answer
correctness, faithfulness, and citation accuracy against a fixed domain-specific
question set. Sprint 0 only establishes the interfaces and local gates needed to
support that work later.
