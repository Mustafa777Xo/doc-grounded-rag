# Sprint 0 Execution Board

Goal: establish a clean, production-minded project foundation before any core RAG logic.

Duration: 1-2 days

Suggested GitHub labels: `sprint-0`, `architecture`, `infra`, `testing`, `docs`

## Scope Boundaries

- In scope: project scaffolding, architecture contracts, config system, quality gates, observability baseline, no-op pipeline.
- Out of scope: real PDF parsing, real embeddings, retrieval algorithms, reranking, LLM generation.

## Proposed File Structure (Target at end of Sprint 0)

```text
doc-grounded-rag/
  README.md
  pyproject.toml
  Makefile
  .gitignore
  .env.example
  src/
    rag/
      __init__.py
      config.py
      logging.py
      contracts/
        __init__.py
        document.py
        chunk.py
        retrieval.py
        answer.py
      ingest/
        __init__.py
        interfaces.py
      index/
        __init__.py
        interfaces.py
      retrieve/
        __init__.py
        interfaces.py
      rerank/
        __init__.py
        interfaces.py
      generate/
        __init__.py
        interfaces.py
      eval/
        __init__.py
        interfaces.py
      pipeline/
        __init__.py
        orchestrator.py
  tests/
    test_contracts.py
    test_config.py
    test_noop_pipeline.py
  docs/
    architecture.md
    decisions/
      0001-system-boundaries.md
```

## Ticket Backlog (Task-by-Task)

### S0-01 - Scaffold Python project and dev tooling

- Type: `infra`
- Dependencies: none
- Tasks:
  - create `pyproject.toml` with runtime and dev dependencies
  - create baseline `Makefile` targets (`install`, `lint`, `test`, `format`, `run-noop`)
  - add `.gitignore` for Python artifacts and local env files
- Definition of Done:
  - `make install` succeeds on a clean machine
  - `make lint` and `make test` run successfully (even with minimal tests)
  - project can be imported as package `rag`

### S0-02 - Define architecture and module boundaries

- Type: `architecture`
- Dependencies: S0-01
- Tasks:
  - write `docs/architecture.md`
  - define responsibilities and inputs/outputs for `ingest`, `index`, `retrieve`, `rerank`, `generate`, `eval`, `pipeline`
  - capture constraints and non-goals for v1
- Definition of Done:
  - each module has a clear purpose, owner interface, and dependency direction
  - no circular dependencies in design
  - architecture doc includes one end-to-end sequence diagram (text/mermaid accepted)

### S0-03 - Create typed data contracts

- Type: `architecture`
- Dependencies: S0-02
- Tasks:
  - implement contracts for `Document`, `Chunk`, `RetrievalResult`, `AnswerWithCitations`
  - include explicit metadata fields for traceability (`doc_id`, `source_file`, `page`, `chunk_id`)
  - add validation constraints and serialization helpers
- Definition of Done:
  - contract tests in `tests/test_contracts.py` cover valid and invalid payloads
  - all contracts are typed and serializable to JSON
  - citation fields are mandatory where required

### S0-04 - Establish configuration and environment strategy

- Type: `infra`
- Dependencies: S0-01
- Tasks:
  - implement `src/rag/config.py` with typed settings
  - add `.env.example` with required keys and safe defaults
  - separate runtime profile values (dev/test)
- Definition of Done:
  - app starts with `.env.example`-derived defaults in local dev
  - missing required config raises explicit startup error
  - config loading behavior is documented in README/docs

### S0-05 - Add logging and error handling baseline

- Type: `infra`
- Dependencies: S0-01
- Tasks:
  - implement structured logger in `src/rag/logging.py`
  - define base exception hierarchy for pipeline stages
  - include correlation id or request id in logs
- Definition of Done:
  - no-op pipeline emits structured logs for each stage transition
  - errors are explicit and stage-aware (not generic stack traces only)
  - tests verify at least one error path emits actionable message

### S0-06 - Implement module interfaces (no-op adapters)

- Type: `architecture`
- Dependencies: S0-02, S0-03
- Tasks:
  - create `interfaces.py` for each domain module
  - define method signatures and return contracts
  - implement no-op adapters returning deterministic mock outputs
- Definition of Done:
  - interfaces compile/type-check
  - no-op implementations can be swapped later without contract changes
  - behavior is deterministic for tests

### S0-07 - Build no-op end-to-end pipeline orchestrator

- Type: `backend`
- Dependencies: S0-03, S0-05, S0-06
- Tasks:
  - implement `src/rag/pipeline/orchestrator.py`
  - wire stages in correct sequence with typed handoffs
  - add `run-noop` entrypoint command
- Definition of Done:
  - one command runs full pipeline from mock input to mock cited answer
  - pipeline logs each stage start/finish and execution time
  - failures in any stage bubble up with clear context

### S0-08 - Add baseline tests and CI checks

- Type: `testing`
- Dependencies: S0-01, S0-03, S0-07
- Tasks:
  - implement `tests/test_config.py`, `tests/test_noop_pipeline.py`
  - enforce lint/type/test in CI (or equivalent local gate if CI not set yet)
  - add coverage threshold target (modest initial bar)
- Definition of Done:
  - tests pass locally with one command
  - CI (or local gate) fails on lint/type/test violations
  - no-op pipeline has at least one happy-path and one failure-path test

### S0-09 - Document developer workflow and contribution rules

- Type: `docs`
- Dependencies: S0-01, S0-08
- Tasks:
  - update `README.md` with setup, commands, and sprint-0 architecture links
  - add short PR checklist referencing contracts, tests, and logging
  - add ADR `docs/decisions/0001-system-boundaries.md`
- Definition of Done:
  - a new contributor can bootstrap and run no-op pipeline in <=15 minutes
  - docs state coding standards and merge requirements
  - all commands in README are verified to work

## Execution Order

1. S0-01
2. S0-02
3. S0-03 + S0-04 + S0-05 (parallel)
4. S0-06
5. S0-07
6. S0-08
7. S0-09

## Sprint 0 Exit Criteria

- Clear architecture with stable module boundaries.
- Typed contracts implemented and tested.
- No-op end-to-end pipeline runnable via single command.
- Lint/type/test gates active.
- Onboarding and decision docs complete.

## Suggested Ticket Template

Use this for each GitHub Project item:

```md
Title: [S0-XX] <short task title>

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
  - `make run-noop`
```
