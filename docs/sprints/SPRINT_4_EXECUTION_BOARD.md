# Sprint 4 Execution Board

Goal: implement grounded answer generation that only uses retrieved evidence and always returns verifiable citations.

Duration: 2-3 days

Suggested GitHub labels: `sprint-4`, `generation`, `citations`, `faithfulness`, `safety`

## Scope Boundaries

- In scope: prompt/context assembly, grounded generation flow, citation attachment rules, faithfulness guardrails, insufficient-evidence handling, and validation tests.
- Out of scope: large-scale answer quality benchmarking (Sprint 5), advanced product UX, model fine-tuning.

## Assumptions and Risks

- Assumption: Sprint 3 returns reranked candidates with stable `chunk_id` and citation metadata.
- Assumption: each candidate includes enough text span/context to support claims.
- Risk: model may produce plausible but uncited statements.
- Risk: citation drift (correct citation object attached to wrong claim).
- Risk: long contexts may exceed model token budget and drop critical evidence.
- Risk: strict grounding can increase abstentions if retrieval quality is weak.

## Proposed File Structure Additions (on top of previous sprints)

```text
doc-grounded-rag/
  src/
    rag/
      generate/
        prompt_builder.py
        context_packer.py
        llm_client.py
        service.py
        guardrails.py
        citation_mapper.py
        validators.py
      pipeline/
        answer_pipeline.py
      eval/
        faithfulness_checks.py
  tests/
    fixtures/
      generation/
        grounded_cases.jsonl
        insufficient_evidence_cases.jsonl
    test_prompt_builder.py
    test_context_packer.py
    test_citation_mapper.py
    test_generation_guardrails.py
    test_answer_pipeline.py
    test_faithfulness_checks.py
  docs/
    grounded-generation-and-citations.md
```

## Ticket Backlog (Task-by-Task)

### S4-01 - Define grounded answer and citation contracts

- Type: `architecture`
- Dependencies: Sprint 1 and 3 contracts
- Tasks:
  - finalize `AnswerWithCitations` schema (`answer_text`, `claims`, `citations`, `confidence`, `status`)
  - define citation object fields (`chunk_id`, `doc_id`, `source_file`, `page`, optional span offsets)
  - define status modes (`grounded_answer`, `insufficient_evidence`, `generation_error`)
- Definition of Done:
  - schema validation rejects uncited claims in strict mode
  - contract tests include invalid/missing citation payloads
  - docs include canonical grounded answer example

### S4-02 - Implement context packing from reranked evidence

- Type: `backend`
- Dependencies: S4-01
- Tasks:
  - implement evidence selection policy (top-n, diversity by doc/page)
  - implement token-budget-aware context packing with deterministic truncation
  - preserve source mapping for every included context segment
- Definition of Done:
  - same query + evidence set yields identical packed context
  - token budget overflow is handled explicitly and logged
  - tests verify ordering, truncation, and source mapping integrity

### S4-03 - Implement grounded prompt templates

- Type: `backend`
- Dependencies: S4-02
- Tasks:
  - implement strict prompt template that forbids using outside knowledge
  - require per-claim citation references in model output format
  - add configurable style options without changing grounding constraints
- Definition of Done:
  - prompts include explicit refusal/abstain behavior when evidence is insufficient
  - output format is machine-parseable and validated
  - tests verify critical instruction blocks are present

### S4-04 - Implement LLM generation adapter and response parser

- Type: `backend`
- Dependencies: S4-03
- Tasks:
  - implement provider-agnostic LLM adapter for completion/chat
  - parse model output into structured claim + citation references
  - add timeout/retry handling and explicit error taxonomy
- Definition of Done:
  - adapter supports deterministic test mode with mock provider
  - parser rejects malformed outputs with actionable errors
  - tests cover timeout, parse error, and provider failure scenarios

### S4-05 - Build citation mapping and validation layer

- Type: `backend`
- Dependencies: S4-02, S4-04
- Tasks:
  - map model citation references back to retrieval evidence metadata
  - validate every claim has at least one valid citation
  - reject citations that do not exist in provided context set
- Definition of Done:
  - invalid/unknown citation refs fail validation explicitly
  - citation metadata in final answer is complete and traceable
  - tests cover one-to-many and many-to-one claim-citation mappings

### S4-06 - Implement grounding guardrails and abstention policy

- Type: `backend`
- Dependencies: S4-05
- Tasks:
  - implement strict mode: block uncited claims
  - implement insufficient-evidence decision logic (confidence/rule-based thresholds)
  - define safe fallback message with suggested user reformulation
- Definition of Done:
  - strict mode never emits uncited factual claims in tests
  - insufficient evidence returns correct status and empty/limited answer
  - logs indicate why abstention was triggered

### S4-07 - Build retrieve-to-answer pipeline command

- Type: `backend`
- Dependencies: S4-02, S4-03, S4-04, S4-05, S4-06
- Tasks:
  - orchestrate reranked evidence input -> context pack -> generate -> validate -> output
  - add CLI command for single query and batch file
  - emit structured metrics (tokens in/out, latency, citation counts, abstentions)
- Definition of Done:
  - command returns structured grounded answer payload
  - all output modes are explicit (`grounded_answer`, `insufficient_evidence`, `generation_error`)
  - failures are stage-specific and actionable

### S4-08 - Add faithfulness and citation integrity checks

- Type: `evaluation`
- Dependencies: S4-07
- Tasks:
  - implement rule-based faithfulness checks against retrieved context
  - implement citation coverage metric (claims with valid citations / total claims)
  - add test fixtures for grounded and ungrounded outputs
- Definition of Done:
  - checks flag uncited or unsupported claims reliably on fixtures
  - per-query report includes faithfulness and citation coverage indicators
  - tests validate expected pass/fail behavior

### S4-09 - Add regression and reliability tests

- Type: `testing`
- Dependencies: S4-07, S4-08
- Tasks:
  - add integration tests for end-to-end answer pipeline
  - add deterministic tests for strict-mode behavior and abstention logic
  - add smoke tests for malformed retrieval payloads
- Definition of Done:
  - happy path and critical failure paths are covered
  - strict grounding regressions fail tests immediately
  - all tests pass in existing quality gate commands

### S4-10 - Document generation/citation design and known limits

- Type: `docs`
- Dependencies: S4-07, S4-09
- Tasks:
  - write `docs/grounded-generation-and-citations.md`
  - update `README.md` with answer pipeline and strict mode commands
  - document known limitations (multi-hop reasoning gaps, ambiguous evidence, long-context truncation)
- Definition of Done:
  - another engineer can run grounded answering flow from docs alone
  - tradeoffs between strictness and answer coverage are documented
  - all documented commands are validated locally

## Execution Order

1. S4-01
2. S4-02
3. S4-03
4. S4-04
5. S4-05
6. S4-06
7. S4-07
8. S4-08
9. S4-09
10. S4-10

## Sprint 4 Exit Criteria

- Grounded answer generation pipeline is runnable for single and batch queries.
- Final answers include complete, traceable citations tied to retrieval evidence.
- Strict mode prevents uncited factual output and handles abstention explicitly.
- Faithfulness and citation-integrity checks are implemented and reproducible.
- Documentation enables reproducible usage and clear understanding of limitations.

## Suggested Ticket Template

```md
Title: [S4-XX] <short task title>

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
  - `<answer pipeline command>`
  - `<faithfulness check command>`
```
