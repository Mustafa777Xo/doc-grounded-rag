# Sprint 5 Execution Board

Goal: build a reproducible evaluation harness and systematic error analysis workflow to drive data-backed RAG improvements.

Duration: 3-4 days

Suggested GitHub labels: `sprint-5`, `evaluation`, `error-analysis`, `metrics`, `experimentation`

## Scope Boundaries

- In scope: evaluation dataset specification, evaluation runner, retrieval and generation metrics, experiment tracking outputs, error taxonomy, and improvement prioritization.
- Out of scope: major architecture rewrites, model fine-tuning pipelines, production dashboarding infrastructure.

## Assumptions and Risks

- Assumption: Sprint 3 retrieval and Sprint 4 grounded generation pipelines are runnable and produce structured outputs.
- Assumption: there is a stable corpus and a fixed question set for comparable benchmarking.
- Risk: weak or inconsistent labels can make metrics misleading.
- Risk: metric-only optimization can hide citation/faithfulness regressions.
- Risk: no versioning of dataset/config can make results non-reproducible.
- Risk: error analysis may be too coarse to map to concrete fixes.

## Proposed File Structure Additions (on top of previous sprints)

```text
doc-grounded-rag/
  eval/
    datasets/
      v1/
        questions.jsonl
        relevance_labels.jsonl
        answer_references.jsonl
    runs/
      .gitkeep
    reports/
      .gitkeep
  src/
    rag/
      eval/
        dataset_schema.py
        loader.py
        runner.py
        metrics_retrieval.py
        metrics_generation.py
        metrics_faithfulness.py
        scorer.py
        report_builder.py
        error_taxonomy.py
        error_analyzer.py
        regression_checks.py
      pipeline/
        evaluation_pipeline.py
  tests/
    fixtures/
      eval/
        mini_questions.jsonl
        mini_labels.jsonl
        mini_answers.jsonl
        mini_run_outputs.jsonl
    test_eval_dataset_schema.py
    test_eval_metrics_retrieval.py
    test_eval_metrics_generation.py
    test_eval_faithfulness.py
    test_error_taxonomy.py
    test_error_analyzer.py
    test_evaluation_pipeline.py
  docs/
    evaluation-and-error-analysis.md
```

## Ticket Backlog (Task-by-Task)

### S5-01 - Define evaluation dataset schema and versioning

- Type: `architecture`
- Dependencies: Sprint 3 and 4 output contracts
- Tasks:
  - define schemas for questions, relevance labels, and reference answers
  - enforce dataset versioning and immutable run inputs
  - define split policy (dev/test) and sampling rules
- Definition of Done:
  - schema validators reject malformed or incomplete labels
  - dataset version is required in every run config
  - docs include canonical example records for each file

### S5-02 - Build evaluation data loader and validation pipeline

- Type: `backend`
- Dependencies: S5-01
- Tasks:
  - implement loaders for JSONL datasets with strict validation
  - add cross-file consistency checks (`query_id`, `chunk_id`, references)
  - fail fast with actionable validation errors
- Definition of Done:
  - invalid IDs or missing links are detected before evaluation starts
  - loader returns typed in-memory objects for runner
  - tests cover schema and consistency failures

### S5-03 - Implement retrieval metric module

- Type: `evaluation`
- Dependencies: S5-02
- Tasks:
  - implement `Precision@k`, `Recall@k`, `MRR`, optional `nDCG@k`
  - support per-query and aggregate scoring
  - include confidence intervals or variance summaries where feasible
- Definition of Done:
  - metrics are verified on hand-crafted fixtures
  - per-query debug output identifies which relevant chunks were missed
  - metric calculations are deterministic across repeated runs

### S5-04 - Implement generation quality metric module

- Type: `evaluation`
- Dependencies: S5-02
- Tasks:
  - implement correctness proxy metrics against reference answers
  - implement answer completeness/coverage indicators
  - keep metric definitions explicit and auditable
- Definition of Done:
  - metric outputs include clear interpretation notes
  - tests validate behavior on correct, partial, and incorrect outputs
  - no hidden heuristics without documentation

### S5-05 - Implement faithfulness and citation-integrity metrics

- Type: `evaluation`
- Dependencies: S5-02, Sprint 4 guardrails
- Tasks:
  - compute citation coverage (cited claims ratio)
  - compute unsupported-claim rate from faithfulness checks
  - track abstention rate and insufficient-evidence outcomes
- Definition of Done:
  - output highlights uncited/unsupported claim counts per query
  - tests verify metric behavior across grounded and ungrounded fixtures
  - metrics integrate with strict-mode expectations

### S5-06 - Build unified evaluation runner and run artifacts

- Type: `backend`
- Dependencies: S5-03, S5-04, S5-05
- Tasks:
  - orchestrate retrieval + generation evaluation in one command
  - persist run artifacts (config snapshot, raw outputs, metrics summary)
  - assign unique run IDs and timestamps
- Definition of Done:
  - one command produces reproducible run folder under `eval/runs/`
  - run artifact includes exact config/dataset/model versions
  - rerun with same inputs yields same metrics (within deterministic bounds)

### S5-07 - Implement error taxonomy and automatic error labeling

- Type: `evaluation`
- Dependencies: S5-06
- Tasks:
  - define error classes: `chunking_miss`, `retrieval_miss`, `rerank_miss`, `generation_miss`, `citation_miss`, `insufficient_evidence_false_negative`
  - map failed queries to one primary and optional secondary error label
  - include confidence/rule rationale for each label
- Definition of Done:
  - every failed query in report has an error category
  - taxonomy rules are documented and test-covered
  - ambiguous cases are explicitly marked (not silently bucketed)

### S5-08 - Build error analysis report generator

- Type: `evaluation`
- Dependencies: S5-07
- Tasks:
  - generate report with metric summary, failure breakdown, and top recurring error patterns
  - include query-level drill-down table (query, expected, observed, error label)
  - output markdown and machine-readable JSON report formats
- Definition of Done:
  - report is generated automatically after each run
  - top 3 improvement opportunities are surfaced with evidence
  - report format is stable and parseable for future automation

### S5-09 - Add regression gates and experiment comparison

- Type: `testing`
- Dependencies: S5-06, S5-08
- Tasks:
  - define minimum thresholds for key metrics (retrieval, faithfulness, citation coverage)
  - implement regression checker against baseline run
  - fail CI/local gate on statistically or materially significant regressions
- Definition of Done:
  - baseline snapshot can be pinned and compared
  - evaluation gate blocks merges on threshold failures
  - override flow is documented for intentional tradeoff experiments

### S5-10 - Document evaluation workflow and decision process

- Type: `docs`
- Dependencies: S5-06, S5-08, S5-09
- Tasks:
  - write `docs/evaluation-and-error-analysis.md`
  - update `README.md` with eval run, compare, and report commands
  - define experiment log template (hypothesis, change, results, decision)
- Definition of Done:
  - another engineer can run full benchmark and interpret outputs from docs
  - decision framework is explicit for prioritizing next sprint improvements
  - all documented commands are verified locally

## Execution Order

1. S5-01
2. S5-02
3. S5-03 + S5-04 + S5-05 (parallel)
4. S5-06
5. S5-07
6. S5-08
7. S5-09
8. S5-10

## Sprint 5 Exit Criteria

- Evaluation harness runs end-to-end with versioned datasets and reproducible artifacts.
- Retrieval, generation, and faithfulness metrics are computed at query and aggregate levels.
- Error taxonomy labels failures consistently and highlights dominant failure modes.
- Regression gates exist to protect quality from silent degradation.
- Report output drives a prioritized backlog of next improvements.

## Suggested Ticket Template

```md
Title: [S5-XX] <short task title>

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
  - `<evaluation run command>`
  - `<evaluation compare/regression command>`
```
