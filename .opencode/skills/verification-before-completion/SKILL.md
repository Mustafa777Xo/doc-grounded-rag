---
name: verification-before-completion
description: Use before claiming work is complete, fixed, correct, or passing; require fresh command output that directly proves each claim.
license: MIT
metadata:
  source: https://github.com/obra/superpowers
---

# Verification Before Completion

Use evidence before assertions.

## Gate

Before making a success claim:

1. Identify the command or inspection that proves the exact claim.
2. Run it freshly and completely.
3. Read the output and exit status, including failure and warning counts.
4. Confirm that the evidence actually supports the claim.
5. Report the result together with the verification performed.

## Project Verification

- Use a focused pytest command while iterating on one behavior.
- Use `make lint` for lint and formatting claims.
- Use `make typecheck` for strict typing claims.
- Use `make test` for test-suite claims.
- Use `make check` for repository-wide completion claims.
- Run relevant ingestion, indexing, or benchmark smoke workflows when a change affects those paths.
- Inspect `git diff` when claiming the requested scope is complete.

A previous run, a partial check, or another agent's report is not fresh evidence.
If verification cannot run, state the blocker and do not imply success.
