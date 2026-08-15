---
name: systematic-debugging
description: Use when encountering a bug, test failure, performance problem, integration failure, or unexpected behavior; investigate and prove the root cause before proposing a fix.
license: MIT
metadata:
  source: https://github.com/obra/superpowers
---

# Systematic Debugging

Find the root cause before changing code. A symptom-level patch is not a fix.

## Workflow

1. Read the complete error, traceback, command, and exit status.
2. Reproduce the failure consistently. If it is intermittent, gather evidence rather than guessing.
3. Inspect relevant recent changes and compare the failing path with a working example.
4. Trace bad data backward through each component boundary until its origin is known.
5. State one falsifiable hypothesis: "X is the cause because Y."
6. Test that hypothesis with the smallest possible experiment and one changed variable.
7. Add a focused failing test when behavior can be reproduced automatically.
8. Implement one root-cause fix, then run the focused test and the appropriate broader quality gate.

For this repository, preserve the separation between ingest, chunk, embed, index,
retrieve, rerank, generate, and eval while tracing failures. Inspect contracts and
metadata at every boundary, especially `doc_id`, `source_file`, `page`,
`chunk_id`, and `chunk_index`.

## Stop Conditions

- Do not propose a fix before collecting evidence.
- Do not bundle several speculative fixes into one experiment.
- If three attempted fixes fail, stop and discuss whether the architecture or assumption is wrong.
- If the cause is external or environmental, document the evidence and add explicit handling or diagnostics.
