---
name: rag-grounding-audit
description: Use when reviewing generated answers, citations, insufficient-evidence behavior, unsupported claims, context assembly, or grounding regressions in this document-grounded RAG project.
---

# RAG Grounding Audit

Audit whether every answer claim is supported by retrieved document evidence and
traceable to its source.

## Workflow

1. Capture the user question, retrieved candidates, reranked order, final context, answer, and citations.
2. Break the answer into independently verifiable claims.
3. Map each claim to exact supporting context and its `source_file`, page, and chunk identifier.
4. Mark claims as supported, partially supported, unsupported, or contradicted.
5. Verify citation correctness, completeness, and placement; a real source that does not support the claim is an incorrect citation.
6. Confirm the generator used only the supplied context and did not fill gaps from general model knowledge.
7. When evidence is insufficient, require the project's explicit no-evidence response instead of guessing.
8. Report unsupported-claim rate and citation accuracy for evaluation sets, with concrete failing examples.
9. Add focused tests for every corrected grounding or citation regression.

Do not hide the retrieved or final context during diagnosis. Preserve inspectability
across retrieval, reranking, context selection, generation, and citation formatting.
