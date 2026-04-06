# doc-grounded-rag

A modular retrieval-augmented generation (RAG) system for querying company knowledge stored in PDF documents.

The system ingests PDFs, extracts and chunks content with metadata, indexes it using embeddings, and answers questions by retrieving and reranking relevant context. Responses are generated using only retrieved evidence and include citations to source documents.


## Features

- PDF ingestion with page-level metadata
- structured text chunking
- embedding-based indexing
- hybrid retrieval (semantic + keyword)
- cross-encoder reranking
- grounded answer generation with citations
- evaluation pipeline for retrieval and answer quality

## Pipeline

1. Ingest PDF documents
2. Extract text and attach metadata (file, page)
3. Split into chunks
4. Generate embeddings and store in database
5. Retrieve relevant chunks (vector + keyword)
6. Rerank results
7. Generate grounded answer using selected context
8. Return answer with citations

## Evaluation

The system includes an evaluation pipeline to measure:

- retrieval quality (Precision@k, Recall@k)
- answer correctness
- faithfulness to retrieved context

Evaluation is performed on a fixed dataset of domain-specific questions.

## Scope

- single-domain document corpus
- PDF input only (v1)
- no external APIs required
- focused on retrieval quality and grounded generation
