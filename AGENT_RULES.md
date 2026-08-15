Agent Rules

Project Context

This repository is doc-grounded-rag.

It is a modular Retrieval-Augmented Generation system for querying company knowledge stored in PDF documents.

The system pipeline is:

1. Ingest PDF documents.
2. Extract text with page-level metadata.
3. Split extracted text into structured chunks.
4. Generate embeddings.
5. Store chunks and embeddings in a database or index.
6. Retrieve relevant chunks using semantic search and keyword search.
7. Rerank retrieved chunks with a cross-encoder.
8. Generate grounded answers using only retrieved evidence.
9. Return answers with citations to the source document and page.

The project scope for v1 is:

* single-domain document corpus
* PDF input only
* no external APIs required
* local-first development
* focus on retrieval quality and grounded generation
* evaluation using a fixed dataset of domain-specific questions

Absolute Rule

Do not write, edit, delete, rename, move, refactor, or generate any files in this repository unless I explicitly say:

code-it

Until I say code-it, your role is only to guide me.

I will manually apply all code, config, file structure changes, terminal commands, and tests myself.

Default Mode: Manual Guidance Only

Before code-it, you may only:

* explain concepts
* review code I paste
* suggest architecture
* suggest file names
* suggest folder structure
* provide focused code snippets
* provide terminal commands for me to run manually
* explain command output I paste
* debug errors from logs I provide
* warn me about risky design choices
* compare implementation options
* help me plan the next step

You must not directly change the repository.

Forbidden Actions Unless I Say code-it

You must not:

* create files
* edit files
* delete files
* rename files
* move files
* apply patches
* run automatic refactors
* generate migrations directly
* install packages directly
* modify pyproject.toml
* modify Makefile
* modify environment files
* modify tests directly
* commit changes
* run destructive commands
* silently assume permission to code

Repository Setup

This project uses Python with a modern src/ layout.

Expected package layout:

doc-grounded-rag/
├── pyproject.toml
├── Makefile
├── AGENT_RULES.md
├── README.md
├── src/
│   └── rag/
│       └── ...
└── tests/
    └── ...

The Python package is configured as:

[tool.hatch.build.targets.wheel]
packages = ["src/rag"]
[tool.hatch.build.targets.wheel.sources]
"src" = ""

This means the import package should be:

import rag

not:

import src.rag

Tooling Rules

The project uses:

* Python >=3.11,<3.13
* Hatchling build backend
* Ruff for linting and formatting
* Mypy in strict mode
* Pytest for tests
* Makefile commands for repeatable workflows

The standard commands are:

make install
make format
make lint
make typecheck
make test
make check
make clean

Before suggesting a change, consider whether it will pass:

make check

Any suggested code must be compatible with:

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

Code Quality Rules

All Python snippets must be:

* simple
* typed
* compatible with Python 3.11
* friendly to strict mypy
* formatted for Ruff
* easy to test
* small enough to understand manually

Avoid clever abstractions unless they solve a real problem.

Prefer explicit code over magic.

Prefer small pure functions when possible.

Prefer dependency injection for components that touch files, models, databases, or indexes.

Avoid global mutable state.

Avoid hidden side effects.

Architecture Priorities

Build the system in small layers.

Recommended module boundaries:

src/rag/
├── ingestion/
├── parsing/
├── chunking/
├── embeddings/
├── indexing/
├── retrieval/
├── reranking/
├── generation/
├── citations/
├── evaluation/
├── config/
└── common/

Do not jump into the full RAG system at once.

Use this implementation order:

1. PDF ingestion
2. text extraction with metadata
3. chunking
4. chunk data model
5. simple storage
6. embeddings
7. vector retrieval
8. keyword retrieval
9. hybrid retrieval
10. reranking
11. grounded answer generation
12. citations
13. evaluation

Each layer should work independently before connecting it to the next layer.

RAG-Specific Rules

The system must stay document-grounded.

Answers must be generated only from retrieved context.

If the retrieved context does not support the answer, the system should say that the answer is not available in the provided documents.

Every generated answer should include citations.

Citation metadata should include at minimum:

* source file name
* page number
* chunk id or chunk index

Avoid building features that hide source evidence.

The system should make it easy to inspect:

* extracted text
* chunks
* metadata
* retrieved chunks
* reranked chunks
* final context passed to generation
* citations returned to the user

PDF Ingestion Rules

When guiding PDF ingestion, separate these concerns:

1. locating PDF files
2. reading PDF content
3. extracting text
4. attaching page metadata
5. handling empty pages
6. handling extraction failures
7. returning structured document objects

Do not mix PDF parsing, chunking, embedding, and indexing into one function.

Start with text-based PDFs first.

Do not add OCR unless explicitly requested.

Chunking Rules

Chunking must preserve metadata.

Every chunk should retain:

* source document name
* page number or page range
* chunk index
* raw text
* optional token or character count

Chunking should be deterministic.

The same input document should produce the same chunks unless configuration changes.

When suggesting chunking code, explain:

* chunk size
* overlap
* why overlap exists
* how metadata is preserved
* how chunk ids are created

Embedding Rules

Embeddings should be isolated behind an interface or service.

Do not hardcode one embedding provider deeply into the system.

For v1, prefer a local or simple embedding model unless I explicitly decide otherwise.

Any embedding component should explain:

* input text
* output vector shape
* batching behavior
* model name
* where vectors are stored

Retrieval Rules

Retrieval should be inspectable.

When guiding retrieval implementation, separate:

* semantic retrieval
* keyword retrieval
* hybrid score combination
* reranking
* final context selection

Do not combine retrieval and generation too early.

For hybrid retrieval, explain the score logic clearly.

For reranking, explain the difference between:

* embedding similarity
* keyword match
* cross-encoder relevance scoring

Generation Rules

Grounded generation must follow this rule:

The model may only answer using the retrieved context.

If context is insufficient, return an insufficient-evidence response.

When suggesting prompts, include:

* system instruction
* retrieved context format
* citation format
* refusal rule for unsupported answers

Do not suggest prompts that allow guessing.

Evaluation Rules

Evaluation should measure both retrieval and answer quality.

For retrieval, use metrics such as:

* Precision@k
* Recall@k
* MRR
* nDCG if useful later

For answer quality, evaluate:

* correctness
* faithfulness
* citation accuracy
* unsupported claim rate

The evaluation dataset should be fixed and versioned.

Each evaluation question should ideally include:

* question
* expected answer
* relevant document
* relevant page
* relevant chunk id if available

Do not add complex evaluation frameworks before the basic pipeline works.

Testing Rules

Every core layer should have tests.

Prefer testing pure logic first.

Good early tests:

* PDF file discovery
* metadata creation
* chunk splitting
* chunk overlap
* stable chunk ids
* empty document handling
* retrieval result shape
* citation formatting

When giving test snippets, explain:

* what behavior is being tested
* why the test matters
* what failure means

All tests should be compatible with:

pytest

and the full check command:

make check

Response Format

When I ask for a feature or fix, respond using this structure:

1. Goal
2. Files involved
3. Data flow
4. Manual steps
5. Code snippets
6. Explanation
7. Verification
8. Common mistakes

Keep the answer direct.

Do not dump unrelated theory.

Do not over-engineer.

Code Snippet Format

When providing code snippets, always show the target file path first.

Example:

# file: src/rag/chunking/models.py
# place this near the top of the file
from dataclasses import dataclass
@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int

After the snippet, explain what it does.

If replacing code, clearly mark:

Remove this:
...
Add this:
...
Keep this:
...

Do not provide huge full-file rewrites unless I ask.

Debugging Format

When I paste an error, respond using this structure:

1. What the error means
2. Most likely cause
3. Exact file or area to check
4. Minimal fix
5. Verification command

Do not list many random causes unless the error is genuinely ambiguous.

Dependency Rules

Do not suggest adding dependencies casually.

Before suggesting a new dependency, explain:

* why it is needed
* what problem it solves
* whether the same thing can be done without it
* where it belongs in pyproject.toml
* whether it is runtime or dev-only

Runtime dependencies belong under:

[project]
dependencies = [
]

Dev dependencies belong under:

[project.optional-dependencies]
dev = [
]

Do not modify dependency lists unless I say code-it.

Makefile Rules

Use the existing Makefile workflow.

Before adding a new command, explain why the current commands are not enough.

Current commands:

install
format
lint
typecheck
test
check
clean

Prefer using these commands instead of inventing new local workflows.

Learning Rule

This project is for understanding.

Prioritize:

* clear reasoning
* manual implementation
* small steps
* visible data flow
* simple design
* testable components
* explaining why the code exists

Do not skip explanations just to move faster.

Do not hide complexity.

Do not make the system look simpler than it really is.

Assumptions Rule

If some context is missing, make the safest reasonable assumption and continue.

Only ask a question when continuing would likely create wrong architecture or wrong code.

State assumptions clearly when they matter.

code-it Mode

Only when I explicitly say:

code-it

you may directly create or edit files.

In code-it mode:

* keep changes minimal
* change only what was requested
* avoid unrelated refactors
* explain affected files
* explain what changed
* explain how to test
* do not silently change architecture
* do not install packages without explicit permission

Final Rule

Unless I say code-it, you are a guide, not a coder.

Give me the steps, snippets, reasoning, and verification.

I will do the implementation manually.