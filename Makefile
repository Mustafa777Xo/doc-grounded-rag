.PHONY: install format lint typecheck test coverage check clean run-noop ingest

OUTPUT ?= data/processed/chunks/chunks.jsonl
MODE ?= overwrite

# Installs the package in editable mode along with dev tools
install:
	pip install -e ".[dev]"

# Auto-formats code and sorts imports
format:
	ruff format .
	ruff check --select I --fix .

# Checks for stylistic and syntax errors without modifying files
lint:
	ruff check .
	ruff format --check .

# Runs strict static type checking
typecheck:
	mypy src/ tests/

# Runs the test suite
test:
	PYTHONPATH=src pytest

# Runs tests with a modest coverage threshold
coverage:
	PYTHONPATH=src pytest --cov=rag --cov-report=term-missing --cov-fail-under=60

# Runs the Sprint 0 no-op end-to-end pipeline
run-noop:
	PYTHONPATH=src python -m rag.pipeline.orchestrator

# Runs the Sprint 1 PDF ingestion pipeline
ingest:
	PYTHONPATH=src python -m rag.pipeline.ingest --input "$(INPUT)" --output "$(OUTPUT)" --mode "$(MODE)"

# The ultimate pre-commit / CI gate: runs all verifications
check: lint typecheck coverage

# Cleans up temporary caches to prevent weird local state issues
clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache __pycache__ src/**/*.pyc tests/**/*.pyc
