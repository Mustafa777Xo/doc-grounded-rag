.PHONY: install format lint typecheck test check clean run-noop

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
	pytest

# Runs the Sprint 0 no-op end-to-end pipeline
run-noop:
	PYTHONPATH=src python -m rag.pipeline.orchestrator

# The ultimate pre-commit / CI gate: runs all verifications
check: lint typecheck test

# Cleans up temporary caches to prevent weird local state issues
clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache __pycache__ src/**/*.pyc tests/**/*.pyc
