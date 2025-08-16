.PHONY: dev test lint type run clean install help

# Default target
help:
	@echo "Available targets:"
	@echo "  dev     - Install development dependencies"
	@echo "  test    - Run tests"
	@echo "  lint    - Run linting (ruff + black)"
	@echo "  type    - Run type checking (mypy)"
	@echo "  run     - Run the application in development mode"
	@echo "  clean   - Clean up build artifacts"
	@echo "  install - Install package in development mode"

# Development setup
dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e . -c constraints.txt

# Install package
install:
	pip install -r requirements.txt
	pip install -e . -c constraints.txt

# Testing
test:
	pytest -q --maxfail=1

test-cov:
	pytest -q --cov=improved_local_assistant --cov-report=xml --fail-under=70

test-smoke:
	pytest -q -m smoke

# Code quality
lint:
	ruff check .
	black --check .

lint-fix:
	ruff check . --fix
	black .

type:
	mypy src

# Run application
run:
	ila api --reload

run-prod:
	ila api --host 0.0.0.0 --port 8000

# Utilities
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# CI simulation
ci-local:
	bash scripts/ci_local.sh

# Dependency conflict check
check-deps:
	python -m pip check
	pip install pipdeptree
	pipdeptree --warn fail -p llama-index,llama-index-core,llama-index-embeddings-ollama

# Multi-version testing
tox:
	tox -q

# Documentation
docs-serve:
	mkdocs serve -f config/mkdocs.yml

docs-build:
	mkdocs build -f config/mkdocs.yml

# Health check
health:
	ila health

# Download graphs
download-graphs:
	ila download-graphs all