.PHONY: help install install-dev test lint format type-check security docs clean build publish

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and ruff"
	@echo "  type-check   Run type checking with mypy"
	@echo "  security     Run security checks"
	@echo "  docs         Build documentation"
	@echo "  docs-serve   Serve documentation locally"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  publish      Publish to PyPI"
	@echo "  pre-commit   Run pre-commit hooks"
	@echo "  setup        Complete development setup"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src/improved_local_assistant --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -x -q

# Code quality
lint:
	ruff check .
	black --check .

format:
	ruff format .
	black .
	isort .

type-check:
	mypy src/

# Security
security:
	bandit -r src/ -f json -o bandit-report.json
	safety check

# Documentation
docs:
	mkdocs build

docs-serve:
	mkdocs serve

# Maintenance
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building and publishing
build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Development workflow
pre-commit:
	pre-commit run --all-files

setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# CI simulation
ci: lint type-check security test
	@echo "All CI checks passed!"

# Quick development check
check: format lint type-check test-fast
	@echo "Quick development check complete!"