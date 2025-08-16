#!/usr/bin/env bash
set -euo pipefail

echo "==> Preflight CI - Local Simulation"
echo "Python: $(python -V)"
echo "Platform: ${OSTYPE:-unknown}"
echo

# Linux audio prereq (no-op elsewhere)
if [[ "${OSTYPE:-}" == "linux-gnu"* ]]; then
  if ! dpkg -s libportaudio2 >/dev/null 2>&1; then
    echo "Installing libportaudio2 (Linux)..."
    sudo apt-get update && sudo apt-get install -y libportaudio2
  fi
fi

echo "==> Install dependencies"
python -m pip install -U pip
pip install -r requirements.txt
pip install -e . -c constraints.txt
pip install pytest-cov
pip install types-requests types-PyYAML types-setuptools

echo "==> Dependency health check"
python -m pip check
pip install -U pipdeptree
pipdeptree --warn fail -p llama-index,llama-index-core,llama-index-embeddings-ollama

echo "==> Pre-commit hooks"
if command -v pre-commit >/dev/null 2>&1; then
    pre-commit run --all-files
else
    echo "pre-commit not installed, installing..."
    pip install pre-commit
    pre-commit run --all-files
fi

echo "==> Lint"
ruff check .
black --check .

echo "==> Type check"
mypy src

echo "==> Sanity check pytest-cov"
pytest --help | grep -- --cov

echo "==> Tests"
pytest -q -vv -ra --cov=improved_local_assistant --cov-report=xml --fail-under=70

echo "✅ All preflight CI checks passed!"
echo "Ready to push to GitHub!"