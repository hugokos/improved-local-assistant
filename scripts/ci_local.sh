#!/usr/bin/env bash
set -euo pipefail

echo "==> Local CI Simulation"
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

echo "==> Install dependencies (with constraints)"
python -m pip install -U pip
pip install -e .[dev] -c constraints.txt

echo "==> Environment check"
python - << 'PY'
import sys, platform
print("Python:", sys.version)
try:
    import networkx as nx; print("networkx:", nx.__version__)
except Exception as e: print("networkx import failed:", e)
try:
    import sounddevice; print("sounddevice: OK")
except Exception as e: print("sounddevice import failed:", e)
try:
    import llama_index; print("llama-index: OK")
except Exception as e: print("llama-index import failed:", e)
PY

echo "==> Lint"
ruff check .
black --check .

echo "==> Type check"
mypy src

echo "==> Tests"
pytest -q --cov=improved_local_assistant --cov-report=xml --fail-under=70

echo "==> Pre-commit hooks (optional)"
if command -v pre-commit >/dev/null 2>&1; then
    pre-commit run --all-files
else
    echo "pre-commit not installed, skipping"
fi

echo "✅ All local CI checks passed!"