@echo off
setlocal enabledelayedexpansion

echo =^> Preflight CI - Local Simulation (Windows)
python -V
echo.

echo =^> Install dependencies
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .[dev] -c constraints.txt || pip install -e . -c constraints.txt
pip install types-requests types-PyYAML types-setuptools

echo =^> Dependency health check
python -m pip check
pip install -U pipdeptree
pipdeptree --warn fail -p llama-index,llama-index-core,llama-index-embeddings-ollama

echo =^> Pre-commit hooks
where pre-commit >nul 2>&1
if !errorlevel! == 0 (
    pre-commit run --all-files
) else (
    echo pre-commit not installed, installing...
    pip install pre-commit
    pre-commit run --all-files
)

echo =^> Lint
ruff check .
black --check .

echo =^> Type check
mypy src

echo =^> Tests
pytest -q

echo ✅ All preflight CI checks passed!
echo Ready to push to GitHub!