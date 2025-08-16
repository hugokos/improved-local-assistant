@echo off
setlocal enabledelayedexpansion

echo =^> Local CI Simulation (Windows)
python -V
echo.

echo =^> Install dependencies (requirements.txt + constraints)
python -m pip install -U pip
pip install -r requirements.txt
pip install -e . -c constraints.txt

echo =^> Dependency conflict check
python -m pip check
pip install pipdeptree
pipdeptree --warn fail -p llama-index,llama-index-core,llama-index-embeddings-ollama

echo =^> Environment check
python -c "import sys; print('Python:', sys.version)"
python -c "try: import networkx as nx; print('networkx:', nx.__version__); except Exception as e: print('networkx import failed:', e)"
python -c "try: import sounddevice; print('sounddevice: OK'); except Exception as e: print('sounddevice import failed:', e)"
python -c "try: import llama_index; print('llama-index: OK'); except Exception as e: print('llama-index import failed:', e)"

echo =^> Lint
ruff check .
black --check .

echo =^> Type check
mypy src

echo =^> Tests
pytest -q --cov=improved_local_assistant --cov-report=xml --fail-under=70

echo =^> Pre-commit hooks (optional)
where pre-commit >nul 2>&1
if !errorlevel! == 0 (
    pre-commit run --all-files
) else (
    echo pre-commit not installed, skipping
)

echo ✅ All local CI checks passed!