# Installation Guide

**Supported Platforms**: Windows 10/11 and Linux (Ubuntu 20.04+ recommended)

> **macOS Note**: Currently unsupported in CI and may not work out of the box due to voice processing dependencies.

## 🪟 Windows Quick Start

Works on Windows 10/11 using PowerShell. Requires Python 3.10+ and Git. Optional: install Ollama and pull models (`ollama pull hermes3:3b`) if you want local LLMs.

### 1) Clone and enter the project

```powershell
git clone https://github.com/hugokos/improved-local-assistant.git
cd improved-local-assistant
```

### 2) Create & activate a virtual environment

```powershell
py -3.12 -m venv .venv        # or py -3.11 / -3.10 if you prefer
.\.venv\Scripts\Activate.ps1  # activates the venv in PowerShell
python -V                     # should show 3.10+ and the .venv path
```

If activation is blocked by policy, either:
- **Run for this session only**: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
- **Use classic CMD activation**: `.\.venv\Scripts\activate.bat`

### 3) Install dependencies + the package (editable)

```powershell
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e . -c constraints.txt
```

### 4) Verify the CLI is installed

```powershell
ila --help

# If 'ila' isn't recognized, run the executable directly:
.\.venv\Scripts\ila.exe --help
```

### 5) Run the server

```powershell
# Dev mode (auto-reload)
ila api --reload --port 8000

# or explicitly:
.\.venv\Scripts\ila.exe api --reload --port 8000
```

Open: http://localhost:8000 (API) and http://localhost:8000/docs (Swagger UI).

**Optional**: download a prebuilt knowledge graph
```powershell
ila download-graphs all
```

### Fallback (don't use ila)

If you prefer the old way or the CLI isn't available yet, run Uvicorn directly:

```powershell
python -m uvicorn improved_local_assistant.api.main:fastapi_app --factory --reload --port 8000
```

### Common Windows issues (and quick fixes)

**`ila: command not found`**
- Ensure the venv is active (`.\.venv\Scripts\Activate.ps1`) and reinstall the package: `pip install -e . -c constraints.txt`
- You can also run it explicitly: `.\.venv\Scripts\ila.exe ...`

**Activation script blocked**
- `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`
- Then re-run `.\.venv\Scripts\Activate.ps1`

**Conda shows (base) in the prompt**
- That's fine—just confirm `where python` shows the `.\.venv\Scripts\python.exe` path

**Port already in use**
- Start on another port: `ila api --port 8080`

**Dependency conflicts**
- Always install with the constraints file: `pip install -e . -c constraints.txt`
- To sanity-check:
  ```powershell
  python -m pip check
  pip install -U pipdeptree
  pipdeptree --warn fail -p llama-index,llama-index-core,llama-index-embeddings-ollama
  ```

### Handy commands

```powershell
# Interactive GraphRAG REPL
ila repl

# Health check / environment sanity
ila health

# Lightweight benchmarks
ila bench

# Stop the server
Ctrl + C
```

### Clean up (optional)

```powershell
deactivate      # leave the venv
rmdir /s /q .venv
```

---

## 🐧 Linux

```bash
# Clone and setup
git clone https://github.com/hugokos/improved-local-assistant.git
cd improved-local-assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (includes PortAudio for voice features)
sudo apt-get update && sudo apt-get install -y libportaudio2
pip install -r requirements.txt
pip install -e . -c constraints.txt

# Run
ila api --reload
```

---

## 🐳 Docker

```bash
# Build and run
docker build -t improved-local-assistant .
docker run -p 8000:8000 improved-local-assistant

# With GPU support
docker run --gpus all -p 8000:8000 improved-local-assistant
```

---

## 🔧 Ollama Setup

1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull models**:
   ```bash
   ollama pull hermes3:3b      # Recommended
   ollama pull phi3:mini       # Faster, smaller
   ```
3. **Verify**: `ollama list` should show your models

---

## ⚡ Verification

After installation:

1. **Start**: `ila api --reload`
2. **Open browser**: http://localhost:8000
3. **API docs**: http://localhost:8000/docs
4. **Health check**: `ila health`

---

## 🚨 Troubleshooting

**Import errors**: `pip install -e . -c constraints.txt --force-reinstall`

**Memory issues**: Use smaller models (`phi3:mini`) or reduce concurrent sessions

**Ollama connection**: Check `ollama list` and ensure Ollama is running

**GPU acceleration**: Install CUDA toolkit + PyTorch with CUDA support

---

## 🔄 Migration from Previous Versions

If you're upgrading from an earlier version:

```bash
# Old way (deprecated)
python run_app.py

# New way
ila api

# Configuration moved
# config.yaml → configs/base.yaml
# Environment variables now use ILA_ prefix
```

---

## 🛠️ Development Setup

For contributors:

```bash
# Install development tools
make dev

# Code quality
make lint          # Run linting
make type          # Type checking
make test          # Run tests
make ci-local      # Simulate CI locally
```