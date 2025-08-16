# Improved Local Assistant

Local‑first GraphRAG assistant with an offline voice interface. Runs entirely on your machine for privacy, speed, and reliability.

[![CI](https://github.com/hugokos/improved-local-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/hugokos/improved-local-assistant/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://hugokos.github.io/improved-local-assistant/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview

**Improved Local Assistant (ILA)** is a production‑oriented, local AI stack that combines a property‑graph knowledge base with hybrid retrieval (graph + vector + BM25) and a voice‑first interface (offline STT/TTS). It’s designed for edge hardware and developer workstations where privacy and low latency matter.

**Highlights**

* **Local‑first privacy:** all inference, retrieval, and voice processing run on your device.
* **Deeper answers:** GraphRAG routing over a property graph preserves semantics and relationships.
* **Voice‑native:** Vosk (STT) + Piper (TTS) for hands‑free chat with streaming responses.
* **Flexible:** Web UI, REST & WebSocket APIs, and CLI tools; configurable models via Ollama.

---

## Architecture

A high‑level look at how the pieces fit together:

```mermaid
flowchart LR
  UI[Web UI / CLI] --> API[FastAPI / WebSocket]
  API --> Router[Graph Router]
  Router -->|graph| KG[(Property Graph)]
  Router -->|vector| VDB[(Vector Store)]
  Router -->|keyword| BM25[(BM25)]
  Router --> LLM[Local LLM (Ollama)]
  LLM --> TTS[TTS (Piper)]
  UI <-->|voice| STT[Vosk]
```

**Key components**

* **Property Graph (KG):** stores entities/relations; ships with optional prebuilt graphs and grows via KG‑Live during conversation.
* **Hybrid Retriever:** fuses graph, vector, and BM25 signals; router selects the smallest useful context window with citations.
* **Local LLM:** model‑agnostic via **Ollama**; pick a small, fast model for TTFT or a heavier model for depth.
* **Voice layer:** streaming STT (Vosk) and TTS (Piper) for low‑latency, fully offline voice chat.

---

## Getting Started

### Prerequisites

* **Python** ≥ 3.10
* **Git**
* **Ollama** (running locally)
* Optional: **CUDA‑capable GPU** for models that can use it

### Install

```bash
# 1) Clone
git clone https://github.com/hugokos/improved-local-assistant.git
cd improved-local-assistant

# 2) Create & activate a virtual environment
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
# source .venv/bin/activate

# 3) Install Python dependencies
pip install -r requirements.txt

# 4) Pull default Ollama models (adjust to taste)
ollama pull hermes3:3b
ollama pull phi3:mini

# 5) (Optional) Download a prebuilt graph
action="survivalist"  # or "all"
python scripts/download_graphs.py "$action"

# 6) Run the app
python run_app.py
```

**Open**

* Web UI: [http://localhost:8000](http://localhost:8000)
* API docs (OpenAPI): [http://localhost:8000/docs](http://localhost:8000/docs)
* Health: [http://localhost:8000/api/health](http://localhost:8000/api/health)

**CLI** (REPL):

```bash
python cli/graphrag_repl.py
```

---

## Configuration

ILA reads configuration from sensible defaults and environment variables. Common knobs:

```bash
# Ollama
export OLLAMA_HOST="http://127.0.0.1:11434"
export ILA_MODEL_CHAT="hermes3:3b"          # chat/inference
export ILA_MODEL_EMBED="nomic-embed-text"   # embedding model name if applicable

# App & storage
export ILA_PORT=8000
export ILA_DATA_DIR="./data"                 # stores graphs, caches, logs
export ILA_PREBUILT_DIR="./data/prebuilt_graphs"

# Router / retrieval (example weights; tune to your taste)
export ILA_USE_GRAPH=true
export ILA_USE_VECTOR=true
export ILA_USE_BM25=true
export ILA_ROUTER_GRAPH_WEIGHT=0.5
export ILA_ROUTER_VECTOR_WEIGHT=0.4
export ILA_ROUTER_BM25_WEIGHT=0.1
```

> Tip: put these in a `.env` file and load with your shell, or use your process manager.

---

## Knowledge Graphs

**KG‑Prime (prebuilt):** a property graph you can ship with the app for a domain (e.g., "survivalist").

**KG‑Live (dynamic):** updates during conversation—new entities/edges extracted from user and assistant turns.

### Add your own documents

```bash
# Ingest a folder of markdown/PDF/text into a new prebuilt graph
python scripts/build_graph.py --input ./my_docs --out ./data/prebuilt_graphs/my_domain

# Point the app at it
export ILA_PREBUILT_DIR="./data/prebuilt_graphs/my_domain"
python run_app.py
```

Chunking and entity extraction are configurable. Start with smaller, semantically coherent chunks for tighter relationships; rely on the retriever to stitch cross‑chunk context.

---

## Retrieval & Routing

* **Graph traversal** finds semantically linked entities and relations.
* **Vector search** (embedding‑based) surfaces semantically similar passages.
* **BM25** ensures exact‑term recall for names, commands, and IDs.
* The **router** balances these signals and builds a small, cited context for the LLM.

To tweak behavior, adjust the `ILA_USE_*` flags and weights, or the per‑retriever limits (k‑values) in your config.

---

## Voice Interface (Offline)

* **STT (Vosk):** per‑session recognizers; real‑time partial and final transcripts; VAD‑assisted utterance boundaries.
* **TTS (Piper):** streaming synthesis; configurable voices; audio chunks are sent to the UI while tokens stream from the LLM.

**Enable voice in the Web UI:** toggle the mic orb to start/stop listening. Configure default devices in your browser/OS.

---

## APIs & Examples

The server exposes a REST API and a WebSocket for streaming.

**Health**

```bash
curl http://localhost:8000/api/health
```

**Chat (REST)**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

**Streaming (WebSocket) — Python snippet**

```python
import asyncio, websockets, json

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        await ws.send(json.dumps({"type": "user", "content": "Explain GraphRAG in 2 lines"}))
        async for msg in ws:
            print(msg)

asyncio.run(main())
```

---

## Performance & Tuning

* **Model choice:** prefer smaller quantized models for low TTFT; switch to larger models for depth.
* **Caching:** enable embedding/result caches to speed repeated queries.
* **Concurrency:** limit concurrent sessions on low‑power systems; configure worker pool size.
* **Retrieval budgets:** cap per‑retriever `k` and token budgets to avoid oversized contexts.

### Benchmarks (reproducible)

Use the included scripts to measure **TTFT**, **tokens/s**, and **end‑to‑end latency**:

```bash
python scripts/run_benchmarks.py
python scripts/benchmark_models.py --model hermes3:3b --contexts 1024 2048 4096 --runs 5
```

Results are stored under `benchmarks/` with CSV/JSON outputs you can plot.

---

## Security & Privacy

* No external API calls are required; network calls can be disabled entirely.
* Logs and caches stay in `ILA_DATA_DIR` on your machine.
* Sanitization hooks are available before content is persisted to the graph.

---

## Troubleshooting

* **Ollama not reachable** → verify `OLLAMA_HOST` and that the daemon is running.
* **Model not found** → `ollama pull <name>` again; confirm tags.
* **Mic or TTS silent** → check OS device permissions and your browser’s site settings.
* **Port in use** → change `ILA_PORT` or free the port.

---

## Roadmap (selected)

* Presets for **speed vs. quality** routing and model bundles.
* Optional Windows/macOS packaged launchers.
* Coverage reporting and Codecov badge in CI.
* Additional docs: developer internals, deployment patterns, and advanced voice controls.

---

## Contributing

We welcome issues and PRs! Please read:

* [CONTRIBUTING.md](CONTRIBUTING.md)
* [CODE\_OF\_CONDUCT.md](CODE_OF_CONDUCT.md)

Use conventional commits when possible (`feat:`, `fix:`, `docs:` …). Run linters before committing if you use pre‑commit.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Thanks to the maintainers of **Ollama**, **LlamaIndex**, **FastAPI**, **Vosk**, and **Piper** for foundational tooling.

<!-- Optional demo: place a short GIF at docs/assets/demo.gif and uncomment below -->

<!-- ![Demo](docs/assets/demo.gif) -->
