# Environment Variables

This document lists all environment variables used by Improved Local Assistant.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ILA_CONFIG` | `configs/base.yaml` | Path to configuration file |

## Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ILA_HOST` | `0.0.0.0` | Server host to bind to |
| `ILA_PORT` | `8000` | Server port |

## Ollama Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ILA_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `ILA_MODEL_CHAT` | `hermes3:3b` | Chat/inference model |
| `ILA_MODEL_EMBED` | `nomic-embed-text` | Embedding model |

## Storage Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ILA_DATA_DIR` | `./data` | Data directory |
| `ILA_PREBUILT_DIR` | `./data/prebuilt_graphs` | Prebuilt graphs directory |

## Router Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ILA_USE_GRAPH` | `true` | Enable graph retrieval |
| `ILA_USE_VECTOR` | `true` | Enable vector retrieval |
| `ILA_USE_BM25` | `true` | Enable BM25 retrieval |
| `ILA_ROUTER_GRAPH_WEIGHT` | `0.5` | Graph retrieval weight |
| `ILA_ROUTER_VECTOR_WEIGHT` | `0.4` | Vector retrieval weight |
| `ILA_ROUTER_BM25_WEIGHT` | `0.1` | BM25 retrieval weight |

## Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ILA_EDGE_OPTIMIZATION` | `true` | Enable edge optimizations |
| `ILA_MAX_CONCURRENT_SESSIONS` | `10` | Max concurrent sessions |

## Voice Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ILA_VOICE_ENABLED` | `true` | Enable voice interface |
| `ILA_STT_MODEL` | `vosk-model-en-us-0.22` | STT model |
| `ILA_TTS_VOICE` | `en_US-lessac-medium` | TTS voice |

## Legacy Environment Variables

These may still be used by some components during the transition:

| Variable | Description |
|----------|-------------|
| `STARTUP_PRELOAD` | What to preload during startup |
| `GRAPH_LAZY_LOAD` | Load graphs in background |
| `OLLAMA_HEALTHCHECK` | Type of Ollama health check |
| `EMBED_MODEL_DEVICE` | Device for embedding model |
| `EMBED_MODEL_INT8` | Use int8 quantization |
| `RESOURCE_MONITOR_INTERVAL` | Resource monitoring interval |
| `MEM_PRESSURE_THRESHOLD` | Memory pressure threshold |
| `SKIP_SURVIVALIST_GRAPH` | Skip loading survivalist graph |

## Usage Examples

```bash
# Basic usage
export ILA_PORT=8080
export ILA_MODEL_CHAT="llama3.2:3b"
ila api

# Development with custom config
export ILA_CONFIG="configs/dev.yaml"
ila api --reload

# Production deployment
export ILA_HOST="0.0.0.0"
export ILA_PORT="80"
export ILA_EDGE_OPTIMIZATION="true"
ila api
```

## Documentation

The project uses MkDocs with Material theme for documentation:

```bash
# Serve docs locally
make docs-serve

# Build docs
make docs-build

# Test docs configuration
make docs-test

# Deploy to GitHub Pages (CI does this automatically)
make docs-deploy
```

Documentation is automatically deployed to GitHub Pages when changes are pushed to the main branch.