# Edge Optimization Guide

This guide covers the edge optimization features designed to maximize performance on CPU-constrained devices like Raspberry Pi, mobile devices, and low-power systems.

## Overview

The edge optimization system introduces a centralized LLM orchestration layer that ensures optimal resource utilization through:

- **Turn-by-turn orchestration**: Hermes streams first, TinyLlama extracts after completion
- **Process-wide semaphore**: Prevents model contention and CPU thrashing
- **Working set caching**: Fast graph retrieval through bounded neighborhood caching
- **Hybrid ensemble retrieval**: Multi-modal retrieval with strict budget controls
- **Resource-aware extraction**: Bounded processing with skip logic under pressure
- **Connection pooling**: Optimized HTTP client with keep-alive connections

## Quick Start

### Enable Edge Optimization

```bash
# Enable edge optimization
python cli/toggle_edge_optimization.py --enable

# Check status
python cli/toggle_edge_optimization.py --status

# Set device-specific mode
python cli/toggle_edge_optimization.py --mode raspberry_pi
```

### Test the System

```bash
# Run comprehensive tests
python scripts/test_edge_optimization.py

# Test orchestration components
python cli/test_orchestration.py

# A/B test with and without knowledge graphs
python -m asyncio cli.graphrag_repl --no-kg
```

## Architecture

### Core Components

1. **LLM Orchestrator** (`services/llm_orchestrator.py`)
   - Process-wide `asyncio.Semaphore(1)` prevents concurrent LLM execution
   - Pre-warms Hermes at startup, unloads TinyLlama after extraction
   - Handles task cancellation when new user turns arrive

2. **Connection Pool Manager** (`services/connection_pool_manager.py`)
   - Shared `httpx.AsyncClient` with connection pooling
   - Explicit timeouts and keep-alive management
   - Model residency verification via `/api/ps`

3. **Working Set Cache** (`services/working_set_cache.py`)
   - Per-session caches of recently accessed node IDs
   - LRU eviction with global memory limits
   - Disk persistence for warm starts

4. **Hybrid Ensemble Retriever** (`services/hybrid_retriever.py`)
   - Combines graph, BM25, and vector retrieval
   - Weighted fusion with configurable budgets
   - PropertyGraphIndex integration with `include_text=False`

5. **Extraction Pipeline** (`services/extraction_pipeline.py`)
   - Bounded TinyLlama processing with time/token limits
   - Resource pressure monitoring with skip logic
   - JSON format enforcement for reliable parsing

### Data Flow

```
User Message → LLM Orchestrator → [Hermes Streams] → [TinyLlama Extracts] → KG Update
                     ↓
            Working Set Cache ← Hybrid Retriever ← Context Assembly
```

## Configuration

### Basic Configuration

```yaml
# Enable edge optimization
edge_optimization:
  enabled: true
  mode: "production"  # development, production, raspberry_pi, iphone

# LLM Orchestration
orchestration:
  llm_semaphore_timeout: 30.0
  keep_alive_hermes: "30m"
  keep_alive_tinyllama: 0
  json_mode_for_extraction: true
  max_extraction_time: 8.0

# Connection pooling
connection_pool:
  max_connections: 10
  max_keepalive_connections: 5
  timeout:
    connect: 5.0
    read: 30.0
    write: 30.0
    pool: 5.0
```

### Device-Specific Configurations

#### Raspberry Pi
```yaml
environment_raspberry_pi:
  working_set_cache:
    nodes_per_session: 50
    global_memory_limit_mb: 128
  hybrid_retriever:
    budget:
      max_chunks: 8
  extraction:
    max_time_seconds: 4.0
    max_tokens: 512
  server_env:
    OLLAMA_NUM_PARALLEL: 1
    OLLAMA_MAX_LOADED_MODELS: 1
```

#### iPhone/Mobile
```yaml
environment_iphone:
  working_set_cache:
    nodes_per_session: 25
    global_memory_limit_mb: 64
  hybrid_retriever:
    budget:
      max_chunks: 6
  extraction:
    max_time_seconds: 2.0
    max_tokens: 256
```

### Advanced Configuration

#### Working Set Cache
```yaml
working_set_cache:
  nodes_per_session: 100        # Max nodes cached per session
  max_edges_per_query: 40       # Max edges in 1-hop neighborhood
  global_memory_limit_mb: 256   # Global cache memory limit
  eviction_threshold: 0.8       # LRU eviction threshold
  include_text: false           # Use facts over raw text
  persist_dir: "./storage"      # Disk persistence directory
```

#### Hybrid Retriever
```yaml
hybrid_retriever:
  weights:
    graph: 0.6    # Prefer graph results
    bm25: 0.2     # Lightweight text search
    vector: 0.2   # Semantic similarity
  budget:
    max_chunks: 12      # Hard limit on retrieved chunks
    graph_depth: 2      # Max graph traversal depth
    bm25_top_k: 3       # BM25 result limit
    vector_top_k: 3     # Vector result limit
  response_mode: "compact"  # Reduce LLM calls
  query_fusion:
    enabled: true
    num_queries: 3      # Query expansion limit
```

#### Extraction Pipeline
```yaml
extraction:
  max_time_seconds: 8.0         # Hard time limit
  max_tokens: 1024              # Token generation limit
  max_triples_per_turn: 10      # Output size limit
  skip_on_memory_pressure: true # Skip under pressure
  skip_on_cpu_pressure: true    # Skip under CPU load
  input_window_turns: 3         # Context window size
  format: "json"                # Enforce JSON output
  unload_after_extraction: true # Free memory after use
```

## Performance Targets

### Latency Goals
- **Time-to-First-Token**: <500ms (with warm Hermes)
- **Graph Retrieval**: <1s (using working set cache)
- **Total Response**: 1-3s typical
- **Extraction Budget**: ≤8s max, skip on pressure

### Resource Constraints
- **Memory**: Stay within configured thresholds (70-95% depending on device)
- **CPU**: Respect existing CPU monitoring limits
- **Concurrency**: Single LLM execution, bounded background tasks

## Monitoring and Metrics

### Built-in Metrics

```python
# Get orchestration metrics
status = await model_manager.get_model_status()

# Orchestrator metrics
orchestrator_metrics = status["orchestrator"]["metrics"]
print(f"Turns processed: {orchestrator_metrics['turns_processed']}")
print(f"Extractions completed: {orchestrator_metrics['extractions_completed']}")

# Working set cache metrics
cache_stats = status["working_set_cache"]
print(f"Cache hit rate: {cache_stats['metrics']['cache_hits'] / cache_stats['metrics']['cache_misses']}")

# Hybrid retriever metrics
retriever_metrics = status["hybrid_retriever"]["metrics"]
print(f"Average retrieval time: {retriever_metrics['avg_retrieval_time']}")
```

### Performance Validation

```bash
# Validate performance gates
python scripts/test_edge_optimization.py

# Check model residency
curl http://localhost:11434/api/ps

# Monitor resource usage
python scripts/system_health_check.py
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check cache usage
python cli/toggle_edge_optimization.py --status

# Reduce cache limits
# Edit config.yaml:
working_set_cache:
  global_memory_limit_mb: 128  # Reduce from 256
  nodes_per_session: 50        # Reduce from 100
```

#### Slow Response Times
```bash
# Check if models are resident
curl http://localhost:11434/api/ps

# Verify pre-warming
# Check logs for "Successfully pre-warmed hermes3:3b"

# Reduce context budget
hybrid_retriever:
  budget:
    max_chunks: 8  # Reduce from 12
```

#### Extraction Timeouts
```bash
# Check extraction metrics
python scripts/test_edge_optimization.py

# Reduce extraction budget
extraction:
  max_time_seconds: 4.0  # Reduce from 8.0
  max_tokens: 512        # Reduce from 1024
```

### Debug Mode

```yaml
# Enable debug logging
logging:
  level: DEBUG

# Disable extraction skip logic for testing
extraction:
  skip_on_memory_pressure: false
  skip_on_cpu_pressure: false
```

## Migration Guide

### From Standard to Orchestrated

1. **Enable edge optimization**:
   ```bash
   python cli/toggle_edge_optimization.py --enable
   ```

2. **Update startup command**:
   ```bash
   # Old
   python app.py

   # New (same command, orchestration auto-detected)
   python app.py
   ```

3. **Verify operation**:
   ```bash
   python scripts/test_edge_optimization.py
   ```

### Backward Compatibility

The orchestrated system maintains full backward compatibility:

- All existing CLI commands work unchanged
- WebSocket API remains identical
- Configuration files are backward compatible
- Graceful degradation when orchestration disabled

### Rollback

```bash
# Disable edge optimization
python cli/toggle_edge_optimization.py --disable

# Restart application
python app.py
```

## Best Practices

### Device-Specific Tuning

#### Raspberry Pi 4 (4GB RAM)
```yaml
environment_raspberry_pi:
  system:
    memory_threshold_percent: 70
  working_set_cache:
    global_memory_limit_mb: 128
  extraction:
    max_time_seconds: 4.0
```

#### Low-Power Devices (2GB RAM)
```yaml
environment_iphone:
  system:
    memory_threshold_percent: 60
  working_set_cache:
    global_memory_limit_mb: 64
  hybrid_retriever:
    budget:
      max_chunks: 6
```

### Production Deployment

1. **Set appropriate thresholds**:
   ```yaml
   system:
     memory_threshold_percent: 80
     cpu_threshold_percent: 80
   ```

2. **Enable persistence**:
   ```yaml
   working_set_cache:
     persist_dir: "/var/lib/assistant/cache"
   ```

3. **Configure server environment**:
   ```bash
   export OLLAMA_NUM_PARALLEL=1
   export OLLAMA_MAX_LOADED_MODELS=1
   ```

4. **Monitor performance**:
   ```bash
   # Set up monitoring
   python scripts/system_health_check.py --continuous
   ```

## API Reference

### Orchestrated Model Manager

```python
from services.orchestrated_model_manager import OrchestratedModelManager

# Create orchestrated manager
manager = OrchestratedModelManager(config)
await manager.initialize_models(model_config)

# Stream conversation with orchestration
async for token in manager.query_conversation_model(
    messages=messages,
    session_id="session_123"
):
    print(token, end="")

# Extract knowledge with bounds
triples = await manager.extract_knowledge_bounded(
    text="Some text to extract from"
)

# Get comprehensive status
status = await manager.get_model_status()
```

### Working Set Cache

```python
from services.working_set_cache import WorkingSetCache

cache = WorkingSetCache(config)
await cache.initialize()

# Get working set for session
nodes = await cache.get_working_set("session_123")

# Update with new nodes
await cache.update_working_set("session_123", ["node1", "node2"])

# Get statistics
stats = cache.get_global_stats()
```

### Hybrid Retriever

```python
from services.hybrid_retriever import HybridEnsembleRetriever

retriever = HybridEnsembleRetriever(config, working_set_cache)
await retriever.initialize(graph_index, document_nodes)

# Retrieve with budget
chunks = await retriever.retrieve(
    query="What is machine learning?",
    session_id="session_123",
    budget=12
)
```

## Contributing

When contributing to edge optimization features:

1. **Test on target devices**: Validate on Raspberry Pi, mobile devices
2. **Measure performance**: Use built-in metrics and profiling
3. **Respect budgets**: All operations must have bounded resource usage
4. **Maintain compatibility**: Ensure backward compatibility with existing APIs
5. **Document configuration**: Add configuration options to this guide

## Support

For issues with edge optimization:

1. Check the troubleshooting section above
2. Run diagnostic tests: `python scripts/test_edge_optimization.py`
3. Review logs for orchestration-specific messages
4. Test with orchestration disabled to isolate issues
