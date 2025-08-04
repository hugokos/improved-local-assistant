# Model Benchmarking

This directory contains tools and results for benchmarking local AI models in the improved-local-assistant.

## Quick Start

### Model Performance Benchmarks
Run benchmarks for all common models:
```bash
python scripts/run_benchmarks.py
```

### GraphRAG Pipeline Benchmarks
Test the complete user experience including knowledge retrieval:
```bash
# Quick user-friendly test
python scripts/quick_graphrag_benchmark.py

# Comprehensive GraphRAG pipeline benchmark
python scripts/benchmark_graphrag_pipeline.py --runs 5
```

## Individual Benchmarking

### Model-Only Benchmarks
Benchmark a specific model:
```bash
python scripts/benchmark_models.py --model hermes3:3b --contexts 512 1024 2048 4096 --runs 5
```

### GraphRAG Pipeline Benchmarks
Test the complete pipeline with knowledge graph integration:
```bash
# Comprehensive pipeline benchmark
python scripts/benchmark_graphrag_pipeline.py --runs 3 --output my_graphrag_results.json

# Quick performance test (user-friendly)
python scripts/quick_graphrag_benchmark.py
```

## Comparing Results

Compare multiple benchmark files:
```bash
python scripts/compare_benchmarks.py --compare benchmark_hermes3_3b_*.json benchmark_tinyllama_*.json
```

Compare all benchmarks in this directory:
```bash
python scripts/compare_benchmarks.py --dir benchmarks/
```

Show only hardware information:
```bash
python scripts/compare_benchmarks.py --dir benchmarks/ --hardware
```

Analyze scaling for a single model:
```bash
python scripts/compare_benchmarks.py --analyze benchmark_hermes3_3b_1234567890.json
```

## Metrics Explained

### Model Performance Metrics
- **TTFT (Time-to-First-Token)**: How long from request submission to the first generated token
- **TTLT (Time-to-Last-Token)**: Total wall-clock time until the final token arrives
- **Throughput**: (total output tokens) ÷ (TTLT − TTFT) in tokens per second

### GraphRAG Pipeline Metrics
- **Retrieval Time**: Time spent querying the knowledge graph for relevant context
- **Prep Time**: Context assembly and prompt preparation overhead
- **Model TTFT**: Time from final prompt to first AI response token
- **Total Time**: Complete end-to-end user experience time
- **Pipeline Breakdown**: Percentage of time spent in each phase

## Benchmark Methodology

### Model Performance Benchmarks
The model benchmarking follows these principles:

1. **Consistent Context**: Uses repeated text to create predictable context sizes
2. **Warm-up Runs**: Excludes the first run to eliminate startup overhead
3. **Multiple Iterations**: Averages results across multiple runs for reliability
4. **Controlled Variables**: Same temperature, max tokens, and system conditions
5. **Real Integration**: Uses your actual model manager and inference pipeline

### GraphRAG Pipeline Benchmarks
The pipeline benchmarking measures real user experience:

1. **End-to-End Testing**: Complete pipeline from query to final response
2. **Component Isolation**: Separate timing for retrieval, prep, and generation
3. **Realistic Queries**: Uses actual AI-related questions that hit the knowledge graph
4. **Multiple Query Types**: Tests different complexity levels and topics
5. **Production Environment**: Uses the same components as the live application

## Interpreting Results

### Good Performance Indicators:
- **Low TTFT** (< 1s): Fast response initiation
- **High Throughput** (> 20 tokens/sec): Efficient token generation
- **Consistent Performance**: Low standard deviation across runs

### Context Size Impact:
- TTFT typically increases with context size (more to process)
- Throughput may decrease with larger contexts (memory pressure)
- Look for models that maintain performance across context sizes

### Model Comparison:
- Compare models at the same context sizes
- Consider both speed and quality for your use case
- Smaller models often have better TTFT, larger models better quality

## Hardware Information

Each benchmark automatically captures detailed hardware information:

### Captured Data:
- **System**: OS, version, architecture, hostname
- **CPU**: Physical/logical cores, frequency, current load
- **Memory**: Total, available, used (including swap)
- **GPU**: Detected GPUs with VRAM, CUDA/Metal support
- **Storage**: Disk space and usage
- **Python**: Version and implementation details

### Hardware Impact on Performance:
- **CPU**: Single-threaded performance matters most for inference
- **RAM**: Larger models need more memory; insufficient RAM causes swapping
- **GPU**: CUDA/Metal acceleration can dramatically improve performance
- **Storage**: SSD vs HDD affects model loading times

## Best Practices

1. **Run benchmarks when system is idle** (no competing workloads)
2. **Use consistent hardware settings** (power profiles, thermal throttling)
3. **Benchmark multiple times** to account for variability
4. **Test realistic context sizes** for your use case
5. **Consider quality vs speed tradeoffs** - fastest isn't always best

## Example Output

```
HARDWARE INFORMATION
================================================================================

hermes3:3b:
  System: Windows 11
  CPU: 8C/16T @ 3600MHz
  Memory: 32.0 GB total, 24.5 GB available
  GPU: NVIDIA GeForce RTX 4070 (12288 MB VRAM)

tinyllama:
  System: Windows 11  
  CPU: 8C/16T @ 3600MHz
  Memory: 32.0 GB total, 24.1 GB available
  GPU: NVIDIA GeForce RTX 4070 (12288 MB VRAM)

================================================================================
MODEL COMPARISON
================================================================================
Model                Avg TTFT     Avg Throughput  Max Throughput  Min TTFT    
--------------------------------------------------------------------------------
hermes3:3b           0.245        28.4            32.1            0.198       
phi3:mini            0.312        24.7            27.3            0.287       
tinyllama            0.156        45.2            48.9            0.134       

Context Size: 1024 tokens
Model                TTFT (s)     TTLT (s)     Throughput      Tokens    
----------------------------------------------------------------------
tinyllama            0.134        2.847        45.2            128.5     
hermes3:3b           0.198        4.234        28.4            119.8     
phi3:mini            0.287        5.123        24.7            126.3     
```

This shows tinyllama is fastest but hermes3:3b might provide better quality responses. The hardware context shows all tests ran on identical hardware, making the comparison valid.