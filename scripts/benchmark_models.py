#!/usr/bin/env python3
"""
Local Model Benchmarking Script

Measures Time-to-First-Token (TTFT), Time-to-Last-Token (TTLT), and throughput
for different context sizes to help compare model performance.
"""

import time
import json
import statistics
import argparse
from typing import List, Tuple, Dict, Any
import sys
import os
import platform
import psutil
import subprocess
import asyncio

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.model_manager import ModelManager
from app.core.config import load_config


class ModelBenchmark:
    def __init__(self, model_name: str = "hermes3:3b"):
        self.config = load_config()
        # ModelManager takes host parameter, not config
        host = self.config.get("ollama", {}).get("host", "http://localhost:11434")
        self.model_manager = ModelManager(host)
        self.model_name = model_name
        self.hardware_info = self.collect_hardware_info()
    
    def collect_hardware_info(self) -> Dict[str, Any]:
        """Collect comprehensive hardware information for benchmark context."""
        hardware = {
            "timestamp": time.time(),
            "system": {},
            "cpu": {},
            "memory": {},
            "gpu": {},
            "storage": {},
            "python": {}
        }
        
        try:
            # System information
            hardware["system"] = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "hostname": platform.node()
            }
            
            # CPU information
            hardware["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
                "cpu_percent": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            hardware["memory"] = {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent_used": memory.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_gb": round(swap.used / (1024**3), 2),
                "swap_percent": swap.percent
            }
            
            # GPU information (try multiple methods)
            hardware["gpu"] = self.detect_gpu_info()
            
            # Storage information for the current drive
            disk = psutil.disk_usage('/')
            hardware["storage"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 1)
            }
            
            # Python environment
            hardware["python"] = {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler()
            }
            
        except Exception as e:
            print(f"Warning: Could not collect complete hardware info: {e}")
            hardware["error"] = str(e)
        
        return hardware
    
    def detect_gpu_info(self) -> Dict[str, Any]:
        """Detect GPU information using multiple methods."""
        gpu_info = {
            "detected": False,
            "gpus": [],
            "cuda_available": False,
            "metal_available": False
        }
        
        try:
            # Try nvidia-smi for NVIDIA GPUs
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_info["gpus"].append({
                                "name": parts[0],
                                "memory_total_mb": int(parts[1]) if parts[1].isdigit() else None,
                                "memory_free_mb": int(parts[2]) if parts[2].isdigit() else None,
                                "utilization_percent": int(parts[3]) if parts[3].isdigit() else None,
                                "type": "NVIDIA"
                            })
                gpu_info["detected"] = len(gpu_info["gpus"]) > 0
                gpu_info["cuda_available"] = True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check for Metal (macOS)
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and "Metal" in result.stdout:
                    gpu_info["metal_available"] = True
                    # Parse GPU info from system_profiler output
                    lines = result.stdout.split('\n')
                    current_gpu = {}
                    for line in lines:
                        line = line.strip()
                        if "Chipset Model:" in line:
                            current_gpu["name"] = line.split(":", 1)[1].strip()
                            current_gpu["type"] = "Metal"
                        elif "VRAM (Total):" in line:
                            vram_str = line.split(":", 1)[1].strip()
                            # Extract number from strings like "8 GB" or "8192 MB"
                            if "GB" in vram_str:
                                current_gpu["memory_total_mb"] = int(float(vram_str.split()[0]) * 1024)
                            elif "MB" in vram_str:
                                current_gpu["memory_total_mb"] = int(vram_str.split()[0])
                        elif current_gpu and ("Resolution:" in line or "Displays:" in line):
                            # End of current GPU section
                            if current_gpu.get("name"):
                                gpu_info["gpus"].append(current_gpu)
                                gpu_info["detected"] = True
                            current_gpu = {}
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        # Try to detect integrated graphics on Windows
        if platform.system() == "Windows":
            try:
                result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                name = ' '.join(parts[1:])
                                ram_bytes = parts[0] if parts[0].isdigit() else None
                                if name and name != "Name":
                                    gpu_info["gpus"].append({
                                        "name": name,
                                        "memory_total_mb": int(ram_bytes) // (1024*1024) if ram_bytes else None,
                                        "type": "Windows"
                                    })
                                    gpu_info["detected"] = True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        return gpu_info
    
    def print_hardware_summary(self):
        """Print a concise hardware summary for the benchmark."""
        hw = self.hardware_info
        
        print("HARDWARE SUMMARY:")
        
        # System
        system = hw.get("system", {})
        print(f"  System: {system.get('system', 'Unknown')} {system.get('release', '')} ({system.get('machine', '')})")
        
        # CPU
        cpu = hw.get("cpu", {})
        cores_info = f"{cpu.get('physical_cores', '?')}C/{cpu.get('logical_cores', '?')}T"
        freq_info = f" @ {cpu.get('max_frequency', '?'):.0f}MHz" if cpu.get('max_frequency') else ""
        print(f"  CPU: {cores_info}{freq_info}")
        
        # Memory
        memory = hw.get("memory", {})
        total_gb = memory.get('total_gb', 0)
        available_gb = memory.get('available_gb', 0)
        used_percent = memory.get('percent_used', 0)
        print(f"  Memory: {total_gb} GB total, {available_gb} GB available ({used_percent:.1f}% used)")
        
        # GPU
        gpu = hw.get("gpu", {})
        if gpu.get("detected") and gpu.get("gpus"):
            gpu_list = []
            for g in gpu["gpus"]:
                name = g.get("name", "Unknown")
                vram = f" ({g['memory_total_mb']} MB)" if g.get("memory_total_mb") else ""
                gpu_list.append(f"{name}{vram}")
            print(f"  GPU: {', '.join(gpu_list)}")
            
            # Acceleration info
            accel = []
            if gpu.get("cuda_available"):
                accel.append("CUDA")
            if gpu.get("metal_available"):
                accel.append("Metal")
            if accel:
                print(f"  Acceleration: {', '.join(accel)}")
        else:
            print(f"  GPU: None detected (CPU-only)")
        
        print()
        
    def generate_test_prompt(self, token_count: int) -> str:
        """Generate a test prompt of approximately the specified token count."""
        # Rough estimate: ~4 characters per token for English text
        words_needed = token_count // 4 * 16  # Adjust for word boundaries
        base_text = "This is a test prompt for benchmarking model performance. "
        repetitions = max(1, words_needed // len(base_text.split()))
        
        prompt = (base_text * repetitions)[:token_count * 4]  # Rough token limit
        return f"Please analyze the following text and provide insights:\n\n{prompt}\n\nProvide a detailed analysis:"
    
    async def benchmark_single_run(self, prompt: str) -> Tuple[float, float, int]:
        """
        Run a single benchmark iteration.
        
        Returns:
            Tuple of (ttft, ttlt, token_count)
        """
        start_time = time.time()
        first_token_time = None
        token_count = 0
        
        try:
            # Create messages for the model manager
            messages = [{"role": "user", "content": prompt}]
            
            # Use the model manager's streaming interface
            response_stream = self.model_manager.query_conversation_model(
                messages=messages,
                temperature=0.7,
                max_tokens=500  # Limit for consistent comparison
            )
            
            async for chunk in response_stream:
                if first_token_time is None:
                    first_token_time = time.time()
                
                # Count tokens (rough approximation)
                if isinstance(chunk, str):
                    token_count += len(chunk.split())
            
            end_time = time.time()
            
            ttft = first_token_time - start_time if first_token_time else 0
            ttlt = end_time - start_time
            
            return ttft, ttlt, token_count
            
        except Exception as e:
            print(f"Error during benchmark run: {e}")
            return 0, 0, 0
    
    async def benchmark_context_size(self, context_tokens: int, num_runs: int = 3) -> Dict[str, float]:
        """
        Benchmark a specific context size multiple times.
        
        Args:
            context_tokens: Approximate number of tokens in the input context
            num_runs: Number of runs to average over
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking {context_tokens} token context ({num_runs} runs)...")
        
        prompt = self.generate_test_prompt(context_tokens)
        
        ttfts = []
        ttlts = []
        token_counts = []
        
        # Warm-up run (not counted)
        print("  Warm-up run...")
        await self.benchmark_single_run(prompt)
        
        # Actual benchmark runs
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")
            ttft, ttlt, tokens = await self.benchmark_single_run(prompt)
            
            if ttft > 0 and ttlt > 0 and tokens > 0:
                ttfts.append(ttft)
                ttlts.append(ttlt)
                token_counts.append(tokens)
        
        if not ttfts:
            return {
                "context_tokens": context_tokens,
                "avg_ttft": 0,
                "avg_ttlt": 0,
                "avg_throughput": 0,
                "runs": 0
            }
        
        # Calculate averages
        avg_ttft = statistics.mean(ttfts)
        avg_ttlt = statistics.mean(ttlts)
        avg_tokens = statistics.mean(token_counts)
        
        # Throughput = tokens per second (excluding TTFT)
        avg_throughput = avg_tokens / (avg_ttlt - avg_ttft) if (avg_ttlt - avg_ttft) > 0 else 0
        
        return {
            "context_tokens": context_tokens,
            "avg_ttft": avg_ttft,
            "avg_ttlt": avg_ttlt,
            "avg_tokens": avg_tokens,
            "avg_throughput": avg_throughput,
            "ttft_std": statistics.stdev(ttfts) if len(ttfts) > 1 else 0,
            "ttlt_std": statistics.stdev(ttlts) if len(ttlts) > 1 else 0,
            "runs": len(ttfts)
        }
    
    async def run_full_benchmark(self, context_sizes: List[int] = None, num_runs: int = 3) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Args:
            context_sizes: List of context sizes to test (in tokens)
            num_runs: Number of runs per context size
            
        Returns:
            Complete benchmark results
        """
        if context_sizes is None:
            context_sizes = [512, 1024, 2048, 4096]
        
        print(f"Starting benchmark for model: {self.model_name}")
        print(f"Context sizes: {context_sizes}")
        print(f"Runs per size: {num_runs}")
        print()
        
        # Display hardware summary
        self.print_hardware_summary()
        print("-" * 50)
        
        # Initialize the model manager
        from services.model_manager import ModelConfig
        config = ModelConfig(
            name=self.model_name,
            type="conversation",
            temperature=0.7,
            max_tokens=500
        )
        
        # Initialize models
        init_success = await self.model_manager.initialize_models(config)
        if not init_success:
            print(f"Failed to initialize model: {self.model_name}")
            return {
                "model": self.model_name,
                "timestamp": time.time(),
                "hardware": self.hardware_info,
                "context_sizes": [],
                "summary": {},
                "error": "Model initialization failed"
            }
        
        # Set the model for the conversation client
        if self.model_name != "hermes3:3b":
            self.model_manager.conversation_model = self.model_name
        
        results = {
            "model": self.model_name,
            "timestamp": time.time(),
            "hardware": self.hardware_info,
            "context_sizes": [],
            "summary": {}
        }
        
        for size in context_sizes:
            result = await self.benchmark_context_size(size, num_runs)
            results["context_sizes"].append(result)
            
            # Print immediate results
            print(f"Results for {size} tokens:")
            print(f"  TTFT: {result['avg_ttft']:.3f}s (±{result['ttft_std']:.3f})")
            print(f"  TTLT: {result['avg_ttlt']:.3f}s (±{result['ttlt_std']:.3f})")
            print(f"  Throughput: {result['avg_throughput']:.1f} tokens/sec")
            print(f"  Avg output tokens: {result['avg_tokens']:.1f}")
            print()
        
        # Calculate summary statistics
        valid_results = [r for r in results["context_sizes"] if r["runs"] > 0]
        if valid_results:
            results["summary"] = {
                "avg_ttft": statistics.mean([r["avg_ttft"] for r in valid_results]),
                "avg_throughput": statistics.mean([r["avg_throughput"] for r in valid_results]),
                "max_throughput": max([r["avg_throughput"] for r in valid_results]),
                "min_ttft": min([r["avg_ttft"] for r in valid_results])
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"benchmark_{self.model_name.replace(':', '_')}_{timestamp}.json"
        
        filepath = os.path.join(project_root, "benchmarks", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath


async def main():
    parser = argparse.ArgumentParser(description="Benchmark local AI models")
    parser.add_argument("--model", default="hermes3:3b", help="Model name to benchmark")
    parser.add_argument("--contexts", nargs="+", type=int, default=[512, 1024, 2048, 4096],
                       help="Context sizes to test (in tokens)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per context size")
    parser.add_argument("--output", help="Output filename for results")
    
    args = parser.parse_args()
    
    try:
        benchmark = ModelBenchmark(args.model)
        results = await benchmark.run_full_benchmark(args.contexts, args.runs)
        
        # Print summary
        print("=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        if results.get("summary"):
            print(f"Model: {results['model']}")
            print(f"Average TTFT: {results['summary']['avg_ttft']:.3f}s")
            print(f"Average Throughput: {results['summary']['avg_throughput']:.1f} tokens/sec")
            print(f"Max Throughput: {results['summary']['max_throughput']:.1f} tokens/sec")
            print(f"Min TTFT: {results['summary']['min_ttft']:.3f}s")
        elif results.get("error"):
            print(f"Benchmark failed: {results['error']}")
        
        # Save results
        benchmark.save_results(results, args.output)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())