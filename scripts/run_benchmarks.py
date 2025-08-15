#!/usr/bin/env python3
"""
Quick benchmark runner for common models in the improved-local-assistant.
"""

import os
import subprocess
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_benchmark(model_name: str, contexts: list = None, runs: int = 3):
    """Run benchmark for a specific model."""
    if contexts is None:
        contexts = [512, 1024, 2048, 4096]

    print(f"Starting benchmark for {model_name}...")

    cmd = (
        [
            sys.executable,
            os.path.join(project_root, "scripts", "benchmark_models.py"),
            "--model",
            model_name,
            "--contexts",
        ]
        + [str(c) for c in contexts]
        + ["--runs", str(runs)]
    )

    try:
        # Run with output visible for debugging
        result = subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed for {model_name}: {e}")
        return False


def main():
    print("Improved Local Assistant - Model Benchmarking Suite")
    print("=" * 60)

    # Common models to benchmark
    models = ["hermes3:3b", "tinyllama", "phi3:mini"]

    # Quick benchmark settings
    contexts = [512, 1024, 2048]  # Smaller set for quick testing
    runs = 3

    print(f"Models to benchmark: {', '.join(models)}")
    print(f"Context sizes: {contexts}")
    print(f"Runs per context: {runs}")
    print()

    successful_benchmarks = []

    for model in models:
        print(f"\n{'='*20} {model} {'='*20}")
        if run_benchmark(model, contexts, runs):
            successful_benchmarks.append(model)
            print(f"✓ Benchmark completed for {model}")
        else:
            print(f"✗ Benchmark failed for {model}")

        # Small delay between models
        time.sleep(2)

    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {len(successful_benchmarks)}/{len(models)}")

    if successful_benchmarks:
        print(f"Completed models: {', '.join(successful_benchmarks)}")

        # Run comparison
        print("\nRunning comparison...")
        benchmarks_dir = os.path.join(project_root, "benchmarks")
        if os.path.exists(benchmarks_dir):
            compare_cmd = [
                sys.executable,
                os.path.join(project_root, "scripts", "compare_benchmarks.py"),
                "--dir",
                benchmarks_dir,
            ]

            try:
                subprocess.run(compare_cmd, check=True)
            except subprocess.CalledProcessError:
                print("Comparison failed, but benchmark files should be available.")

    print(f"\nBenchmark files saved in: {os.path.join(project_root, 'benchmarks')}")


if __name__ == "__main__":
    main()
