#!/usr/bin/env python3
"""
Benchmark Comparison Tool

Compares benchmark results from multiple models or runs.
"""

import argparse
import glob
import json
import os
from typing import Any
from typing import Dict
from typing import List


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def compare_models(result_files: List[str]):
    """Compare benchmark results from multiple models."""
    results = []

    for filepath in result_files:
        try:
            result = load_benchmark_results(filepath)
            results.append(result)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if not results:
        print("No valid benchmark results found.")
        return

    # Display hardware information first
    print("HARDWARE INFORMATION")
    print("=" * 80)

    for i, result in enumerate(results):
        model = result.get("model", "Unknown")
        hardware = result.get("hardware", {})

        print(f"\n{model}:")
        if hardware:
            # System info
            system = hardware.get("system", {})
            cpu = hardware.get("cpu", {})
            memory = hardware.get("memory", {})
            gpu = hardware.get("gpu", {})

            print(f"  System: {system.get('system', 'Unknown')} {system.get('release', '')}")
            print(
                f"  CPU: {cpu.get('physical_cores', '?')}C/{cpu.get('logical_cores', '?')}T @ {cpu.get('max_frequency', '?')}MHz"
            )
            print(
                f"  Memory: {memory.get('total_gb', '?')} GB total, {memory.get('available_gb', '?')} GB available"
            )

            if gpu.get("detected") and gpu.get("gpus"):
                gpu_names = [g.get("name", "Unknown") for g in gpu["gpus"]]
                print(f"  GPU: {', '.join(gpu_names)}")
                for g in gpu["gpus"]:
                    if g.get("memory_total_mb"):
                        print(f"    - {g['name']}: {g['memory_total_mb']} MB VRAM")
            else:
                print("  GPU: None detected")
        else:
            print("  Hardware info not available")

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Header
    print(
        f"{'Model':<20} {'Avg TTFT':<12} {'Avg Throughput':<15} {'Max Throughput':<15} {'Min TTFT':<12}"
    )
    print("-" * 80)

    # Sort by average throughput (descending)
    results.sort(key=lambda x: x.get("summary", {}).get("avg_throughput", 0), reverse=True)

    for result in results:
        model = result.get("model", "Unknown")
        summary = result.get("summary", {})

        avg_ttft = summary.get("avg_ttft", 0)
        avg_throughput = summary.get("avg_throughput", 0)
        max_throughput = summary.get("max_throughput", 0)
        min_ttft = summary.get("min_ttft", 0)

        print(
            f"{model:<20} {avg_ttft:<12.3f} {avg_throughput:<15.1f} {max_throughput:<15.1f} {min_ttft:<12.3f}"
        )

    print("\nDETAILED BREAKDOWN BY CONTEXT SIZE")
    print("=" * 80)

    # Get all context sizes
    all_contexts = set()
    for result in results:
        for ctx_result in result.get("context_sizes", []):
            all_contexts.add(ctx_result["context_tokens"])

    all_contexts = sorted(all_contexts)

    for context in all_contexts:
        print(f"\nContext Size: {context} tokens")
        print(f"{'Model':<20} {'TTFT (s)':<12} {'TTLT (s)':<12} {'Throughput':<15} {'Tokens':<10}")
        print("-" * 70)

        context_results = []
        for result in results:
            model = result.get("model", "Unknown")

            # Find matching context size
            ctx_data = None
            for ctx_result in result.get("context_sizes", []):
                if ctx_result["context_tokens"] == context:
                    ctx_data = ctx_result
                    break

            if ctx_data and ctx_data["runs"] > 0:
                context_results.append((model, ctx_data))

        # Sort by throughput for this context
        context_results.sort(key=lambda x: x[1]["avg_throughput"], reverse=True)

        for model, ctx_data in context_results:
            ttft = ctx_data["avg_ttft"]
            ttlt = ctx_data["avg_ttlt"]
            throughput = ctx_data["avg_throughput"]
            tokens = ctx_data["avg_tokens"]

            print(f"{model:<20} {ttft:<12.3f} {ttlt:<12.3f} {throughput:<15.1f} {tokens:<10.1f}")


def analyze_scaling(result_file: str):
    """Analyze how a single model scales with context size."""
    try:
        result = load_benchmark_results(result_file)
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return

    model = result.get("model", "Unknown")
    context_results = result.get("context_sizes", [])

    if not context_results:
        print("No context size data found.")
        return

    print(f"SCALING ANALYSIS: {model}")
    print("=" * 60)

    # Sort by context size
    context_results.sort(key=lambda x: x["context_tokens"])

    print(f"{'Context':<10} {'TTFT':<10} {'TTLT':<10} {'Throughput':<12} {'Efficiency':<12}")
    print("-" * 60)

    baseline_throughput = None

    for ctx_data in context_results:
        if ctx_data["runs"] == 0:
            continue

        context = ctx_data["context_tokens"]
        ttft = ctx_data["avg_ttft"]
        ttlt = ctx_data["avg_ttlt"]
        throughput = ctx_data["avg_throughput"]

        if baseline_throughput is None:
            baseline_throughput = throughput
            efficiency = 100.0
        else:
            efficiency = (throughput / baseline_throughput) * 100

        print(f"{context:<10} {ttft:<10.3f} {ttlt:<10.3f} {throughput:<12.1f} {efficiency:<12.1f}%")

    # Calculate scaling trends
    contexts = [r["context_tokens"] for r in context_results if r["runs"] > 0]
    ttfts = [r["avg_ttft"] for r in context_results if r["runs"] > 0]
    throughputs = [r["avg_throughput"] for r in context_results if r["runs"] > 0]

    if len(contexts) > 1:
        print("\nSCALING TRENDS:")
        print(f"Context range: {min(contexts)} - {max(contexts)} tokens")
        print(f"TTFT range: {min(ttfts):.3f} - {max(ttfts):.3f}s")
        print(f"Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} tokens/sec")

        # Simple linear correlation
        if max(throughputs) > 0:
            throughput_degradation = (1 - min(throughputs) / max(throughputs)) * 100
            print(f"Throughput degradation: {throughput_degradation:.1f}%")


def show_hardware_only(result_files: List[str]):
    """Show only hardware information from benchmark files."""
    results = []

    for filepath in result_files:
        try:
            result = load_benchmark_results(filepath)
            results.append((os.path.basename(filepath), result))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if not results:
        print("No valid benchmark results found.")
        return

    print("HARDWARE COMPARISON")
    print("=" * 100)

    for filename, result in results:
        model = result.get("model", "Unknown")
        hardware = result.get("hardware", {})

        print(f"\n{filename} ({model}):")
        if hardware:
            system = hardware.get("system", {})
            cpu = hardware.get("cpu", {})
            memory = hardware.get("memory", {})
            gpu = hardware.get("gpu", {})

            print(f"  Platform: {system.get('platform', 'Unknown')}")
            print(f"  CPU: {cpu.get('physical_cores', '?')}C/{cpu.get('logical_cores', '?')}T")
            if cpu.get("max_frequency"):
                print(f"       Max Frequency: {cpu['max_frequency']:.0f} MHz")
            print(f"  Memory: {memory.get('total_gb', '?')} GB total")

            if gpu.get("detected") and gpu.get("gpus"):
                print("  GPU(s):")
                for g in gpu["gpus"]:
                    name = g.get("name", "Unknown")
                    vram = f" ({g['memory_total_mb']} MB VRAM)" if g.get("memory_total_mb") else ""
                    gpu_type = f" [{g.get('type', 'Unknown')}]" if g.get("type") else ""
                    print(f"    - {name}{vram}{gpu_type}")

                accel = []
                if gpu.get("cuda_available"):
                    accel.append("CUDA")
                if gpu.get("metal_available"):
                    accel.append("Metal")
                if accel:
                    print(f"  Acceleration: {', '.join(accel)}")
            else:
                print("  GPU: None detected")
        else:
            print("  Hardware info not available")


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--compare", nargs="+", help="Benchmark files to compare")
    parser.add_argument("--analyze", help="Single benchmark file to analyze scaling")
    parser.add_argument("--dir", help="Directory containing benchmark files")
    parser.add_argument("--hardware", action="store_true", help="Show only hardware information")

    args = parser.parse_args()

    if args.analyze:
        analyze_scaling(args.analyze)
    elif args.compare:
        if args.hardware:
            show_hardware_only(args.compare)
        else:
            compare_models(args.compare)
    elif args.dir:
        # Find all benchmark files in directory
        pattern = os.path.join(args.dir, "benchmark_*.json")
        files = glob.glob(pattern)
        if files:
            print(f"Found {len(files)} benchmark files:")
            for f in files:
                print(f"  {os.path.basename(f)}")
            print()
            if args.hardware:
                show_hardware_only(files)
            else:
                compare_models(files)
        else:
            print(f"No benchmark files found in {args.dir}")
    else:
        print("Please specify --compare, --analyze, or --dir option")


if __name__ == "__main__":
    main()
