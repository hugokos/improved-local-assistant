#!/usr/bin/env python3
"""
Memory optimization script to free up system resources.

This script helps optimize memory usage before running the assistant.
"""

import gc
import logging
import os
import sys

import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_memory_info():
    """Get current memory information."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_percent": memory.percent,
        "free_gb": memory.free / (1024**3),
    }


def optimize_python_memory():
    """Optimize Python memory usage."""
    logger.info("🧹 Optimizing Python memory...")

    # Force garbage collection
    collected = gc.collect()
    logger.info(f"🗑️  Collected {collected} objects")

    # Set garbage collection thresholds more aggressively
    gc.set_threshold(700, 10, 10)
    logger.info("⚙️  Set aggressive garbage collection thresholds")


def suggest_system_optimizations():
    """Suggest system-level optimizations."""
    logger.info("💡 System Optimization Suggestions:")

    memory_info = get_memory_info()

    if memory_info["available_gb"] < 2:
        logger.warning("⚠️  Very low memory available!")
        logger.info("💡 Consider:")
        logger.info("   - Closing other applications")
        logger.info(
            "   - Using the optimized config: python run_app.py --config config_optimized.yaml"
        )
        logger.info("   - Setting OLLAMA_NUM_GPU=0 for CPU-only mode")

    # Check for memory-heavy processes
    logger.info("🔍 Top memory-consuming processes:")
    processes = []
    for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Sort by memory usage and show top 5
    processes.sort(key=lambda x: x["memory_percent"], reverse=True)
    for proc in processes[:5]:
        logger.info(f"   {proc['name']}: {proc['memory_percent']:.1f}% (PID: {proc['pid']})")


def set_environment_optimizations():
    """Set environment variables for better memory management."""
    logger.info("⚙️  Setting memory optimization environment variables...")

    optimizations = {
        "PYTHONHASHSEED": "0",  # Consistent hashing
        "PYTHONUNBUFFERED": "1",  # Unbuffered output
        "OMP_NUM_THREADS": "2",  # Limit OpenMP threads
        "MKL_NUM_THREADS": "2",  # Limit MKL threads
        "NUMEXPR_NUM_THREADS": "2",  # Limit NumExpr threads
        "TOKENIZERS_PARALLELISM": "false",  # Disable tokenizer parallelism
    }

    for key, value in optimizations.items():
        os.environ[key] = value
        logger.info(f"   {key}={value}")


def main():
    """Main optimization function."""
    logger.info("🚀 Starting Memory Optimization")
    logger.info("=" * 50)

    # Show initial memory state
    initial_memory = get_memory_info()
    logger.info(
        f"💾 Initial Memory: {initial_memory['available_gb']:.1f}GB available "
        f"({initial_memory['used_percent']:.1f}% used)"
    )

    # Apply optimizations
    optimize_python_memory()
    set_environment_optimizations()
    suggest_system_optimizations()

    # Show final memory state
    final_memory = get_memory_info()
    logger.info(
        f"💾 Final Memory: {final_memory['available_gb']:.1f}GB available "
        f"({final_memory['used_percent']:.1f}% used)"
    )

    improvement = final_memory["available_gb"] - initial_memory["available_gb"]
    if improvement > 0:
        logger.info(f"✅ Freed up {improvement:.2f}GB of memory")

    logger.info("=" * 50)
    logger.info("🎯 Optimization complete!")

    # Recommendations based on available memory
    if final_memory["available_gb"] < 1:
        logger.warning("⚠️  Still very low memory. Consider restarting your system.")
        return 1
    elif final_memory["available_gb"] < 2:
        logger.warning("⚠️  Low memory. Use CPU-only mode for better stability.")
        logger.info("💡 Run with: OLLAMA_NUM_GPU=0 python run_app.py --config config_optimized.yaml")
        return 0
    else:
        logger.info("✅ Memory looks good for running the assistant!")
        return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Optimization interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during optimization: {e}")
        sys.exit(1)
