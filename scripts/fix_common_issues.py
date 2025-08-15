#!/usr/bin/env python3
"""
Quick Fix Script for Common Issues

This script attempts to automatically fix common issues identified in the system.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def fix_missing_packages():
    """Install missing required packages."""
    logger.info("🔧 Checking and installing missing packages...")

    try:
        # Install the corrected requirements
        requirements_file = Path(__file__).parent.parent / "requirements.txt"

        if requirements_file.exists():
            logger.info("📦 Installing packages from requirements.txt...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ Successfully installed packages")
                return True
            else:
                logger.error(f"❌ Failed to install packages: {result.stderr}")
                return False
        else:
            logger.error("❌ requirements.txt not found")
            return False

    except Exception as e:
        logger.error(f"❌ Error installing packages: {e}")
        return False


def create_missing_directories():
    """Create missing data directories."""
    logger.info("🔧 Creating missing directories...")

    base_dir = Path(__file__).parent.parent
    required_dirs = [
        "data",
        "data/prebuilt_graphs",
        "data/dynamic_graph",
        "logs",
    ]

    created_dirs = []

    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        if not full_path.exists():
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {dir_path}")
                created_dirs.append(dir_path)
            except Exception as e:
                logger.error(f"❌ Failed to create directory {dir_path}: {e}")
                return False

    if created_dirs:
        logger.info(f"📁 Created {len(created_dirs)} directories")
    else:
        logger.info("ℹ️  All required directories already exist")

    return True


def set_environment_variables():
    """Set recommended environment variables for better performance."""
    logger.info("🔧 Setting environment variables for better performance...")

    env_vars = {
        "SKIP_ALL_GRAPHS": "0",  # Enable graphs by default
        "SKIP_SURVIVALIST_GRAPH": "0",  # Enable survivalist graph
        "CUDA_VISIBLE_DEVICES": "0",  # Use first GPU only
        "TOKENIZERS_PARALLELISM": "false",  # Avoid tokenizer warnings
    }

    env_file = Path(__file__).parent.parent / ".env"

    try:
        # Read existing .env file if it exists
        existing_vars = {}
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        existing_vars[key.strip()] = value.strip()

        # Add new variables that don't exist
        new_vars = {}
        for key, value in env_vars.items():
            if key not in existing_vars:
                new_vars[key] = value

        if new_vars:
            with open(env_file, "a") as f:
                f.write("\n# Auto-generated environment variables for performance\n")
                for key, value in new_vars.items():
                    f.write(f"{key}={value}\n")
                    logger.info(f"✅ Added environment variable: {key}={value}")

            logger.info(f"📝 Added {len(new_vars)} environment variables to .env")
        else:
            logger.info("ℹ️  All recommended environment variables already set")

        return True

    except Exception as e:
        logger.error(f"❌ Error setting environment variables: {e}")
        return False


def optimize_ollama_settings():
    """Provide recommendations for Ollama optimization."""
    logger.info("🔧 Checking Ollama optimization...")

    recommendations = [
        "💡 For better GPU memory management, consider setting OLLAMA_GPU_MEMORY_FRACTION=0.8",
        "💡 To reduce memory usage, try smaller models like 'llama3.2:1b' instead of 'hermes3:3b'",
        "💡 If experiencing CUDA errors, try CPU-only mode with OLLAMA_NUM_GPU=0",
        "💡 For Windows users, ensure Ollama service is running: 'ollama serve'",
    ]

    for rec in recommendations:
        logger.info(rec)

    return True


def create_sample_config():
    """Create a sample configuration file with optimized settings."""
    logger.info("🔧 Creating optimized configuration...")

    config_file = Path(__file__).parent.parent / "config_optimized.yaml"

    if config_file.exists():
        logger.info("ℹ️  Optimized config already exists")
        return True

    optimized_config = """# Optimized Configuration for Improved Local Assistant
# This configuration is designed to work better with limited resources

# Model configuration with smaller, more efficient models
models:
  conversation:
    name: "llama3.2:1b"  # Smaller model for better memory usage
    context_window: 4096  # Reduced context window
    temperature: 0.7
  knowledge:
    name: "tinyllama"
    context_window: 2048
    temperature: 0.3

# Ollama configuration with conservative timeouts
ollama:
  base_url: "http://localhost:11434"
  timeout: 60  # Reduced timeout
  max_retries: 2

# Knowledge graph configuration with reduced complexity
knowledge_graphs:
  prebuilt_directory: "./data/prebuilt_graphs"
  dynamic_storage: "./data/dynamic_graph"
  max_triplets_per_chunk: 2  # Reduced for better performance
  enable_visualization: true
  max_nodes_per_query: 50  # Limit query complexity

# System configuration for better resource management
system:
  max_parallel_models: 1  # Reduce parallel model loading
  max_loaded_models: 1
  memory_limit_mb: 4096  # 4GB memory limit
  enable_gpu: true
  gpu_memory_fraction: 0.7  # Reserve some GPU memory

# Logging configuration
logging:
  level: "INFO"
  file: "./logs/app.log"
  max_size_mb: 100
  backup_count: 3

# Web interface configuration
web:
  host: "127.0.0.1"
  port: 8000
  reload: false  # Disable auto-reload for production
"""

    try:
        with open(config_file, "w") as f:
            f.write(optimized_config)

        logger.info(f"✅ Created optimized configuration: {config_file.name}")
        logger.info("💡 Use this config with: python run_app.py --config config_optimized.yaml")
        return True

    except Exception as e:
        logger.error(f"❌ Error creating optimized config: {e}")
        return False


def main():
    """Run all fixes."""
    logger.info("🔧 Starting Automatic Fix for Common Issues")
    logger.info("=" * 60)

    fixes = [
        ("Missing Packages", fix_missing_packages),
        ("Missing Directories", create_missing_directories),
        ("Environment Variables", set_environment_variables),
        ("Ollama Optimization", optimize_ollama_settings),
        ("Sample Config", create_sample_config),
    ]

    results = {}

    for fix_name, fix_func in fixes:
        logger.info(f"\n🔧 Applying {fix_name} fix...")
        try:
            results[fix_name] = fix_func()
        except Exception as e:
            logger.error(f"❌ Error during {fix_name} fix: {e}")
            results[fix_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 FIX SUMMARY")
    logger.info("=" * 60)

    applied = sum(1 for result in results.values() if result)
    total = len(results)

    for fix_name, result in results.items():
        status = "✅ APPLIED" if result else "❌ FAILED"
        logger.info(f"{status} {fix_name}")

    logger.info(f"\n🎯 Overall: {applied}/{total} fixes applied successfully")

    if applied == total:
        logger.info("🎉 All fixes applied successfully!")
        logger.info("💡 Restart the application to ensure all changes take effect.")
    else:
        logger.info("⚠️  Some fixes failed. Check the logs above for details.")

    return applied == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
