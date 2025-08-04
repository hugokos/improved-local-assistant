#!/usr/bin/env python
"""
Cleanup script for LlamaIndex installation issues.
This script helps diagnose and fix LlamaIndex installation problems.
"""

import sys
import site
import importlib
import subprocess
import os
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print_header("LlamaIndex Installation Diagnostic")
    
    # Check if llama_index is installed
    try:
        import llama_index
        print(f"LlamaIndex version: {llama_index.__version__}")
        print(f"Loaded from: {llama_index.__file__}")
    except ImportError:
        print("LlamaIndex is not installed")
    except Exception as e:
        print(f"Error importing llama_index: {e}")
    
    # Check for duplicate installations
    print_header("Checking for duplicate installations")
    try:
        result = subprocess.run("pip list | grep llama-index", shell=True, capture_output=True, text=True)
        print(result.stdout)
    except Exception:
        print("Could not check pip list")
    
    # Clean up LlamaIndex installations
    print_header("Cleaning up LlamaIndex installations")
    
    # Uninstall all llama-index related packages
    run_command("pip uninstall -y llama-index llama_index llama-index-core llama_index-core")
    run_command("pip uninstall -y llama-index-llms-ollama llama-index-embeddings-ollama")
    
    # Clean up site-packages
    site_packages = site.getsitepackages()[0]
    print(f"Cleaning up site-packages directory: {site_packages}")
    
    for pattern in ["llama_index*", "llama-index*"]:
        for path in Path(site_packages).glob(pattern):
            if path.is_dir():
                print(f"Removing directory: {path}")
                try:
                    import shutil
                    shutil.rmtree(path)
                except Exception as e:
                    print(f"Error removing {path}: {e}")
    
    # Clean up __pycache__ files
    print_header("Cleaning up __pycache__ files")
    for path in Path(".").rglob("*.pyc"):
        try:
            path.unlink()
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")
    
    # Reinstall LlamaIndex
    print_header("Reinstalling LlamaIndex")
    run_command("pip install llama-index[llms-ollama]==0.9.44 llama-index-core==0.9.44 llama-index-llms-ollama==0.1.2 llama-index-embeddings-ollama==0.1.2")
    
    # Verify installation
    print_header("Verifying installation")
    try:
        run_command("python -c \"from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader, StorageContext, Settings; from llama_index.core.graph_stores import SimpleGraphStore; from llama_index.llms.ollama import Ollama; from llama_index.query_engine.ensemble import EnsembleQueryEngine; print('Import successful')\"")
    except Exception as e:
        print(f"Verification failed: {e}")
    
    print_header("Cleanup complete")
    print("Please try running your script again.")

if __name__ == "__main__":
    main()