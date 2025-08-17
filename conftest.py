# conftest.py (repo root)
# Legacy import compatibility for tests, cli, and scripts
import importlib
import sys

# Create aliases for legacy import paths
sys.modules["services"] = importlib.import_module("improved_local_assistant.services")
sys.modules["app"] = importlib.import_module("improved_local_assistant.app")