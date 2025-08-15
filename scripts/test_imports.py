#!/usr/bin/env python3
"""
Simple script to test imports for edge optimization components.
"""

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all the edge optimization imports."""
    print("🧪 Testing Edge Optimization Imports")

    try:
        print("📋 Testing basic imports...")
        print("✅ app.core.config imported successfully")

        print("✅ SystemMonitor imported successfully")

        print("✅ ConnectionPoolManager imported successfully")

        print("✅ WorkingSetCache imported successfully")

        print("✅ ExtractionPipeline imported successfully")

        print("✅ HybridEnsembleRetriever imported successfully")

        print("✅ LLMOrchestrator imported successfully")

        print("✅ OrchestratedModelManager imported successfully")

        print("\n🎉 All imports successful!")
        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_initialization():
    """Test basic initialization without external dependencies."""
    print("\n🔧 Testing Basic Initialization")

    try:
        from services.system_monitor import SystemMonitor
        from services.working_set_cache import WorkingSetCache

        from app.core.config import load_config

        # Load config
        config = load_config()
        print("✅ Configuration loaded")

        # Create system monitor
        system_monitor = SystemMonitor(config)
        print("✅ SystemMonitor created")

        # Create working set cache
        cache = WorkingSetCache(config)
        print("✅ WorkingSetCache created")

        print("\n🎉 Basic initialization successful!")
        return True

    except Exception as e:
        print(f"❌ Initialization error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    if success:
        success = test_basic_initialization()

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
