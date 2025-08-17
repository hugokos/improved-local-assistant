#!/usr/bin/env python3
"""
Import smoke test that tries to import everything under the package
and reports failures (including circular imports that Uvicorn sometimes hides).
"""

import importlib
import pkgutil
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    import improved_local_assistant as pkg
except ImportError as e:
    print(f"❌ Could not import main package: {e}")
    sys.exit(1)

def test_all_imports():
    """Test importing all modules in the package."""
    errors = []
    success_count = 0
    
    print("🔍 Scanning package modules...")
    
    # Walk through all modules in the package
    for mod_info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        name = mod_info.name
        try:
            importlib.import_module(name)
            success_count += 1
            print(f"✅ {name}")
        except Exception as e:
            errors.append((name, repr(e)))
            print(f"❌ {name} -> {e}")
    
    print(f"\n📊 Scanned {success_count + len(errors)} modules.")
    print(f"✅ {success_count} successful imports")
    
    if errors:
        print(f"❌ {len(errors)} import failures:")
        for name, error in errors:
            print(f"   - {name} -> {error}")
        return False
    else:
        print("🎉 All modules import cleanly!")
        return True

def test_key_components():
    """Test importing key components that should always work."""
    key_imports = [
        "improved_local_assistant.core.settings",
        "improved_local_assistant.api.main",
        "improved_local_assistant.cli",
        "improved_local_assistant.services",
        "improved_local_assistant.models.schemas",
    ]
    
    print("\n🔑 Testing key component imports...")
    errors = []
    
    for module_name in key_imports:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            errors.append((module_name, str(e)))
            print(f"❌ {module_name} -> {e}")
    
    if errors:
        print(f"\n❌ {len(errors)} key component failures:")
        for name, error in errors:
            print(f"   - {name} -> {error}")
        return False
    else:
        print("🎉 All key components import successfully!")
        return True

def test_fastapi_app_creation():
    """Test that we can create the FastAPI app."""
    print("\n🚀 Testing FastAPI app creation...")
    try:
        from improved_local_assistant.api.main import create_app
        app = create_app()
        print("✅ FastAPI app created successfully")
        print(f"   - Title: {app.title}")
        print(f"   - Routes: {len(app.routes)}")
        return True
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_entry_point():
    """Test that CLI entry point works."""
    print("\n⌨️  Testing CLI entry point...")
    try:
        from improved_local_assistant.cli import app as cli_app
        print("✅ CLI app imported successfully")
        print(f"   - Commands: {len(cli_app.commands)}")
        return True
    except Exception as e:
        print(f"❌ CLI entry point failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Import Smoke Test for Improved Local Assistant")
    print("=" * 50)
    
    all_passed = True
    
    # Test all imports
    if not test_all_imports():
        all_passed = False
    
    # Test key components
    if not test_key_components():
        all_passed = False
    
    # Test FastAPI app
    if not test_fastapi_app_creation():
        all_passed = False
    
    # Test CLI
    if not test_cli_entry_point():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All import tests passed!")
        sys.exit(0)
    else:
        print("❌ Some import tests failed!")
        sys.exit(1)