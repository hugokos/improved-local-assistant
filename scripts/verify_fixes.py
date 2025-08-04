#!/usr/bin/env python3
"""
Verification script for all critical fixes.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout.strip():
                # Only show count for lint checks
                if "count" in cmd or "violations" in result.stdout:
                    print(f"   Result: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🚀 Starting verification of critical fixes...")
    print("=" * 60)
    
    checks = [
        # Test infrastructure
        ("python -c 'from tests.mock_ollama import MockAsyncClient; print(\"Mock client available\")'", 
         "Mock Ollama client import"),
        
        # Lint checks
        ("python -m flake8 app/main.py --select=E722 --count", 
         "Bare except statements check"),
        
        ("python -m flake8 app/ services/ --select=F401 --count", 
         "Unused imports check"),
        
        ("python -m flake8 app/ services/ --max-line-length=120 --count", 
         "Overall lint violations"),
        
        # Type checks
        ("python -m mypy services/constants.py --ignore-missing-imports --no-error-summary", 
         "Constants type check"),
        
        ("python -m mypy services/error_handler.py --ignore-missing-imports --no-error-summary", 
         "Error handler type check"),
        
        ("python -m mypy services/circuit_breaker.py --ignore-missing-imports --no-error-summary", 
         "Circuit breaker type check"),
        
        # Import structure
        ("python -c 'from services import ModelManager, KnowledgeGraphManager; print(\"Services import successfully\")'", 
         "Services package imports"),
        
        ("python -c 'from app.main import app; print(\"App imports successfully\")'", 
         "App imports"),
        
        # Basic functionality
        ("python -c 'import sys; print(\"Python path:\", len([p for p in sys.path if \"sys.path.append\" not in p]))'", 
         "Python path check"),
    ]
    
    passed = 0
    total = len(checks)
    
    for cmd, desc in checks:
        if run_command(cmd, desc):
            passed += 1
        print()  # Add spacing between checks
    
    print("=" * 60)
    print(f"📊 RESULTS: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All critical fixes verified successfully!")
        print("✅ Ready for production deployment!")
        return 0
    else:
        print("⚠️  Some issues remain - check output above")
        print(f"🔧 {total - passed} issues need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())