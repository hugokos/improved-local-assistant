#!/usr/bin/env python3
"""
Quick fix script for the most critical issues.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - COMPLETED")
            return True
        else:
            print(f"⚠️ {description} - ISSUES FOUND")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run quick fixes for critical issues."""
    print("🚀 Running quick fixes for critical issues...")
    
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    # Quick fixes
    fixes = [
        # Remove trailing whitespace and fix blank lines
        ("python -m autopep8 --in-place --select=W291,W292,W293 --recursive services/", 
         "Fixing whitespace issues"),
        
        # Remove unused imports automatically
        ("python -m autoflake --remove-all-unused-imports --in-place --recursive services/", 
         "Removing unused imports"),
        
        # Fix import order
        ("python -m isort services/ --profile black", 
         "Sorting imports"),
        
        # Basic formatting
        ("python -m black services/ --line-length 120 --quiet", 
         "Basic code formatting"),
    ]
    
    for cmd, desc in fixes:
        run_command(cmd, desc)
        print()
    
    print("🔍 Checking results...")
    
    # Check if basic issues are fixed
    result = subprocess.run(
        "python -m flake8 services/ --select=F401,E722,W291,W292,W293 --count", 
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        violations = result.stdout.strip()
        print(f"📊 Remaining critical violations: {violations}")
        if violations == "0":
            print("🎉 Critical issues fixed!")
        else:
            print("⚠️ Some critical issues remain")
    
    # Test basic imports
    try:
        from tests.mock_ollama import MockAsyncClient
        print("✅ Mock client import works")
    except ImportError as e:
        print(f"❌ Mock client import failed: {e}")
    
    try:
        from services import ModelManager
        print("✅ Services import works")
    except ImportError as e:
        print(f"❌ Services import failed: {e}")

if __name__ == "__main__":
    main()