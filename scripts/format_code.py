#!/usr/bin/env python3
"""
Code formatting script to fix style violations.
"""

import shlex
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print(f"✅ {description} - COMPLETED")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"⚠️ {description} - ISSUES FOUND")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run code formatting tools."""
    print("🚀 Starting code formatting...")

    # Change to project directory
    project_root = Path(__file__).parent.parent
    print(f"Working in: {project_root}")

    # Commands to run
    commands = [
        # Remove unused imports
        (
            "python -m autoflake --remove-all-unused-imports --in-place --recursive app/ services/ cli/ tests/",
            "Removing unused imports",
        ),
        # Fix import order
        ("python -m isort app/ services/ cli/ tests/", "Sorting imports"),
        # Format code
        (
            "python -m black app/ services/ cli/ tests/ --line-length 120",
            "Formatting code with Black",
        ),
        # Fix specific PEP8 issues
        (
            "python -m autopep8 --in-place --aggressive --aggressive --recursive app/ services/ cli/ tests/",
            "Fixing PEP8 issues",
        ),
    ]

    success_count = 0
    for cmd, desc in commands:
        if run_command(cmd, desc):
            success_count += 1
        print()  # Add spacing

    print(f"📊 Completed {success_count}/{len(commands)} formatting steps")

    # Final lint check
    print("🔍 Running final lint check...")
    run_command(
        "python -m flake8 app/ services/ cli/ tests/ --max-line-length=120 --count",
        "Final lint check",
    )


if __name__ == "__main__":
    main()
