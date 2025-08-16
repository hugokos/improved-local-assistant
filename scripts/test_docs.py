#!/usr/bin/env python3
"""
Test script to validate MkDocs configuration.
"""
import subprocess
import sys
from pathlib import Path


def test_mkdocs_config():
    """Test that MkDocs configuration is valid."""
    config_path = Path("config/mkdocs.yml")

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        # Test that the config is valid
        result = subprocess.run(
            ["mkdocs", "build", "--config-file", str(config_path), "--site-dir", "test_site"],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            print("✅ MkDocs configuration is valid")
            print("✅ Documentation builds successfully")

            # Clean up test site
            import shutil
            test_site = Path("test_site")
            if test_site.exists():
                shutil.rmtree(test_site)
                print("✅ Cleaned up test build")

            return True
        else:
            print("❌ MkDocs build failed:")
            print(result.stderr)
            return False

    except FileNotFoundError:
        print("❌ MkDocs not installed. Install with: pip install mkdocs mkdocs-material mkdocstrings[python]")
        return False
    except Exception as e:
        print(f"❌ Error testing MkDocs: {e}")
        return False


def main():
    """Main test function."""
    print("🔍 Testing MkDocs configuration...")

    if test_mkdocs_config():
        print("🎉 All documentation tests passed!")
        return 0
    else:
        print("💥 Documentation tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
