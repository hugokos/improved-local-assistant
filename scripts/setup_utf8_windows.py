#!/usr/bin/env python3
"""
Setup UTF-8 as default encoding on Windows.
This provides an additional safeguard beyond the runtime patch.
"""

import os
import platform
import subprocess
import sys


def set_python_utf8_env():
    """Set PYTHONUTF8=1 environment variable on Windows"""
    if platform.system() != "Windows":
        print("ℹ️ This script is for Windows only")
        return True

    try:
        # Set for current session
        os.environ["PYTHONUTF8"] = "1"
        print("✅ Set PYTHONUTF8=1 for current session")

        # Set permanently for user
        result = subprocess.run(
            ["setx", "PYTHONUTF8", "1"], capture_output=True, text=True, check=True
        )

        print("✅ Set PYTHONUTF8=1 permanently for user")
        print("💡 This will take effect for new Python processes")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set environment variable: {e}")
        print("💡 You can manually run: setx PYTHONUTF8 1")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def verify_utf8_setting():
    """Verify UTF-8 settings are working"""
    print("\n🔍 Verifying UTF-8 settings...")

    # Check environment variable
    pythonutf8 = os.environ.get("PYTHONUTF8")
    if pythonutf8 == "1":
        print("✅ PYTHONUTF8 environment variable is set")
    else:
        print("⚠️ PYTHONUTF8 environment variable not set")

    # Check default encoding
    import locale

    default_encoding = locale.getpreferredencoding()
    print(f"📊 System default encoding: {default_encoding}")

    # Test UTF-8 file operations
    import json
    import tempfile

    test_data = {"test": "Hello 世界 café naïve"}

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f, ensure_ascii=False)
            temp_path = f.name

        # Read back without explicit encoding
        with open(temp_path) as f:
            loaded_data = json.load(f)

        if loaded_data == test_data:
            print("✅ UTF-8 file operations working correctly")
            success = True
        else:
            print("❌ UTF-8 file operations failed - data corruption")
            success = False

        os.unlink(temp_path)
        return success

    except Exception as e:
        print(f"❌ UTF-8 test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🔧 Windows UTF-8 Setup")
    print("=" * 40)

    print("This script will:")
    print("1. Set PYTHONUTF8=1 environment variable")
    print("2. Verify UTF-8 operations work correctly")
    print("3. Provide additional safeguards for knowledge graph loading")

    if platform.system() != "Windows":
        print("\nℹ️ This setup is only needed on Windows")
        print("✅ Your system should handle UTF-8 correctly by default")
        return 0

    print(f"\n🖥️ Detected Windows system: {platform.platform()}")

    # Set environment variable
    env_success = set_python_utf8_env()

    # Verify settings
    verify_success = verify_utf8_setting()

    print("\n" + "=" * 40)
    if env_success and verify_success:
        print("🎉 UTF-8 setup completed successfully!")
        print("🚀 Knowledge graphs will load correctly with Unicode content")
        print("💡 Restart your terminal/IDE for the changes to take full effect")
        return 0
    else:
        print("⚠️ UTF-8 setup had some issues")
        print("💡 The runtime patch in the application should still work")
        return 1


if __name__ == "__main__":
    sys.exit(main())
