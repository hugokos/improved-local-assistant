#!/usr/bin/env python3
"""
CLI tool to toggle quiet monitoring mode.

This script allows users to enable or disable resource usage warnings
that can interrupt the chat experience.
"""

import argparse
import sys
from pathlib import Path

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config_file():
    """Load the configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return None, None

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config, config_path
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None, None


def save_config_file(config, config_path):
    """Save the configuration file."""
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False


def show_status(config):
    """Show current quiet monitoring status."""
    print("🔇 Current Quiet Monitoring Status:")
    print("=" * 40)

    system_config = config.get("system", {})
    quiet_monitoring = system_config.get("quiet_monitoring", False)

    print(f"📊 Status: {'✅ ENABLED' if quiet_monitoring else '❌ DISABLED'}")

    if quiet_monitoring:
        print("\n🤫 Resource usage warnings are muted in CLI")
        print("   - High memory usage warnings: MUTED")
        print("   - High CPU usage warnings: MUTED")
        print("   - High system load warnings: MUTED")
        print("\n💡 This provides a cleaner chat experience without interruptions.")
    else:
        print("\n📢 Resource usage warnings are visible in CLI")
        print("   - High memory usage warnings: VISIBLE")
        print("   - High CPU usage warnings: VISIBLE")
        print("   - High system load warnings: VISIBLE")
        print("\n💡 Use --enable to mute these warnings for a cleaner chat experience.")

    print("\n💡 Use --enable or --disable to change the status.")


def enable_quiet_monitoring(config):
    """Enable quiet monitoring."""
    print("🤫 Enabling Quiet Monitoring...")

    # Ensure system section exists
    if "system" not in config:
        config["system"] = {}

    config["system"]["quiet_monitoring"] = True

    print("✅ Quiet monitoring enabled successfully!")
    print("💡 Resource usage warnings will no longer interrupt your chat experience.")
    print("💡 Restart the application for changes to take effect.")


def disable_quiet_monitoring(config):
    """Disable quiet monitoring."""
    print("📢 Disabling Quiet Monitoring...")

    # Ensure system section exists
    if "system" not in config:
        config["system"] = {}

    config["system"]["quiet_monitoring"] = False

    print("✅ Quiet monitoring disabled successfully!")
    print("💡 Resource usage warnings will now be visible in CLI.")
    print("💡 Restart the application for changes to take effect.")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Toggle quiet monitoring mode for the Improved Local AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli/toggle_quiet_monitoring.py --status
    python cli/toggle_quiet_monitoring.py --enable
    python cli/toggle_quiet_monitoring.py --disable
        """,
    )

    parser.add_argument(
        "--status", action="store_true", help="Show current quiet monitoring status"
    )

    parser.add_argument(
        "--enable", action="store_true", help="Enable quiet monitoring (mute resource warnings)"
    )

    parser.add_argument(
        "--disable", action="store_true", help="Disable quiet monitoring (show resource warnings)"
    )

    args = parser.parse_args()

    # Load configuration
    config, config_path = load_config_file()
    if config is None:
        return 1

    # Handle commands
    if args.status or (not args.enable and not args.disable):
        show_status(config)
        return 0

    config_changed = False

    if args.enable:
        enable_quiet_monitoring(config)
        config_changed = True

    if args.disable:
        disable_quiet_monitoring(config)
        config_changed = True

    # Save configuration if changed
    if config_changed:
        if save_config_file(config, config_path):
            print(f"💾 Configuration saved to: {config_path}")
        else:
            return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
