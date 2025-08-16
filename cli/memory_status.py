#!/usr/bin/env python3
"""
Quick memory status checker for the assistant.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def check_memory_status():
    """Check current memory status and model availability."""
    try:
        from app.core.config import load_config
        from services.llm_orchestrator import LLMOrchestrator
        from services.system_monitor import SystemMonitor

        # Load configuration
        config = load_config()

        # Initialize services
        system_monitor = SystemMonitor(config)
        await system_monitor.start_monitoring()

        orchestrator = LLMOrchestrator(config, system_monitor)
        await orchestrator.initialize()

        # Get status
        status = await orchestrator.get_status()

        print("🔍 Memory & Model Status")
        print("=" * 30)
        print(f"Conversation Model: {status['conversation_model_status']}")
        print(f"Using Fallback: {'Yes' if status['using_fallback'] else 'No'}")
        print(
            f"Primary Model ({config['models']['conversation']['name']}): {'Loaded' if status['hermes_resident'] else 'Not Loaded'}"
        )
        print(
            f"Fallback Model ({status['fallback_model']}): {'Loaded' if status['tinyllama_resident'] else 'Not Loaded'}"
        )
        print()

        # Memory threshold information
        current_memory = status.get("current_memory_percent", 0)
        threshold = status.get("proactive_threshold", 98)
        print(f"Memory Threshold: {threshold}% (proactive fallback)")
        print(
            f"Current Memory: {current_memory:.1f}% {'⚠️ EXCEEDS THRESHOLD' if current_memory >= threshold else '✅ Within limits'}"
        )

        # System resources
        import psutil

        memory = psutil.virtual_memory()
        print(
            f"System Memory: {memory.percent:.1f}% used ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)"
        )
        print(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
        print()

        # Recommendations
        threshold = status.get("proactive_threshold", 98)
        if status["using_fallback"]:
            if current_memory >= threshold:
                print("💡 Recommendations:")
                print(f"  - Memory usage ({current_memory:.1f}%) exceeds threshold ({threshold}%)")
                print("  - System automatically using lightweight model")
                print("  - Close other applications to free memory")
                print(f"  - Memory must drop below {threshold}% to use primary model again")
            else:
                print("💡 Recommendations:")
                print("  - System using fallback due to previous memory errors")
                print("  - Consider closing other applications to free memory")
                print("  - Run 'python cli/memory_status.py reset' to retry primary model")
        elif current_memory >= threshold - 5:  # Within 5% of threshold
            print("⚠️  Warning: Approaching memory threshold")
            print(f"  - Current: {current_memory:.1f}%, Threshold: {threshold}%")
            print("  - Consider closing other applications")
            print("  - System will automatically switch to fallback if threshold exceeded")
        else:
            print("✅ System operating normally with primary model")

        # Cleanup
        await orchestrator.shutdown()
        await system_monitor.stop_monitoring()

    except Exception as e:
        print(f"❌ Error checking status: {e}")


async def reset_status():
    """Reset model status to retry primary model."""
    try:
        from app.core.config import load_config
        from services.llm_orchestrator import LLMOrchestrator
        from services.system_monitor import SystemMonitor

        config = load_config()
        system_monitor = SystemMonitor(config)
        await system_monitor.start_monitoring()

        orchestrator = LLMOrchestrator(config, system_monitor)
        await orchestrator.initialize()

        await orchestrator.reset_model_status()
        print("✅ Model status reset - will retry primary model on next request")

        await orchestrator.shutdown()
        await system_monitor.stop_monitoring()

    except Exception as e:
        print(f"❌ Error resetting status: {e}")


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1].lower() == "reset":
        asyncio.run(reset_status())
    else:
        asyncio.run(check_memory_status())


if __name__ == "__main__":
    main()
