#!/usr/bin/env python3
"""
Test script for memory-aware model fallback system.

This script allows testing and managing the memory fallback functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from improved_local_assistant.app.core.config import load_config  # noqa: E402
from improved_local_assistant.services.graceful_degradation import ComponentStatus  # noqa: E402
from improved_local_assistant.services.graceful_degradation import degradation_manager  # noqa: E402
from improved_local_assistant.services.llm_orchestrator import LLMOrchestrator  # noqa: E402
from improved_local_assistant.services.system_monitor import SystemMonitor  # noqa: E402


async def test_memory_fallback():
    """Test the memory fallback system."""
    print("🧪 Testing Memory Fallback System")
    print("=" * 50)

    # Load configuration
    config = load_config()

    # Initialize system monitor
    system_monitor = SystemMonitor(config)
    await system_monitor.initialize()

    # Initialize orchestrator
    orchestrator = LLMOrchestrator(config, system_monitor)
    await orchestrator.initialize()

    try:
        # Get initial status
        status = await orchestrator.get_status()
        print("Initial Status:")
        print(f"  Conversation Model: {status['conversation_model_status']}")
        print(f"  Fallback Enabled: {status['fallback_enabled']}")
        print(f"  Fallback Model: {status['fallback_model']}")
        print(f"  Using Fallback: {status['using_fallback']}")
        print()

        # Test normal conversation
        print("🔄 Testing normal conversation...")
        messages = [{"role": "user", "content": "Hello, how are you?"}]

        response_tokens = []
        async for token in orchestrator._stream_conversation(messages):
            response_tokens.append(token)
            print(token, end="", flush=True)

        print("\n")
        print(f"Response length: {len(''.join(response_tokens))} characters")

        # Check status after normal conversation
        status = await orchestrator.get_status()
        print(f"Status after normal conversation: {status['conversation_model_status']}")
        print()

        # Simulate memory error by setting component to degraded
        print("🚨 Simulating memory error...")
        await degradation_manager.set_component_status(
            "conversation_model", ComponentStatus.DEGRADED
        )

        # Test fallback conversation
        print("🔄 Testing fallback conversation...")
        messages = [{"role": "user", "content": "Can you help me with a simple question?"}]

        response_tokens = []
        async for token in orchestrator._stream_conversation(messages):
            response_tokens.append(token)
            print(token, end="", flush=True)

        print("\n")
        print(f"Fallback response length: {len(''.join(response_tokens))} characters")

        # Check final status
        status = await orchestrator.get_status()
        print("Final Status:")
        print(f"  Conversation Model: {status['conversation_model_status']}")
        print(f"  Using Fallback: {status['using_fallback']}")
        print()

        # Reset status
        print("🔄 Resetting model status...")
        await orchestrator.reset_model_status()

        status = await orchestrator.get_status()
        print(f"Status after reset: {status['conversation_model_status']}")

    finally:
        await orchestrator.shutdown()
        await system_monitor.shutdown()


async def reset_model_status():
    """Reset the conversation model status."""
    print("🔄 Resetting conversation model status...")

    # Load configuration
    config = load_config()

    # Initialize system monitor
    system_monitor = SystemMonitor(config)
    await system_monitor.initialize()

    # Initialize orchestrator
    orchestrator = LLMOrchestrator(config, system_monitor)
    await orchestrator.initialize()

    try:
        # Reset status
        await orchestrator.reset_model_status()

        # Get status
        status = await orchestrator.get_status()
        print(f"Model status reset to: {status['conversation_model_status']}")
        print("Primary model will be tried on next request.")

    finally:
        await orchestrator.shutdown()
        await system_monitor.shutdown()


async def show_status():
    """Show current system status."""
    print("📊 Current System Status")
    print("=" * 30)

    # Load configuration
    config = load_config()

    # Initialize system monitor
    system_monitor = SystemMonitor(config)
    await system_monitor.initialize()

    # Initialize orchestrator
    orchestrator = LLMOrchestrator(config, system_monitor)
    await orchestrator.initialize()

    try:
        # Get status
        status = await orchestrator.get_status()

        print(f"Conversation Model Status: {status['conversation_model_status']}")
        print(f"Fallback Enabled: {status['fallback_enabled']}")
        print(f"Fallback Model: {status['fallback_model']}")
        print(f"Currently Using Fallback: {status['using_fallback']}")
        print(f"Hermes Resident: {status['hermes_resident']}")
        print(f"TinyLlama Resident: {status['tinyllama_resident']}")
        print()

        print("All Component Statuses:")
        for component, comp_status in status["degradation_status"].items():
            print(f"  {component}: {comp_status}")

    finally:
        await orchestrator.shutdown()
        await system_monitor.shutdown()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_memory_fallback.py test     - Test fallback system")
        print("  python test_memory_fallback.py reset    - Reset model status")
        print("  python test_memory_fallback.py status   - Show current status")
        return

    command = sys.argv[1].lower()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if command == "test":
        asyncio.run(test_memory_fallback())
    elif command == "reset":
        asyncio.run(reset_model_status())
    elif command == "status":
        asyncio.run(show_status())
    else:
        print(f"Unknown command: {command}")
        print("Use 'test', 'reset', or 'status'")


if __name__ == "__main__":
    main()
