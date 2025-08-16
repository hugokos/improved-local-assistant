#!/usr/bin/env python3
"""
Simple test script for memory fallback functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import load_config
from services.graceful_degradation import ComponentStatus
from services.graceful_degradation import degradation_manager


async def test_simple_fallback():
    """Test the memory fallback system with a simple conversation."""
    print("🧪 Testing Memory Fallback System")
    print("=" * 40)

    try:
        # Load configuration
        config = load_config()
        print("✅ Configuration loaded")

        # Check memory fallback config
        fallback_config = config.get("memory_fallback", {})
        print(f"Memory Fallback Enabled: {fallback_config.get('enabled', False)}")
        print(f"Primary Model: {fallback_config.get('primary_model', 'N/A')}")
        print(f"Fallback Model: {fallback_config.get('fallback_model', 'N/A')}")
        print()

        # Import and initialize services
        from services.llm_orchestrator import LLMOrchestrator
        from services.system_monitor import SystemMonitor

        # Initialize system monitor
        system_monitor = SystemMonitor(config)
        await system_monitor.start_monitoring()
        print("✅ System monitor initialized")

        # Initialize orchestrator
        orchestrator = LLMOrchestrator(config, system_monitor)
        await orchestrator.initialize()
        print("✅ LLM orchestrator initialized")

        # Get initial status
        status = await orchestrator.get_status()
        print(f"Initial conversation model status: {status['conversation_model_status']}")
        print(f"Using fallback: {status['using_fallback']}")
        print()

        # Test 1: Normal conversation
        print("🔄 Test 1: Normal conversation")
        messages = [{"role": "user", "content": "Say hello in one sentence."}]

        print("Response: ", end="")
        async for token in orchestrator._stream_conversation(messages):
            print(token, end="", flush=True)
        print("\n")

        # Test 2: Force degraded status and test fallback
        print("🚨 Test 2: Simulating memory error")
        await degradation_manager.set_component_status(
            "conversation_model", ComponentStatus.DEGRADED
        )

        status = await orchestrator.get_status()
        print(f"Model status after degradation: {status['conversation_model_status']}")
        print(f"Will use fallback: {status['using_fallback']}")
        print()

        print("🔄 Testing fallback conversation")
        messages = [{"role": "user", "content": "Explain what 2+2 equals in one sentence."}]

        print("Fallback Response: ", end="")
        async for token in orchestrator._stream_conversation(messages):
            print(token, end="", flush=True)
        print("\n")

        # Test 3: Reset and verify
        print("🔄 Test 3: Resetting model status")
        await orchestrator.reset_model_status()

        status = await orchestrator.get_status()
        print(f"Model status after reset: {status['conversation_model_status']}")
        print(f"Will use fallback: {status['using_fallback']}")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()

    finally:
        try:
            await orchestrator.shutdown()
            await system_monitor.stop_monitoring()
            print("✅ Cleanup completed")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_simple_fallback())
