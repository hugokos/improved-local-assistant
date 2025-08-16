#!/usr/bin/env python3
"""
Test script for the 98% memory threshold fallback.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_memory_threshold():
    """Test the 98% memory threshold functionality."""
    print("🧪 Testing 98% Memory Threshold")
    print("=" * 40)

    try:
        from app.core.config import load_config
        from services.llm_orchestrator import LLMOrchestrator
        from services.system_monitor import SystemMonitor

        # Load configuration
        config = load_config()
        threshold = config.get("memory_fallback", {}).get("proactive_threshold_percent", 98)
        print(f"✅ Configuration loaded - threshold: {threshold}%")

        # Initialize services
        system_monitor = SystemMonitor(config)
        await system_monitor.start_monitoring()

        orchestrator = LLMOrchestrator(config, system_monitor)
        await orchestrator.initialize()

        # Test 1: Normal memory usage
        print("\n🔄 Test 1: Normal memory usage")
        status = await orchestrator.get_status()
        current_memory = status["current_memory_percent"]
        print(f"Current memory: {current_memory:.1f}%")
        print(f"Should use fallback: {status['using_fallback']}")

        # Test 2: Simulate high memory usage (99%)
        print("\n🔄 Test 2: Simulating 99% memory usage")

        # Mock psutil to return 99% memory usage
        with patch("psutil.virtual_memory") as mock_memory:
            # Create a mock memory object
            mock_memory_obj = type("MockMemory", (), {"percent": 99.0})()
            mock_memory.return_value = mock_memory_obj

            # Check if fallback is triggered
            should_fallback = orchestrator._should_use_fallback()
            memory_usage = orchestrator._get_current_memory_usage()

            print(f"Simulated memory: {memory_usage:.1f}%")
            print(f"Should use fallback: {should_fallback}")

            if should_fallback:
                print("✅ Threshold working - fallback triggered at 99%")

                # Test streaming with fallback
                messages = [{"role": "user", "content": "Hello"}]
                print("Response: ", end="")
                async for token in orchestrator._stream_conversation(messages):
                    print(token, end="", flush=True)
                print()
            else:
                print("❌ Threshold not working - fallback should be triggered")

        # Test 3: Simulate memory just below threshold (97%)
        print("\n🔄 Test 3: Simulating 97% memory usage (below threshold)")

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory_obj = type("MockMemory", (), {"percent": 97.0})()
            mock_memory.return_value = mock_memory_obj

            should_fallback = orchestrator._should_use_fallback()
            memory_usage = orchestrator._get_current_memory_usage()

            print(f"Simulated memory: {memory_usage:.1f}%")
            print(f"Should use fallback: {should_fallback}")

            if not should_fallback:
                print("✅ Threshold working - no fallback at 97%")
            else:
                print("❌ Unexpected fallback at 97%")

        print("\n✅ Memory threshold tests completed!")

        # Cleanup
        await orchestrator.shutdown()
        await system_monitor.stop_monitoring()

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_memory_threshold())
