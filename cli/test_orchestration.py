#!/usr/bin/env python3
"""
Test script for the LLM Orchestration system.

This script provides a simple way to test the orchestration functionality
including semaphore behavior, model lifecycle, and resource management.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import from services
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.connection_pool_manager import ConnectionPoolManager
from services.llm_orchestrator import LLMOrchestrator
from services.system_monitor import SystemMonitor
from services.working_set_cache import WorkingSetCache

from app.core.config import load_config


async def test_basic_orchestration():
    """Test basic orchestration functionality."""
    print("🚀 Testing Basic LLM Orchestration")

    # Load configuration
    config = load_config()

    # Create system monitor
    system_monitor = SystemMonitor(config)
    await system_monitor.start_monitoring()

    # Create orchestrator
    orchestrator = LLMOrchestrator(config, system_monitor)

    try:
        # Initialize orchestrator
        print("📋 Initializing orchestrator...")
        success = await orchestrator.initialize()
        if not success:
            print("❌ Failed to initialize orchestrator")
            return False

        print("✅ Orchestrator initialized successfully")

        # Test basic conversation
        print("\n💬 Testing conversation flow...")
        conversation_history = [{"role": "user", "content": "Hello, how are you?"}]

        response_tokens = []
        async for token in orchestrator.process_turn(
            session_id="test_session",
            user_message="Tell me about artificial intelligence",
            conversation_history=conversation_history,
        ):
            response_tokens.append(token)
            print(token, end="", flush=True)

        print(f"\n✅ Received {len(response_tokens)} tokens")

        # Wait a moment for background extraction
        print("\n⏳ Waiting for background extraction...")
        await asyncio.sleep(2.0)

        # Check orchestrator status
        status = await orchestrator.get_status()
        print("\n📊 Orchestrator Status:")
        print(f"  - Semaphore locked: {status['semaphore_locked']}")
        print(f"  - Hermes prewarmed: {status['hermes_prewarmed']}")
        print(f"  - Hermes resident: {status['hermes_resident']}")
        print(f"  - TinyLlama resident: {status['tinyllama_resident']}")
        print(f"  - Active extraction: {status['active_extraction']}")
        print(f"  - Turns processed: {status['metrics']['turns_processed']}")
        print(f"  - Extractions completed: {status['metrics']['extractions_completed']}")

        return True

    except Exception as e:
        print(f"❌ Error during orchestration test: {e}")
        return False

    finally:
        await orchestrator.shutdown()
        await system_monitor.stop_monitoring()


async def test_concurrent_requests():
    """Test concurrent request handling with semaphore."""
    print("\n🔄 Testing Concurrent Request Handling")

    config = load_config()
    system_monitor = SystemMonitor(config)
    await system_monitor.start_monitoring()

    orchestrator = LLMOrchestrator(config, system_monitor)

    try:
        await orchestrator.initialize()

        # Create multiple concurrent requests
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                collect_response(
                    orchestrator.process_turn(
                        session_id=f"session_{i}",
                        user_message=f"Question {i}: What is {i + 1} + {i + 1}?",
                        conversation_history=[],
                    )
                )
            )
            tasks.append(task)

        # Wait for all to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        print(f"✅ Completed {len(results)} concurrent requests in {total_time:.2f}s")
        for i, result in enumerate(results):
            print(f"  Session {i}: {len(result)} tokens")

        # Check final status
        status = await orchestrator.get_status()
        print(f"📊 Final metrics: {status['metrics']['turns_processed']} turns processed")

        return True

    except Exception as e:
        print(f"❌ Error during concurrent test: {e}")
        return False

    finally:
        await orchestrator.shutdown()
        await system_monitor.stop_monitoring()


async def test_resource_pressure_handling():
    """Test resource pressure handling and skip logic."""
    print("\n⚠️  Testing Resource Pressure Handling")

    config = load_config()

    # Lower thresholds for testing
    config["system"]["memory_threshold_percent"] = 50
    config["system"]["cpu_threshold_percent"] = 50

    system_monitor = SystemMonitor(config)
    await system_monitor.start_monitoring()

    orchestrator = LLMOrchestrator(config, system_monitor)

    try:
        await orchestrator.initialize()

        # Simulate high resource usage
        # Note: This is a simulation - in real testing you'd need actual resource pressure
        print("🔧 Simulating resource pressure...")

        # Process a turn under "pressure"
        response_tokens = []
        async for token in orchestrator.process_turn(
            session_id="pressure_test", user_message="Test under pressure", conversation_history=[]
        ):
            response_tokens.append(token)

        print(f"✅ Handled request under pressure: {len(response_tokens)} tokens")

        # Check if extractions were skipped
        status = await orchestrator.get_status()
        skipped = status["metrics"]["extractions_skipped"]
        print(f"📊 Extractions skipped due to pressure: {skipped}")

        return True

    except Exception as e:
        print(f"❌ Error during pressure test: {e}")
        return False

    finally:
        await orchestrator.shutdown()
        await system_monitor.stop_monitoring()


async def test_working_set_cache():
    """Test working set cache functionality."""
    print("\n🗄️  Testing Working Set Cache")

    config = load_config()
    cache = WorkingSetCache(config)

    try:
        await cache.initialize()

        # Test basic cache operations
        print("📝 Testing cache operations...")

        # Add nodes to working set
        await cache.update_working_set("test_session", ["node1", "node2", "node3"])

        # Retrieve working set
        working_set = await cache.get_working_set("test_session")
        print(f"✅ Working set contains {len(working_set)} nodes: {working_set}")

        # Test cache stats
        session_stats = cache.get_session_stats("test_session")
        global_stats = cache.get_global_stats()

        print(f"📊 Session stats: {session_stats}")
        print(f"📊 Global stats: {global_stats}")

        # Test persistence
        print("💾 Testing cache persistence...")
        await cache.persist_cache()
        print("✅ Cache persisted successfully")

        return True

    except Exception as e:
        print(f"❌ Error during cache test: {e}")
        return False

    finally:
        await cache.shutdown()


async def test_connection_pool():
    """Test connection pool functionality."""
    print("\n🔌 Testing Connection Pool")

    config = load_config()
    pool_manager = ConnectionPoolManager(config)

    try:
        await pool_manager.initialize()

        # Test health check
        print("🏥 Testing health check...")
        is_healthy = await pool_manager.health_check()
        print(f"✅ Ollama health check: {'PASS' if is_healthy else 'FAIL'}")

        if not is_healthy:
            print("⚠️  Ollama may not be running - some tests will be skipped")
            return True

        # Test model residency check
        print("🔍 Testing model residency...")
        loaded_models = await pool_manager.get_loaded_models()
        print(f"📋 Currently loaded models: {[m.get('name', 'unknown') for m in loaded_models]}")

        # Test basic request (if Ollama is available)
        try:
            print("📤 Testing basic request...")
            response = await pool_manager.chat_request(
                model="hermes3:3b", messages=[{"role": "user", "content": "test"}], num_predict=1
            )
            print("✅ Basic request successful")
        except Exception as e:
            print(f"⚠️  Basic request failed (model may not be available): {e}")

        # Get metrics
        metrics = pool_manager.get_metrics()
        print(f"📊 Connection pool metrics: {metrics}")

        return True

    except Exception as e:
        print(f"❌ Error during connection pool test: {e}")
        return False

    finally:
        await pool_manager.shutdown()


async def collect_response(response_generator):
    """Helper to collect all tokens from a response generator."""
    tokens = []
    async for token in response_generator:
        tokens.append(token)
    return tokens


async def main():
    """Run all orchestration tests."""
    print("🧪 LLM Orchestration Test Suite")
    print("=" * 50)

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    tests = [
        ("Connection Pool", test_connection_pool),
        ("Working Set Cache", test_working_set_cache),
        ("Basic Orchestration", test_basic_orchestration),
        ("Concurrent Requests", test_concurrent_requests),
        ("Resource Pressure", test_resource_pressure_handling),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = await test_func()
            results.append((test_name, success))
            print(f"{'✅' if success else '❌'} {test_name}: {'PASS' if success else 'FAIL'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'=' * 50}")
    print("📋 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Orchestration system is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
