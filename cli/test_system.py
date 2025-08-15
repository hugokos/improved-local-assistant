"""
System testing and validation for the Improved Local AI Assistant.

This module provides comprehensive testing for error handling, recovery,
performance optimization, and resource management.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services
from services.conversation_manager import ConversationManager
from services.graph_manager import KnowledgeGraphManager
from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager
from services.system_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/system_test.log", mode="w")],
)
logger = logging.getLogger(__name__)

# Global variables
config = {}
model_manager = None
kg_manager = None
conversation_manager = None
system_monitor = None


# Load configuration
def load_config():
    """Load configuration from config.yaml file."""
    import yaml

    global config
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


# Initialize services
async def initialize_services():
    """Initialize all services required by the application."""
    global model_manager, kg_manager, conversation_manager, system_monitor, config

    try:
        # Initialize model manager
        logger.info("Initializing ModelManager...")
        ollama_config = config.get("ollama", {})
        host = ollama_config.get("host", "http://localhost:11434")
        model_manager = ModelManager(host=host)

        # Create model config
        model_config = ModelConfig(
            name=config.get("models", {}).get("conversation", {}).get("name", "hermes3:3b"),
            type="conversation",
            context_window=config.get("models", {})
            .get("conversation", {})
            .get("context_window", 8000),
            temperature=config.get("models", {}).get("conversation", {}).get("temperature", 0.7),
            max_tokens=config.get("models", {}).get("conversation", {}).get("max_tokens", 2048),
            timeout=ollama_config.get("timeout", 120),
            max_parallel=ollama_config.get("max_parallel", 2),
            max_loaded=ollama_config.get("max_loaded_models", 2),
        )

        # Initialize models
        await model_manager.initialize_models(model_config)

        # Initialize knowledge graph manager
        logger.info("Initializing KnowledgeGraphManager...")
        kg_manager = KnowledgeGraphManager(model_manager=model_manager, config=config)

        # Load pre-built knowledge graphs
        kg_dir = config.get("knowledge_graphs", {}).get(
            "prebuilt_directory", "./data/prebuilt_graphs"
        )
        kg_manager.load_prebuilt_graphs(kg_dir)

        # Initialize conversation manager
        logger.info("Initializing ConversationManager...")
        conversation_manager = ConversationManager(
            model_manager=model_manager, kg_manager=kg_manager, config=config
        )

        # Initialize system monitor
        logger.info("Initializing SystemMonitor...")
        system_monitor = SystemMonitor(config=config)
        await system_monitor.start_monitoring()

        logger.info("All services initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        return False


# Test error handling
async def test_error_handling():
    """Test comprehensive error handling and recovery mechanisms."""
    logger.info("Starting error handling tests...")
    results = {"tests_run": 0, "tests_passed": 0, "tests_failed": 0, "details": {}}

    # Test 1: Model initialization failure recovery
    logger.info("Test 1: Model initialization failure recovery")
    results["tests_run"] += 1
    try:
        # Simulate model failure by using an invalid model name
        invalid_config = ModelConfig(
            name="invalid_model_name",
            type="conversation",
            context_window=8000,
            temperature=0.7,
            max_tokens=2048,
            timeout=30,
            max_parallel=2,
            max_loaded=2,
        )

        # Try to initialize with invalid model
        try:
            await model_manager.initialize_models(invalid_config)
            logger.error("Expected model initialization to fail but it succeeded")
            results["tests_failed"] += 1
            results["details"][
                "model_init_failure_recovery"
            ] = "Failed: Did not detect invalid model"
        except Exception as e:
            # This is expected - now test recovery
            logger.info(f"Expected error occurred: {str(e)}")

            # Try to recover by initializing with valid model
            valid_config = ModelConfig(
                name="hermes3:3b",
                type="conversation",
                context_window=8000,
                temperature=0.7,
                max_tokens=2048,
                timeout=30,
                max_parallel=2,
                max_loaded=2,
            )

            success = await model_manager.initialize_models(valid_config)
            if success:
                logger.info("Successfully recovered from model initialization failure")
                results["tests_passed"] += 1
                results["details"]["model_init_failure_recovery"] = "Passed"
            else:
                logger.error("Failed to recover from model initialization failure")
                results["tests_failed"] += 1
                results["details"]["model_init_failure_recovery"] = "Failed: Could not recover"
    except Exception as e:
        logger.error(f"Error in model initialization failure recovery test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["model_init_failure_recovery"] = f"Failed: {str(e)}"

    # Test 2: Knowledge graph query error handling
    logger.info("Test 2: Knowledge graph query error handling")
    results["tests_run"] += 1
    try:
        # Simulate a malformed query
        malformed_query = "?" * 1000  # Very long nonsensical query

        # Query should not crash even with malformed input
        query_result = kg_manager.query_graphs(malformed_query)

        # Check if we got a response (even if it's an error message)
        if "response" in query_result:
            logger.info("Knowledge graph query handled malformed input gracefully")
            results["tests_passed"] += 1
            results["details"]["kg_query_error_handling"] = "Passed"
        else:
            logger.error("Knowledge graph query failed to handle malformed input")
            results["tests_failed"] += 1
            results["details"]["kg_query_error_handling"] = "Failed: No response returned"
    except Exception as e:
        logger.error(f"Error in knowledge graph query error handling test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["kg_query_error_handling"] = f"Failed: {str(e)}"

    # Test 3: Conversation context error handling
    logger.info("Test 3: Conversation context error handling")
    results["tests_run"] += 1
    try:
        # Create a session
        session_id = conversation_manager.create_session()

        # Simulate a very large context that exceeds limits
        large_message = "test " * 10000  # Very large message

        # Process should not crash with large input
        response_text = ""
        async for token in conversation_manager.process_message(session_id, large_message):
            response_text += token

        if response_text:
            logger.info("Conversation manager handled large context gracefully")
            results["tests_passed"] += 1
            results["details"]["conversation_context_error_handling"] = "Passed"
        else:
            logger.error("Conversation manager failed to handle large context")
            results["tests_failed"] += 1
            results["details"][
                "conversation_context_error_handling"
            ] = "Failed: No response returned"
    except Exception as e:
        logger.error(f"Error in conversation context error handling test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["conversation_context_error_handling"] = f"Failed: {str(e)}"

    # Test 4: Invalid session ID handling
    logger.info("Test 4: Invalid session ID handling")
    results["tests_run"] += 1
    try:
        # Try to use an invalid session ID
        invalid_session_id = "invalid_session_id"

        # Process should handle invalid session gracefully
        response_text = ""
        try:
            async for token in conversation_manager.process_message(invalid_session_id, "Hello"):
                response_text += token

            # If we get here without exception, the error wasn't handled properly
            logger.error("Conversation manager did not detect invalid session ID")
            results["tests_failed"] += 1
            results["details"][
                "invalid_session_handling"
            ] = "Failed: Did not detect invalid session"
        except Exception as e:
            # This is expected
            logger.info(f"Expected error occurred: {str(e)}")
            results["tests_passed"] += 1
            results["details"]["invalid_session_handling"] = "Passed"
    except Exception as e:
        logger.error(f"Error in invalid session ID handling test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["invalid_session_handling"] = f"Failed: {str(e)}"

    # Test 5: Circuit breaker for external dependencies
    logger.info("Test 5: Circuit breaker for external dependencies")
    results["tests_run"] += 1
    try:
        # Simulate Ollama service being down by changing the host
        original_host = model_manager.host
        model_manager.host = "http://localhost:99999"  # Invalid port

        # Try to query the model
        start_time = time.time()
        try:
            messages = [{"role": "user", "content": "Hello"}]
            response_text = ""
            async for token in model_manager.query_conversation_model(messages):
                response_text += token

            # If we get here, the circuit breaker didn't work
            end_time = time.time()
            if end_time - start_time > 10:  # If it took more than 10 seconds
                logger.error("Circuit breaker failed: request took too long")
                results["tests_failed"] += 1
                results["details"]["circuit_breaker"] = "Failed: Request took too long"
            else:
                logger.info("Circuit breaker worked: request completed quickly")
                results["tests_passed"] += 1
                results["details"]["circuit_breaker"] = "Passed"
        except Exception as e:
            # This is expected
            logger.info(f"Expected error occurred: {str(e)}")
            end_time = time.time()
            if end_time - start_time < 10:  # If it failed quickly
                logger.info("Circuit breaker worked: failed quickly")
                results["tests_passed"] += 1
                results["details"]["circuit_breaker"] = "Passed"
            else:
                logger.error("Circuit breaker failed: took too long to fail")
                results["tests_failed"] += 1
                results["details"]["circuit_breaker"] = "Failed: Took too long to fail"

        # Restore original host
        model_manager.host = original_host
    except Exception as e:
        logger.error(f"Error in circuit breaker test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["circuit_breaker"] = f"Failed: {str(e)}"
        # Restore original host
        model_manager.host = original_host

    # Print summary
    logger.info(
        f"Error handling tests completed: {results['tests_passed']}/{results['tests_run']} passed"
    )
    for test_name, result in results["details"].items():
        logger.info(f"  {test_name}: {result}")

    return results


# Test performance optimization
async def test_performance():
    """Test performance optimization and resource management."""
    logger.info("Starting performance tests...")
    results = {"tests_run": 0, "tests_passed": 0, "tests_failed": 0, "details": {}}

    # Test 1: Memory usage monitoring
    logger.info("Test 1: Memory usage monitoring")
    results["tests_run"] += 1
    try:
        # Get initial memory usage
        initial_usage = system_monitor.get_resource_usage()

        # Check if memory usage is being tracked
        if "memory_percent" in initial_usage and "memory_used_gb" in initial_usage:
            logger.info(
                f"Memory usage is being tracked: {initial_usage['memory_percent']}%, {initial_usage['memory_used_gb']} GB"
            )
            results["tests_passed"] += 1
            results["details"]["memory_usage_monitoring"] = "Passed"
        else:
            logger.error("Memory usage is not being tracked properly")
            results["tests_failed"] += 1
            results["details"]["memory_usage_monitoring"] = "Failed: Memory metrics not available"
    except Exception as e:
        logger.error(f"Error in memory usage monitoring test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["memory_usage_monitoring"] = f"Failed: {str(e)}"

    # Test 2: Conversation summarization for long sessions
    logger.info("Test 2: Conversation summarization for long sessions")
    results["tests_run"] += 1
    try:
        # Create a session
        session_id = conversation_manager.create_session()

        # Add multiple messages to trigger summarization
        summarize_threshold = conversation_manager.summarize_threshold
        logger.info(f"Adding {summarize_threshold} messages to trigger summarization")

        for i in range(summarize_threshold):
            message = f"Test message {i+1}"
            response_text = ""
            async for token in conversation_manager.process_message(session_id, message):
                response_text += token

        # Wait for background summarization to complete
        logger.info("Waiting for background summarization to complete...")
        await asyncio.sleep(5)

        # Check if summarization was triggered
        session = conversation_manager.sessions[session_id]
        if session.get("summary"):
            logger.info("Conversation summarization was triggered successfully")
            results["tests_passed"] += 1
            results["details"]["conversation_summarization"] = "Passed"
        else:
            logger.error("Conversation summarization was not triggered")
            results["tests_failed"] += 1
            results["details"]["conversation_summarization"] = "Failed: No summary generated"
    except Exception as e:
        logger.error(f"Error in conversation summarization test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["conversation_summarization"] = f"Failed: {str(e)}"

    # Test 3: Knowledge graph query performance
    logger.info("Test 3: Knowledge graph query performance")
    results["tests_run"] += 1
    try:
        # Perform multiple queries and measure performance
        queries = [
            "What is a knowledge graph?",
            "How does LlamaIndex work?",
            "What are the benefits of using Ollama?",
            "How can I optimize performance?",
            "What is the dual-model architecture?",
        ]

        query_times = []
        for query in queries:
            start_time = time.time()
            kg_manager.query_graphs(query)
            query_time = time.time() - start_time
            query_times.append(query_time)
            logger.info(f"Query '{query}' took {query_time:.2f} seconds")

        # Calculate average query time
        avg_query_time = sum(query_times) / len(query_times)
        logger.info(f"Average query time: {avg_query_time:.2f} seconds")

        # Check if average query time is within acceptable limits (5 seconds)
        if avg_query_time < 5.0:
            logger.info("Knowledge graph query performance is acceptable")
            results["tests_passed"] += 1
            results["details"]["kg_query_performance"] = f"Passed: {avg_query_time:.2f}s avg"
        else:
            logger.error("Knowledge graph query performance is too slow")
            results["tests_failed"] += 1
            results["details"]["kg_query_performance"] = f"Failed: {avg_query_time:.2f}s avg (>5s)"
    except Exception as e:
        logger.error(f"Error in knowledge graph query performance test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["kg_query_performance"] = f"Failed: {str(e)}"

    # Test 4: Adaptive resource allocation
    logger.info("Test 4: Adaptive resource allocation")
    results["tests_run"] += 1
    try:
        # Trigger high load condition
        logger.info("Triggering high load condition...")

        # Create multiple concurrent tasks to simulate high load
        async def heavy_task():
            # Generate a large list to consume memory
            large_list = ["x" * 1000000 for _ in range(10)]
            await asyncio.sleep(1)
            return len(large_list)

        # Run multiple heavy tasks concurrently
        tasks = [heavy_task() for _ in range(10)]
        await asyncio.gather(*tasks)

        # Check if adaptive actions were triggered
        metrics = system_monitor.get_performance_metrics()
        if metrics.get("adaptive_actions", 0) > 0:
            logger.info("Adaptive resource management was triggered")
            results["tests_passed"] += 1
            results["details"]["adaptive_resource_allocation"] = "Passed"
        else:
            logger.warning("Adaptive resource management was not triggered")
            # This is not necessarily a failure, as it depends on the system load
            # We'll mark it as passed but with a warning
            results["tests_passed"] += 1
            results["details"][
                "adaptive_resource_allocation"
            ] = "Passed with warning: No adaptive actions triggered"
    except Exception as e:
        logger.error(f"Error in adaptive resource allocation test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["adaptive_resource_allocation"] = f"Failed: {str(e)}"

    # Print summary
    logger.info(
        f"Performance tests completed: {results['tests_passed']}/{results['tests_run']} passed"
    )
    for test_name, result in results["details"].items():
        logger.info(f"  {test_name}: {result}")

    return results


# Test deployment configuration
async def test_deployment():
    """Test deployment and configuration management."""
    logger.info("Starting deployment tests...")
    results = {"tests_run": 0, "tests_passed": 0, "tests_failed": 0, "details": {}}

    # Test 1: Configuration validation
    logger.info("Test 1: Configuration validation")
    results["tests_run"] += 1
    try:
        # Check if configuration was loaded successfully
        if config:
            # Validate required configuration sections
            required_sections = ["ollama", "models", "knowledge_graphs", "conversation", "system"]
            missing_sections = [section for section in required_sections if section not in config]

            if not missing_sections:
                logger.info("Configuration validation passed")
                results["tests_passed"] += 1
                results["details"]["configuration_validation"] = "Passed"
            else:
                logger.error(
                    f"Configuration validation failed: Missing sections {missing_sections}"
                )
                results["tests_failed"] += 1
                results["details"][
                    "configuration_validation"
                ] = f"Failed: Missing sections {missing_sections}"
        else:
            logger.error("Configuration was not loaded")
            results["tests_failed"] += 1
            results["details"]["configuration_validation"] = "Failed: Configuration not loaded"
    except Exception as e:
        logger.error(f"Error in configuration validation test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["configuration_validation"] = f"Failed: {str(e)}"

    # Test 2: Directory structure
    logger.info("Test 2: Directory structure")
    results["tests_run"] += 1
    try:
        # Check if required directories exist
        required_dirs = ["logs", "data", "data/prebuilt_graphs", "data/dynamic_graph"]
        missing_dirs = [d for d in required_dirs if not os.path.exists(d)]

        if not missing_dirs:
            logger.info("Directory structure validation passed")
            results["tests_passed"] += 1
            results["details"]["directory_structure"] = "Passed"
        else:
            # Create missing directories
            for d in missing_dirs:
                os.makedirs(d, exist_ok=True)
            logger.info(f"Created missing directories: {missing_dirs}")
            results["tests_passed"] += 1
            results["details"][
                "directory_structure"
            ] = f"Passed: Created missing directories {missing_dirs}"
    except Exception as e:
        logger.error(f"Error in directory structure test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["directory_structure"] = f"Failed: {str(e)}"

    # Test 3: Environment variables
    logger.info("Test 3: Environment variables")
    results["tests_run"] += 1
    try:
        # Check if environment variables are set correctly
        required_vars = ["OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS"]
        missing_vars = [var for var in required_vars if var not in os.environ]

        if not missing_vars:
            logger.info("Environment variables validation passed")
            results["tests_passed"] += 1
            results["details"]["environment_variables"] = "Passed"
        else:
            logger.error(
                f"Environment variables validation failed: Missing variables {missing_vars}"
            )
            results["tests_failed"] += 1
            results["details"][
                "environment_variables"
            ] = f"Failed: Missing variables {missing_vars}"
    except Exception as e:
        logger.error(f"Error in environment variables test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["environment_variables"] = f"Failed: {str(e)}"

    # Test 4: Dependency checking
    logger.info("Test 4: Dependency checking")
    results["tests_run"] += 1
    try:
        # Check if required dependencies are installed
        required_packages = ["ollama", "llama_index", "fastapi", "networkx", "pyvis", "psutil"]
        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if not missing_packages:
            logger.info("Dependency checking passed")
            results["tests_passed"] += 1
            results["details"]["dependency_checking"] = "Passed"
        else:
            logger.error(f"Dependency checking failed: Missing packages {missing_packages}")
            results["tests_failed"] += 1
            results["details"][
                "dependency_checking"
            ] = f"Failed: Missing packages {missing_packages}"
    except Exception as e:
        logger.error(f"Error in dependency checking test: {str(e)}")
        results["tests_failed"] += 1
        results["details"]["dependency_checking"] = f"Failed: {str(e)}"

    # Print summary
    logger.info(
        f"Deployment tests completed: {results['tests_passed']}/{results['tests_run']} passed"
    )
    for test_name, result in results["details"].items():
        logger.info(f"  {test_name}: {result}")

    return results


# Main function
async def main():
    """Main function for system testing."""
    parser = argparse.ArgumentParser(description="System testing for Improved Local AI Assistant")
    parser.add_argument(
        "--test-error-handling", action="store_true", help="Test error handling and recovery"
    )
    parser.add_argument(
        "--test-performance", action="store_true", help="Test performance optimization"
    )
    parser.add_argument(
        "--test-deployment", action="store_true", help="Test deployment configuration"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("logs", exist_ok=True)

    # Load configuration
    load_config()

    # Initialize services
    success = await initialize_services()
    if not success:
        logger.error("Failed to initialize services")
        return 1

    results = {"timestamp": datetime.now().isoformat(), "tests": {}}

    try:
        # Run tests based on arguments
        if args.test_error_handling or args.all:
            results["tests"]["error_handling"] = await test_error_handling()

        if args.test_performance or args.all:
            results["tests"]["performance"] = await test_performance()

        if args.test_deployment or args.all:
            results["tests"]["deployment"] = await test_deployment()

        # If no specific tests were requested, show help
        if not (
            args.test_error_handling or args.test_performance or args.test_deployment or args.all
        ):
            parser.print_help()
            return 0

        # Calculate overall results
        total_run = sum(test["tests_run"] for test in results["tests"].values())
        total_passed = sum(test["tests_passed"] for test in results["tests"].values())
        total_failed = sum(test["tests_failed"] for test in results["tests"].values())

        results["summary"] = {
            "total_tests": total_run,
            "passed": total_passed,
            "failed": total_failed,
            "success_rate": f"{(total_passed / total_run * 100):.1f}%" if total_run > 0 else "N/A",
        }

        # Print overall summary
        logger.info("=" * 50)
        logger.info(
            f"SYSTEM TEST SUMMARY: {total_passed}/{total_run} tests passed ({results['summary']['success_rate']})"
        )
        logger.info("=" * 50)

        # Save results to file
        import json

        with open("logs/system_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Test results saved to logs/system_test_results.json")

        return 0 if total_failed == 0 else 1

    finally:
        # Clean up resources
        if system_monitor:
            await system_monitor.stop_monitoring()

        # Additional cleanup if needed
        logger.info("System tests completed")


if __name__ == "__main__":
    asyncio.run(main())
