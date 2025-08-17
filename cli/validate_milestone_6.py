"""
Interactive validation CLI for Milestone 6.

This module provides an interactive CLI for validating the complete integrated system,
including end-to-end testing, performance benchmarking, and stability testing.
"""

import argparse
import asyncio
import json
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
from improved_local_assistant.services.conversation_manager import ConversationManager
from improved_local_assistant.services.graph_manager import KnowledgeGraphManager
from improved_local_assistant.services.model_mgr import ModelConfig
from improved_local_assistant.services.model_mgr import ModelManager
from improved_local_assistant.services.system_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/validate_milestone_6.log", mode="w"),
    ],
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


# Interactive validation functions
async def interactive_system_test():
    """Run interactive system testing."""
    print("\n" + "=" * 50)
    print("INTERACTIVE SYSTEM TESTING")
    print("=" * 50)

    # Create a session
    session_id = conversation_manager.create_session()
    print(f"\nCreated session: {session_id}")

    # Main interaction loop
    while True:
        print("\n" + "-" * 50)
        print("SYSTEM TEST MENU")
        print("-" * 50)
        print("1. Chat with assistant")
        print("2. View system status")
        print("3. View knowledge graph")
        print("4. Run performance test")
        print("5. Run stability test")
        print("6. Run error handling test")
        print("7. Exit")

        choice = input("\nEnter your choice (1-7): ")

        if choice == "1":
            await chat_with_assistant(session_id)
        elif choice == "2":
            await view_system_status()
        elif choice == "3":
            await view_knowledge_graph()
        elif choice == "4":
            await run_performance_test()
        elif choice == "5":
            await run_stability_test()
        elif choice == "6":
            await run_error_handling_test()
        elif choice == "7":
            print("\nExiting interactive system test...")
            break
        else:
            print("\nInvalid choice. Please try again.")


async def chat_with_assistant(session_id):
    """Chat with the assistant."""
    print("\n" + "-" * 50)
    print("CHAT WITH ASSISTANT")
    print("-" * 50)
    print("Type 'exit' to return to the main menu.")

    while True:
        # Get user input
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        # Process message
        print("\nAssistant: ", end="", flush=True)
        start_time = time.time()

        try:
            async for token in conversation_manager.converse_with_context(session_id, user_input):
                print(token, end="", flush=True)
        except Exception as e:
            print(f"\nError: {str(e)}")

        # Print processing time
        end_time = time.time()
        print(f"\n[Processed in {end_time - start_time:.2f} seconds]")

        # Show system resource usage
        if system_monitor:
            resource_usage = system_monitor.get_resource_usage()
            print(
                f"[CPU: {resource_usage['cpu_percent']}%, Memory: {resource_usage['memory_percent']}%]"
            )


async def view_system_status():
    """View detailed system status."""
    print("\n" + "-" * 50)
    print("SYSTEM STATUS")
    print("-" * 50)

    if not system_monitor:
        print("System monitor not initialized.")
        return

    # Get all metrics
    metrics = system_monitor.get_all_metrics()

    # Print system info
    print("\nSYSTEM INFO:")
    system_info = metrics["system_info"]
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Print resource usage
    print("\nRESOURCE USAGE:")
    resource_usage = metrics["resource_usage"]
    for key, value in resource_usage.items():
        print(f"  {key}: {value}")

    # Print performance metrics
    print("\nPERFORMANCE METRICS:")
    performance_metrics = metrics["performance_metrics"]
    for key, value in performance_metrics.items():
        print(f"  {key}: {value}")

    # Print health status
    print("\nHEALTH STATUS:")
    health = metrics["health"]
    print(f"  Overall status: {health['status']}")
    print("  Components:")
    for component, status in health["components"].items():
        print(f"    {component}: {status}")

    # Print model status
    print("\nMODEL STATUS:")
    if model_manager:
        model_status = await model_manager.get_model_status()
        print(f"  Conversation model: {model_status.get('conversation_model', 'unknown')}")
        print(f"  Knowledge model: {model_status.get('knowledge_model', 'unknown')}")

        # Print model metrics
        if "metrics" in model_status:
            print("  Metrics:")
            for model_type, model_metrics in model_status["metrics"].items():
                print(f"    {model_type}:")
                for metric, value in model_metrics.items():
                    print(f"      {metric}: {value}")
    else:
        print("  Model manager not initialized.")

    # Print knowledge graph status
    print("\nKNOWLEDGE GRAPH STATUS:")
    if kg_manager:
        kg_stats = kg_manager.get_graph_statistics()
        print(f"  Total graphs: {kg_stats.get('total_graphs', 0)}")
        print(f"  Total nodes: {kg_stats.get('total_nodes', 0)}")
        print(f"  Total edges: {kg_stats.get('total_edges', 0)}")

        # Print individual graph stats
        if "graphs" in kg_stats:
            print("  Graphs:")
            for graph_id, graph_info in kg_stats["graphs"].items():
                print(f"    {graph_id}:")
                for metric, value in graph_info.items():
                    print(f"      {metric}: {value}")
    else:
        print("  Knowledge graph manager not initialized.")

    # Print conversation status
    print("\nCONVERSATION STATUS:")
    if conversation_manager:
        sessions = conversation_manager.list_sessions()
        print(f"  Active sessions: {len(sessions)}")

        # Print session info
        if sessions:
            print("  Sessions:")
            for session in sessions:
                print(f"    {session['session_id']}:")
                for key, value in session.items():
                    if key != "session_id":
                        print(f"      {key}: {value}")
    else:
        print("  Conversation manager not initialized.")

    # Wait for user to continue
    input("\nPress Enter to continue...")


async def view_knowledge_graph():
    """View knowledge graph visualization."""
    print("\n" + "-" * 50)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("-" * 50)

    if not kg_manager:
        print("Knowledge graph manager not initialized.")
        return

    # Get available graphs
    kg_stats = kg_manager.get_graph_statistics()
    graphs = kg_stats.get("graphs", {})

    if not graphs:
        print("No knowledge graphs available.")
        return

    # Print available graphs
    print("\nAvailable graphs:")
    for i, graph_id in enumerate(graphs.keys()):
        print(
            f"{i+1}. {graph_id} ({graphs[graph_id].get('nodes', 0)} nodes, {graphs[graph_id].get('edges', 0)} edges)"
        )

    # Add option for dynamic graph
    if kg_manager.dynamic_kg:
        print(f"{len(graphs)+1}. dynamic (Dynamic knowledge graph)")

    # Get user choice
    try:
        choice = int(input("\nEnter graph number to visualize (0 to cancel): "))
        if choice == 0:
            return

        if choice <= len(graphs):
            graph_id = list(graphs.keys())[choice - 1]
        elif choice == len(graphs) + 1 and kg_manager.dynamic_kg:
            graph_id = "dynamic"
        else:
            print("Invalid choice.")
            return
    except ValueError:
        print("Invalid input.")
        return

    # Generate visualization
    print(f"\nGenerating visualization for graph {graph_id}...")
    html_content = kg_manager.visualize_graph(graph_id)

    # Save to file
    os.makedirs("temp", exist_ok=True)
    filename = f"temp/graph_{graph_id}_{int(time.time())}.html"
    with open(filename, "w") as f:
        f.write(html_content)

    print(f"\nVisualization saved to {filename}")
    print("Open this file in a web browser to view the graph.")

    # Wait for user to continue
    input("\nPress Enter to continue...")


async def run_performance_test():
    """Run performance benchmarking."""
    print("\n" + "-" * 50)
    print("PERFORMANCE BENCHMARKING")
    print("-" * 50)

    # Get number of concurrent users
    try:
        num_users = int(input("\nEnter number of concurrent users (1-10): "))
        num_users = max(1, min(10, num_users))
    except ValueError:
        print("Invalid input. Using default of 3 users.")
        num_users = 3

    # Get number of messages per user
    try:
        num_messages = int(input("Enter number of messages per user (1-10): "))
        num_messages = max(1, min(10, num_messages))
    except ValueError:
        print("Invalid input. Using default of 3 messages.")
        num_messages = 3

    print(
        f"\nRunning performance test with {num_users} concurrent users, {num_messages} messages each..."
    )

    # Create sessions
    session_ids = [conversation_manager.create_session() for _ in range(num_users)]

    # Define test messages
    test_messages = [
        "Tell me about artificial intelligence",
        "What are knowledge graphs?",
        "How does natural language processing work?",
        "Explain machine learning algorithms",
        "What is deep learning?",
        "How do neural networks work?",
        "What is transfer learning?",
        "Explain reinforcement learning",
        "What are the applications of AI?",
        "How is AI used in healthcare?",
    ]

    # Ensure we have enough messages
    while len(test_messages) < num_messages:
        test_messages.extend(test_messages)

    # Select messages for each user
    user_messages = [test_messages[:num_messages] for _ in range(num_users)]

    # Process messages concurrently
    async def process_user_messages(user_id, session_id, messages):
        results = []
        for i, message in enumerate(messages):
            start_time = time.time()
            response = ""
            try:
                async for token in conversation_manager.converse_with_context(session_id, message):
                    response += token
            except Exception as e:
                response = f"Error: {str(e)}"

            end_time = time.time()
            processing_time = end_time - start_time

            results.append(
                {
                    "user_id": user_id,
                    "message_id": i + 1,
                    "message": message,
                    "response_length": len(response),
                    "processing_time": processing_time,
                }
            )

            # Print progress
            print(f"User {user_id}, Message {i+1}: {processing_time:.2f} seconds")

        return results

    # Create tasks
    tasks = [
        process_user_messages(i + 1, sid, msgs)
        for i, (sid, msgs) in enumerate(zip(session_ids, user_messages, strict=False))
    ]

    # Run tasks concurrently and measure time
    start_time = time.time()
    all_results = await asyncio.gather(*tasks)
    end_time = time.time()

    # Flatten results
    results = [item for sublist in all_results for item in sublist]

    # Calculate statistics
    total_time = end_time - start_time
    total_messages = num_users * num_messages
    avg_time = sum(r["processing_time"] for r in results) / len(results)
    max_time = max(r["processing_time"] for r in results)
    min_time = min(r["processing_time"] for r in results)

    # Print results
    print("\nPERFORMANCE TEST RESULTS:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total messages: {total_messages}")
    print(f"Messages per second: {total_messages / total_time:.2f}")
    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Maximum processing time: {max_time:.2f} seconds")
    print(f"Minimum processing time: {min_time:.2f} seconds")

    # Print resource usage
    if system_monitor:
        resource_usage = system_monitor.get_resource_usage()
        print("\nRESOURCE USAGE:")
        print(f"CPU: {resource_usage['cpu_percent']}%")
        print(f"Memory: {resource_usage['memory_percent']}%")
        print(f"Memory used: {resource_usage['memory_used_gb']:.2f} GB")

    # Save results to file
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/performance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "num_users": num_users,
                    "num_messages": num_messages,
                    "total_messages": total_messages,
                },
                "results": {
                    "total_time": total_time,
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time,
                    "messages_per_second": total_messages / total_time,
                },
                "resource_usage": system_monitor.get_resource_usage() if system_monitor else {},
                "detailed_results": results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to {filename}")

    # Wait for user to continue
    input("\nPress Enter to continue...")


async def run_stability_test():
    """Run stability testing."""
    print("\n" + "-" * 50)
    print("STABILITY TESTING")
    print("-" * 50)

    # Get test duration
    try:
        duration_minutes = float(input("\nEnter test duration in minutes (0.5-30): "))
        duration_minutes = max(0.5, min(30, duration_minutes))
    except ValueError:
        print("Invalid input. Using default of 2 minutes.")
        duration_minutes = 2

    # Get message interval
    try:
        interval_seconds = float(input("Enter message interval in seconds (1-30): "))
        interval_seconds = max(1, min(30, interval_seconds))
    except ValueError:
        print("Invalid input. Using default of 5 seconds.")
        interval_seconds = 5

    duration_seconds = duration_minutes * 60
    estimated_messages = int(duration_seconds / interval_seconds)

    print(
        f"\nRunning stability test for {duration_minutes:.1f} minutes with {interval_seconds:.1f} second intervals"
    )
    print(f"Estimated messages: {estimated_messages}")

    # Create a session
    session_id = conversation_manager.create_session()

    # Define test messages
    test_messages = [
        "Tell me about artificial intelligence",
        "What are knowledge graphs?",
        "How does natural language processing work?",
        "Explain machine learning algorithms",
        "What is deep learning?",
        "How do neural networks work?",
        "What is transfer learning?",
        "Explain reinforcement learning",
        "What are the applications of AI?",
        "How is AI used in healthcare?",
    ]

    # Ensure we have enough messages
    while len(test_messages) < estimated_messages:
        test_messages.extend(test_messages)

    # Select messages for the test
    test_messages = test_messages[:estimated_messages]

    # Run the test
    start_time = time.time()
    end_time = start_time + duration_seconds

    results = []
    message_index = 0

    print("\nStarting stability test...")
    print("Press Ctrl+C to stop the test early")

    try:
        while time.time() < end_time and message_index < len(test_messages):
            # Get current message
            message = test_messages[message_index]

            # Process message
            message_start_time = time.time()
            print(f"\nMessage {message_index+1}: {message}")
            print("Response: ", end="", flush=True)

            response = ""
            try:
                async for token in conversation_manager.converse_with_context(session_id, message):
                    response += token
                    print(token, end="", flush=True)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                response = error_msg
                print(error_msg)

            message_end_time = time.time()
            processing_time = message_end_time - message_start_time

            # Record result
            results.append(
                {
                    "message_index": message_index,
                    "message": message,
                    "response_length": len(response),
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "error": None if "Error:" not in response else response,
                }
            )

            # Print resource usage
            if system_monitor:
                resource_usage = system_monitor.get_resource_usage()
                print(
                    f"\n[CPU: {resource_usage['cpu_percent']}%, Memory: {resource_usage['memory_percent']}%]"
                )

            # Print progress
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            print(
                f"\n[Progress: {elapsed:.1f}s / {duration_seconds:.1f}s, Remaining: {remaining:.1f}s]"
            )

            # Increment message index
            message_index += 1

            # Wait for next interval
            next_message_time = start_time + (message_index * interval_seconds)
            wait_time = max(0, next_message_time - time.time())
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n\nTest stopped by user")

    # Calculate statistics
    total_time = time.time() - start_time
    total_messages = len(results)
    successful_messages = sum(1 for r in results if r["error"] is None)
    error_messages = total_messages - successful_messages

    if total_messages > 0:
        avg_time = sum(r["processing_time"] for r in results) / total_messages
        max_time = max(r["processing_time"] for r in results)
        min_time = min(r["processing_time"] for r in results)
    else:
        avg_time = max_time = min_time = 0

    # Print results
    print("\nSTABILITY TEST RESULTS:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total messages: {total_messages}")
    print(f"Successful messages: {successful_messages}")
    print(f"Error messages: {error_messages}")
    print(f"Success rate: {successful_messages / total_messages * 100:.1f}%")
    print(f"Average processing time: {avg_time:.2f} seconds")
    print(f"Maximum processing time: {max_time:.2f} seconds")
    print(f"Minimum processing time: {min_time:.2f} seconds")

    # Print resource usage
    if system_monitor:
        resource_usage = system_monitor.get_resource_usage()
        print("\nFINAL RESOURCE USAGE:")
        print(f"CPU: {resource_usage['cpu_percent']}%")
        print(f"Memory: {resource_usage['memory_percent']}%")
        print(f"Memory used: {resource_usage['memory_used_gb']:.2f} GB")

    # Save results to file
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/stability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "duration_minutes": duration_minutes,
                    "interval_seconds": interval_seconds,
                    "estimated_messages": estimated_messages,
                },
                "results": {
                    "total_time": total_time,
                    "total_messages": total_messages,
                    "successful_messages": successful_messages,
                    "error_messages": error_messages,
                    "success_rate": (
                        successful_messages / total_messages if total_messages > 0 else 0
                    ),
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time,
                },
                "resource_usage": system_monitor.get_resource_usage() if system_monitor else {},
                "detailed_results": results,
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to {filename}")

    # Wait for user to continue
    input("\nPress Enter to continue...")


async def run_error_handling_test():
    """Run error handling test."""
    print("\n" + "-" * 50)
    print("ERROR HANDLING TEST")
    print("-" * 50)

    results = {"tests": [], "passed": 0, "failed": 0}

    # Test 1: Invalid session ID
    print("\nTest 1: Invalid session ID")
    invalid_session_id = "invalid_session_id"

    try:
        response = ""
        async for token in conversation_manager.converse_with_context(invalid_session_id, "Hello"):
            response += token

        print("❌ Failed: Expected an error but got a response")
        results["tests"].append(
            {
                "name": "Invalid session ID",
                "result": "failed",
                "details": "Expected an error but got a response",
            }
        )
        results["failed"] += 1
    except Exception as e:
        print(f"✅ Passed: Got expected error: {str(e)}")
        results["tests"].append(
            {
                "name": "Invalid session ID",
                "result": "passed",
                "details": f"Got expected error: {str(e)}",
            }
        )
        results["passed"] += 1

    # Test 2: Very large input
    print("\nTest 2: Very large input")
    session_id = conversation_manager.create_session()
    large_message = "test " * 1000  # Very large message

    try:
        response = ""
        async for token in conversation_manager.converse_with_context(session_id, large_message):
            response += token

        if response:
            print("✅ Passed: Handled large input gracefully")
            results["tests"].append(
                {
                    "name": "Very large input",
                    "result": "passed",
                    "details": "Handled large input gracefully",
                }
            )
            results["passed"] += 1
        else:
            print("❌ Failed: Got empty response")
            results["tests"].append(
                {"name": "Very large input", "result": "failed", "details": "Got empty response"}
            )
            results["failed"] += 1
    except Exception as e:
        print(f"❌ Failed: Got unexpected error: {str(e)}")
        results["tests"].append(
            {
                "name": "Very large input",
                "result": "failed",
                "details": f"Got unexpected error: {str(e)}",
            }
        )
        results["failed"] += 1

    # Test 3: Malformed query
    print("\nTest 3: Malformed query")
    malformed_query = "?" * 1000  # Very long nonsensical query

    try:
        # Query should not crash even with malformed input
        query_result = kg_manager.query_graphs(malformed_query)

        # Check if we got a response (even if it's an error message)
        if "response" in query_result:
            print("✅ Passed: Handled malformed query gracefully")
            results["tests"].append(
                {
                    "name": "Malformed query",
                    "result": "passed",
                    "details": "Handled malformed query gracefully",
                }
            )
            results["passed"] += 1
        else:
            print("❌ Failed: No response returned")
            results["tests"].append(
                {"name": "Malformed query", "result": "failed", "details": "No response returned"}
            )
            results["failed"] += 1
    except Exception as e:
        print(f"❌ Failed: Got unexpected error: {str(e)}")
        results["tests"].append(
            {
                "name": "Malformed query",
                "result": "failed",
                "details": f"Got unexpected error: {str(e)}",
            }
        )
        results["failed"] += 1

    # Test 4: Concurrent requests
    print("\nTest 4: Concurrent requests")

    try:
        # Create multiple sessions
        session_ids = [conversation_manager.create_session() for _ in range(5)]

        # Define messages for each session
        messages = ["Hello"] * 5

        # Process messages concurrently
        async def process_message(session_id, message):
            response = ""
            async for token in conversation_manager.converse_with_context(session_id, message):
                response += token
            return response

        # Create tasks
        tasks = [process_message(sid, msg) for sid, msg in zip(session_ids, messages, strict=False)]

        # Run tasks concurrently
        responses = await asyncio.gather(*tasks)

        # Check responses
        if all(len(response) > 0 for response in responses):
            print("✅ Passed: Handled concurrent requests gracefully")
            results["tests"].append(
                {
                    "name": "Concurrent requests",
                    "result": "passed",
                    "details": "Handled concurrent requests gracefully",
                }
            )
            results["passed"] += 1
        else:
            print("❌ Failed: Some responses were empty")
            results["tests"].append(
                {
                    "name": "Concurrent requests",
                    "result": "failed",
                    "details": "Some responses were empty",
                }
            )
            results["failed"] += 1
    except Exception as e:
        print(f"❌ Failed: Got unexpected error: {str(e)}")
        results["tests"].append(
            {
                "name": "Concurrent requests",
                "result": "failed",
                "details": f"Got unexpected error: {str(e)}",
            }
        )
        results["failed"] += 1

    # Print summary
    print("\nERROR HANDLING TEST RESULTS:")
    print(f"Total tests: {results['passed'] + results['failed']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")

    # Save results to file
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/error_handling_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)

    print(f"\nDetailed results saved to {filename}")

    # Wait for user to continue
    input("\nPress Enter to continue...")


# Main function
async def main():
    """Main function for interactive validation."""
    parser = argparse.ArgumentParser(description="Interactive validation for Milestone 6")
    parser.add_argument("--interactive", action="store_true", help="Run interactive validation")
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

    try:
        # Run interactive validation
        if args.interactive:
            await interactive_system_test()
        else:
            parser.print_help()
            return 0

        return 0

    finally:
        # Clean up resources
        if system_monitor:
            await system_monitor.stop_monitoring()

        # Additional cleanup if needed
        logger.info("Interactive validation completed")


if __name__ == "__main__":
    asyncio.run(main())
