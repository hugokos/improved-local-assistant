#!/usr/bin/env python
"""
Test script for the ModelManager class.

This script provides command-line functionality to test the ModelManager
implementation, including model initialization, streaming responses,
and background model querying. It also includes resource usage monitoring,
model switching, and error handling capabilities.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict

import psutil
from dotenv import load_dotenv

# Add parent directory to path to import from services
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resource usage during model operations."""

    def __init__(self, interval: float = 1.0):
        """Initialize the resource monitor.

        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_task = None
        self.process = psutil.Process()
        self.baseline = self._get_current_usage()
        self.peak = self.baseline.copy()
        self.current = self.baseline.copy()
        self.history = []

    def _get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        memory_info = self.process.memory_info()
        return {
            "cpu_percent": self.process.cpu_percent(),
            "memory_percent": self.process.memory_percent(),
            "rss_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # VMS in MB
            "timestamp": time.time(),
        }

    async def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self.current = self._get_current_usage()
                self.history.append(self.current)

                # Update peak values
                for key in self.peak:
                    if key != "timestamp" and self.current[key] > self.peak[key]:
                        self.peak[key] = self.current[key]

                # Keep history to a reasonable size (last 60 measurements)
                if len(self.history) > 60:
                    self.history = self.history[-60:]

                await asyncio.sleep(self.interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                await asyncio.sleep(self.interval)

    async def start(self):
        """Start resource monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.baseline = self._get_current_usage()
            self.peak = self.baseline.copy()
            self.history = [self.baseline]
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Resource monitoring started")

    async def stop(self):
        """Stop resource monitoring and return summary."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_task:
                await self.monitor_task
                self.monitor_task = None

            # Calculate final statistics
            summary = {
                "baseline": self.baseline,
                "peak": self.peak,
                "duration_seconds": self.peak["timestamp"] - self.baseline["timestamp"],
                "cpu_increase": self.peak["cpu_percent"] - self.baseline["cpu_percent"],
                "memory_increase_mb": (self.peak["rss_mb"] - self.baseline["rss_mb"]),
            }

            logger.info("Resource monitoring stopped")
            return summary
        return None

    def get_current_usage(self) -> Dict[str, float]:
        """Get the current resource usage."""
        return self.current

    def get_peak_usage(self) -> Dict[str, float]:
        """Get the peak resource usage."""
        return self.peak

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of resource usage."""
        if not self.history:
            return {}

        return {
            "baseline": self.baseline,
            "current": self.current,
            "peak": self.peak,
            "duration_seconds": self.current["timestamp"] - self.baseline["timestamp"],
            "cpu_increase": self.current["cpu_percent"] - self.baseline["cpu_percent"],
            "memory_increase_mb": (self.current["rss_mb"] - self.baseline["rss_mb"]),
        }


async def verify_installation():
    """Verify that Ollama is installed and models are available."""
    logger.info("Verifying Ollama installation and model availability...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    success = await model_manager.initialize_models(config)

    if success:
        logger.info("✅ Ollama installation verified successfully!")
        logger.info("✅ Required models are available!")

        # Get and display model status
        status = await model_manager.get_model_status()
        logger.info(f"Conversation model: {status['conversation_model']}")
        logger.info(f"Knowledge model: {status['knowledge_model']}")

        return True
    else:
        logger.error("❌ Failed to verify Ollama installation or models!")
        logger.error("Please ensure Ollama is installed and the required models are pulled.")
        logger.error("Run: ollama pull hermes:3b")
        logger.error("Run: ollama pull tinyllama")
        return False


async def test_streaming():
    """Test streaming responses from the conversation model."""
    logger.info("Testing streaming responses from the conversation model...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Test streaming response
    logger.info("Generating streaming response for test prompt...")
    messages = [{"role": "user", "content": "Write a short poem about AI assistants."}]

    print("\n--- Streaming Response ---")
    async for token in model_manager.query_conversation_model(messages):
        print(token, end="", flush=True)
    print("\n------------------------\n")

    # Get model status after streaming
    status = await model_manager.get_model_status()
    logger.info(f"Conversation metrics: {status['metrics']['conversation']}")

    logger.info("✅ Streaming test completed successfully!")
    return True


async def test_background_model():
    """Test background model for knowledge extraction."""
    logger.info("Testing background model for knowledge extraction...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Test knowledge extraction
    test_text = "The Hermes 3:3B model was developed by Nous Research. It runs efficiently on local hardware and is used for conversational AI."

    logger.info("Extracting knowledge from test text...")
    result = await model_manager.query_knowledge_model(test_text)

    print("\n--- Knowledge Extraction Result ---")
    print(result["content"])
    print(f"Elapsed time: {result['elapsed_time']:.2f} seconds")
    print("----------------------------------\n")

    # Get model status after extraction
    status = await model_manager.get_model_status()
    logger.info(f"Knowledge metrics: {status['metrics']['knowledge']}")

    logger.info("✅ Background model test completed successfully!")
    return True


async def test_concurrent_operation():
    """Test concurrent operation of both models."""
    logger.info("Testing concurrent operation of both models...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Test concurrent operation
    test_message = (
        "Explain how knowledge graphs can be used with LLMs for better context retrieval."
    )

    logger.info("Running concurrent queries...")
    conversation_stream, bg_task = await model_manager.run_concurrent_queries(test_message)

    print("\n--- Conversation Response ---")
    async for token in conversation_stream:
        print(token, end="", flush=True)
    print("\n---------------------------\n")

    # Wait for background task to complete
    bg_result = await bg_task

    print("\n--- Background Knowledge Extraction ---")
    print(bg_result["content"])
    print(f"Elapsed time: {bg_result['elapsed_time']:.2f} seconds")
    print("-------------------------------------\n")

    # Get model status after concurrent operation
    status = await model_manager.get_model_status()
    logger.info(f"Conversation metrics: {status['metrics']['conversation']}")
    logger.info(f"Knowledge metrics: {status['metrics']['knowledge']}")

    logger.info("✅ Concurrent operation test completed successfully!")
    return True


async def interactive_chat():
    """Run an interactive chat session with the model."""
    logger.info("Starting interactive chat session...")

    # Initialize resource monitor
    resource_monitor = ResourceMonitor()
    await resource_monitor.start()

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "2048"))

    # Initialize model manager with configuration from environment
    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(
        name="hermes3:3b",
        type="conversation",
        max_parallel=max_parallel,
        max_loaded=max_loaded,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Initialize models
    try:
        success = await model_manager.initialize_models(config)
        if not success:
            logger.error("❌ Failed to initialize models!")
            print("\nTrying to recover by checking Ollama service...")

            # Attempt recovery
            if await attempt_recovery():
                print("Recovery successful, retrying initialization...")
                success = await model_manager.initialize_models(config)
                if not success:
                    logger.error("❌ Recovery failed. Please check Ollama installation.")
                    await resource_monitor.stop()
                    return False
            else:
                logger.error("❌ Recovery failed. Please check Ollama installation.")
                await resource_monitor.stop()
                return False
    except Exception as e:
        logger.error(f"❌ Error during initialization: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Attempting recovery...")

        if await attempt_recovery():
            print("Recovery successful, retrying initialization...")
            try:
                success = await model_manager.initialize_models(config)
                if not success:
                    logger.error("❌ Recovery failed. Please check Ollama installation.")
                    await resource_monitor.stop()
                    return False
            except Exception as e2:
                logger.error(f"❌ Recovery failed: {str(e2)}")
                await resource_monitor.stop()
                return False
        else:
            logger.error("❌ Recovery failed. Please check Ollama installation.")
            await resource_monitor.stop()
            return False

    # Track conversation history
    conversation_history = []

    print("\n=== Interactive Chat Session ===")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'status' to see model status.")
    print("Type 'resources' to see resource usage.")
    print("Type 'switch model <model_name>' to switch the conversation model.")
    print("Type 'config <param> <value>' to change model configuration.")
    print("Type 'help' to see all available commands.")
    print("================================\n")

    while True:
        try:
            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                break

            # Handle special commands
            if user_input.lower() == "help":
                print_help_menu()
                continue

            elif user_input.lower() == "system":
                check_system_requirements()
                continue

            elif user_input.lower() == "status":
                status = await model_manager.get_model_status()
                print("\n--- Model Status ---")
                print(f"Conversation model: {status['conversation_model']}")
                print(f"Knowledge model: {status['knowledge_model']}")
                print(f"Conversation metrics: {status['metrics']['conversation']}")
                print(f"Knowledge metrics: {status['metrics']['knowledge']}")
                print("-------------------\n")
                continue

            elif user_input.lower() == "resources":
                usage = resource_monitor.get_usage_summary()
                print("\n--- Resource Usage ---")
                print(
                    f"CPU: {usage['current']['cpu_percent']:.1f}% (Peak: {usage['peak']['cpu_percent']:.1f}%)"
                )
                print(
                    f"Memory: {usage['current']['rss_mb']:.1f} MB (Peak: {usage['peak']['rss_mb']:.1f} MB)"
                )
                print(
                    f"Memory %: {usage['current']['memory_percent']:.1f}% (Peak: {usage['peak']['memory_percent']:.1f}%)"
                )
                print(f"Duration: {usage['duration_seconds']:.1f} seconds")
                print("---------------------\n")
                continue

            elif user_input.lower().startswith("switch model "):
                new_model = user_input[13:].strip()
                print(f"\nAttempting to switch conversation model to: {new_model}")

                try:
                    success = model_manager.swap_model("chat", new_model)
                    if success:
                        print(f"✅ Successfully switched to model: {new_model}")
                    else:
                        print(f"❌ Failed to switch to model: {new_model}")
                except Exception as e:
                    print(f"❌ Error switching model: {str(e)}")
                continue

            elif user_input.lower().startswith("config "):
                parts = user_input.split()
                if len(parts) >= 3:
                    param = parts[1].lower()
                    value = parts[2]

                    if param == "temperature":
                        try:
                            config.temperature = float(value)
                            print(f"✅ Temperature set to: {config.temperature}")
                        except ValueError:
                            print("❌ Invalid temperature value. Must be a float.")
                    elif param == "max_tokens":
                        try:
                            config.max_tokens = int(value)
                            print(f"✅ Max tokens set to: {config.max_tokens}")
                        except ValueError:
                            print("❌ Invalid max_tokens value. Must be an integer.")
                    else:
                        print(f"❌ Unknown configuration parameter: {param}")
                else:
                    print("❌ Invalid config command. Format: config <param> <value>")
                continue

            elif user_input.lower() == "list models":
                try:
                    status = await model_manager.get_model_status()
                    print("\n--- Available Models ---")
                    for model_name, model_info in status.get("models", {}).items():
                        print(f"- {model_name} ({model_info['size'] / (1024*1024):.1f} MB)")
                    print("----------------------\n")
                except Exception as e:
                    print(f"❌ Error listing models: {str(e)}")
                continue

            # Add message to history
            conversation_history.append({"role": "user", "content": user_input})

            print("\nAssistant: ", end="")

            # Start resource monitoring for this request
            request_start = time.time()
            baseline_usage = resource_monitor.get_current_usage()

            # Stream the response
            response_text = ""
            try:
                async for token in model_manager.query_conversation_model(
                    conversation_history,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                ):
                    print(token, end="", flush=True)
                    response_text += token
            except Exception as e:
                error_msg = f"\n\n❌ Error: {str(e)}"
                print(error_msg)
                response_text = error_msg

                # Attempt recovery if needed
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    print("\nAttempting to recover connection...")
                    if await attempt_recovery():
                        print("✅ Connection recovered")
                    else:
                        print("❌ Recovery failed")

            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response_text})

            # Show resource usage for this request
            current_usage = resource_monitor.get_current_usage()
            elapsed = time.time() - request_start
            cpu_delta = current_usage["cpu_percent"] - baseline_usage["cpu_percent"]
            memory_delta = current_usage["rss_mb"] - baseline_usage["rss_mb"]

            print(
                f"\n[Response time: {elapsed:.2f}s | CPU: +{cpu_delta:.1f}% | Memory: +{memory_delta:.1f} MB]"
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {str(e)}")
            logger.error(f"Unexpected error in interactive chat: {str(e)}")

    # Stop resource monitoring and show summary
    summary = await resource_monitor.stop()
    if summary:
        print("\n=== Resource Usage Summary ===")
        print(f"Peak CPU: {summary['peak']['cpu_percent']:.1f}%")
        print(f"Peak Memory: {summary['peak']['rss_mb']:.1f} MB")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print("============================\n")

    print("\nEnding chat session.")
    return True


def print_help_menu():
    """Print the help menu for interactive mode."""
    print("\n=== Available Commands ===")
    print("- exit, quit: End the session")
    print("- status: Show model status")
    print("- resources: Show resource usage")
    print("- switch model <model_name>: Switch conversation model")
    print("- config temperature <value>: Set temperature (0.0-1.0)")
    print("- config max_tokens <value>: Set max tokens for response")
    print("- list models: Show available models")
    print("- system: Show system information and requirements check")
    print("- help: Show this help menu")
    print("========================\n")


def check_system_requirements():
    """Check if the system meets the minimum requirements for running the models."""
    print("\n=== System Requirements Check ===")

    # Check CPU
    cpu_count = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count(logical=True)
    print(f"CPU: {cpu_count} physical cores, {cpu_logical} logical cores")
    if cpu_count < 4:
        print("⚠️  Warning: Recommended at least 4 physical CPU cores")
    else:
        print("✅ CPU: Sufficient cores available")

    # Check memory
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    print(f"Memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")

    min_memory_gb = float(os.getenv("MAX_MEMORY_GB", "12"))
    if total_gb < min_memory_gb:
        print(f"⚠️  Warning: System has {total_gb:.1f} GB RAM, recommended {min_memory_gb} GB")
    else:
        print("✅ Memory: Sufficient RAM available")

    # Check disk space
    disk = psutil.disk_usage("/")
    free_gb = disk.free / (1024**3)
    print(f"Disk: {free_gb:.1f} GB free space")
    if free_gb < 10:
        print("⚠️  Warning: Less than 10 GB free disk space")
    else:
        print("✅ Disk: Sufficient free space")

    # Check Ollama installation
    try:
        ollama_found = False
        for proc in psutil.process_iter(["pid", "name"]):
            if "ollama" in proc.info["name"].lower():
                ollama_found = True
                print(f"✅ Ollama: Running (PID: {proc.info['pid']})")
                break

        if not ollama_found:
            print("⚠️  Warning: Ollama process not found")
    except Exception as e:
        print(f"⚠️  Error checking Ollama process: {str(e)}")

    print("===============================\n")


async def attempt_recovery():
    """Attempt to recover from errors by checking Ollama service."""
    logger.info("Attempting recovery...")

    try:
        # Check if Ollama process is running
        for proc in psutil.process_iter(["pid", "name"]):
            if "ollama" in proc.info["name"].lower():
                logger.info(f"Found Ollama process: {proc.info}")
                return True

        # If not found, try to check if the service is responding
        from ollama import AsyncClient

        client = AsyncClient()
        await client.list()
        logger.info("Ollama service is responding")
        return True
    except Exception as e:
        logger.error(f"Recovery failed: {str(e)}")
        return False


async def test_error_handling():
    """Test error handling and recovery mechanisms."""
    logger.info("Testing error handling and recovery...")

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        return False

    # Test error handling with invalid model
    print("\n--- Testing Error Handling ---")
    print("1. Testing invalid model name...")

    try:
        model_manager.swap_model("chat", "nonexistent-model")
        messages = [{"role": "user", "content": "This should fail with an invalid model"}]
        async for token in model_manager.query_conversation_model(messages):
            print(token, end="", flush=True)
        print("\n❌ Expected an error but got a response!")
    except Exception as e:
        print(f"\n✅ Correctly caught error: {str(e)}")

    # Test recovery by switching back to a valid model
    print("\n2. Testing recovery by switching back to valid model...")

    try:
        success = model_manager.swap_model("chat", "hermes:3b")
        if success:
            print("✅ Successfully switched back to hermes:3b")

            # Test that the model works again
            messages = [{"role": "user", "content": "Test after recovery"}]
            print("\nResponse after recovery: ", end="")
            async for token in model_manager.query_conversation_model(messages):
                print(token, end="", flush=True)
            print("\n✅ Model recovered successfully!")
        else:
            print("❌ Failed to switch back to valid model")
    except Exception as e:
        print(f"\n❌ Error during recovery: {str(e)}")

    print("---------------------------\n")
    logger.info("Error handling test completed")
    return True


async def test_resource_monitoring():
    """Test resource monitoring during model operations."""
    logger.info("Testing resource monitoring...")

    # Initialize resource monitor
    resource_monitor = ResourceMonitor()
    await resource_monitor.start()

    # Get configuration from environment variables
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
    max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))

    model_manager = ModelManager(host=ollama_host)
    config = ModelConfig(name="test", type="test", max_parallel=max_parallel, max_loaded=max_loaded)

    # Initialize models
    success = await model_manager.initialize_models(config)
    if not success:
        logger.error("❌ Failed to initialize models!")
        await resource_monitor.stop()
        return False

    print("\n--- Testing Resource Monitoring ---")
    print("Running a series of model operations while monitoring resources...")

    # Baseline usage
    baseline = resource_monitor.get_current_usage()
    print(f"Baseline CPU: {baseline['cpu_percent']:.1f}% | Memory: {baseline['rss_mb']:.1f} MB")

    # Test 1: Simple query
    print("\n1. Testing simple query...")
    messages = [{"role": "user", "content": "Write a short poem about AI."}]

    print("Response: ", end="")
    async for token in model_manager.query_conversation_model(messages):
        print(token, end="", flush=True)

    # Check resource usage after simple query
    usage1 = resource_monitor.get_current_usage()
    print(
        f"\nAfter simple query - CPU: {usage1['cpu_percent']:.1f}% | Memory: {usage1['rss_mb']:.1f} MB"
    )
    print(
        f"Delta - CPU: {usage1['cpu_percent'] - baseline['cpu_percent']:.1f}% | Memory: {usage1['rss_mb'] - baseline['rss_mb']:.1f} MB"
    )

    # Test 2: Complex query
    print("\n2. Testing complex query...")
    messages = [
        {
            "role": "user",
            "content": "Explain quantum computing and its potential applications in AI in detail.",
        }
    ]

    print("Response: ", end="")
    async for token in model_manager.query_conversation_model(messages):
        print(token, end="", flush=True)

    # Check resource usage after complex query
    usage2 = resource_monitor.get_current_usage()
    print(
        f"\nAfter complex query - CPU: {usage2['cpu_percent']:.1f}% | Memory: {usage2['rss_mb']:.1f} MB"
    )
    print(
        f"Delta from baseline - CPU: {usage2['cpu_percent'] - baseline['cpu_percent']:.1f}% | Memory: {usage2['rss_mb'] - baseline['rss_mb']:.1f} MB"
    )

    # Test 3: Background knowledge extraction
    print("\n3. Testing background knowledge extraction...")
    result = await model_manager.query_knowledge_model(
        "The Hermes 3:3B model was developed by Nous Research. It runs efficiently on local hardware and is used for conversational AI."
    )

    print(f"Extraction result: {result['content']}")

    # Check resource usage after knowledge extraction
    usage3 = resource_monitor.get_current_usage()
    print(
        f"\nAfter knowledge extraction - CPU: {usage3['cpu_percent']:.1f}% | Memory: {usage3['rss_mb']:.1f} MB"
    )
    print(
        f"Delta from baseline - CPU: {usage3['cpu_percent'] - baseline['cpu_percent']:.1f}% | Memory: {usage3['rss_mb'] - baseline['rss_mb']:.1f} MB"
    )

    # Test 4: Concurrent operation
    print("\n4. Testing concurrent operation...")
    conversation_stream, bg_task = await model_manager.run_concurrent_queries(
        "Explain how knowledge graphs can be used with LLMs for better context retrieval."
    )

    print("Conversation response: ", end="")
    async for token in conversation_stream:
        print(token, end="", flush=True)

    bg_result = await bg_task
    print(f"\n\nBackground extraction: {bg_result['content']}")

    # Check resource usage after concurrent operation
    usage4 = resource_monitor.get_current_usage()
    print(
        f"\nAfter concurrent operation - CPU: {usage4['cpu_percent']:.1f}% | Memory: {usage4['rss_mb']:.1f} MB"
    )
    print(
        f"Delta from baseline - CPU: {usage4['cpu_percent'] - baseline['cpu_percent']:.1f}% | Memory: {usage4['rss_mb'] - baseline['rss_mb']:.1f} MB"
    )

    # Stop monitoring and get summary
    summary = await resource_monitor.stop()

    print("\n--- Resource Monitoring Summary ---")
    print(f"Peak CPU: {summary['peak']['cpu_percent']:.1f}%")
    print(f"Peak Memory: {summary['peak']['rss_mb']:.1f} MB")
    print(f"Duration: {summary['duration_seconds']:.1f} seconds")
    print("---------------------------------\n")

    logger.info("Resource monitoring test completed")
    return True


async def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test the ModelManager implementation")

    parser.add_argument(
        "--verify-installation",
        action="store_true",
        help="Verify Ollama installation and model availability",
    )
    parser.add_argument(
        "--test-streaming",
        action="store_true",
        help="Test streaming responses from the conversation model",
    )
    parser.add_argument(
        "--test-background",
        action="store_true",
        help="Test background model for knowledge extraction",
    )
    parser.add_argument(
        "--test-concurrent", action="store_true", help="Test concurrent operation of both models"
    )
    parser.add_argument(
        "--test-error-handling",
        action="store_true",
        help="Test error handling and recovery mechanisms",
    )
    parser.add_argument(
        "--test-resource-monitoring",
        action="store_true",
        help="Test resource monitoring during model operations",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive chat session with model switching and monitoring",
    )
    parser.add_argument("--all", action="store_true", help="Run all tests except interactive mode")

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Run tests based on arguments
    if args.verify_installation or args.all:
        await verify_installation()
        print()

    if args.test_streaming or args.all:
        await test_streaming()
        print()

    if args.test_background or args.all:
        await test_background_model()
        print()

    if args.test_concurrent or args.all:
        await test_concurrent_operation()
        print()

    if args.test_error_handling or args.all:
        await test_error_handling()
        print()

    if args.test_resource_monitoring or args.all:
        await test_resource_monitoring()
        print()

    if args.interactive:
        await interactive_chat()


if __name__ == "__main__":
    asyncio.run(main())
