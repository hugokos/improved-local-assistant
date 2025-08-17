#!/usr/bin/env python
"""
Interactive validation CLI for Milestone 1: Model Integration and Basic Testing.

This script provides a comprehensive interactive testing interface for model functionality,
including menu-driven testing for all model operations, resource monitoring, performance
validation, and clear pass/fail indicators for each test.

Usage:
    python cli/validate_milestone_1.py --interactive
"""

import argparse
import asyncio
import logging
import os
import sys
import time

import psutil

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from collections.abc import Awaitable
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# We're using the real Ollama implementation for testing
from improved_local_assistant.services import ModelConfig  # noqa: E402
from improved_local_assistant.services import ModelManager  # noqa: E402

# Load environment variables from .env file
load_dotenv()

# Configuration
USE_MOCK = False  # Using real Ollama implementation

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

    def _get_current_usage(self) -> dict[str, float]:
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

    def get_current_usage(self) -> dict[str, float]:
        """Get the current resource usage."""
        return self.current

    def get_peak_usage(self) -> dict[str, float]:
        """Get the peak resource usage."""
        return self.peak

    def get_usage_summary(self) -> dict[str, Any]:
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


class ValidationResult:
    """Store and display test validation results."""

    def __init__(self, name: str):
        """Initialize validation result.

        Args:
            name: Test name
        """
        self.name = name
        self.passed = False
        self.message = ""
        self.details = {}
        self.duration = 0.0
        self.timestamp = time.time()

    def set_passed(self, passed: bool, message: str = "", details: dict[str, Any] = None):
        """Set the test result.

        Args:
            passed: Whether the test passed
            message: Result message
            details: Additional details
        """
        self.passed = passed
        self.message = message
        self.details = details or {}

    def set_duration(self, duration: float):
        """Set the test duration.

        Args:
            duration: Test duration in seconds
        """
        self.duration = duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dict: Result as dictionary
        """
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "duration": self.duration,
            "timestamp": self.timestamp,
        }

    def display(self):
        """Display the test result."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        print(f"\n{status}: {self.name} ({self.duration:.2f}s)")
        if self.message:
            print(f"  {self.message}")
        if self.details:
            for key, value in self.details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")


class MilestoneValidator:
    """Interactive validator for Milestone 1."""

    def __init__(self):
        """Initialize the milestone validator."""
        self.model_manager = None
        self.resource_monitor = ResourceMonitor()
        self.results = {}
        self.config = None
        self.conversation_history = []

        # Load configuration from environment variables
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.max_parallel = int(os.getenv("OLLAMA_NUM_PARALLEL", "2"))
        self.max_loaded = int(os.getenv("OLLAMA_MAX_LOADED_MODELS", "2"))
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        self.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "2048"))

    async def initialize(self) -> bool:
        """Initialize the model manager and resource monitor.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Start resource monitoring
            await self.resource_monitor.start()

            # Initialize model manager
            self.model_manager = ModelManager(host=self.ollama_host)
            self.config = ModelConfig(
                name="hermes3:3b",
                type="conversation",
                max_parallel=self.max_parallel,
                max_loaded=self.max_loaded,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Initialize models
            success = await self.model_manager.initialize_models(self.config)
            if not success:
                print("❌ Failed to initialize models!")
                print(
                    "\nPlease ensure Ollama is installed and running, and the required models are pulled:"
                )
                print("1. Install Ollama from https://ollama.ai/")
                print("2. Start the Ollama service")
                print("3. Pull the required models:")
                print("   ollama pull hermes:3b")
                print("   ollama pull tinyllama")
                print("\nYou can also set custom model names using environment variables:")
                print("   CONVERSATION_MODEL=your-model-name")
                print("   KNOWLEDGE_MODEL=your-model-name")
                return False

            print("✅ Models initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error during initialization: {str(e)}")
            print(
                "\nPlease ensure Ollama is installed and running, and the required models are pulled:"
            )
            print("1. Install Ollama from https://ollama.ai/")
            print("2. Start the Ollama service")
            print("3. Pull the required models:")
            print("   ollama pull hermes:3b")
            print("   ollama pull tinyllama")
            return False

    async def cleanup(self):
        """Clean up resources."""
        # Stop resource monitoring
        summary = await self.resource_monitor.stop()
        if summary:
            print("\n=== Resource Usage Summary ===")
            print(f"Peak CPU: {summary['peak']['cpu_percent']:.1f}%")
            print(f"Peak Memory: {summary['peak']['rss_mb']:.1f} MB")
            print(f"Duration: {summary['duration_seconds']:.1f} seconds")
            print("============================\n")

    async def run_test(
        self, name: str, test_func: Callable[[], Awaitable[bool]]
    ) -> ValidationResult:
        """Run a test and record the result.

        Args:
            name: Test name
            test_func: Test function

        Returns:
            ValidationResult: Test result
        """
        result = ValidationResult(name)

        try:
            print(f"\n{'=' * 80}")
            print(f"Running Test: {name}")
            print(f"{'=' * 80}\n")

            start_time = time.time()
            success = await test_func()
            elapsed = time.time() - start_time

            result.set_duration(elapsed)

            if success:
                result.set_passed(True, "Test completed successfully")
            else:
                result.set_passed(False, "Test failed")
        except Exception as e:
            elapsed = time.time() - start_time
            result.set_duration(elapsed)
            result.set_passed(False, f"Error: {str(e)}")
            logger.error(f"Error in test {name}: {str(e)}")

        # Record result
        self.results[name] = result

        # Display result
        result.display()

        return result

    async def test_model_initialization(self) -> bool:
        """Test model initialization.

        Returns:
            bool: True if test passed
        """
        print("Testing model initialization...")

        try:
            # Get model status
            status = await self.model_manager.get_model_status()

            print(f"Conversation model: {status['conversation_model']}")
            print(f"Knowledge model: {status['knowledge_model']}")

            # Check available models
            if "models" in status:
                print("\nAvailable models:")
                for model_name, model_info in status["models"].items():
                    print(f"- {model_name} ({model_info.get('size', 0) / (1024*1024):.1f} MB)")

            return True
        except Exception as e:
            print(f"❌ Error getting model status: {str(e)}")
            return False

    async def test_streaming_response(self) -> bool:
        """Test streaming responses from the conversation model.

        Returns:
            bool: True if test passed
        """
        print("Testing streaming responses from the conversation model...")

        try:
            # Test streaming response
            messages = [{"role": "user", "content": "Write a short poem about AI assistants."}]

            print("\n--- Streaming Response ---")
            response_text = ""
            start_time = time.time()

            async for token in self.model_manager.query_conversation_model(messages):
                print(token, end="", flush=True)
                response_text += token

            elapsed = time.time() - start_time
            print(f"\n\nResponse generated in {elapsed:.2f} seconds")
            print("------------------------\n")

            # Get model status after streaming
            status = await self.model_manager.get_model_status()
            print(f"Conversation metrics: {status['metrics']['conversation']}")

            return len(response_text) > 0
        except Exception as e:
            print(f"❌ Error during streaming test: {str(e)}")
            return False

    async def test_background_model(self) -> bool:
        """Test background model for knowledge extraction.

        Returns:
            bool: True if test passed
        """
        print("Testing background model for knowledge extraction...")

        try:
            # Test knowledge extraction
            test_text = "The Hermes 3:3B model was developed by Nous Research. It runs efficiently on local hardware and is used for conversational AI."

            print("Extracting knowledge from test text...")
            start_time = time.time()
            result = await self.model_manager.query_knowledge_model(test_text)
            elapsed = time.time() - start_time

            print("\n--- Knowledge Extraction Result ---")
            print(result["content"])
            print(f"Elapsed time: {elapsed:.2f} seconds")
            print("----------------------------------\n")

            # Get model status after extraction
            status = await self.model_manager.get_model_status()
            print(f"Knowledge metrics: {status['metrics']['knowledge']}")

            return "content" in result and result["content"]
        except Exception as e:
            print(f"❌ Error during background model test: {str(e)}")
            return False

    async def test_concurrent_operation(self) -> bool:
        """Test concurrent operation of both models.

        Returns:
            bool: True if test passed
        """
        print("Testing concurrent operation of both models...")

        try:
            # Test concurrent operation
            test_message = (
                "Explain how knowledge graphs can be used with LLMs for better context retrieval."
            )

            print("Running concurrent queries...")
            start_time = time.time()
            conversation_stream, bg_task = await self.model_manager.run_concurrent_queries(
                test_message
            )

            print("\n--- Conversation Response ---")
            response_text = ""
            async for token in conversation_stream:
                print(token, end="", flush=True)
                response_text += token

            conversation_elapsed = time.time() - start_time
            print(f"\n\nConversation completed in {conversation_elapsed:.2f} seconds")
            print("---------------------------\n")

            # Wait for background task to complete
            try:
                bg_result = await asyncio.wait_for(bg_task, timeout=10)
                bg_elapsed = time.time() - start_time

                print("\n--- Background Knowledge Extraction ---")
                print(bg_result["content"])
                print(f"Background task completed in {bg_elapsed:.2f} seconds")
                print("-------------------------------------\n")

                # Check if background task took longer than conversation
                if bg_elapsed > conversation_elapsed:
                    print("✅ Background task completed after conversation (expected behavior)")
                else:
                    print("⚠️ Background task completed before conversation (unexpected)")
            except asyncio.TimeoutError:
                print("\n--- Background task is taking too long ---")
                print("This may be expected behavior for complex extraction tasks")
                print("Cancelling background task...")
                bg_task.cancel()
            except asyncio.CancelledError:
                print("\n--- Background task was cancelled ---")
                print("This demonstrates the fire-and-forget pattern where background tasks")
                print("can be cancelled if system resources are needed elsewhere.")

            # Get model status after concurrent operation
            status = await self.model_manager.get_model_status()
            print(f"Conversation metrics: {status['metrics']['conversation']}")
            print(f"Knowledge metrics: {status['metrics']['knowledge']}")

            return len(response_text) > 0
        except Exception as e:
            print(f"❌ Error during concurrent operation test: {str(e)}")
            return False

    async def test_fire_and_forget(self) -> bool:
        """Test fire-and-forget background task pattern.

        Returns:
            bool: True if test passed
        """
        print("Testing fire-and-forget background task pattern...")

        try:
            # Test fire-and-forget pattern
            test_message = "The TinyLlama model is efficient for background processing tasks."

            print("Running fire-and-forget background task...")
            conversation_stream, bg_task = await self.model_manager.run_concurrent_queries(
                test_message
            )

            print("\n--- Processing conversation without waiting for background task ---")
            response_text = ""
            async for token in conversation_stream:
                print(token, end="", flush=True)
                response_text += token
            print("\n----------------------------------------------------------\n")

            print("Conversation completed. Background task is still running...")
            print(f"Background task done: {bg_task.done()}")

            # Wait a bit to see background task progress
            await asyncio.sleep(2)
            print(f"After 2 seconds - Background task done: {bg_task.done()}")

            # Now wait for background task to complete or timeout after 5 seconds
            try:
                bg_result = await asyncio.wait_for(bg_task, timeout=5)

                print("\n--- Background Knowledge Extraction (completed after conversation) ---")
                print(bg_result["content"])
                print(f"Elapsed time: {bg_result['elapsed_time']:.2f} seconds")
                print("----------------------------------------------------------------\n")
            except asyncio.TimeoutError:
                print("\n--- Background task is taking too long ---")
                print("This may be expected behavior for complex extraction tasks")
                print("Cancelling background task...")
                bg_task.cancel()
            except asyncio.CancelledError:
                print("\n--- Background task was cancelled ---")
                print("This demonstrates the fire-and-forget pattern where background tasks")
                print("can be cancelled if system resources are needed elsewhere.")

            return len(response_text) > 0
        except Exception as e:
            print(f"❌ Error during fire-and-forget test: {str(e)}")
            return False

    async def test_resource_isolation(self) -> bool:
        """Test resource isolation between models.

        Returns:
            bool: True if test passed
        """
        print("Testing resource isolation between models...")

        try:
            # Test resource isolation with multiple concurrent tasks
            print("\n--- Testing resource isolation with multiple concurrent tasks ---")

            # Create multiple background tasks
            bg_tasks = []
            for i in range(3):
                text = f"Topic {i+1} is about testing resource isolation between multiple model instances."
                task = asyncio.create_task(self.model_manager.query_knowledge_model(text))
                bg_tasks.append(task)

            # While background tasks are running, run a conversation task
            messages = [
                {"role": "user", "content": "Write a short paragraph about resource isolation."}
            ]

            print("Running conversation task while background tasks are processing...")
            response_text = ""
            start_time = time.time()

            async for token in self.model_manager.query_conversation_model(messages):
                print(token, end="", flush=True)
                response_text += token

            conversation_elapsed = time.time() - start_time
            print(f"\n\nConversation completed in {conversation_elapsed:.2f} seconds")
            print("----------------------------------------------------------------------\n")

            # Wait for all background tasks to complete or timeout after 10 seconds
            try:
                results = await asyncio.wait_for(asyncio.gather(*bg_tasks), timeout=10)

                print("\n--- Background Task Results ---")
                for i, result in enumerate(results):
                    print(f"Task {i+1} completed in {result['elapsed_time']:.2f} seconds")
                print("-----------------------------\n")
            except asyncio.TimeoutError:
                print("\n--- Some background tasks are taking too long ---")
                print("Cancelling remaining background tasks...")
                for task in bg_tasks:
                    if not task.done():
                        task.cancel()

            return len(response_text) > 0
        except Exception as e:
            print(f"❌ Error during resource isolation test: {str(e)}")
            return False

    async def test_non_blocking(self) -> bool:
        """Test that operations don't block each other.

        Returns:
            bool: True if test passed
        """
        print("Testing non-blocking operations...")

        try:
            # Start a long-running background task
            print("\n--- Testing non-blocking operations ---")
            print("Starting a long-running task...")

            long_text = (
                "Write a detailed essay about the history of artificial intelligence, "
                "including major milestones, key researchers, and future directions."
            )
            messages = [{"role": "user", "content": long_text}]

            # Start the long-running task as a background task
            long_task = asyncio.create_task(
                self._collect_stream(self.model_manager.query_conversation_model(messages))
            )

            # Start a short task immediately after
            print("Starting a short task while long task is running...")
            short_messages = [{"role": "user", "content": "What is 2+2?"}]
            short_task_start = time.time()

            short_response = ""
            async for token in self.model_manager.query_conversation_model(short_messages):
                print(token, end="", flush=True)
                short_response += token

            short_task_duration = time.time() - short_task_start

            print(f"\n\nShort task completed in {short_task_duration:.2f} seconds")
            print("--------------------------\n")

            # Clean up the long-running task by consuming its output
            print("Now waiting for long task to complete...")

            try:
                # Wait for the long task with a timeout
                long_response = await asyncio.wait_for(long_task, timeout=10)

                print("\n--- Long Task Completed ---")
                print(f"Long response length: {len(long_response)} characters")
                print("-------------------------\n")
            except asyncio.TimeoutError:
                print("\n--- Long task is taking too long ---")
                print("Cancelling long task...")
                long_task.cancel()

            # Check if short task was significantly faster than the long task would be
            return len(short_response) > 0 and short_task_duration < 10.0
        except Exception as e:
            print(f"❌ Error during non-blocking test: {str(e)}")
            return False

    async def _collect_stream(self, stream):
        """Helper method to collect all tokens from a stream."""
        result = ""
        async for token in stream:
            result += token
        return result

    async def test_performance_benchmarking(self) -> bool:
        """Test performance benchmarking.

        Returns:
            bool: True if test passed
        """
        print("Running performance benchmarking...")

        try:
            # Define benchmark prompts of increasing complexity
            benchmarks = [
                {"name": "Simple Query", "prompt": "What is the capital of France?"},
                {
                    "name": "Medium Query",
                    "prompt": "Explain the concept of knowledge graphs in a few sentences.",
                },
                {
                    "name": "Complex Query",
                    "prompt": "Compare and contrast transformer models with recurrent neural networks, including their strengths and weaknesses for different NLP tasks.",
                },
            ]

            results = []

            for benchmark in benchmarks:
                print(f"\n--- Benchmarking: {benchmark['name']} ---")
                messages = [{"role": "user", "content": benchmark["prompt"]}]

                # Measure response time
                start_time = time.time()
                response_text = ""
                token_count = 0

                print("Response: ", end="")
                async for token in self.model_manager.query_conversation_model(messages):
                    print(token, end="", flush=True)
                    response_text += token
                    token_count += 1

                elapsed = time.time() - start_time
                tokens_per_second = token_count / elapsed if elapsed > 0 else 0

                print(f"\n\nGenerated {token_count} tokens in {elapsed:.2f} seconds")
                print(f"Speed: {tokens_per_second:.2f} tokens/second")

                # Record benchmark result
                results.append(
                    {
                        "name": benchmark["name"],
                        "elapsed_seconds": elapsed,
                        "token_count": token_count,
                        "tokens_per_second": tokens_per_second,
                    }
                )

                # Get resource usage
                usage = self.resource_monitor.get_current_usage()
                print(f"CPU: {usage['cpu_percent']:.1f}% | Memory: {usage['rss_mb']:.1f} MB")

                # Wait a bit between benchmarks
                await asyncio.sleep(2)

            # Display summary
            print("\n--- Benchmark Summary ---")
            for result in results:
                print(
                    f"{result['name']}: {result['elapsed_seconds']:.2f}s, {result['tokens_per_second']:.2f} tokens/s"
                )
            print("------------------------\n")

            return True
        except Exception as e:
            print(f"❌ Error during performance benchmarking: {str(e)}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling and recovery mechanisms.

        Returns:
            bool: True if test passed
        """
        print("Testing error handling and recovery...")

        try:
            # Test error handling with invalid model
            print("\n--- Testing Error Handling ---")
            print("1. Testing invalid model name...")

            try:
                self.model_manager.swap_model("chat", "nonexistent-model")
                messages = [{"role": "user", "content": "This should fail with an invalid model"}]

                response_text = ""
                async for token in self.model_manager.query_conversation_model(messages):
                    print(token, end="", flush=True)
                    response_text += token

                print("\n❌ Expected an error but got a response!")
                error_handled = False
            except Exception as e:
                print(f"\n✅ Correctly caught error: {str(e)}")
                error_handled = True

            # Test recovery by switching back to a valid model
            print("\n2. Testing recovery by switching back to valid model...")

            try:
                success = self.model_manager.swap_model("chat", "hermes3:3b")
                if success:
                    print("✅ Successfully switched back to hermes3:3b")

                    # Test that the model works again
                    messages = [{"role": "user", "content": "Test after recovery"}]
                    print("\nResponse after recovery: ", end="")

                    response_text = ""
                    async for token in self.model_manager.query_conversation_model(messages):
                        print(token, end="", flush=True)
                        response_text += token

                    print("\n✅ Model recovered successfully!")
                    recovery_success = True
                else:
                    print("❌ Failed to switch back to valid model")
                    recovery_success = False
            except Exception as e:
                print(f"\n❌ Error during recovery: {str(e)}")
                recovery_success = False

            print("---------------------------\n")

            return error_handled and recovery_success
        except Exception as e:
            print(f"❌ Error during error handling test: {str(e)}")
            return False

    async def test_system_requirements(self) -> bool:
        """Check if the system meets the minimum requirements.

        Returns:
            bool: True if system meets requirements
        """
        print("\n=== System Requirements Check ===")

        requirements_met = True

        # Check CPU
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        print(f"CPU: {cpu_count} physical cores, {cpu_logical} logical cores")
        if cpu_count < 4:
            print("⚠️  Warning: Recommended at least 4 physical CPU cores")
            requirements_met = False
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
            requirements_met = False
        else:
            print("✅ Memory: Sufficient RAM available")

        # Check disk space
        disk = psutil.disk_usage("/")
        free_gb = disk.free / (1024**3)
        print(f"Disk: {free_gb:.1f} GB free space")
        if free_gb < 10:
            print("⚠️  Warning: Less than 10 GB free disk space")
            requirements_met = False
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

            if not ollama_found and not USE_MOCK:
                print("⚠️  Warning: Ollama process not found")
                requirements_met = False
        except Exception as e:
            print(f"⚠️  Error checking Ollama process: {str(e)}")

        print("===============================\n")

        return requirements_met

    async def interactive_chat(self) -> bool:
        """Run an interactive chat session with the model.

        Returns:
            bool: True if chat session completed successfully
        """
        print("\n=== Interactive Chat Session ===")
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'status' to see model status.")
        print("Type 'resources' to see resource usage.")
        print("Type 'switch model <model_name>' to switch the conversation model.")
        print("Type 'help' to see all available commands.")
        print("================================\n")

        # Track conversation history
        self.conversation_history = []

        while True:
            try:
                user_input = input("\nYou: ")

                if user_input.lower() in ["exit", "quit"]:
                    break

                # Handle special commands
                if user_input.lower() == "help":
                    self._print_help_menu()
                    continue

                elif user_input.lower() == "status":
                    status = await self.model_manager.get_model_status()
                    print("\n--- Model Status ---")
                    print(f"Conversation model: {status['conversation_model']}")
                    print(f"Knowledge model: {status['knowledge_model']}")
                    print(f"Conversation metrics: {status['metrics']['conversation']}")
                    print(f"Knowledge metrics: {status['metrics']['knowledge']}")
                    print("-------------------\n")
                    continue

                elif user_input.lower() == "resources":
                    usage = self.resource_monitor.get_usage_summary()
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
                        success = self.model_manager.swap_model("chat", new_model)
                        if success:
                            print(f"✅ Successfully switched to model: {new_model}")
                        else:
                            print(f"❌ Failed to switch to model: {new_model}")
                    except Exception as e:
                        print(f"❌ Error switching model: {str(e)}")
                    continue

                elif user_input.lower() == "list models":
                    try:
                        status = await self.model_manager.get_model_status()
                        print("\n--- Available Models ---")
                        for model_name, model_info in status.get("models", {}).items():
                            print(f"- {model_name} ({model_info['size'] / (1024*1024):.1f} MB)")
                        print("----------------------\n")
                    except Exception as e:
                        print(f"❌ Error listing models: {str(e)}")
                    continue

                # Add message to history
                self.conversation_history.append({"role": "user", "content": user_input})

                print("\nAssistant: ", end="")

                # Start resource monitoring for this request
                request_start = time.time()
                baseline_usage = self.resource_monitor.get_current_usage()

                # Stream the response
                response_text = ""
                try:
                    async for token in self.model_manager.query_conversation_model(
                        self.conversation_history,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    ):
                        print(token, end="", flush=True)
                        response_text += token
                except Exception as e:
                    error_msg = f"\n\n❌ Error: {str(e)}"
                    print(error_msg)
                    response_text = error_msg

                # Add assistant response to history
                self.conversation_history.append({"role": "assistant", "content": response_text})

                # Show resource usage for this request
                current_usage = self.resource_monitor.get_current_usage()
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

        print("\nEnding chat session.")
        return True

    def _print_help_menu(self):
        """Print the help menu for interactive mode."""
        print("\n=== Available Commands ===")
        print("- exit, quit: End the session")
        print("- status: Show model status")
        print("- resources: Show resource usage")
        print("- switch model <model_name>: Switch conversation model")
        print("- list models: Show available models")
        print("- help: Show this help menu")
        print("========================\n")

    async def run_interactive_validation(self):
        """Run the interactive validation menu."""
        if not await self.initialize():
            print("❌ Failed to initialize. Exiting.")
            return

        # Check system requirements
        await self.test_system_requirements()

        while True:
            try:
                print("\n=== Milestone 1 Validation Menu ===")
                print("1. Test Model Initialization")
                print("2. Test Streaming Response")
                print("3. Test Background Model")
                print("4. Test Concurrent Operation")
                print("5. Test Fire-and-Forget Pattern")
                print("6. Test Resource Isolation")
                print("7. Test Non-Blocking Operations")
                print("8. Test Performance Benchmarking")
                print("9. Test Error Handling")
                print("10. Interactive Chat Session")
                print("11. Run All Tests")
                print("12. Show Test Results Summary")
                print("0. Exit")
                print("================================\n")

                choice = input("Enter your choice (0-12): ")

                if choice == "0":
                    break
                elif choice == "1":
                    await self.run_test("Model Initialization", self.test_model_initialization)
                elif choice == "2":
                    await self.run_test("Streaming Response", self.test_streaming_response)
                elif choice == "3":
                    await self.run_test("Background Model", self.test_background_model)
                elif choice == "4":
                    await self.run_test("Concurrent Operation", self.test_concurrent_operation)
                elif choice == "5":
                    await self.run_test("Fire-and-Forget Pattern", self.test_fire_and_forget)
                elif choice == "6":
                    await self.run_test("Resource Isolation", self.test_resource_isolation)
                elif choice == "7":
                    await self.run_test("Non-Blocking Operations", self.test_non_blocking)
                elif choice == "8":
                    await self.run_test(
                        "Performance Benchmarking", self.test_performance_benchmarking
                    )
                elif choice == "9":
                    await self.run_test("Error Handling", self.test_error_handling)
                elif choice == "10":
                    await self.interactive_chat()
                elif choice == "11":
                    await self.run_all_tests()
                elif choice == "12":
                    self.show_results_summary()
                else:
                    print("Invalid choice. Please try again.")

                # Wait for user to press Enter before continuing
                input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"\n\n❌ Unexpected error: {str(e)}")
                logger.error(f"Unexpected error in menu: {str(e)}")

                # Wait for user to press Enter before continuing
                input("\nPress Enter to continue...")

        # Clean up resources
        await self.cleanup()

    async def run_all_tests(self):
        """Run all tests in sequence."""
        tests = [
            ("Model Initialization", self.test_model_initialization),
            ("Streaming Response", self.test_streaming_response),
            ("Background Model", self.test_background_model),
            ("Concurrent Operation", self.test_concurrent_operation),
            ("Fire-and-Forget Pattern", self.test_fire_and_forget),
            ("Resource Isolation", self.test_resource_isolation),
            ("Non-Blocking Operations", self.test_non_blocking),
            ("Performance Benchmarking", self.test_performance_benchmarking),
            ("Error Handling", self.test_error_handling),
        ]

        for name, test_func in tests:
            await self.run_test(name, test_func)

        # Show summary
        self.show_results_summary()

    def show_results_summary(self):
        """Show a summary of all test results."""
        if not self.results:
            print("\nNo test results available.")
            return

        print("\n=== Test Results Summary ===")

        passed = 0
        failed = 0

        for name, result in self.results.items():
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"{status}: {name} ({result.duration:.2f}s)")

            if result.passed:
                passed += 1
            else:
                failed += 1

        print(f"\nTotal: {len(self.results)} tests, {passed} passed, {failed} failed")
        print("==========================\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Interactive validation CLI for Milestone 1: Model Integration and Basic Testing"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive validation menu"
    )

    args = parser.parse_args()

    # Default to interactive mode if no arguments are provided
    if not args.interactive:
        args.interactive = True

    async def run_validation():
        validator = MilestoneValidator()

        if args.interactive:
            await validator.run_interactive_validation()
        else:
            # Initialize
            if await validator.initialize():
                # Run all tests
                await validator.run_all_tests()
                # Clean up
                await validator.cleanup()
            else:
                print("❌ Failed to initialize. Exiting.")

    asyncio.run(run_validation())


if __name__ == "__main__":
    main()
