"""
Model Manager for handling Ollama model operations.

This module provides the ModelManager class that handles initialization,
configuration, and communication with Ollama models using best practices
from the reference guide.
"""

import asyncio
import logging
import os
import sys

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import contextlib
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from ollama import AsyncClient


@dataclass
class ModelConfig:
    """Configuration for a model."""

    name: str
    type: str  # "conversation" or "knowledge_extraction"
    context_window: int = 8000
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    max_parallel: int = 2
    max_loaded: int = 2


class ModelManager:
    """
    Manages Ollama models with dual-model architecture.

    Uses persistent AsyncClient instances for production to reuse HTTP/2 connections.
    Implements resource management via environment variables and supports
    concurrent dual-model operations without blocking.
    """

    def __init__(self, host: str = "http://localhost:11434", healthcheck_mode: str = "version"):
        """Initialize ModelManager with Ollama host."""
        self.host = host
        self.healthcheck_mode = healthcheck_mode  # "version" or "chat"
        self.chat_client = AsyncClient(host=host)  # Hermes 3:3B - user-facing
        self.bg_client = AsyncClient(host=host)  # TinyLlama - background jobs

        # Use environment variables to allow overriding default models
        self.conversation_model = os.getenv("CONVERSATION_MODEL", "hermes3:3b")
        self.knowledge_model = os.getenv("KNOWLEDGE_MODEL", "tinyllama:latest")

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Performance metrics
        self.metrics = {
            "conversation": {
                "requests": 0,
                "tokens_generated": 0,
                "avg_response_time": 0,
                "last_response_time": 0,
            },
            "knowledge": {
                "requests": 0,
                "tokens_generated": 0,
                "avg_response_time": 0,
                "last_response_time": 0,
            },
        }

        # Resource management
        self._config = None
        self._background_tasks = set()
        self._is_under_load = False
        self._load_check_interval = 5.0  # seconds
        self._load_check_task = None
        self._max_background_tasks = 5

    async def _resolve_model_names(self) -> None:
        """Resolve actual model names from Ollama to handle tag variations."""
        try:
            models = await self.chat_client.list()
            available_models = []

            for m in models.get("models", []):
                model_name = m.get("model") or m.get("name", "")
                if model_name:
                    available_models.append(model_name)

            # Resolve conversation model
            if self.conversation_model not in available_models:
                # Try to find with different tags
                base_name = self.conversation_model.split(":")[0]
                for model in available_models:
                    if model.startswith(base_name + ":"):
                        self.logger.info(
                            f"Resolved conversation model: {self.conversation_model} -> {model}"
                        )
                        self.conversation_model = model
                        break

            # Resolve knowledge model
            if self.knowledge_model not in available_models:
                # Try to find with different tags
                base_name = self.knowledge_model.split(":")[0]
                for model in available_models:
                    if model.startswith(base_name + ":"):
                        self.logger.info(
                            f"Resolved knowledge model: {self.knowledge_model} -> {model}"
                        )
                        self.knowledge_model = model
                        break

        except Exception as e:
            self.logger.warning(f"Could not resolve model names: {e}")

    async def initialize_models(self, config: ModelConfig) -> bool:
        """
        Initialize Ollama models with proper configuration.

        Args:
            config: ModelConfig object with resource settings

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Set environment variables for resource management
            os.environ["OLLAMA_NUM_PARALLEL"] = str(config.max_parallel)
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(config.max_loaded)

            self.logger.info(
                f"Initializing models with max_parallel={config.max_parallel}, max_loaded={config.max_loaded}"
            )

            # Always use hermes3:3b and tinyllama:latest regardless of what's passed in config
            # This ensures we're using the models that are actually available
            self.conversation_model = "hermes3:3b"
            self.knowledge_model = "tinyllama:latest"
            self.logger.info(f"Using conversation model: {self.conversation_model}")
            self.logger.info(f"Using knowledge model: {self.knowledge_model}")

            # Store configuration for resource management
            self._config = config

            # Resolve actual model names from Ollama
            await self._resolve_model_names()

            # Configure resource allocation
            await self._configure_resource_allocation()

            # Test model availability based on healthcheck mode
            if self.healthcheck_mode == "version":
                # Light health check - just check if Ollama is running
                self.logger.info("Performing light health check (version)")
                try:
                    # Check if models exist in Ollama
                    models = await self.chat_client.list()
                    available_models = []

                    for m in models.get("models", []):
                        # Handle both old and new Ollama API formats
                        model_name = m.get("model") or m.get("name", "")
                        if model_name:
                            available_models.append(model_name)

                    if self.conversation_model not in available_models:
                        self.logger.warning(
                            f"Conversation model {self.conversation_model} not found in Ollama"
                        )
                    if self.knowledge_model not in available_models:
                        self.logger.warning(
                            f"Knowledge model {self.knowledge_model} not found in Ollama"
                        )

                    self.logger.info(f"Available models: {available_models}")
                except Exception as e:
                    self.logger.warning(f"Could not list models: {e}")
            else:
                # Heavy health check - actually test generation
                self.logger.info(f"Testing conversation model: {self.conversation_model}")
                await self.chat_client.chat(
                    model=self.conversation_model,
                    messages=[{"role": "user", "content": "test"}],
                    options={
                        "temperature": config.temperature,
                        "num_predict": 1,
                    },  # Minimal response for testing
                )

                self.logger.info(f"Testing knowledge model: {self.knowledge_model}")
                await self.bg_client.chat(
                    model=self.knowledge_model,
                    messages=[{"role": "user", "content": "test"}],
                    options={
                        "temperature": config.temperature,
                        "num_predict": 1,
                    },  # Minimal response for testing
                )

            self.logger.info("Both models initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            return False

    async def _configure_resource_allocation(self) -> None:
        """
        Configure resource allocation between models.

        This method implements resource isolation and priority management
        between the conversation and background models.
        """
        # Note: AsyncClient in newer Ollama versions may not support custom headers
        # We'll use other methods for resource isolation

        # Set resource limits for background model
        # This is a conceptual implementation - actual resource limits
        # would depend on the specific Ollama version and capabilities
        try:
            # Check if we can set model-specific resource limits
            # This is an example and may not work with all Ollama versions
            await self.bg_client.chat(
                model=self.knowledge_model,
                messages=[{"role": "system", "content": "test"}],
                options={
                    "num_ctx": min(2048, getattr(self._config, "context_window", 8000) // 2),
                    "num_thread": 2,  # Limit threads for background model
                    "num_predict": 1,  # Just for testing
                },
            )
            self.logger.info("Resource limits configured for background model")
        except Exception as e:
            # If setting resource limits fails, log but continue
            self.logger.warning(f"Could not set resource limits for background model: {str(e)}")
            self.logger.warning("Continuing with default resource allocation")

    async def query_conversation_model(
        self, messages: list[dict[str, str]], temperature: float = 0.7, max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from conversation model (Hermes 3:3B).

        Args:
            messages: List of message dictionaries with role and content
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate

        Yields:
            str: Response tokens as they are generated
        """
        start_time = time.time()
        token_count = 0

        # Import circuit breaker here to avoid circular imports
        from improved_local_assistant.services.circuit_breaker import CircuitBreakerOpenError
        from improved_local_assistant.services.circuit_breaker import with_circuit_breaker
        from improved_local_assistant.services.error_handler import handle_error

        try:
            # Define the actual query function
            async def _query_model():
                return await self.chat_client.chat(
                    model=self.conversation_model,
                    messages=messages,
                    stream=True,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )

            # Use circuit breaker pattern
            try:
                # Execute with circuit breaker protection
                stream = await with_circuit_breaker(
                    _query_model,
                    name=f"ollama_conversation_{self.conversation_model}",
                    failure_threshold=3,
                    recovery_timeout=10.0,
                )
            except CircuitBreakerOpenError as e:
                self.logger.error(f"Circuit breaker open for conversation model: {str(e)}")
                error_response = handle_error(
                    e, context={"model": self.conversation_model}, error_code="CIRCUIT_BREAKER_OPEN"
                )
                yield f"I'm sorry, the AI model is temporarily unavailable. {error_response['suggestion']}"
                return

            # Process the stream
            try:
                async for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        token_count += 1
                        yield content
            finally:
                # Ensure stream is properly closed to avoid generator issues
                if hasattr(stream, "aclose"):
                    try:
                        await stream.aclose()
                    except Exception:
                        pass  # Ignore cleanup errors

            # Update metrics
            elapsed = time.time() - start_time
            self._update_metrics("conversation", token_count, elapsed)

        except Exception as e:
            self.logger.error(f"Error querying conversation model: {str(e)}")

            # Import error handler
            from improved_local_assistant.services.error_handler import handle_error

            # Get user-friendly error message
            error_response = handle_error(
                e, context={"model": self.conversation_model}, error_code="MODEL_QUERY_ERROR"
            )

            yield f"I'm sorry, there was an issue with the AI model. {error_response['suggestion']}"

    async def query_knowledge_model(
        self, text: str, temperature: float = 0.2, max_tokens: int = 1024
    ) -> dict[str, Any]:
        """
        Query knowledge model (TinyLlama) for entity extraction.

        Args:
            text: Text to extract entities from
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate

        Returns:
            Dict: Response from the model
        """
        start_time = time.time()

        # Import circuit breaker and graceful degradation here to avoid circular imports
        from improved_local_assistant.services.circuit_breaker import CircuitBreakerOpenError
        from improved_local_assistant.services.circuit_breaker import with_circuit_breaker
        from improved_local_assistant.services.error_handler import handle_error
        from improved_local_assistant.services.graceful_degradation import degradation_manager
        from improved_local_assistant.services.graceful_degradation import with_degradation

        try:
            prompt = f"""Extract entities and relationships from the following text.
            Format the output as a list of triples (subject, relation, object):

            {text}

            Triples:"""

            # Define the actual query function
            async def _query_model():
                return await self.bg_client.chat(
                    model=self.knowledge_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature, "num_predict": max_tokens},
                )

            # Define fallback function for graceful degradation
            async def _fallback_query():
                # Simple fallback that extracts basic entities without relationships
                # This is less sophisticated but more reliable
                simple_prompt = (
                    f"Extract key entities (people, places, concepts) from this text: {text}"
                )

                try:
                    # Try with simplified prompt and lower max tokens
                    simple_response = await self.bg_client.chat(
                        model=self.knowledge_model,
                        messages=[{"role": "user", "content": simple_prompt}],
                        options={
                            "temperature": 0.1,  # Lower temperature for more deterministic results
                            "num_predict": 512,  # Lower token limit
                        },
                    )

                    content = simple_response["message"]["content"]
                    # Format as simple triples
                    entities = [entity.strip() for entity in content.split(",") if entity.strip()]
                    formatted_content = "\n".join(
                        [f"({entity}, mentioned_in, text)" for entity in entities]
                    )

                    return {
                        "content": formatted_content,
                        "model": self.knowledge_model,
                        "elapsed_time": time.time() - start_time,
                        "degraded": True,
                    }
                except Exception as e:
                    self.logger.error(f"Fallback query also failed: {str(e)}")
                    return {
                        "content": "No entities extracted.",
                        "model": self.knowledge_model,
                        "elapsed_time": time.time() - start_time,
                        "error": str(e),
                        "degraded": True,
                    }

            # Register fallback for graceful degradation
            degradation_manager.register_fallback("knowledge_extraction", _fallback_query)

            try:
                # Execute with circuit breaker protection
                response = await with_circuit_breaker(
                    _query_model,
                    name=f"ollama_knowledge_{self.knowledge_model}",
                    failure_threshold=3,
                    recovery_timeout=10.0,
                )

                # Update metrics
                elapsed = time.time() - start_time
                content = response["message"]["content"]
                token_count = len(content.split())
                self._update_metrics("knowledge", token_count, elapsed)

                return {"content": content, "model": self.knowledge_model, "elapsed_time": elapsed}

            except CircuitBreakerOpenError as e:
                self.logger.warning(
                    f"Circuit breaker open for knowledge model, using fallback: {str(e)}"
                )
                # Use graceful degradation
                return await with_degradation("knowledge_extraction", _query_model)

        except Exception as e:
            self.logger.error(f"Error querying knowledge model: {str(e)}")

            # Get user-friendly error information
            error_response = handle_error(
                e,
                context={"model": self.knowledge_model, "operation": "entity_extraction"},
                error_code="KNOWLEDGE_MODEL_ERROR",
            )

            return {
                "content": f"Error extracting entities: {error_response['message']}",
                "model": self.knowledge_model,
                "error": str(e),
                "suggestion": error_response["suggestion"],
            }

    def swap_model(self, client_type: str, new_model: str) -> bool:
        """
        Hot-swap model for a client.

        Args:
            client_type: "chat" or "background"
            new_model: Name of the new model to use

        Returns:
            bool: True if swap was successful
        """
        try:
            if client_type == "chat":
                # Update the model name for future requests
                self.conversation_model = new_model
                self.logger.info(f"Swapped conversation model to {new_model}")
            elif client_type == "background":
                # Update the model name for future requests
                self.knowledge_model = new_model
                self.logger.info(f"Swapped knowledge model to {new_model}")
            else:
                self.logger.error(f"Unknown client type: {client_type}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error swapping model: {str(e)}")
            return False

    async def get_model_status(self) -> dict[str, Any]:
        """
        Get current status of all models.

        Returns:
            Dict: Status information for all models
        """
        try:
            # Use Ollama API to get model list and status
            response = await self.chat_client.list()

            models = response.get("models", [])
            model_info = {}

            for model in models:
                model_name = model.get("name", "unknown")
                model_info[model_name] = {
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", ""),
                    "is_conversation_model": model_name == self.conversation_model,
                    "is_knowledge_model": model_name == self.knowledge_model,
                }

            return {
                "models": model_info,
                "conversation_model": self.conversation_model,
                "knowledge_model": self.knowledge_model,
                "metrics": self.metrics,
            }

        except Exception as e:
            self.logger.error(f"Error getting model status: {str(e)}")
            return {
                "error": str(e),
                "conversation_model": self.conversation_model,
                "knowledge_model": self.knowledge_model,
                "metrics": self.metrics,
            }

    async def run_concurrent_queries(
        self, user_message: str, priority_mode: bool = True
    ) -> tuple[AsyncGenerator[str, None], asyncio.Task]:
        """
        Run conversation and knowledge queries concurrently.

        This implements the dual-model architecture where the conversation
        model handles the user-facing responses while the knowledge model
        processes the message for entity extraction in the background.

        Args:
            user_message: User message to process
            priority_mode: If True, conversation model gets priority over background tasks

        Returns:
            Tuple: (conversation_stream, background_task)
        """
        # Create conversation stream
        messages = [{"role": "user", "content": user_message}]
        conversation_stream = self.query_conversation_model(messages)

        # Create background task for knowledge extraction with lower priority
        if priority_mode:
            # Set lower priority for background task using environment variables
            # This is a fire-and-forget pattern where we don't wait for the background task
            bg_task = asyncio.create_task(self._run_background_task(user_message))
            bg_task.set_name(f"background_knowledge_extraction_{int(time.time())}")

            # Register the background task for management
            self.register_background_task(bg_task)

            # Start load monitoring if not already started
            if self._load_check_task is None:
                asyncio.create_task(self.start_load_monitoring())
        else:
            # Run with equal priority if priority mode is disabled
            bg_task = asyncio.create_task(self.query_knowledge_model(user_message))
            bg_task.set_name(f"knowledge_extraction_{int(time.time())}")

        return conversation_stream, bg_task

    async def _run_background_task(self, text: str) -> dict[str, Any]:
        """
        Run a background task with resource isolation.

        This method implements the fire-and-forget pattern with proper resource
        management to ensure background tasks don't interfere with conversation
        responsiveness.

        Args:
            text: Text to process in the background

        Returns:
            Dict: Result from the background processing
        """
        try:
            # Use a separate event loop policy for background tasks if needed
            # This helps with resource isolation on some platforms

            # Set lower resource priority for background task
            original_parallel = os.environ.get("OLLAMA_NUM_PARALLEL", "2")
            original_loaded = os.environ.get("OLLAMA_MAX_LOADED_MODELS", "2")

            # Temporarily reduce resource allocation for background task
            os.environ["OLLAMA_NUM_PARALLEL"] = "1"

            # Execute the background task
            self.logger.debug("Starting background knowledge extraction task")
            result = await self.query_knowledge_model(text, temperature=0.2)

            # Restore original resource settings
            os.environ["OLLAMA_NUM_PARALLEL"] = original_parallel
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = original_loaded

            return result
        except Exception as e:
            self.logger.error(f"Background task error: {str(e)}")
            return {
                "content": f"Error in background task: {str(e)}",
                "model": self.knowledge_model,
                "error": str(e),
            }

    def _update_metrics(self, model_type: str, tokens: int, elapsed: float) -> None:
        """
        Update performance metrics for a model.

        Args:
            model_type: "conversation" or "knowledge"
            tokens: Number of tokens generated
            elapsed: Time elapsed in seconds
        """
        metrics = self.metrics.get(model_type, {})
        metrics["requests"] += 1
        metrics["tokens_generated"] += tokens
        metrics["last_response_time"] = elapsed

        # Update rolling average
        if metrics["requests"] > 1:
            metrics["avg_response_time"] = (
                metrics["avg_response_time"] * (metrics["requests"] - 1) + elapsed
            ) / metrics["requests"]
        else:
            metrics["avg_response_time"] = elapsed

    async def run_multiple_background_tasks(
        self, texts: list[str], max_concurrent: int = 2
    ) -> list[dict[str, Any]]:
        """
        Run multiple background tasks with controlled concurrency.

        This method implements resource management for multiple background tasks,
        ensuring they don't overwhelm system resources or block conversation tasks.

        Args:
            texts: List of texts to process in the background
            max_concurrent: Maximum number of concurrent background tasks

        Returns:
            List[Dict]: Results from all background tasks
        """
        self.logger.info(
            f"Starting {len(texts)} background tasks with max_concurrent={max_concurrent}"
        )

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded_task(text: str) -> dict[str, Any]:
            """Run a single task with semaphore bounds."""
            async with semaphore:
                return await self._run_background_task(text)

        # Create tasks with bounded concurrency
        tasks = [_bounded_task(text) for text in texts]

        # Execute all tasks and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, converting exceptions to error dictionaries
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Background task {i} failed: {str(result)}")
                processed_results.append(
                    {
                        "content": f"Error: {str(result)}",
                        "model": self.knowledge_model,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def start_load_monitoring(self):
        """
        Start monitoring system load to adaptively manage resources.

        This implements adaptive resource management to ensure conversation
        responsiveness even under high system load.
        """
        if self._load_check_task is None:
            self._load_check_task = asyncio.create_task(self._monitor_system_load())
            self.logger.info("Started system load monitoring")

    async def stop_load_monitoring(self):
        """Stop the system load monitoring."""
        if self._load_check_task:
            self._load_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._load_check_task
            self._load_check_task = None
            self.logger.info("Stopped system load monitoring")

    async def _monitor_system_load(self):
        """
        Monitor system load and adjust resource allocation.

        This background task periodically checks system load and adjusts
        resource allocation between conversation and background models.
        """
        import psutil

        while True:
            try:
                # Check CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent

                # Determine if system is under high load
                high_load = cpu_percent > 80 or memory_percent > 80

                if high_load and not self._is_under_load:
                    # System just went into high load state
                    self._is_under_load = True
                    self.logger.warning(
                        f"High system load detected: CPU {cpu_percent}%, Memory {memory_percent}%"
                    )
                    await self._adjust_for_high_load()
                elif not high_load and self._is_under_load:
                    # System recovered from high load
                    self._is_under_load = False
                    self.logger.info(
                        f"System load normalized: CPU {cpu_percent}%, Memory {memory_percent}%"
                    )
                    await self._adjust_for_normal_load()

                # Check and clean up completed background tasks
                self._cleanup_background_tasks()

                # Wait before next check
                await asyncio.sleep(self._load_check_interval)

            except Exception as e:
                self.logger.error(f"Error in load monitoring: {str(e)}")
                await asyncio.sleep(self._load_check_interval)

    async def _adjust_for_high_load(self):
        """
        Adjust resource allocation for high system load.

        This method implements priority management by reducing resources
        allocated to background tasks during high system load.
        """
        self.logger.info("Adjusting for high system load")

        # Pause or cancel non-critical background tasks
        for task in list(self._background_tasks):
            if not task.done() and task.get_name().startswith("background_"):
                task.cancel()
                self.logger.info(f"Cancelled background task {task.get_name()} due to high load")

        # Reduce resource allocation for background model
        os.environ["OLLAMA_NUM_PARALLEL"] = "1"

        # Prioritize conversation model
        await self._configure_resource_allocation()

    async def _adjust_for_normal_load(self):
        """
        Restore normal resource allocation after high load.

        This method restores the default resource allocation when
        system load returns to normal levels.
        """
        self.logger.info("Restoring normal resource allocation")

        # Restore original resource settings
        if self._config:
            os.environ["OLLAMA_NUM_PARALLEL"] = str(self._config.max_parallel)
            os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(self._config.max_loaded)

        # Reconfigure resource allocation
        await self._configure_resource_allocation()

    def _cleanup_background_tasks(self):
        """Clean up completed background tasks."""
        done_tasks = {task for task in self._background_tasks if task.done()}
        self._background_tasks -= done_tasks

        # Process any exceptions in completed tasks
        for task in done_tasks:
            if task.exception():
                self.logger.error(f"Background task {task.get_name()} failed: {task.exception()}")

    def register_background_task(self, task: asyncio.Task):
        """
        Register a background task for tracking and management.

        Args:
            task: The background task to register
        """
        self._background_tasks.add(task)

        # Set up callback to remove task when done
        def _on_task_done(t):
            if t in self._background_tasks:
                self._background_tasks.remove(t)

        task.add_done_callback(_on_task_done)

    async def cleanup(self):
        """Clean up resources and close clients."""
        try:
            if hasattr(self, "chat_client") and self.chat_client:
                await self.chat_client.aclose()
                self.logger.info("Chat client closed")
        except Exception as e:
            self.logger.warning(f"Error closing chat client: {e}")

        try:
            if hasattr(self, "bg_client") and self.bg_client:
                await self.bg_client.aclose()
                self.logger.info("Background client closed")
        except Exception as e:
            self.logger.warning(f"Error closing background client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
