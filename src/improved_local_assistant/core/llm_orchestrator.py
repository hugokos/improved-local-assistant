"""
LLM Orchestrator for edge-optimized turn-by-turn model coordination.

This module provides the LLMOrchestrator class that ensures only one LLM runs
at a time across the entire process, preventing CPU/memory contention while
maintaining optimal performance on edge devices.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Optional

# Fix nested async issues
try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    pass

from improved_local_assistant.services.connection_pool_manager import ConnectionPoolManager
from improved_local_assistant.services.graceful_degradation import ComponentStatus
from improved_local_assistant.services.graceful_degradation import degradation_manager
from improved_local_assistant.core.system_monitor import SystemMonitor


class LLMOrchestrator:
    """
    Orchestrates LLM execution with process-wide semaphore and turn-by-turn coordination.

    Ensures Hermes streams first, TinyLlama extracts after completion, with proper
    resource management and model lifecycle control.
    """

    # Process-wide semaphore to ensure only one LLM runs at a time
    _llm_semaphore = None

    @classmethod
    def _get_semaphore(cls):
        """Get or create the process-wide semaphore."""
        if cls._llm_semaphore is None:
            cls._llm_semaphore = asyncio.Semaphore(1)
        return cls._llm_semaphore

    def __init__(self, config: dict[str, Any], system_monitor: SystemMonitor):
        """Initialize LLM Orchestrator with configuration and system monitor."""
        self.config = config
        self.system_monitor = system_monitor
        self.logger = logging.getLogger(__name__)

        # Extract orchestration configuration
        orchestration_config = config.get("orchestration", {})
        self.semaphore_timeout = orchestration_config.get("llm_semaphore_timeout", 30.0)
        self.skip_on_pressure = orchestration_config.get("extraction_skip_on_pressure", True)
        self.keep_alive_hermes = orchestration_config.get("keep_alive_hermes", "30m")
        self.keep_alive_tinyllama = orchestration_config.get("keep_alive_tinyllama", 0)
        self.json_mode = orchestration_config.get("json_mode_for_extraction", True)
        self.max_extraction_time = orchestration_config.get("max_extraction_time", 8.0)

        # Model configuration
        models_config = config.get("models", {})
        self.conversation_model = models_config.get("conversation", {}).get("name", "hermes3:3b")
        self.knowledge_model = models_config.get("knowledge", {}).get("name", "tinyllama")

        # Connection pool manager
        self.connection_pool = ConnectionPoolManager(config)

        # Metrics
        self.metrics = {
            "turns_processed": 0,
            "extractions_completed": 0,
            "extractions_skipped": 0,
            "avg_turn_time": 0.0,
            "avg_extraction_time": 0.0,
        }

        # Active extraction task for cancellation
        self._active_extraction_task: Optional[asyncio.Task] = None

        # Pre-warming status
        self._hermes_prewarmed = False

        # Memory fallback configuration
        fallback_config = config.get("memory_fallback", {})
        self._memory_fallback_enabled = fallback_config.get("enabled", True)
        self._fallback_model = fallback_config.get("fallback_model", self.knowledge_model)
        self._proactive_threshold = fallback_config.get("proactive_threshold_percent", 98)
        self._memory_error_patterns = fallback_config.get(
            "error_patterns",
            [
                "model requires more system memory",
                "500 Internal Server Error",
                "out of memory",
                "insufficient memory",
            ],
        )
        self._auto_reset_minutes = fallback_config.get("auto_reset_after_minutes", 10)

    async def initialize(self) -> bool:
        """Initialize the orchestrator and pre-warm Hermes."""
        try:
            self.logger.info("Initializing LLM Orchestrator")

            # Initialize connection pool
            await self.connection_pool.initialize()

            # Initialize memory fallback tracking
            if self._memory_fallback_enabled:
                self.logger.info(
                    f"Memory fallback enabled: {self.conversation_model} -> {self._fallback_model}"
                )

            # Pre-warm Hermes
            await self._prewarm_hermes()

            self.logger.info("LLM Orchestrator initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize LLM Orchestrator: {e}")
            return False

    async def process_turn(
        self, session_id: str, user_message: str, conversation_history: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Process a complete turn with orchestrated LLM execution.

        Args:
            session_id: Session identifier
            user_message: User's message
            conversation_history: Previous conversation messages

        Yields:
            str: Response tokens from conversation model
        """
        turn_start_time = time.time()

        try:
            # Cancel any active extraction from previous turn
            await self._cancel_active_extraction()

            # Acquire the LLM semaphore with timeout
            semaphore = self._get_semaphore()
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=self.semaphore_timeout)
            except asyncio.TimeoutError:
                self.logger.error(
                    f"Failed to acquire LLM semaphore within {self.semaphore_timeout}s"
                )
                yield "I'm sorry, the system is currently busy. Please try again in a moment."
                return

            try:
                # Stream conversation response (Hermes)
                conversation_response = ""
                async for token in self._stream_conversation(conversation_history):
                    conversation_response += token
                    yield token

                # Release semaphore after conversation streaming completes
                semaphore.release()

                # Start background extraction (TinyLlama) - re-acquire semaphore
                extraction_text = f"User: {user_message}\nAssistant: {conversation_response}"
                self._active_extraction_task = asyncio.create_task(
                    self._extract_knowledge_background(extraction_text, session_id)
                )

                # Update metrics
                turn_time = time.time() - turn_start_time
                self.metrics["turns_processed"] += 1
                self._update_avg_metric("avg_turn_time", turn_time)

            except Exception as e:
                # Ensure semaphore is released on error
                semaphore.release()
                raise e

        except Exception as e:
            self.logger.error(f"Error in process_turn: {e}")
            yield f"I'm sorry, there was an error processing your request: {str(e)}"

    async def _stream_conversation(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from conversation model (Hermes) with memory-aware fallback.

        Args:
            messages: Conversation messages for the model

        Yields:
            str: Response tokens
        """
        try:
            self.logger.debug(f"Streaming conversation with {self.conversation_model}")

            # Primary function for streaming
            async def primary_stream():
                async for token in self.connection_pool.stream_chat(
                    model=self.conversation_model,
                    messages=messages,
                    keep_alive=self.keep_alive_hermes,
                    stream=True,
                ):
                    yield token

            # Check if we should use fallback due to memory threshold or previous issues
            if self._should_use_fallback():
                current_memory = self._get_current_memory_usage()
                if current_memory >= self._proactive_threshold:
                    reason = f"high memory usage ({current_memory:.1f}%)"
                else:
                    reason = "previous memory issues"

                self.logger.warning(f"Using fallback model due to {reason}")
                async for token in self._stream_with_fallback(messages, reason):
                    yield token
                return

            # Try primary model first
            try:
                async for token in primary_stream():
                    yield token

                # If successful, mark component as operational
                await degradation_manager.set_component_status(
                    "conversation_model", ComponentStatus.OPERATIONAL
                )

            except Exception as e:
                # Check if this is a memory-related error
                if self._is_memory_error(str(e)):
                    self.logger.warning(f"Memory error detected: {e}")
                    await degradation_manager.set_component_status(
                        "conversation_model", ComponentStatus.DEGRADED
                    )

                    # Use fallback model
                    async for token in self._stream_with_fallback(messages, "memory error"):
                        yield token
                else:
                    # Re-raise non-memory errors
                    raise e

        except Exception as e:
            self.logger.error(f"Error streaming conversation: {e}")
            await degradation_manager.set_component_status(
                "conversation_model", ComponentStatus.FAILED
            )
            yield f"Error generating response: {str(e)}"

    async def _extract_knowledge_background(
        self, text: str, session_id: str
    ) -> dict[str, Any] | None:
        """
        Extract knowledge in background after conversation completes.

        Args:
            text: Text to extract knowledge from
            session_id: Session identifier for context

        Returns:
            Optional[Dict]: Extraction results or None if skipped/failed
        """
        extraction_start_time = time.time()

        try:
            # Check if we should skip extraction due to resource pressure
            if self.should_skip_extraction():
                self.logger.info(
                    f"Skipping extraction for session {session_id} due to resource pressure"
                )
                self.metrics["extractions_skipped"] += 1
                return None

            # Acquire semaphore for extraction
            semaphore = self._get_semaphore()
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=self.semaphore_timeout)
            except asyncio.TimeoutError:
                self.logger.warning(f"Extraction timeout for session {session_id}")
                self.metrics["extractions_skipped"] += 1
                return None

            try:
                # Perform bounded extraction
                result = await self._extract_knowledge_bounded(text)

                # Unload TinyLlama after extraction
                await self._unload_tinyllama()

                # Update metrics
                extraction_time = time.time() - extraction_start_time
                self.metrics["extractions_completed"] += 1
                self._update_avg_metric("avg_extraction_time", extraction_time)

                return result

            finally:
                # Always release semaphore
                semaphore.release()

        except asyncio.CancelledError:
            self.logger.debug(f"Extraction cancelled for session {session_id}")
            self.metrics["extractions_skipped"] += 1
            # Re-raise to properly handle cancellation
            raise

        except Exception as e:
            self.logger.error(f"Error in background extraction: {e}")
            self.metrics["extractions_skipped"] += 1
            return None

    async def _extract_knowledge_bounded(self, text: str) -> dict[str, Any] | None:
        """
        Perform bounded knowledge extraction with time and token limits.

        Args:
            text: Text to extract from

        Returns:
            Optional[Dict]: Extraction results
        """
        try:
            # Create extraction prompt
            prompt = f"""Extract entities and relationships from the following text.
            Return ONLY a JSON array of triples in this exact format:
            [{{"subject": "entity1", "predicate": "relationship", "object": "entity2", "confidence": 0.9}}]

            Maximum 10 triples. Focus on the most important entities and relationships.

            Text: {text}

            JSON:"""

            # Prepare request options
            options = {
                "temperature": 0.2,
                "num_predict": 1024,  # Token limit
            }

            if self.json_mode:
                options["format"] = "json"

            # Perform extraction with timeout
            result = await asyncio.wait_for(
                self.connection_pool.chat_request(
                    model=self.knowledge_model,
                    messages=[{"role": "user", "content": prompt}],
                    keep_alive=self.keep_alive_tinyllama,
                    **options,
                ),
                timeout=self.max_extraction_time,
            )

            return {
                "content": result.get("message", {}).get("content", ""),
                "model": self.knowledge_model,
                "bounded": True,
            }

        except asyncio.TimeoutError:
            self.logger.warning(f"Extraction timed out after {self.max_extraction_time}s")
            return None

        except Exception as e:
            self.logger.error(f"Error in bounded extraction: {e}")
            return None

    async def _prewarm_hermes(self) -> None:
        """Pre-warm Hermes model by sending an empty request."""
        try:
            if self._hermes_prewarmed:
                return

            self.logger.info(f"Pre-warming {self.conversation_model}")

            # Send minimal request to load model
            await self.connection_pool.chat_request(
                model=self.conversation_model,
                messages=[{"role": "user", "content": "test"}],
                keep_alive=self.keep_alive_hermes,
                temperature=0.1,
                num_predict=1,
            )

            self._hermes_prewarmed = True
            self.logger.info(f"Successfully pre-warmed {self.conversation_model}")

        except Exception as e:
            self.logger.error(f"Failed to pre-warm Hermes: {e}")

    async def _unload_tinyllama(self) -> None:
        """Unload TinyLlama model to free resources."""
        try:
            self.logger.debug(f"Unloading {self.knowledge_model}")

            # Send request with keep_alive=0 to unload
            await self.connection_pool.chat_request(
                model=self.knowledge_model,
                messages=[{"role": "user", "content": "unload"}],
                keep_alive=0,
                num_predict=1,
            )

        except Exception as e:
            self.logger.debug(f"Error unloading TinyLlama (may already be unloaded): {e}")

    async def _cancel_active_extraction(self) -> None:
        """Cancel any active extraction task."""
        if self._active_extraction_task and not self._active_extraction_task.done():
            self.logger.debug("Cancelling active extraction task")
            self._active_extraction_task.cancel()

            try:
                await self._active_extraction_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            except Exception as e:
                self.logger.error(f"Error during extraction cancellation: {e}")

    def should_skip_extraction(self) -> bool:
        """
        Determine if extraction should be skipped based on resource pressure.

        Returns:
            bool: True if extraction should be skipped
        """
        if not self.skip_on_pressure:
            return False

        try:
            # Check memory pressure
            memory_threshold = self.config.get("system", {}).get("memory_threshold_percent", 95)
            current_memory = self.system_monitor.metrics.get("system", {}).get("memory_percent", 0)
            if current_memory > memory_threshold:
                return True

            # Check CPU pressure
            cpu_threshold = self.config.get("system", {}).get("cpu_threshold_percent", 90)
            current_cpu = self.system_monitor.metrics.get("system", {}).get("cpu_percent", 0)
            if current_cpu > cpu_threshold:
                return True

        except Exception as e:
            self.logger.debug(f"Error checking resource pressure: {e}")

        return False

    def _update_avg_metric(self, metric_name: str, new_value: float) -> None:
        """Update rolling average metric."""
        current_avg = self.metrics.get(metric_name, 0.0)
        count = self.metrics.get("turns_processed", 1)

        if count > 1:
            self.metrics[metric_name] = (current_avg * (count - 1) + new_value) / count
        else:
            self.metrics[metric_name] = new_value

    async def get_status(self) -> dict[str, Any]:
        """Get orchestrator status and metrics."""
        # Check model residency
        hermes_resident = await self.connection_pool.verify_model_residency(self.conversation_model)
        tinyllama_resident = await self.connection_pool.verify_model_residency(self.knowledge_model)

        # Get component status
        conversation_status = degradation_manager.get_component_status("conversation_model")

        semaphore = self._get_semaphore()
        # Get current memory usage
        current_memory = self._get_current_memory_usage()

        return {
            "semaphore_locked": semaphore.locked(),
            "hermes_prewarmed": self._hermes_prewarmed,
            "hermes_resident": hermes_resident,
            "tinyllama_resident": tinyllama_resident,
            "active_extraction": self._active_extraction_task is not None
            and not self._active_extraction_task.done(),
            "conversation_model_status": conversation_status.value,
            "fallback_enabled": self._memory_fallback_enabled,
            "fallback_model": self._fallback_model,
            "proactive_threshold": self._proactive_threshold,
            "current_memory_percent": current_memory,
            "memory_threshold_exceeded": current_memory >= self._proactive_threshold,
            "using_fallback": self._should_use_fallback(),
            "metrics": self.metrics.copy(),
            "degradation_status": degradation_manager.get_all_statuses(),
        }

    def _is_memory_error(self, error_message: str) -> bool:
        """
        Check if an error message indicates a memory-related issue.

        Args:
            error_message: Error message to check

        Returns:
            bool: True if this appears to be a memory error
        """
        error_lower = error_message.lower()
        return any(pattern.lower() in error_lower for pattern in self._memory_error_patterns)

    def _get_current_memory_usage(self) -> float:
        """
        Get current system memory usage percentage.

        Returns:
            float: Memory usage percentage (0-100)
        """
        try:
            import psutil

            return psutil.virtual_memory().percent
        except Exception as e:
            self.logger.debug(f"Error getting memory usage: {e}")
            return 0.0

    def _should_use_fallback(self) -> bool:
        """
        Determine if we should immediately use the fallback model.

        Returns:
            bool: True if fallback should be used
        """
        if not self._memory_fallback_enabled:
            return False

        # Check proactive memory threshold
        current_memory = self._get_current_memory_usage()
        if current_memory >= self._proactive_threshold:
            self.logger.warning(
                f"Memory usage {current_memory:.1f}% exceeds threshold {self._proactive_threshold}% - using fallback model"
            )
            return True

        # Check component status
        status = degradation_manager.get_component_status("conversation_model")
        return status in [ComponentStatus.DEGRADED, ComponentStatus.FAILED]

    async def _stream_with_fallback(
        self, messages: list[dict[str, str]], reason: str = "memory constraints"
    ) -> AsyncGenerator[str, None]:
        """
        Stream response using the fallback model.

        Args:
            messages: Conversation messages for the model
            reason: Reason for using fallback (for user notification)

        Yields:
            str: Response tokens from fallback model
        """
        try:
            self.logger.info(
                f"Streaming with fallback model: {self._fallback_model} (reason: {reason})"
            )

            # Add a prefix to indicate fallback is being used
            current_memory = self._get_current_memory_usage()
            if current_memory >= self._proactive_threshold:
                yield f"[Using lightweight model - memory usage at {current_memory:.1f}%]\n\n"
            else:
                yield f"[Using lightweight model due to {reason}]\n\n"

            # Stream with fallback model
            async for token in self.connection_pool.stream_chat(
                model=self._fallback_model,
                messages=messages,
                keep_alive=0,  # Don't keep fallback model loaded
                temperature=0.7,  # Use similar temperature to main model
                stream=True,
            ):
                yield token

        except Exception as e:
            self.logger.error(f"Fallback model also failed: {e}")
            yield "I'm sorry, both models are currently unavailable due to memory constraints. Please try again in a moment."

    async def reset_model_status(self) -> None:
        """Reset the conversation model status to allow retry of primary model."""
        await degradation_manager.set_component_status(
            "conversation_model", ComponentStatus.OPERATIONAL
        )
        self.logger.info(
            "Reset conversation model status - will retry primary model on next request"
        )

    async def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup resources."""
        try:
            self.logger.info("Shutting down LLM Orchestrator")

            # Cancel active extraction
            await self._cancel_active_extraction()

            # Shutdown connection pool
            await self.connection_pool.shutdown()

            self.logger.info("LLM Orchestrator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during orchestrator shutdown: {e}")
