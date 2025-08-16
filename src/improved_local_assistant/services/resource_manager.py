"""
Resource management and performance optimization.

This module provides mechanisms for monitoring and optimizing resource usage,
including memory management, conversation summarization, and adaptive resource allocation.
"""

import asyncio
import contextlib
import gc
import logging
import os
import sys
import time
from collections.abc import Callable
from typing import Any

import psutil

# Configure logging
logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages system resources and optimizes performance.

    This class monitors resource usage, implements adaptive resource allocation,
    and provides mechanisms for optimizing performance under various conditions.
    """

    def __init__(self, config=None):
        """
        Initialize the resource manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Extract configuration values
        system_config = self.config.get("system", {})
        self.max_memory_gb = system_config.get("max_memory_gb", 12)
        self.cpu_cores = system_config.get("cpu_cores", psutil.cpu_count())
        self.memory_threshold_percent = system_config.get("memory_threshold_percent", 80)
        self.cpu_threshold_percent = system_config.get("cpu_threshold_percent", 80)
        self.quiet_monitoring = system_config.get("quiet_monitoring", False)

        # Resource allocation settings
        self.conversation_memory_percent = system_config.get("conversation_memory_percent", 60)
        self.knowledge_memory_percent = system_config.get("knowledge_memory_percent", 30)
        self.system_memory_percent = system_config.get("system_memory_percent", 10)

        # Performance metrics
        self.metrics = {
            "memory_cleanup_count": 0,
            "last_cleanup_time": None,
            "high_load_events": 0,
            "last_high_load_time": None,
            "adaptive_actions": 0,
        }

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task = None
        self.check_interval = 5.0  # seconds

        # Callbacks for resource-intensive components
        self.memory_pressure_callbacks = []

        logger.debug("Resource manager initialized")

    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.is_monitoring:
            logger.warning("Resource monitoring is already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.debug("Resource monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self.is_monitoring:
            logger.warning("Resource monitoring is not running")
            return

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task
            self.monitoring_task = None

        logger.info("Resource monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background task for monitoring resources."""
        while self.is_monitoring:
            try:
                # Check resource usage
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Check for high memory usage
                if memory_percent > self.memory_threshold_percent:
                    if not self.quiet_monitoring:
                        logger.warning(f"High memory usage detected: {memory_percent}%")
                    else:
                        logger.debug(f"High memory usage detected: {memory_percent}%")
                    await self._handle_high_memory()

                # Check for high CPU usage
                if cpu_percent > self.cpu_threshold_percent:
                    if not self.quiet_monitoring:
                        logger.warning(f"High CPU usage detected: {cpu_percent}%")
                    else:
                        logger.debug(f"High CPU usage detected: {cpu_percent}%")
                    await self._handle_high_cpu()

                # Periodic cleanup (every 5 minutes)
                current_time = time.time()
                if (
                    self.metrics["last_cleanup_time"] is None
                    or current_time - self.metrics["last_cleanup_time"] > 300
                ):
                    await self._perform_cleanup()
                    self.metrics["last_cleanup_time"] = current_time

            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {str(e)}")

            # Wait before next check
            await asyncio.sleep(self.check_interval)

    async def _handle_high_memory(self) -> None:
        """Handle high memory usage."""
        logger.debug("Handling high memory usage")

        # Update metrics
        self.metrics["high_load_events"] += 1
        self.metrics["last_high_load_time"] = time.time()

        # Perform memory cleanup
        await self._perform_memory_cleanup()

        # Notify components about memory pressure
        for callback in self.memory_pressure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(memory_percent=psutil.virtual_memory().percent)
                else:
                    callback(memory_percent=psutil.virtual_memory().percent)
            except Exception as e:
                logger.error(f"Error in memory pressure callback: {str(e)}")

        # Update metrics
        self.metrics["adaptive_actions"] += 1

    async def _handle_high_cpu(self) -> None:
        """Handle high CPU usage."""
        logger.debug("Handling high CPU usage")

        # Update metrics
        self.metrics["high_load_events"] += 1
        self.metrics["last_high_load_time"] = time.time()

        # Reduce background task priority (skip on Windows to avoid errors)
        if sys.platform != "win32" and not os.environ.get("SKIP_PROCESS_PRIORITY"):
            try:
                process = psutil.Process(os.getpid())
                process.nice(10)
                logger.debug("Adjusted process priority to reduce CPU usage")
            except Exception as e:
                logger.debug(f"Could not adjust process priority: {str(e)}")
        else:
            logger.debug("Skipping process priority adjustment on Windows")

        # Update metrics
        self.metrics["adaptive_actions"] += 1

    async def _perform_cleanup(self) -> None:
        """Perform periodic cleanup."""
        logger.debug("Performing periodic cleanup")

        # Perform memory cleanup
        await self._perform_memory_cleanup()

        # Clean up temporary files
        try:
            temp_dirs = ["temp", "logs/temp"]
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                # Check if file is older than 1 hour
                                if time.time() - os.path.getmtime(file_path) > 3600:
                                    os.unlink(file_path)
                                    logger.debug(f"Deleted old temporary file: {file_path}")
                        except Exception as e:
                            logger.error(f"Error deleting temporary file {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

    async def _perform_memory_cleanup(self) -> None:
        """Perform memory cleanup."""
        logger.debug("Performing memory cleanup")

        # Force garbage collection
        gc.collect()

        # Update metrics
        self.metrics["memory_cleanup_count"] += 1

    def register_memory_pressure_callback(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        """
        Register a callback for memory pressure events.

        Args:
            callback: Callback function that takes a dictionary of memory metrics
        """
        self.memory_pressure_callbacks.append(callback)
        logger.info(f"Registered memory pressure callback: {callback.__name__}")

    def get_resource_limits(self) -> dict[str, Any]:
        """
        Get resource limits.

        Returns:
            Dict[str, Any]: Resource limits
        """
        return {
            "max_memory_gb": self.max_memory_gb,
            "cpu_cores": self.cpu_cores,
            "memory_threshold_percent": self.memory_threshold_percent,
            "cpu_threshold_percent": self.cpu_threshold_percent,
            "conversation_memory_percent": self.conversation_memory_percent,
            "knowledge_memory_percent": self.knowledge_memory_percent,
            "system_memory_percent": self.system_memory_percent,
        }

    def get_resource_usage(self) -> dict[str, Any]:
        """
        Get current resource usage.

        Returns:
            Dict[str, Any]: Resource usage
        """
        memory = psutil.virtual_memory()

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
        }

    def get_metrics(self) -> dict[str, Any]:
        """
        Get resource manager metrics.

        Returns:
            Dict[str, Any]: Metrics
        """
        return self.metrics

    def get_all_metrics(self) -> dict[str, Any]:
        """
        Get all metrics and resource information.

        Returns:
            Dict[str, Any]: All metrics and resource information
        """
        return {
            "resource_usage": self.get_resource_usage(),
            "resource_limits": self.get_resource_limits(),
            "metrics": self.metrics,
        }

    async def optimize_conversation_memory(self, conversation_manager) -> None:
        """
        Optimize memory usage for conversations.

        Args:
            conversation_manager: ConversationManager instance
        """
        logger.info("Optimizing conversation memory")

        try:
            # Get all sessions
            sessions = conversation_manager.sessions

            # Sort sessions by last updated time
            sorted_sessions = sorted(
                sessions.items(), key=lambda x: x[1]["updated_at"], reverse=True
            )

            # Keep only the 10 most recent sessions
            if len(sorted_sessions) > 10:
                for session_id, _ in sorted_sessions[10:]:
                    conversation_manager.delete_session(session_id)
                    logger.info(f"Deleted old session: {session_id}")

            # For remaining sessions, ensure they are summarized
            for session_id, session in list(sessions.items()):
                if len(session["messages"]) > conversation_manager.summarize_threshold:
                    await conversation_manager._maybe_summarize_conversation(session_id)

            logger.info("Conversation memory optimization complete")

        except Exception as e:
            logger.error(f"Error optimizing conversation memory: {str(e)}")

    async def optimize_knowledge_graph_memory(self, kg_manager) -> None:
        """
        Optimize memory usage for knowledge graphs.

        Args:
            kg_manager: KnowledgeGraphManager instance
        """
        logger.info("Optimizing knowledge graph memory")

        try:
            # Prune dynamic graph if it's too large
            if kg_manager.dynamic_kg:
                try:
                    G = kg_manager.dynamic_kg.get_networkx_graph()
                    node_count = G.number_of_nodes()

                    # If the graph is too large, prune it
                    max_nodes = 1000
                    if node_count > max_nodes:
                        logger.info(f"Dynamic graph has {node_count} nodes, pruning to {max_nodes}")

                        # Create a new graph store
                        graph_store = kg_manager.dynamic_kg._graph_store.__class__()
                        storage_ctx = (
                            kg_manager.dynamic_kg._storage_context.__class__.from_defaults(
                                graph_store=graph_store
                            )
                        )

                        # Create a new knowledge graph index
                        new_kg = kg_manager.dynamic_kg.__class__(storage_context=storage_ctx)

                        # Get the most important nodes (based on degree)
                        important_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[
                            :max_nodes
                        ]
                        important_node_ids = [node_id for node_id, _ in important_nodes]

                        # Get subgraph of important nodes
                        subgraph = G.subgraph(important_node_ids)

                        # Add triplets to the new graph
                        for u, v, data in subgraph.edges(data=True):
                            relation = data.get("relation", "related_to")
                            new_kg.upsert_triplet_and_node((u, relation, v))

                        # Replace the old graph with the new one
                        kg_manager.dynamic_kg = new_kg

                        logger.info(f"Pruned dynamic graph to {max_nodes} nodes")
                except Exception as e:
                    logger.error(f"Error pruning dynamic graph: {str(e)}")

            logger.info("Knowledge graph memory optimization complete")

        except Exception as e:
            logger.error(f"Error optimizing knowledge graph memory: {str(e)}")


# Global resource manager instance
resource_manager = ResourceManager()


async def start_resource_monitoring(config=None) -> None:
    """
    Start resource monitoring using the global resource manager.

    Args:
        config: Configuration dictionary
    """
    global resource_manager

    if config:
        resource_manager = ResourceManager(config)

    await resource_manager.start_monitoring()


async def optimize_memory_usage(conversation_manager=None, kg_manager=None) -> None:
    """
    Optimize memory usage using the global resource manager.

    Args:
        conversation_manager: ConversationManager instance
        kg_manager: KnowledgeGraphManager instance
    """
    if conversation_manager:
        await resource_manager.optimize_conversation_memory(conversation_manager)

    if kg_manager:
        await resource_manager.optimize_knowledge_graph_memory(kg_manager)

    # Force garbage collection
    gc.collect()
