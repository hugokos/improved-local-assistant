"""
Working Set Cache Manager for fast graph retrieval on edge devices.

This module provides the WorkingSetCache class that maintains per-session
caches of recently touched graph nodes with LRU eviction and disk persistence
for optimal performance on CPU-constrained devices.
"""

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any

try:
    from llama_index.core.storage import StorageContext
except ImportError:
    try:
        from llama_index.storage import StorageContext
    except ImportError:
        StorageContext = None


class WorkingSetCache:
    """
    Manages per-session working set caches for fast graph neighborhood retrieval.

    Maintains bounded caches of recently accessed node IDs with LRU eviction,
    disk persistence, and integration with PropertyGraphIndex for efficient
    1-hop neighbor retrieval.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Working Set Cache with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract cache configuration
        cache_config = config.get("working_set_cache", {})
        self.nodes_per_session = cache_config.get("nodes_per_session", 100)
        self.max_edges_per_query = cache_config.get("max_edges_per_query", 40)
        self.global_memory_limit_mb = cache_config.get("global_memory_limit_mb", 256)
        self.eviction_threshold = cache_config.get("eviction_threshold", 0.8)
        self.include_text = cache_config.get("include_text", False)
        self.persist_dir = cache_config.get("persist_dir", "./storage")

        # Session caches: session_id -> OrderedDict[node_id, access_time]
        self._session_caches: dict[str, OrderedDict[str, float]] = {}

        # Global memory tracking
        self._total_cached_nodes = 0
        self._max_total_nodes = (
            self.global_memory_limit_mb * 1024 * 1024
        ) // 1024  # Rough estimate

        # Persistence
        self._cache_file = os.path.join(self.persist_dir, "working_set_cache.json")

        # Metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "sessions_created": 0,
            "sessions_evicted": 0,
            "total_nodes_cached": 0,
        }

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the working set cache and load from disk if available."""
        try:
            self.logger.info("Initializing Working Set Cache")

            # Ensure persist directory exists
            os.makedirs(self.persist_dir, exist_ok=True)

            # Load cache from disk if available
            await self._load_cache()

            self.logger.info(
                f"Working Set Cache initialized with {len(self._session_caches)} sessions"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Working Set Cache: {e}")
            raise

    async def get_working_set(self, session_id: str) -> set[str]:
        """
        Get the working set of node IDs for a session.

        Args:
            session_id: Session identifier

        Returns:
            Set[str]: Set of recently accessed node IDs
        """
        async with self._lock:
            if session_id not in self._session_caches:
                self._session_caches[session_id] = OrderedDict()
                self.metrics["sessions_created"] += 1

            # Return copy of node IDs (most recent first)
            cache = self._session_caches[session_id]
            return set(cache.keys())

    async def update_working_set(self, session_id: str, node_ids: list[str]) -> None:
        """
        Update the working set with newly accessed node IDs.

        Args:
            session_id: Session identifier
            node_ids: List of node IDs that were accessed
        """
        async with self._lock:
            current_time = time.time()

            # Ensure session cache exists
            if session_id not in self._session_caches:
                self._session_caches[session_id] = OrderedDict()
                self.metrics["sessions_created"] += 1

            cache = self._session_caches[session_id]

            # Update access times for existing nodes and add new ones
            for node_id in node_ids:
                if node_id in cache:
                    # Move to end (most recent)
                    cache.move_to_end(node_id)
                    cache[node_id] = current_time
                    self.metrics["cache_hits"] += 1
                else:
                    # Add new node
                    cache[node_id] = current_time
                    self._total_cached_nodes += 1
                    self.metrics["cache_misses"] += 1

            # Enforce per-session limit
            while len(cache) > self.nodes_per_session:
                cache.popitem(last=False)[0]
                self._total_cached_nodes -= 1
                self.metrics["evictions"] += 1

            # Check global memory limit and evict if necessary
            await self._maybe_evict_sessions()

            # Update metrics
            self.metrics["total_nodes_cached"] = self._total_cached_nodes

    async def get_neighborhood(
        self, session_id: str, node_ids: set[str] | None = None, max_edges: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Get 1-hop neighborhood for nodes in the working set.

        Args:
            session_id: Session identifier
            node_ids: Specific node IDs to get neighborhood for (or None for all in working set)
            max_edges: Maximum edges to return (defaults to config value)

        Returns:
            List[Dict]: List of edges/relationships in the neighborhood
        """
        if max_edges is None:
            max_edges = self.max_edges_per_query

        try:
            # Get working set if no specific nodes provided
            if node_ids is None:
                node_ids = await self.get_working_set(session_id)

            if not node_ids:
                return []

            # This is a placeholder for actual graph retrieval
            # In the full implementation, this would integrate with PropertyGraphIndex
            # to retrieve 1-hop neighbors for the specified nodes

            self.logger.debug(
                f"Getting neighborhood for {len(node_ids)} nodes (max {max_edges} edges)"
            )

            # Placeholder return - in real implementation, this would query the graph
            return []

        except Exception as e:
            self.logger.error(f"Error getting neighborhood: {e}")
            return []

    async def _maybe_evict_sessions(self) -> None:
        """Evict least recently used sessions if over global memory limit."""
        if self._total_cached_nodes <= self._max_total_nodes * self.eviction_threshold:
            return

        self.logger.info(
            f"Global cache limit reached ({self._total_cached_nodes} nodes), evicting sessions"
        )

        # Sort sessions by last access time (oldest first)
        session_access_times = []
        for session_id, cache in self._session_caches.items():
            if cache:
                last_access = max(cache.values())
                session_access_times.append((last_access, session_id, len(cache)))

        session_access_times.sort()  # Oldest first

        # Evict sessions until we're under the threshold
        target_nodes = int(
            self._max_total_nodes * self.eviction_threshold * 0.8
        )  # 80% of threshold

        for _, session_id, node_count in session_access_times:
            if self._total_cached_nodes <= target_nodes:
                break

            # Evict entire session
            del self._session_caches[session_id]
            self._total_cached_nodes -= node_count
            self.metrics["sessions_evicted"] += 1

            self.logger.debug(f"Evicted session {session_id} with {node_count} nodes")

    async def evict_lru_sessions(self, count: int = 1) -> None:
        """
        Manually evict least recently used sessions.

        Args:
            count: Number of sessions to evict
        """
        async with self._lock:
            if len(self._session_caches) <= count:
                return

            # Sort sessions by last access time
            session_access_times = []
            for session_id, cache in self._session_caches.items():
                if cache:
                    last_access = max(cache.values())
                    session_access_times.append((last_access, session_id, len(cache)))

            session_access_times.sort()  # Oldest first

            # Evict the oldest sessions
            for i in range(min(count, len(session_access_times))):
                _, session_id, node_count = session_access_times[i]
                del self._session_caches[session_id]
                self._total_cached_nodes -= node_count
                self.metrics["sessions_evicted"] += 1

                self.logger.debug(f"Manually evicted session {session_id}")

    async def persist_cache(self) -> None:
        """Persist cache to disk for warm starts."""
        try:
            async with self._lock:
                # Convert OrderedDict to regular dict for JSON serialization
                cache_data = {
                    "session_caches": {
                        session_id: dict(cache)
                        for session_id, cache in self._session_caches.items()
                    },
                    "metrics": self.metrics.copy(),
                    "timestamp": time.time(),
                }

                # Write to temporary file first, then rename for atomicity
                temp_file = self._cache_file + ".tmp"
                with open(temp_file, "w") as f:
                    json.dump(cache_data, f, indent=2)

                # On Windows, remove the target file first if it exists
                if os.path.exists(self._cache_file):
                    os.remove(self._cache_file)

                os.rename(temp_file, self._cache_file)

                self.logger.debug(f"Persisted cache with {len(self._session_caches)} sessions")

        except Exception as e:
            self.logger.error(f"Error persisting cache: {e}")

    async def _load_cache(self) -> None:
        """Load cache from disk if available."""
        try:
            if not os.path.exists(self._cache_file):
                self.logger.debug("No cache file found, starting with empty cache")
                return

            with open(self._cache_file) as f:
                cache_data = json.load(f)

            # Restore session caches
            session_caches = cache_data.get("session_caches", {})
            for session_id, cache_dict in session_caches.items():
                # Convert back to OrderedDict, sorted by access time
                sorted_items = sorted(cache_dict.items(), key=lambda x: x[1])
                self._session_caches[session_id] = OrderedDict(sorted_items)

            # Restore metrics
            if "metrics" in cache_data:
                self.metrics.update(cache_data["metrics"])

            # Recalculate total cached nodes
            self._total_cached_nodes = sum(len(cache) for cache in self._session_caches.values())

            cache_age = time.time() - cache_data.get("timestamp", 0)
            self.logger.info(
                f"Loaded cache from disk: {len(self._session_caches)} sessions, "
                f"{self._total_cached_nodes} nodes, {cache_age:.1f}s old"
            )

        except Exception as e:
            self.logger.error(f"Error loading cache from disk: {e}")
            # Start with empty cache on error
            self._session_caches = {}
            self._total_cached_nodes = 0

    def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """
        Get statistics for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dict: Session statistics
        """
        if session_id not in self._session_caches:
            return {"exists": False}

        cache = self._session_caches[session_id]
        if not cache:
            return {"exists": True, "node_count": 0}

        access_times = list(cache.values())
        return {
            "exists": True,
            "node_count": len(cache),
            "oldest_access": min(access_times),
            "newest_access": max(access_times),
            "age_seconds": time.time() - max(access_times),
        }

    def get_global_stats(self) -> dict[str, Any]:
        """Get global cache statistics."""
        return {
            "total_sessions": len(self._session_caches),
            "total_nodes": self._total_cached_nodes,
            "memory_usage_mb": (self._total_cached_nodes * 1024) // (1024 * 1024),  # Rough estimate
            "memory_limit_mb": self.global_memory_limit_mb,
            "memory_utilization": self._total_cached_nodes / self._max_total_nodes
            if self._max_total_nodes > 0
            else 0,
            "metrics": self.metrics.copy(),
        }

    async def clear_session(self, session_id: str) -> bool:
        """
        Clear cache for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            bool: True if session was found and cleared
        """
        async with self._lock:
            if session_id in self._session_caches:
                node_count = len(self._session_caches[session_id])
                del self._session_caches[session_id]
                self._total_cached_nodes -= node_count
                self.logger.debug(f"Cleared session {session_id} with {node_count} nodes")
                return True
            return False

    async def clear_all(self) -> None:
        """Clear all cached data."""
        async with self._lock:
            self._session_caches.clear()
            self._total_cached_nodes = 0
            self.logger.info("Cleared all cached data")

    async def shutdown(self) -> None:
        """Shutdown cache and persist to disk."""
        try:
            self.logger.info("Shutting down Working Set Cache")

            # Persist cache to disk
            await self.persist_cache()

            # Clear memory
            await self.clear_all()

            self.logger.info("Working Set Cache shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during cache shutdown: {e}")
