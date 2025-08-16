"""
Knowledge graph optimization for query performance.

This module provides mechanisms for optimizing knowledge graph queries,
including caching, indexing, and pruning strategies.
"""

import json
import logging
import os
import time
from typing import Any

# Configure logging
logger = logging.getLogger(__name__)


class KnowledgeGraphOptimizer:
    """
    Optimizes knowledge graph operations for performance.

    This class provides methods for optimizing knowledge graph queries,
    including caching, indexing, and pruning strategies.
    """

    def __init__(self, kg_manager=None):
        """
        Initialize the knowledge graph optimizer.

        Args:
            kg_manager: KnowledgeGraphManager instance
        """
        self.kg_manager = kg_manager

        # Query cache
        self.query_cache: dict[str, dict[str, Any]] = {}
        self.max_cache_size = 100
        self.cache_ttl = 3600  # 1 hour

        # Performance metrics
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "pruning_operations": 0,
            "last_pruning_time": None,
            "optimized_queries": 0,
        }

        logger.info("Knowledge graph optimizer initialized")

    def get_cached_query(self, query: str) -> dict[str, Any] | None:
        """
        Get a cached query result if available and not expired.

        Args:
            query: Query string

        Returns:
            Optional[Dict[str, Any]]: Cached result or None
        """
        # Normalize query for cache lookup
        cache_key = self._normalize_query(query)

        if cache_key in self.query_cache:
            cached_item = self.query_cache[cache_key]

            # Check if cache is still valid
            if time.time() - cached_item["timestamp"] <= self.cache_ttl:
                self.metrics["cache_hits"] += 1
                logger.debug(f"Cache hit for query: {query}")
                return cached_item["result"]
            else:
                # Cache expired
                del self.query_cache[cache_key]

        self.metrics["cache_misses"] += 1
        return None

    def cache_query_result(self, query: str, result: dict[str, Any]) -> None:
        """
        Cache a query result.

        Args:
            query: Query string
            result: Query result
        """
        # Normalize query for cache lookup
        cache_key = self._normalize_query(query)

        # Add to cache
        self.query_cache[cache_key] = {"result": result, "timestamp": time.time()}

        # Prune cache if it's too large
        if len(self.query_cache) > self.max_cache_size:
            self._prune_cache()

        logger.debug(f"Cached result for query: {query}")

    def _normalize_query(self, query: str) -> str:
        """
        Normalize a query string for cache lookup.

        Args:
            query: Query string

        Returns:
            str: Normalized query string
        """
        # Simple normalization: lowercase and remove extra whitespace
        return " ".join(query.lower().split())

    def _prune_cache(self) -> None:
        """Prune the query cache by removing oldest entries."""
        # Sort cache items by timestamp
        sorted_items = sorted(self.query_cache.items(), key=lambda x: x[1]["timestamp"])

        # Remove oldest 20% of entries
        num_to_remove = max(1, len(self.query_cache) // 5)
        for i in range(num_to_remove):
            if i < len(sorted_items):
                del self.query_cache[sorted_items[i][0]]

        logger.debug(f"Pruned {num_to_remove} entries from query cache")

    async def optimize_query(self, query: str) -> str:
        """
        Optimize a query for better performance.

        Args:
            query: Original query string

        Returns:
            str: Optimized query string
        """
        # This is a placeholder for more sophisticated query optimization
        # In a real implementation, this could analyze the query and rewrite it
        # for better performance based on graph structure

        # Simple optimization: remove unnecessary words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
        }

        words = query.split()
        optimized_words = [
            word for word in words if word.lower() not in stop_words or len(words) <= 3
        ]

        optimized_query = " ".join(optimized_words)

        if optimized_query != query:
            self.metrics["optimized_queries"] += 1
            logger.debug(f"Optimized query: '{query}' -> '{optimized_query}'")

        return optimized_query

    async def prune_dynamic_graph(self, max_nodes: int = 1000) -> bool:
        """
        Prune the dynamic knowledge graph to reduce size.

        Args:
            max_nodes: Maximum number of nodes to keep

        Returns:
            bool: True if pruning was successful
        """
        if not self.kg_manager or not self.kg_manager.dynamic_kg:
            logger.warning("No dynamic knowledge graph available for pruning")
            return False

        try:
            logger.info(f"Pruning dynamic knowledge graph to {max_nodes} nodes")

            # Get the current graph
            G = self.kg_manager.dynamic_kg.get_networkx_graph()
            node_count = G.number_of_nodes()

            if node_count <= max_nodes:
                logger.info(f"Dynamic graph has {node_count} nodes, no pruning needed")
                return True

            # Get node importance scores
            # Use a combination of degree centrality and recency
            node_scores = {}

            # Get node degrees (connectivity importance)
            for node, degree in G.degree():
                node_scores[node] = degree

            # Adjust scores based on recency (if available)
            # This is a placeholder - in a real implementation, you would
            # track when nodes were added and use that information

            # Sort nodes by score
            sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep the top nodes
            nodes_to_keep = [node for node, _ in sorted_nodes[:max_nodes]]

            # Create a new graph with only the important nodes

            subgraph = G.subgraph(nodes_to_keep)

            # Create a new graph store
            graph_store = self.kg_manager.dynamic_kg._graph_store.__class__()
            storage_ctx = self.kg_manager.dynamic_kg._storage_context.__class__.from_defaults(
                graph_store=graph_store
            )

            # Create a new knowledge graph index
            new_kg = self.kg_manager.dynamic_kg.__class__(storage_context=storage_ctx)

            # Add triplets to the new graph
            for u, v, data in subgraph.edges(data=True):
                relation = data.get("relation", "related_to")
                new_kg.upsert_triplet_and_node((u, relation, v))

            # Replace the old graph with the new one
            self.kg_manager.dynamic_kg = new_kg

            # Update metrics
            self.metrics["pruning_operations"] += 1
            self.metrics["last_pruning_time"] = time.time()

            logger.info(f"Pruned dynamic graph from {node_count} to {len(nodes_to_keep)} nodes")
            return True

        except Exception as e:
            logger.error(f"Error pruning dynamic graph: {str(e)}")
            return False

    def save_cache_to_disk(self, cache_file: str = "data/kg_cache.json") -> bool:
        """
        Save the query cache to disk.

        Args:
            cache_file: Path to cache file

        Returns:
            bool: True if save was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

            # Convert cache to serializable format
            serializable_cache = {}
            for key, value in self.query_cache.items():
                serializable_cache[key] = {
                    "result": value["result"],
                    "timestamp": value["timestamp"],
                }

            # Save to file
            with open(cache_file, "w") as f:
                json.dump(serializable_cache, f)

            logger.info(f"Saved query cache to {cache_file}")
            return True

        except Exception as e:
            logger.error(f"Error saving query cache: {str(e)}")
            return False

    def load_cache_from_disk(self, cache_file: str = "data/kg_cache.json") -> bool:
        """
        Load the query cache from disk.

        Args:
            cache_file: Path to cache file

        Returns:
            bool: True if load was successful
        """
        try:
            if not os.path.exists(cache_file):
                logger.warning(f"Cache file {cache_file} does not exist")
                return False

            # Load from file
            with open(cache_file) as f:
                serializable_cache = json.load(f)

            # Convert to cache format
            for key, value in serializable_cache.items():
                self.query_cache[key] = {"result": value["result"], "timestamp": value["timestamp"]}

            logger.info(f"Loaded query cache from {cache_file} ({len(self.query_cache)} entries)")
            return True

        except Exception as e:
            logger.error(f"Error loading query cache: {str(e)}")
            return False

    def get_metrics(self) -> dict[str, Any]:
        """
        Get optimizer metrics.

        Returns:
            Dict[str, Any]: Metrics
        """
        return {
            **self.metrics,
            "cache_size": len(self.query_cache),
            "cache_hit_ratio": self.metrics["cache_hits"]
            / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
            if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
            else 0,
        }


# Global knowledge graph optimizer instance
optimizer = None


def initialize_optimizer(kg_manager=None):
    """
    Initialize the global knowledge graph optimizer.

    Args:
        kg_manager: KnowledgeGraphManager instance
    """
    global optimizer
    optimizer = KnowledgeGraphOptimizer(kg_manager)

    # Try to load cache from disk
    optimizer.load_cache_from_disk()

    return optimizer


async def optimize_query(query: str) -> str:
    """
    Optimize a query using the global optimizer.

    Args:
        query: Original query string

    Returns:
        str: Optimized query string
    """
    global optimizer
    if not optimizer:
        logger.debug("Knowledge graph optimizer not initialized")
        return query

    return await optimizer.optimize_query(query)


def get_cached_query(query: str) -> dict[str, Any] | None:
    """
    Get a cached query result using the global optimizer.

    Args:
        query: Query string

    Returns:
        Optional[Dict[str, Any]]: Cached result or None
    """
    global optimizer
    if not optimizer:
        logger.debug("Knowledge graph optimizer not initialized")
        return None

    return optimizer.get_cached_query(query)


def cache_query_result(query: str, result: dict[str, Any]) -> None:
    """
    Cache a query result using the global optimizer.

    Args:
        query: Query string
        result: Query result
    """
    global optimizer
    if not optimizer:
        logger.debug("Knowledge graph optimizer not initialized")
        return

    optimizer.cache_query_result(query, result)
