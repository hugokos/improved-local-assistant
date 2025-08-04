"""
Statistics module for KnowledgeGraphManager.

Handles metrics and graph statistics collection and reporting.
"""

from typing import Any, Dict

import networkx as nx


class KnowledgeGraphStats:
    """Handles statistics and metrics for knowledge graphs."""

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for all knowledge graphs.

        Returns:
            Dict[str, Any]: Graph statistics
        """
        stats = {
            "graphs": {},
            "total_graphs": len(self.kg_indices) + (1 if self.dynamic_kg else 0),
            "total_nodes": 0,
            "total_edges": 0,
        }

        # Get statistics for pre-built graphs
        for graph_id, kg_index in self.kg_indices.items():
            G = self._safe_get_networkx_graph(kg_index, graph_id)
            if G is not None:
                nodes = G.number_of_nodes()
                edges = G.number_of_edges()

                stats["graphs"][graph_id] = {
                    "nodes": nodes,
                    "edges": edges,
                    "density": nx.density(G) if nodes > 1 else 0,
                }

                stats["total_nodes"] += nodes
                stats["total_edges"] += edges
            else:
                stats["graphs"][graph_id] = {"error": "NetworkX export not supported"}

        # Get statistics for dynamic graph
        if self.dynamic_kg:
            G = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
            if G is not None:
                nodes = G.number_of_nodes()
                edges = G.number_of_edges()

                stats["graphs"]["dynamic"] = {
                    "nodes": nodes,
                    "edges": edges,
                    "density": nx.density(G) if nodes > 1 else 0,
                }

                stats["total_nodes"] += nodes
                stats["total_edges"] += edges
            else:
                stats["graphs"]["dynamic"] = {"error": "NetworkX export not supported"}

        # Add performance metrics
        stats["metrics"] = self.metrics

        return stats