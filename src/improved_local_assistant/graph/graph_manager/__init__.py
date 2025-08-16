"""
Graph Manager module for handling knowledge graph operations.

This module provides comprehensive knowledge graph management including
initialization, persistence, construction, querying, maintenance, 
visualization, and statistics.
"""

from .construction import KnowledgeGraphConstruction
from .init_config import KnowledgeGraphInitializer
from .maintenance import KnowledgeGraphMaintenance
from .persistence_simple import KnowledgeGraphPersistence
from .query import KnowledgeGraphQuery
from .stats import KnowledgeGraphStats
from .visualization import KnowledgeGraphVisualization


class KnowledgeGraphManager(
    KnowledgeGraphInitializer,
    KnowledgeGraphPersistence,
    KnowledgeGraphConstruction,
    KnowledgeGraphQuery,
    KnowledgeGraphMaintenance,
    KnowledgeGraphVisualization,
    KnowledgeGraphStats,
):
    """
    Manages knowledge graphs with LlamaIndex integration.

    Handles loading pre-built knowledge graphs, constructing new graphs,
    updating graphs dynamically, and performing graph-based retrieval.

    This class combines functionality from multiple mixins:
    - KnowledgeGraphInitializer: Setup and configuration
    - KnowledgeGraphPersistence: Saving, loading, and persistence
    - KnowledgeGraphConstruction: Graph creation and dynamic updates
    - KnowledgeGraphQuery: Querying graphs and subgraph extraction
    - KnowledgeGraphMaintenance: Graph merging and rebuilding
    - KnowledgeGraphVisualization: HTML and PyVis visualization
    - KnowledgeGraphStats: Metrics and graph statistics
    """

    pass


__all__ = [
    "KnowledgeGraphManager",
    "KnowledgeGraphInitializer",
    "KnowledgeGraphPersistence",
    "KnowledgeGraphConstruction",
    "KnowledgeGraphQuery",
    "KnowledgeGraphMaintenance",
    "KnowledgeGraphVisualization",
    "KnowledgeGraphStats",
]
