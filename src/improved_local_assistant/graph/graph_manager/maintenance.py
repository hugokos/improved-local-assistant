"""
Maintenance module for KnowledgeGraphManager.

Handles graph merging, rebuilding, and maintenance operations including
NetworkX graph operations and embedding updates.
"""

import os
from datetime import datetime
from typing import Dict

from llama_index.core import KnowledgeGraphIndex
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.graph_stores import SimpleGraphStore


class KnowledgeGraphMaintenance:
    """Handles maintenance operations for knowledge graphs."""

    async def rebuild_graphs_with_new_embeddings(self) -> Dict[str, bool]:
        """
        Rebuild all knowledge graphs with new embeddings.

        Returns:
            Dict[str, bool]: Results of rebuilding each graph
        """
        results = {}

        try:
            self.logger.info("Starting rebuild of all graphs with new embeddings")

            # Rebuild modular graphs
            for graph_id, kg_index in self.kg_indices.items():
                try:
                    self.logger.info(f"Rebuilding modular graph: {graph_id}")

                    # Get the NetworkX representation
                    G = self._safe_get_networkx_graph(kg_index, graph_id)
                    triplets = []

                    if G is not None:
                        for u, v, data in G.edges(data=True):
                            relation = data.get("relation", "related_to")
                            triplets.append((u, relation, v))

                    # Create a new graph store and storage context
                    from ..utf8_import_helper import get_utf8_filesystem

                    graph_store = SimpleGraphStore()
                    storage_ctx = StorageContext.from_defaults(
                        graph_store=graph_store, fs=get_utf8_filesystem()
                    )

                    # Create a new knowledge graph index
                    new_kg_index = KnowledgeGraphIndex(storage_context=storage_ctx)

                    # Add all triplets to the new graph
                    for triplet in triplets:
                        new_kg_index.upsert_triplet_and_node(triplet)

                    # Replace the old graph with the new one
                    self.kg_indices[graph_id] = new_kg_index

                    # Persist the updated graph
                    persist_dir = os.path.join(self.prebuilt_directory, graph_id)
                    os.makedirs(persist_dir, exist_ok=True)
                    new_kg_index.storage_context.persist(persist_dir=persist_dir)

                    self.logger.info(f"Successfully rebuilt modular graph: {graph_id}")
                    results[graph_id] = True

                except Exception as e:
                    self.logger.error(f"Error rebuilding modular graph {graph_id}: {str(e)}")
                    results[graph_id] = False

            # Rebuild dynamic graph if it exists
            if self.dynamic_kg:
                try:
                    self.logger.info("Rebuilding dynamic graph")

                    # Get the NetworkX representation
                    G = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
                    triplets = []

                    if G is not None:
                        for u, v, data in G.edges(data=True):
                            relation = data.get("relation", "related_to")
                            triplets.append((u, relation, v))

                    # Create a new graph store and storage context
                    from ..utf8_import_helper import get_utf8_filesystem

                    graph_store = SimpleGraphStore()
                    storage_ctx = StorageContext.from_defaults(
                        graph_store=graph_store, fs=get_utf8_filesystem()
                    )

                    # Create a new knowledge graph index
                    new_dynamic_kg = KnowledgeGraphIndex(storage_context=storage_ctx)

                    # Add all triplets to the new graph
                    for triplet in triplets:
                        new_dynamic_kg.upsert_triplet_and_node(triplet)

                    # Replace the old dynamic graph with the new one
                    self.dynamic_kg = new_dynamic_kg

                    self.logger.info("Successfully rebuilt dynamic graph")
                    results["dynamic"] = True

                except Exception as e:
                    self.logger.error(f"Error rebuilding dynamic graph: {str(e)}")
                    results["dynamic"] = False

            return results

        except Exception as e:
            self.logger.error(f"Error in rebuild_graphs_with_new_embeddings: {str(e)}")
            return results

    async def _merge_dynamic_graph(self, incoming_persist_dir: str, merge_strategy: str = "union"):
        """
        Merge an incoming graph with the existing dynamic graph.

        Args:
            incoming_persist_dir: Directory containing the incoming graph persistence data
            merge_strategy: Strategy for handling conflicts
        """
        try:
            # Load the incoming graph
            from ..utf8_import_helper import get_utf8_filesystem

            incoming_storage = StorageContext.from_defaults(
                persist_dir=incoming_persist_dir, fs=get_utf8_filesystem()
            )
            incoming_kg = load_index_from_storage(storage_context=incoming_storage)

            # Get NetworkX representations for merging
            existing_graph = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
            incoming_graph = self._safe_get_networkx_graph(incoming_kg, "incoming")

            if existing_graph is None or incoming_graph is None:
                self.logger.error("Cannot merge graphs: NetworkX representation not available")
                return False

            # Merge the graphs
            merged_graph = self._merge_networkx_graphs(
                existing_graph, incoming_graph, merge_strategy
            )

            # Rebuild the dynamic knowledge graph from merged NetworkX graph
            await self._rebuild_dynamic_from_networkx(merged_graph)

            self.logger.info(f"Successfully merged dynamic graph using strategy: {merge_strategy}")

        except Exception as e:
            self.logger.error(f"Error merging dynamic graph: {str(e)}")
            raise

    async def _merge_modular_graph(
        self, incoming_persist_dir: str, graph_id: str, merge_strategy: str = "union"
    ):
        """
        Merge an incoming graph with an existing modular graph.

        Args:
            incoming_persist_dir: Directory containing the incoming graph persistence data
            graph_id: ID of the existing modular graph to merge with
            merge_strategy: Strategy for handling conflicts
        """
        try:
            # Load the incoming graph
            from ..utf8_import_helper import get_utf8_filesystem

            incoming_storage = StorageContext.from_defaults(
                persist_dir=incoming_persist_dir, fs=get_utf8_filesystem()
            )
            incoming_kg = load_index_from_storage(storage_context=incoming_storage)

            # Get the existing modular graph
            if graph_id not in self.kg_indices:
                raise ValueError(f"Modular graph {graph_id} not found for merging")

            existing_kg = self.kg_indices[graph_id]

            # Get NetworkX representations for merging
            existing_graph = self._safe_get_networkx_graph(existing_kg, graph_id)
            incoming_graph = self._safe_get_networkx_graph(incoming_kg, "incoming")

            if existing_graph is None or incoming_graph is None:
                self.logger.error("Cannot merge graphs: NetworkX representation not available")
                return False

            # Merge the graphs
            merged_graph = self._merge_networkx_graphs(
                existing_graph, incoming_graph, merge_strategy
            )

            # Rebuild the modular knowledge graph from merged NetworkX graph
            await self._rebuild_modular_from_networkx(merged_graph, graph_id)

            self.logger.info(
                f"Successfully merged modular graph {graph_id} using strategy: {merge_strategy}"
            )

        except Exception as e:
            self.logger.error(f"Error merging modular graph {graph_id}: {str(e)}")
            raise

    def _merge_networkx_graphs(self, base_graph, incoming_graph, strategy: str = "union"):
        """
        Merge two NetworkX graphs using the specified strategy.

        Args:
            base_graph: Base NetworkX graph
            incoming_graph: Incoming NetworkX graph to merge
            strategy: Merge strategy

        Returns:
            Merged NetworkX graph
        """
        # Create a copy of the base graph
        merged = base_graph.copy()

        # Add nodes from incoming graph
        for node, data in incoming_graph.nodes(data=True):
            if node in merged.nodes():
                # Node exists, handle conflict based on strategy
                existing_data = merged.nodes[node]

                if strategy == "union":
                    # Merge attributes, keeping both
                    merged_data = {**existing_data, **data}
                    # Add provenance information
                    merged_data["sources"] = existing_data.get("sources", ["base"]) + ["incoming"]
                    merged_data["last_updated"] = datetime.now().isoformat()
                elif strategy == "prefer_base":
                    # Keep existing data, add provenance
                    merged_data = existing_data.copy()
                    merged_data["sources"] = existing_data.get("sources", ["base"])
                elif strategy == "prefer_incoming":
                    # Use incoming data, add provenance
                    merged_data = data.copy()
                    merged_data["sources"] = ["incoming"]
                    merged_data["last_updated"] = datetime.now().isoformat()

                merged.nodes[node].update(merged_data)
            else:
                # New node, add with provenance
                node_data = data.copy()
                node_data["sources"] = ["incoming"]
                node_data["added"] = datetime.now().isoformat()
                merged.add_node(node, **node_data)

        # Add edges from incoming graph
        for source, target, data in incoming_graph.edges(data=True):
            if merged.has_edge(source, target):
                # Edge exists, handle conflict
                existing_data = merged.edges[source, target]

                if strategy == "union":
                    merged_data = {**existing_data, **data}
                    merged_data["sources"] = existing_data.get("sources", ["base"]) + ["incoming"]
                elif strategy == "prefer_base":
                    merged_data = existing_data.copy()
                elif strategy == "prefer_incoming":
                    merged_data = data.copy()
                    merged_data["sources"] = ["incoming"]

                merged.edges[source, target].update(merged_data)
            else:
                # New edge, add with provenance
                edge_data = data.copy()
                edge_data["sources"] = ["incoming"]
                edge_data["added"] = datetime.now().isoformat()
                merged.add_edge(source, target, **edge_data)

        return merged

    async def _rebuild_dynamic_from_networkx(self, networkx_graph):
        """
        Rebuild the dynamic knowledge graph from a NetworkX graph.

        Args:
            networkx_graph: NetworkX graph to rebuild from
        """
        try:
            # Create new graph store and storage context
            from ..utf8_import_helper import get_utf8_filesystem

            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(
                graph_store=graph_store, fs=get_utf8_filesystem()
            )

            # Create new knowledge graph index
            new_kg = KnowledgeGraphIndex(storage_context=storage_ctx)

            # Add all edges as triplets
            for source, target, data in networkx_graph.edges(data=True):
                relation = data.get("relation", "connected_to")
                triplet = (str(source), str(relation), str(target))
                new_kg.upsert_triplet_and_node(triplet)

            # Replace the dynamic graph
            self.dynamic_kg = new_kg

            # Persist the updated dynamic graph
            persist_dir = os.path.join(self.dynamic_storage, "main")
            os.makedirs(persist_dir, exist_ok=True)
            self.dynamic_kg.storage_context.persist(persist_dir=persist_dir)

            self.logger.info("Successfully rebuilt dynamic graph from NetworkX")

        except Exception as e:
            self.logger.error(f"Error rebuilding dynamic graph: {str(e)}")
            raise

    async def _rebuild_modular_from_networkx(self, networkx_graph, graph_id: str):
        """
        Rebuild a modular knowledge graph from a NetworkX graph.

        Args:
            networkx_graph: NetworkX graph to rebuild from
            graph_id: ID of the modular graph
        """
        try:
            # Create new graph store and storage context
            from ..utf8_import_helper import get_utf8_filesystem

            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(
                graph_store=graph_store, fs=get_utf8_filesystem()
            )

            # Create new knowledge graph index
            new_kg = KnowledgeGraphIndex(storage_context=storage_ctx)

            # Add all edges as triplets
            for source, target, data in networkx_graph.edges(data=True):
                relation = data.get("relation", "connected_to")
                triplet = (str(source), str(relation), str(target))
                new_kg.upsert_triplet_and_node(triplet)

            # Replace the modular graph
            self.kg_indices[graph_id] = new_kg

            # Persist the updated modular graph
            persist_dir = os.path.join(self.prebuilt_directory, graph_id)
            os.makedirs(persist_dir, exist_ok=True)
            new_kg.storage_context.persist(persist_dir=persist_dir)

            self.logger.info(f"Successfully rebuilt modular graph {graph_id} from NetworkX")

        except Exception as e:
            self.logger.error(f"Error rebuilding modular graph {graph_id}: {str(e)}")
            raise
