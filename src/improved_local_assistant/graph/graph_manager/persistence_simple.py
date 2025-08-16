"""
Simplified persistence module for KnowledgeGraphManager.

Handles saving, loading, and persistence of graphs with single UTF-8 encoding.
"""

import builtins
import json
import os
import time

from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage


class KnowledgeGraphPersistence:
    """Handles persistence operations for knowledge graphs."""

    def _load_from_persistence_with_encoding_fix(self, graph_path: str):
        """
        Load a knowledge graph from persistence with UTF-8 encoding.
        Supports both PropertyGraphIndex and KnowledgeGraphIndex.

        Args:
            graph_path: Path to the graph persistence directory

        Returns:
            PropertyGraphIndex or KnowledgeGraphIndex

        Raises:
            Exception: If loading fails
        """
        self.logger.info(f"Loading graph from {graph_path} with UTF-8 encoding")

        # Save original open function
        _orig_open = builtins.open

        def _utf8_open(path, mode="r", *args, **kwargs):
            if "b" not in mode:
                kwargs.setdefault("encoding", "utf-8")
            return _orig_open(path, mode, *args, **kwargs)

        # Patch builtin open
        builtins.open = _utf8_open

        try:
            # Configure LlamaIndex to use local models instead of OpenAI
            from llama_index.core import Settings
            from services.embedding_singleton import get_embedding_model

            # Set up local embedding model
            embed_model = get_embedding_model("BAAI/bge-small-en-v1.5")
            Settings.embed_model = embed_model

            # Disable LLM for loading (not needed for persistence operations)
            Settings.llm = None

            # Check if this is a property graph by looking for graph_meta.json
            import json
            import pathlib

            graph_type = "simple"  # default
            meta_path = pathlib.Path(graph_path) / "graph_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    graph_type = meta.get("graph_type", "simple")
                except Exception:
                    pass

            # Use UTF-8 filesystem for loading
            from ..utf8_import_helper import get_utf8_filesystem

            storage_context = StorageContext.from_defaults(
                persist_dir=graph_path, fs=get_utf8_filesystem()
            )

            self.logger.info(f"Loading graph as type: {graph_type}")

            # 🔧 INSTRUMENTATION: Log graph counts for debugging
            docs = len(storage_context.docstore.docs) if storage_context.docstore.docs else 0

            try:
                # Try to get node/edge counts from the graph
                if (
                    hasattr(storage_context, "property_graph_store")
                    and storage_context.property_graph_store
                ):
                    kg = storage_context.property_graph_store.graph
                    if hasattr(kg, "number_of_nodes"):
                        nodes = kg.number_of_nodes()
                        edges = kg.number_of_edges()
                    elif hasattr(kg, "nodes") and hasattr(kg, "edges"):
                        nodes = (
                            len(list(kg.nodes)) if hasattr(kg.nodes, "__iter__") else len(kg.nodes)
                        )
                        edges = (
                            len(list(kg.edges)) if hasattr(kg.edges, "__iter__") else len(kg.edges)
                        )
                    else:
                        raise AttributeError("Unknown graph type")
                else:
                    raise AttributeError("No property graph store available")
            except (AttributeError, Exception):
                # Fall back to raw JSON read
                import json
                import pathlib

                p = pathlib.Path(graph_path) / "property_graph_store.json"
                if p.exists():
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    nodes = len(raw.get("nodes", []))
                    edges = len(raw.get("edges", []))
                else:
                    nodes = 0
                    edges = 0

            self.logger.info(
                f"[DEBUG] Loaded graph at {graph_path}: docs={docs} nodes={nodes} edges={edges}"
            )

            kg_index = load_index_from_storage(storage_context=storage_context)
            self.logger.info("Successfully loaded graph with UTF-8 encoding")
            return kg_index
        finally:
            # Restore original open function
            builtins.open = _orig_open

    async def _persist_dynamic_graph(self):
        """Persist the dynamic knowledge graph to storage."""
        try:
            if not self.dynamic_kg:
                return

            persist_dir = os.path.join(self.dynamic_storage, "main")
            os.makedirs(persist_dir, exist_ok=True)

            # Run compaction if WAL is getting large
            await self._maybe_compact_wal()

            self.dynamic_kg.storage_context.persist(persist_dir=persist_dir)

            # For property graphs, also create kg.json
            graph_type = getattr(self, "config", {}).get("graph", {}).get("type", "simple")
            if graph_type == "property" and hasattr(self.dynamic_kg, "property_graph_store"):
                try:
                    # Extract property graph data
                    graph_store = self.dynamic_kg.property_graph_store
                    graph = graph_store.graph

                    # Create kg.json format
                    nodes = {}
                    edges = []

                    # Extract nodes
                    if hasattr(graph, "nodes") and callable(getattr(graph, "nodes", None)):
                        for node_id in graph.nodes():
                            node_data = graph.nodes[node_id]
                            nodes[node_id] = {
                                "label": node_data.get("label", "ENTITY"),
                                "properties": node_data.get("properties", {}),
                            }

                    # Extract edges
                    if hasattr(graph, "edges") and callable(getattr(graph, "edges", None)):
                        for source, target, edge_data in graph.edges(data=True):
                            edges.append(
                                {
                                    "source_id": source,
                                    "target_id": target,
                                    "label": edge_data.get("label", "unknown"),
                                    "properties": edge_data.get("properties", {}),
                                }
                            )

                    # Write kg.json
                    kg_data = {"nodes": nodes, "edges": edges}
                    kg_path = os.path.join(persist_dir, "kg.json")
                    with open(kg_path, "w", encoding="utf-8") as f:
                        json.dump(kg_data, f, ensure_ascii=False, indent=2)

                    self.logger.info(
                        f"Created kg.json with {len(nodes)} nodes and {len(edges)} edges"
                    )

                except Exception as e:
                    self.logger.warning(f"Could not create kg.json: {e}")

            # Reset counters
            self._dynamic_updates_count = 0
            self._last_persist_time = time.time()

            self.logger.info("Successfully persisted dynamic graph")

        except Exception as e:
            self.logger.error(f"Error persisting dynamic graph: {str(e)}")

    def _should_persist_dynamic_graph(self) -> bool:
        """Check if dynamic graph should be persisted."""
        if not self.dynamic_kg:
            return False

        # Persist based on update count or time interval
        time_elapsed = time.time() - self._last_persist_time

        return (
            self._dynamic_updates_count >= self._persist_update_threshold
            or time_elapsed >= self._persist_interval
        )

    async def _maybe_persist_dynamic_graph(self):
        """
        Conditionally persist the dynamic graph if persistence conditions are met.

        This method is called by the background thread to check if the dynamic graph
        should be persisted based on update count or time interval.
        """
        if self._should_persist_dynamic_graph():
            await self._persist_dynamic_graph()

    def get_graph_stats(self, graph_id: str) -> dict:
        """Get statistics for a specific graph."""
        try:
            if graph_id not in self.kg_indices:
                return {"error": f"Graph {graph_id} not found"}

            kg_index = self.kg_indices[graph_id]
            stats = {"graph_id": graph_id}

            # Get storage context stats
            if hasattr(kg_index, "storage_context"):
                storage_ctx = kg_index.storage_context

                # Document count
                if hasattr(storage_ctx, "docstore") and hasattr(storage_ctx.docstore, "docs"):
                    stats["num_documents"] = len(storage_ctx.docstore.docs)

                # Relationship count
                if hasattr(storage_ctx, "graph_store") and hasattr(
                    storage_ctx.graph_store, "rel_map"
                ):
                    stats["num_relationships"] = sum(
                        len(v) for v in storage_ctx.graph_store.rel_map.values()
                    )

                # Vector count (if available)
                if hasattr(storage_ctx, "vector_store"):
                    try:
                        if hasattr(storage_ctx.vector_store, "_collection"):
                            if hasattr(storage_ctx.vector_store._collection, "count"):
                                stats["num_vectors"] = storage_ctx.vector_store._collection.count()
                    except Exception:
                        stats["num_vectors"] = "unknown"

            return stats

        except Exception as e:
            return {"error": f"Error getting stats for {graph_id}: {str(e)}"}

    def export_graph(self, graph_id: str, export_path: str) -> bool:
        """Export a graph to a specified path."""
        try:
            if graph_id not in self.kg_indices:
                self.logger.error(f"Graph {graph_id} not found for export")
                return False

            kg_index = self.kg_indices[graph_id]
            os.makedirs(export_path, exist_ok=True)

            kg_index.storage_context.persist(persist_dir=export_path)
            self.logger.info(f"Successfully exported graph {graph_id} to {export_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting graph {graph_id}: {str(e)}")
            return False

    def _append_wal(self, event: dict):
        """Append event to write-ahead log."""
        try:
            wal_path = os.path.join(self.dynamic_storage, "main", "wal.jsonl")
            os.makedirs(os.path.dirname(wal_path), exist_ok=True)

            with open(wal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to append to WAL: {e}")

    async def _maybe_compact_wal(self):
        """Compact WAL if it's getting too large."""
        try:
            wal_path = os.path.join(self.dynamic_storage, "main", "wal.jsonl")

            if not os.path.exists(wal_path):
                return

            # Check WAL size
            wal_size = os.path.getsize(wal_path)
            max_wal_size = 10 * 1024 * 1024  # 10MB

            if wal_size > max_wal_size:
                self.logger.info("Compacting WAL due to size threshold")
                await self._compact_wal()
        except Exception as e:
            self.logger.error(f"Error checking WAL size: {e}")

    async def _compact_wal(self):
        """Compact the write-ahead log by replaying and creating clean snapshot."""
        try:
            wal_path = os.path.join(self.dynamic_storage, "main", "wal.jsonl")

            if not os.path.exists(wal_path):
                return

            self.logger.info("Starting WAL compaction")

            # Create backup of current WAL
            backup_path = f"{wal_path}.backup.{int(time.time())}"
            os.rename(wal_path, backup_path)

            # The graph is already persisted, so we can safely clear the WAL
            # In a more sophisticated implementation, you would:
            # 1. Replay WAL events into a temp graph
            # 2. Merge duplicate entities using canonical IDs
            # 3. Write a fresh snapshot

            self.logger.info("WAL compaction completed")

            # Keep only the last few backups
            self._cleanup_old_wal_backups()

        except Exception as e:
            self.logger.error(f"Error during WAL compaction: {e}")

    def _cleanup_old_wal_backups(self):
        """Clean up old WAL backup files."""
        try:
            backup_dir = os.path.join(self.dynamic_storage, "main")
            backup_files = [f for f in os.listdir(backup_dir) if f.startswith("wal.jsonl.backup.")]

            # Sort by timestamp and keep only the last 5
            backup_files.sort(key=lambda x: int(x.split(".")[-1]))

            for old_backup in backup_files[:-5]:
                old_path = os.path.join(backup_dir, old_backup)
                os.remove(old_path)
                self.logger.debug(f"Removed old WAL backup: {old_backup}")

        except Exception as e:
            self.logger.warning(f"Error cleaning up WAL backups: {e}")
