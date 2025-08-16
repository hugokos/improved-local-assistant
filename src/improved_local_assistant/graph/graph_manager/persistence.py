"""
Persistence module for KnowledgeGraphManager.

Handles saving, loading, and persistence of graphs including encoding fixes
and import/export functionality.
"""

import builtins
import contextlib
import os
import shutil
import tempfile
import threading
import time
import zipfile

from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage

# Import filesystem utilities and define UTF8FileSystem
try:
    from fsspec.implementations.local import LocalFileSystem

    class UTF8FileSystem(LocalFileSystem):
        """Custom filesystem that enforces UTF-8 encoding for text files."""

        def open(self, path, mode="rb", **kwargs):
            if "b" not in mode:
                kwargs.setdefault("encoding", "utf-8")
            return super().open(path, mode=mode, **kwargs)

    _UTF8_FILESYSTEM_AVAILABLE = True
except ImportError:
    # Fallback if fsspec is not available
    class UTF8FileSystem:
        """Dummy UTF8FileSystem for when fsspec is not available."""

        pass

    _UTF8_FILESYSTEM_AVAILABLE = False


class KnowledgeGraphPersistence:
    """Handles persistence operations for knowledge graphs."""

    def _load_from_persistence_with_encoding_fix(self, graph_path: str):
        """
        Load a knowledge graph from persistence with simplified UTF-8 handling.

        Args:
            graph_path: Path to the graph persistence directory

        Returns:
            KnowledgeGraphIndex or None if loading failed
        """
        # Method 1: Try standard loading first
        try:
            self.logger.info("Attempting UTF-8 persistence loading")
            from ..utf8_import_helper import get_utf8_filesystem

            storage_context = StorageContext.from_defaults(
                persist_dir=graph_path, fs=get_utf8_filesystem()
            )
            kg_index = load_index_from_storage(storage_context=storage_context)
            self.logger.info("Successfully loaded graph using standard method")
            return kg_index
        except Exception as e:
            self.logger.warning(f"Standard loading failed: {str(e)}")

        # Method 2: Single retry with explicit UTF-8 encoding
        try:
            self.logger.info("Attempting UTF-8 retry loading")

            # Save original open function
            _orig_open = builtins.open

            def _utf8_open(path, mode="r", *args, **kwargs):
                if "b" not in mode:
                    kwargs.setdefault("encoding", "utf-8")
                    kwargs.setdefault("errors", "replace")
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

                # Import UTF8FileSystem from kg_builder if available
                # Use UTF-8 filesystem for loading
                from ..utf8_import_helper import get_utf8_filesystem

                storage_context = StorageContext.from_defaults(
                    persist_dir=graph_path, fs=get_utf8_filesystem()
                )

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
                                len(list(kg.nodes))
                                if hasattr(kg.nodes, "__iter__")
                                else len(kg.nodes)
                            )
                            edges = (
                                len(list(kg.edges))
                                if hasattr(kg.edges, "__iter__")
                                else len(kg.edges)
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
                self.logger.info("Successfully loaded graph using UTF-8 retry")
                return kg_index
            finally:
                # Restore original open function
                builtins.open = _orig_open

        except Exception as e:
            self.logger.error(f"UTF-8 retry loading failed: {str(e)}")

        # If both methods fail, return None
        try:
            self.logger.info("Attempting loading with cleaned JSON files")

            temp_dir = os.path.join(tempfile.gettempdir(), f"graph_fix_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)

            # Process each JSON file
            for filename in os.listdir(graph_path):
                if filename.endswith(".json"):
                    src_path = os.path.join(graph_path, filename)
                    dst_path = os.path.join(temp_dir, filename)

                    # Read in binary mode and clean up encoding
                    with open(src_path, "rb") as src_file:
                        binary_content = src_file.read()

                    # Try to decode with various encodings
                    content = None
                    for encoding in ["utf-8", "latin1", "cp1252", "ascii"]:
                        try:
                            content = binary_content.decode(encoding, errors="replace")
                            break
                        except:
                            continue

                    if content is None:
                        self.logger.error(f"Could not decode {filename}")
                        continue

                    # Clean up problematic characters
                    content = content.replace("\x00", "")  # Remove null bytes
                    content = content.replace("\\u0000", "")  # Remove escaped null bytes
                    content = content.replace("\ufffd", "")  # Remove replacement characters

                    # Validate JSON and write
                    try:
                        data = json.loads(content)
                        with open(dst_path, "w", encoding="utf-8") as dst_file:
                            json.dump(data, dst_file, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        # Copy non-JSON files as-is
                        with open(dst_path, "w", encoding="utf-8") as dst_file:
                            dst_file.write(content)
                else:
                    # Copy non-JSON files as-is
                    shutil.copy2(
                        os.path.join(graph_path, filename), os.path.join(temp_dir, filename)
                    )

            # Try loading from cleaned directory
            from ..utf8_import_helper import get_utf8_filesystem

            storage_context = StorageContext.from_defaults(
                persist_dir=temp_dir, fs=get_utf8_filesystem()
            )
            kg_index = load_index_from_storage(storage_context=storage_context)
            self.logger.info("Successfully loaded graph using cleaned JSON files")

            # Clean up temp directory in background
            def cleanup_temp():
                with contextlib.suppress(builtins.BaseException):
                    shutil.rmtree(temp_dir)

            threading.Timer(60, cleanup_temp).start()

            return kg_index

        except Exception as e:
            self.logger.error(f"Cleaned JSON loading failed: {str(e)}")

        # All methods failed
        self.logger.error(f"All loading methods failed for {graph_path}")
        return None

    async def _persist_dynamic_graph(self):
        """Persist the dynamic knowledge graph to storage."""
        try:
            if not self.dynamic_kg:
                return

            persist_dir = os.path.join(self.dynamic_storage, "main")
            os.makedirs(persist_dir, exist_ok=True)

            self.dynamic_kg.storage_context.persist(persist_dir=persist_dir)

            # Reset counters
            self._dynamic_updates_count = 0
            self._last_persist_time = time.time()

            self.logger.info("Successfully persisted dynamic graph")

        except Exception as e:
            self.logger.error(f"Error persisting dynamic graph: {str(e)}")

    async def _maybe_persist_dynamic_graph(self):
        """
        Persist dynamic graph if conditions are met (time interval or update count).
        """
        try:
            current_time = time.time()
            time_since_last_persist = current_time - self._last_persist_time

            should_persist = (
                self._dynamic_updates_count >= self._persist_update_threshold
                or time_since_last_persist >= self._persist_interval
            )

            if should_persist and self.dynamic_kg:
                await self._persist_dynamic_graph()

        except Exception as e:
            self.logger.error(f"Error in _maybe_persist_dynamic_graph: {str(e)}")

    async def export_native(self, graph_id: str) -> str:
        """
        Export a knowledge graph in native LlamaIndex persistence format.

        Args:
            graph_id: ID of the graph to export

        Returns:
            str: Path to the created zip file
        """
        try:
            # Determine the persist directory
            if graph_id == "dynamic" and self.dynamic_kg:
                # For dynamic graph, persist it first
                persist_dir = os.path.join(self.dynamic_storage, "temp_export")
                os.makedirs(persist_dir, exist_ok=True)
                self.dynamic_kg.storage_context.persist(persist_dir=persist_dir)
            elif graph_id in self.kg_indices:
                # For modular graphs, check if they have a persist directory
                persist_dir = os.path.join(self.prebuilt_directory, graph_id)
                if not os.path.exists(persist_dir):
                    # If no persist dir exists, create one
                    persist_dir = os.path.join(self.prebuilt_directory, f"{graph_id}_export")
                    os.makedirs(persist_dir, exist_ok=True)
                    self.kg_indices[graph_id].storage_context.persist(persist_dir=persist_dir)
            else:
                raise ValueError(f"Graph {graph_id} not found")

            # Create temporary zip file
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            temp_zip.close()

            # Create zip archive
            with zipfile.ZipFile(temp_zip.name, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _dirs, files in os.walk(persist_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, persist_dir)
                        zipf.write(file_path, arcname)

            # Clean up temporary export directory if created
            if graph_id == "dynamic" or not os.path.exists(
                os.path.join(self.prebuilt_directory, graph_id)
            ):
                shutil.rmtree(persist_dir, ignore_errors=True)

            self.logger.info(f"Successfully exported graph {graph_id} to {temp_zip.name}")
            return temp_zip.name

        except Exception as e:
            self.logger.error(f"Error exporting graph {graph_id}: {str(e)}")
            raise

    async def export_native_partial(
        self, graph_id: str, query: str = "", max_hops: int = 2, max_nodes: int = 100
    ) -> str:
        """
        Export a partial knowledge graph (subgraph) in native LlamaIndex persistence format.

        Args:
            graph_id: ID of the graph to export
            query: Query to find relevant nodes (empty string gets all nodes)
            max_hops: Maximum hops from relevant nodes
            max_nodes: Maximum number of nodes in subgraph

        Returns:
            str: Path to the created zip file
        """
        try:
            # Get the subgraph
            subgraph_data = self.get_subgraph(query or ".*", max_hops, max_nodes)

            if not subgraph_data["nodes"]:
                raise ValueError(f"No nodes found for partial export of graph {graph_id}")

            # Create a temporary graph from the subgraph data
            from llama_index.core import KnowledgeGraphIndex
            from llama_index.core.graph_stores import SimpleGraphStore

            from ..utf8_import_helper import get_utf8_filesystem

            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(
                graph_store=graph_store, fs=get_utf8_filesystem()
            )

            # Create new knowledge graph index from subgraph
            partial_kg = KnowledgeGraphIndex(storage_context=storage_ctx)

            # Add nodes and edges from subgraph
            for edge in subgraph_data["edges"]:
                source = edge["source"]
                target = edge["target"]
                relation = edge["data"].get("relation", "connected_to")
                triplet = (str(source), str(relation), str(target))
                partial_kg.upsert_triplet_and_node(triplet)

            # Create temporary directory for persistence
            temp_persist_dir = tempfile.mkdtemp()

            try:
                # Persist the partial graph
                partial_kg.storage_context.persist(persist_dir=temp_persist_dir)

                # Create temporary zip file
                temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                temp_zip.close()

                # Create zip archive
                with zipfile.ZipFile(temp_zip.name, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _dirs, files in os.walk(temp_persist_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_persist_dir)
                            zipf.write(file_path, arcname)

                self.logger.info(
                    f"Successfully exported partial graph {graph_id} "
                    f"({len(subgraph_data['nodes'])} nodes, {len(subgraph_data['edges'])} edges)"
                )
                return temp_zip.name

            finally:
                # Clean up temporary persist directory
                shutil.rmtree(temp_persist_dir, ignore_errors=True)

        except Exception as e:
            self.logger.error(f"Error exporting partial graph {graph_id}: {str(e)}")
            raise

    async def import_native(
        self,
        zip_path: str,
        graph_id: str,
        graph_type: str = "modular",
        merge_strategy: str = "union",
        replace: bool = True,
    ) -> str:
        """
        Import a knowledge graph from native LlamaIndex persistence format.

        Args:
            zip_path: Path to the zip file containing the persistence data
            graph_id: ID for the imported graph
            graph_type: "dynamic" or "modular"
            merge_strategy: "union", "prefer_base", or "prefer_incoming" (for dynamic graphs)
            replace: Whether to replace existing graph (modular only)

        Returns:
            str: ID of the imported graph
        """
        try:
            # Create temporary directory for extraction
            temp_dir = tempfile.mkdtemp()

            try:
                # Extract zip file
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    zipf.extractall(temp_dir)

                if graph_type == "modular":
                    # For modular graphs: replace or merge based on replace parameter
                    target_dir = os.path.join(self.prebuilt_directory, graph_id)

                    if replace:
                        # Remove existing directory if it exists
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                    else:
                        # If not replacing and graph exists, merge it
                        if os.path.exists(target_dir) and graph_id in self.kg_indices:
                            # Merge the incoming graph with existing modular graph
                            await self._merge_modular_graph(temp_dir, graph_id, merge_strategy)
                            return graph_id

                    # Move extracted content to target directory
                    shutil.move(temp_dir, target_dir)

                    # Load the index from storage
                    from ..utf8_import_helper import get_utf8_filesystem

                    storage_context = StorageContext.from_defaults(
                        persist_dir=target_dir, fs=get_utf8_filesystem()
                    )
                    kg_index = load_index_from_storage(storage_context=storage_context)

                    # Register the index
                    self.kg_indices[graph_id] = kg_index
                    self.metrics["graphs_loaded"] += 1

                    self.logger.info(f"Successfully imported modular graph {graph_id}")

                elif graph_type == "dynamic":
                    # For dynamic graphs: merge with existing dynamic graph
                    if self.dynamic_kg is None:
                        # If no dynamic graph exists, create one from the import
                        target_dir = os.path.join(self.dynamic_storage, "main")
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        shutil.move(temp_dir, target_dir)

                        from ..utf8_import_helper import get_utf8_filesystem

                        storage_context = StorageContext.from_defaults(
                            persist_dir=target_dir, fs=get_utf8_filesystem()
                        )
                        self.dynamic_kg = load_index_from_storage(storage_context=storage_context)

                        self.logger.info(f"Created new dynamic graph from import {graph_id}")
                    else:
                        # Merge with existing dynamic graph
                        await self._merge_dynamic_graph(temp_dir, merge_strategy)
                        self.logger.info(
                            f"Merged dynamic graph import {graph_id} with existing dynamic graph"
                        )

                else:
                    raise ValueError(f"Unsupported graph_type: {graph_type}")

                return graph_id

            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            self.logger.error(f"Error importing graph {graph_id}: {str(e)}")
            raise
