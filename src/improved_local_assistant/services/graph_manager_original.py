"""
Knowledge Graph Manager for handling LlamaIndex knowledge graph operations.

This module provides the KnowledgeGraphManager class that handles initialization,
configuration, and operations with knowledge graphs using LlamaIndex.
"""

import asyncio
import logging
import os
import sys
import time

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import networkx as nx

# Import LlamaIndex components with consistent style to avoid circular imports
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.ollama import Ollama
from pyvis.network import Network

# Import embedding model (will be used if available)
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    # This will be handled gracefully in the code
    pass

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


class KnowledgeGraphManager:
    """
    Manages knowledge graphs with LlamaIndex integration.

    Handles loading pre-built knowledge graphs, constructing new graphs,
    updating graphs dynamically, and performing graph-based retrieval.
    """

    def __init__(self, model_manager=None, config=None):
        """
        Initialize KnowledgeGraphManager with model manager and configuration.

        Args:
            model_manager: ModelManager instance for model operations
            config: Configuration dictionary
        """
        self.model_manager = model_manager
        self.config = config or {}
        self.kg_indices = {}  # Store multiple knowledge graphs
        self.dynamic_kg = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Extract configuration values
        kg_config = self.config.get("knowledge_graphs", {})
        self.prebuilt_directory = kg_config.get("prebuilt_directory", "./data/prebuilt_graphs")
        self.dynamic_storage = kg_config.get("dynamic_storage", "./data/dynamic_graph")
        self.max_triplets_per_chunk = kg_config.get("max_triplets_per_chunk", 4)
        self.enable_visualization = kg_config.get("enable_visualization", True)

        # Performance metrics
        self.metrics = {
            "graphs_loaded": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "queries_processed": 0,
            "avg_query_time": 0,
            "last_query_time": 0,
        }

        # Dynamic graph persistence tracking
        self._dynamic_updates_count = 0
        self._last_persist_time = time.time()
        self._persist_interval = 300  # 5 minutes
        self._persist_update_threshold = 10  # persist after 10 updates

        # Configure LlamaIndex to use Ollama
        self._configure_llama_index()

    def _safe_get_networkx_graph(self, kg_index, graph_id: str = "unknown") -> Optional[nx.Graph]:
        """
        Safely get NetworkX graph from a knowledge graph index.

        Args:
            kg_index: The knowledge graph index (KnowledgeGraphIndex or PropertyGraphIndex)
            graph_id: ID of the graph for logging purposes

        Returns:
            Optional[nx.Graph]: NetworkX graph or None if not supported
        """
        try:
            # Check if the index has get_networkx_graph method
            if hasattr(kg_index, "get_networkx_graph"):
                return kg_index.get_networkx_graph()
            else:
                # For PropertyGraphIndex, we might need a different approach
                index_type = type(kg_index).__name__
                self.logger.debug(
                    f"Graph {graph_id} ({index_type}) does not support get_networkx_graph()"
                )
                return None
        except Exception as e:
            index_type = type(kg_index).__name__
            self.logger.error(
                f"Error getting NetworkX graph from {graph_id} ({index_type}): {str(e)}"
            )
            return None

    def _configure_llama_index(self):
        """Configure LlamaIndex to use Ollama models and local embeddings."""
        try:
            # Get model configuration
            model_config = self.config.get("models", {}).get("conversation", {})
            model_name = model_config.get("name", "hermes3:3b")
            context_window = model_config.get("context_window", 8000)
            request_timeout = self.config.get("ollama", {}).get("timeout", 120)

            # Configure LlamaIndex to use Ollama for LLM with more resilient settings
            try:
                Settings.llm = Ollama(
                    model=model_name,
                    request_timeout=float(request_timeout),
                    context_window=context_window,
                    num_retries=3,  # Add retries for resilience
                    temperature=0.7,  # Set a reasonable temperature
                )
                self.logger.info(f"Configured LlamaIndex to use Ollama model: {model_name}")
            except Exception as e:
                self.logger.error(f"Error configuring Ollama LLM: {str(e)}")
                self.logger.warning(
                    "Knowledge graph operations may be limited without a working LLM"
                )
                # Don't set a fallback - we'll handle operations without LLM if needed

            # Configure LlamaIndex to use local embeddings
            try:
                # Use a small, efficient embedding model
                EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # quality/speed sweet spot
                Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
                self.logger.info(
                    f"Configured LlamaIndex to use local embeddings: {EMBED_MODEL_NAME}"
                )
            except ImportError:
                self.logger.warning(
                    "Could not import HuggingFaceEmbedding. Make sure sentence-transformers is installed."
                )
                self.logger.warning("Falling back to default embeddings.")

        except Exception as e:
            self.logger.error(f"Error configuring LlamaIndex: {str(e)}")
            self.logger.warning(
                "Knowledge graph operations may be limited due to configuration errors"
            )

    def _load_from_persistence_with_encoding_fix(self, graph_path: str):
        """
        Load a knowledge graph from persistence with robust encoding handling.

        Args:
            graph_path: Path to the graph persistence directory

        Returns:
            KnowledgeGraphIndex or None if loading failed
        """
        import builtins
        import json
        import shutil
        import tempfile

        # Method 1: Try standard loading first
        try:
            self.logger.info("Attempting standard persistence loading")
            storage_context = StorageContext.from_defaults(persist_dir=graph_path)
            kg_index = load_index_from_storage(storage_context=storage_context)
            self.logger.info("Successfully loaded graph using standard method")
            return kg_index
        except Exception as e:
            self.logger.warning(f"Standard loading failed: {str(e)}")

        # Method 2: Try with UTF-8 filesystem if available
        if _UTF8_FILESYSTEM_AVAILABLE:
            try:
                self.logger.info("Attempting UTF-8 filesystem loading")
                fs = UTF8FileSystem()
                storage_context = StorageContext.from_defaults(persist_dir=graph_path, fs=fs)
                kg_index = load_index_from_storage(storage_context=storage_context)
                self.logger.info("Successfully loaded graph using UTF-8 filesystem")
                return kg_index
            except Exception as e:
                self.logger.warning(f"UTF-8 filesystem loading failed: {str(e)}")

        # Method 3: Try with patched builtin open function
        try:
            self.logger.info("Attempting loading with patched open function")

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
                storage_context = StorageContext.from_defaults(persist_dir=graph_path)
                kg_index = load_index_from_storage(storage_context=storage_context)
                self.logger.info("Successfully loaded graph using patched open")
                return kg_index
            finally:
                # Restore original open function
                builtins.open = _orig_open

        except Exception as e:
            self.logger.warning(f"Patched open loading failed: {str(e)}")

        # Method 4: Create clean copies of JSON files and load from temp directory
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
            storage_context = StorageContext.from_defaults(persist_dir=temp_dir)
            kg_index = load_index_from_storage(storage_context=storage_context)
            self.logger.info("Successfully loaded graph using cleaned JSON files")

            # Clean up temp directory in background
            def cleanup_temp():
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

            import threading

            threading.Timer(60, cleanup_temp).start()

            return kg_index

        except Exception as e:
            self.logger.error(f"Cleaned JSON loading failed: {str(e)}")

        # All methods failed
        self.logger.error(f"All loading methods failed for {graph_path}")
        return None

    def load_prebuilt_graphs(self, directory: Optional[str] = None) -> List[str]:
        """
        Load all pre-built knowledge graphs from directory.

        Args:
            directory: Directory containing pre-built knowledge graphs

        Returns:
            List[str]: List of loaded graph IDs
        """
        # Skip all graphs if requested via environment variable
        if os.environ.get("SKIP_ALL_GRAPHS") == "1":
            self.logger.info("Skipping all knowledge graphs as requested via environment variable")
            return []

        directory = directory or self.prebuilt_directory
        loaded_graphs = []

        try:
            # Ensure directory exists
            if not os.path.exists(directory):
                self.logger.warning(f"Pre-built graph directory does not exist: {directory}")
                os.makedirs(directory, exist_ok=True)
                return loaded_graphs

            # Get subdirectories (each should be a separate graph)
            subdirs = [
                d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
            ]

            if not subdirs:
                self.logger.warning(f"No graph directories found in {directory}")
                return loaded_graphs

            # Load each graph
            for subdir in subdirs:
                try:
                    # Skip survivalist graph if requested via environment variable
                    if subdir == "survivalist" and os.environ.get("SKIP_SURVIVALIST_GRAPH") == "1":
                        self.logger.info(
                            "Skipping survivalist graph as requested via environment variable"
                        )
                        continue

                    graph_path = os.path.join(directory, subdir)
                    graph_id = f"prebuilt_{subdir}"

                    # Skip if already loaded
                    if graph_id in self.kg_indices:
                        self.logger.info(f"Graph {graph_id} already loaded, skipping")
                        loaded_graphs.append(graph_id)
                        continue

                    # Load the graph
                    self.logger.info(f"Loading pre-built graph from {graph_path}")

                    # Check if documents exist in the directory
                    if not os.listdir(graph_path):
                        self.logger.warning(f"No documents found in {graph_path}, skipping")
                        continue

                    # Check if this is a pre-built LlamaIndex persistence directory
                    if os.path.exists(os.path.join(graph_path, "graph_store.json")):
                        self.logger.info(f"Found LlamaIndex persistence files in {graph_path}")
                        kg_index = self._load_from_persistence_with_encoding_fix(graph_path)
                        if kg_index is None:
                            continue
                    else:
                        try:
                            # Load documents and create graph
                            docs = SimpleDirectoryReader(graph_path).load_data()

                            # Create graph store and storage context
                            graph_store = SimpleGraphStore()
                            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

                            # Build knowledge graph index - use direct LlamaIndex methods instead of relying on Ollama
                            # This avoids the Ollama API call that's failing
                            kg_index = KnowledgeGraphIndex.from_documents(
                                docs,
                                max_triplets_per_chunk=self.max_triplets_per_chunk,
                                storage_context=storage_ctx,
                                show_progress=True,  # Show progress for long operations
                            )
                        except Exception as e:
                            self.logger.error(f"Error creating graph from documents: {str(e)}")
                            continue

                    # Store the index
                    self.kg_indices[graph_id] = kg_index
                    loaded_graphs.append(graph_id)

                    # Update metrics
                    self.metrics["graphs_loaded"] += 1

                    # Get graph statistics
                    G = self._safe_get_networkx_graph(kg_index, graph_id)
                    if G is not None:
                        self.metrics["total_nodes"] += G.number_of_nodes()
                        self.metrics["total_edges"] += G.number_of_edges()

                    self.logger.info(f"Successfully loaded graph {graph_id}")
                except Exception as e:
                    # Catch exceptions for individual graphs to prevent one bad graph from stopping all loads
                    self.logger.error(f"Error loading graph {subdir}: {str(e)}")
                    continue

            return loaded_graphs

        except Exception as e:
            self.logger.error(f"Error loading pre-built graphs: {str(e)}")
            return loaded_graphs

    def create_graph_from_documents(
        self, docs_path: str, graph_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new knowledge graph from documents.

        Args:
            docs_path: Path to directory containing documents
            graph_id: Optional ID for the graph

        Returns:
            Optional[str]: ID of the created graph, or None if creation failed
        """
        try:
            # Generate graph ID if not provided
            if graph_id is None:
                graph_id = f"graph_{int(time.time())}"

            self.logger.info(f"Creating graph {graph_id} from documents in {docs_path}")

            # Load documents
            docs = SimpleDirectoryReader(docs_path).load_data()

            # Create graph store and storage context
            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

            # Build knowledge graph index
            kg_index = KnowledgeGraphIndex.from_documents(
                docs,
                max_triplets_per_chunk=self.max_triplets_per_chunk,
                storage_context=storage_ctx,
            )

            # Store the index
            self.kg_indices[graph_id] = kg_index

            # Update metrics
            self.metrics["graphs_loaded"] += 1

            # Get graph statistics
            G = self._safe_get_networkx_graph(kg_index, graph_id)
            if G is not None:
                self.metrics["total_nodes"] += G.number_of_nodes()
                self.metrics["total_edges"] += G.number_of_edges()

            self.logger.info(f"Successfully created graph {graph_id}")
            return graph_id

        except Exception as e:
            self.logger.error(f"Error creating graph from documents: {str(e)}")
            return None

    def initialize_dynamic_graph(self) -> bool:
        """
        Initialize the dynamic knowledge graph.

        Returns:
            bool: True if initialization was successful
        """
        try:
            self.logger.info("Initializing dynamic knowledge graph")

            # Ensure we have local embeddings configured
            try:
                # Check if embeddings are already configured
                if Settings.embed_model is None:
                    # Use a small, efficient embedding model
                    EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # quality/speed sweet spot
                    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
                    self.logger.info(
                        f"Configured LlamaIndex to use local embeddings: {EMBED_MODEL_NAME}"
                    )
            except ImportError:
                self.logger.warning(
                    "Could not import HuggingFaceEmbedding. Make sure sentence-transformers is installed."
                )
                self.logger.warning("Falling back to default embeddings.")

            # Create graph store and storage context
            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

            # Create empty knowledge graph index
            self.dynamic_kg = KnowledgeGraphIndex(storage_context=storage_ctx)

            self.logger.info("Successfully initialized dynamic knowledge graph")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing dynamic graph: {str(e)}")
            return False

    async def update_dynamic_graph(self, conversation_text: str) -> bool:
        """
        Update dynamic knowledge graph with new conversation data.

        Args:
            conversation_text: Text from conversation to extract entities from

        Returns:
            bool: True if update was successful
        """
        try:
            # Initialize dynamic graph if it doesn't exist
            if not self.dynamic_kg:
                if not self.initialize_dynamic_graph():
                    return False

            # Extract entities in background using TinyLlama
            if self.model_manager:
                self.logger.info("Extracting entities from conversation text")
                entities_text = await self.model_manager.query_knowledge_model(conversation_text)

                # Process entities in background
                asyncio.create_task(self._process_entities(entities_text.get("content", "")))

                # Check if we need to persist
                self._dynamic_updates_count += 1
                await self._maybe_persist_dynamic_graph()

                return True
            else:
                self.logger.warning("Model manager not available, skipping entity extraction")
                return False

        except Exception as e:
            self.logger.error(f"Error updating dynamic graph: {str(e)}")
            return False

    async def _process_entities(self, entities_text: str) -> None:
        """
        Process extracted entities and update graph.

        Args:
            entities_text: Text containing extracted entities
        """
        try:
            # Parse entities and create triplets
            # This is a simplified implementation - in a real system,
            # you would need more sophisticated parsing
            lines = entities_text.strip().split("\n")
            triplets = []

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("Triples:"):
                    continue

                # Try to extract triplet from line
                # Format could be: (subject, relation, object) or subject, relation, object
                line = line.strip("()")
                parts = [p.strip().strip("\"'") for p in line.split(",")]

                if len(parts) >= 3:
                    subject = parts[0]
                    relation = parts[1]
                    obj = parts[2]

                    if subject and relation and obj:
                        triplets.append((subject, relation, obj))

            # Update graph with triplets
            for triplet in triplets:
                self.logger.debug(f"Adding triplet to dynamic graph: {triplet}")
                self.dynamic_kg.upsert_triplet_and_node(triplet)

            self.logger.info(f"Added {len(triplets)} triplets to dynamic graph")

            # Update metrics
            if triplets:
                G = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
                if G is not None:
                    self.metrics["total_nodes"] = G.number_of_nodes()
                    self.metrics["total_edges"] = G.number_of_edges()

        except Exception as e:
            self.logger.error(f"Error processing entities: {str(e)}")

    def query_graphs(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Query all knowledge graphs using ensemble method.

        Args:
            query: Query string
            context: Optional context for the query

        Returns:
            Dict[str, Any]: Query results
        """
        start_time = time.time()

        # Try to use optimizer if available
        cached_result = None
        optimized_query = query

        try:
            # Import optimizer here to avoid circular imports
            from services.kg_optimizer import cache_query_result
            from services.kg_optimizer import get_cached_query
            from services.kg_optimizer import optimize_query

            # Check cache first
            cached_result = get_cached_query(query)
            if cached_result:
                self.logger.info(f"Using cached result for query: {query}")
                return cached_result

            # Optimize query if possible
            try:
                import asyncio

                if asyncio.get_event_loop().is_running():
                    optimized_query = asyncio.run_coroutine_threadsafe(
                        optimize_query(query), asyncio.get_event_loop()
                    ).result()
                else:
                    optimized_query = asyncio.run(optimize_query(query))

                if optimized_query != query:
                    self.logger.info(f"Optimized query: '{query}' -> '{optimized_query}'")
            except Exception as e:
                self.logger.warning(f"Error optimizing query: {str(e)}")

        except Exception as e:
            self.logger.warning(f"Knowledge graph optimizer not available: {str(e)}")
            # Continue without optimization

        try:
            query_engines = []
            graph_sources = []

            # Add all knowledge graph query engines
            for graph_id, kg_index in self.kg_indices.items():
                qe = kg_index.as_query_engine(
                    include_text=False, response_mode="tree_summarize"
                )  # KG-only answers
                query_engines.append(qe)
                graph_sources.append(graph_id)

            # Add dynamic graph if available
            if self.dynamic_kg:
                dynamic_qe = self.dynamic_kg.as_query_engine(
                    include_text=False, response_mode="tree_summarize"
                )
                query_engines.append(dynamic_qe)
                graph_sources.append("dynamic")

            # If no query engines available, return empty result
            if not query_engines:
                self.logger.warning("No knowledge graphs available for querying")
                empty_result = {
                    "response": "No knowledge graphs available to answer the query.",
                    "source_nodes": [],
                    "metadata": {"graph_sources": [], "query_time": 0},
                }
                return empty_result

            # Use ensemble for weighted voting if multiple engines
            if len(query_engines) > 1:
                # Create a combined query engine with weighted voting
                from llama_index.query_engine.ensemble import EnsembleQueryEngine

                # Use EnsembleQueryEngine for weighted voting across multiple engines
                weights = [1.0 / len(query_engines)] * len(query_engines)
                ensemble = EnsembleQueryEngine(query_engines, weights=weights)
                result = ensemble.query(optimized_query)
            else:
                result = query_engines[0].query(optimized_query)

            # Calculate query time
            query_time = time.time() - start_time

            # Update metrics
            self.metrics["queries_processed"] += 1
            self.metrics["last_query_time"] = query_time

            # Update rolling average
            if self.metrics["queries_processed"] > 1:
                self.metrics["avg_query_time"] = (
                    self.metrics["avg_query_time"] * (self.metrics["queries_processed"] - 1)
                    + query_time
                ) / self.metrics["queries_processed"]
            else:
                self.metrics["avg_query_time"] = query_time

            # Prepare response
            query_result = {
                "response": result.response,
                "source_nodes": result.source_nodes if hasattr(result, "source_nodes") else [],
                "metadata": {
                    "graph_sources": graph_sources,
                    "query_time": query_time,
                    "optimized_query": optimized_query if optimized_query != query else None,
                },
            }

            # Cache the result for future use if optimizer is available
            try:
                cache_query_result(query, query_result)
            except Exception as e:
                self.logger.debug(f"Could not cache query result: {str(e)}")

            return query_result

        except Exception as e:
            self.logger.error(f"Error querying graphs: {str(e)}")
            query_time = time.time() - start_time

            # Import error handler
            from services.error_handler import handle_error

            # Get user-friendly error message
            error_response = handle_error(
                e, context={"query": query}, error_code="KNOWLEDGE_GRAPH_QUERY_ERROR"
            )

            error_result = {
                "response": (
                    f"I couldn't find relevant information in my knowledge base. {error_response['suggestion']}"
                ),
                "source_nodes": [],
                "metadata": {"error": str(e), "query_time": query_time},
            }

            return error_result

    def visualize_graph(self, graph_id: Optional[str] = None) -> str:
        """
        Generate HTML visualization of knowledge graph.

        Args:
            graph_id: ID of the graph to visualize, or None for dynamic graph

        Returns:
            str: HTML visualization
        """
        try:
            # Select graph to visualize
            if graph_id and graph_id in self.kg_indices:
                kg_index = self.kg_indices[graph_id]
                title = f"Knowledge Graph: {graph_id}"
            elif self.dynamic_kg:
                kg_index = self.dynamic_kg
                title = "Dynamic Knowledge Graph"
            else:
                return "<p>No graph available for visualization</p>"

            # Export to NetworkX
            G = self._safe_get_networkx_graph(kg_index, graph_id or "dynamic")
            if G is None:
                return "<p>This graph type does not support NetworkX visualization</p>"

            # Create PyVis network
            net = Network(height="600px", width="100%", directed=True)
            net.from_nx(G)

            # Add title
            net.set_options(
                """
            var options = {
                "nodes": {
                    "font": {
                        "size": 12
                    }
                },
                "edges": {
                    "font": {
                        "size": 10
                    },
                    "smooth": false
                },
                "physics": {
                    "stabilization": true,
                    "barnesHut": {
                        "springLength": 150
                    }
                }
            }
            """
            )

            # Generate HTML
            html_file = "temp_graph.html"
            net.save_graph(html_file)

            with open(html_file) as f:
                html_content = f.read()

            # Add title to HTML
            html_content = html_content.replace("<body>", f"<body><h2>{title}</h2>")

            # Clean up temporary file
            try:
                os.remove(html_file)
            except (OSError, FileNotFoundError):
                pass

            return html_content

        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            return f"<p>Error visualizing graph: {str(e)}</p>"

    def add_new_graph(self, graph_path: str, graph_id: Optional[str] = None) -> Optional[str]:
        """
        Add a new pre-built graph at runtime.

        Args:
            graph_path: Path to the graph directory
            graph_id: Optional ID for the graph

        Returns:
            Optional[str]: ID of the added graph, or None if addition failed
        """
        try:
            # Generate graph ID if not provided
            if graph_id is None:
                graph_id = f"runtime_{len(self.kg_indices)}"

            self.logger.info(f"Adding new graph {graph_id} from {graph_path}")

            # Check if graph ID already exists
            if graph_id in self.kg_indices:
                self.logger.warning(f"Graph ID {graph_id} already exists, using a different ID")
                graph_id = f"{graph_id}_{int(time.time())}"

            # Load documents and create graph
            docs = SimpleDirectoryReader(graph_path).load_data()
            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

            kg_index = KnowledgeGraphIndex.from_documents(
                docs,
                max_triplets_per_chunk=self.max_triplets_per_chunk,
                storage_context=storage_ctx,
            )

            # Store the index
            self.kg_indices[graph_id] = kg_index

            # Update metrics
            self.metrics["graphs_loaded"] += 1

            # Get graph statistics
            G = self._safe_get_networkx_graph(kg_index, graph_id)
            if G is not None:
                self.metrics["total_nodes"] += G.number_of_nodes()
                self.metrics["total_edges"] += G.number_of_edges()

            self.logger.info(f"Successfully added graph {graph_id}")
            return graph_id

        except Exception as e:
            self.logger.error(f"Error adding new graph: {str(e)}")
            return None

    async def export_native(self, graph_id: str) -> str:
        """
        Export a knowledge graph in native LlamaIndex persistence format.

        Args:
            graph_id: ID of the graph to export

        Returns:
            str: Path to the created zip file
        """
        try:
            import shutil
            import tempfile
            import zipfile

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
                for root, dirs, files in os.walk(persist_dir):
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
            import shutil
            import tempfile
            import zipfile

            # Get the subgraph
            subgraph_data = self.get_subgraph(query or ".*", max_hops, max_nodes)

            if not subgraph_data["nodes"]:
                raise ValueError(f"No nodes found for partial export of graph {graph_id}")

            # Create a temporary graph from the subgraph data
            from llama_index.core.graph_stores import SimpleGraphStore

            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

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
                    for root, dirs, files in os.walk(temp_persist_dir):
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
            import shutil
            import tempfile
            import zipfile

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
                    storage_context = StorageContext.from_defaults(persist_dir=target_dir)
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

                        storage_context = StorageContext.from_defaults(persist_dir=target_dir)
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

    async def _merge_dynamic_graph(self, incoming_persist_dir: str, merge_strategy: str = "union"):
        """
        Merge an incoming graph with the existing dynamic graph.

        Args:
            incoming_persist_dir: Directory containing the incoming graph persistence data
            merge_strategy: Strategy for handling conflicts
        """
        try:
            # Load the incoming graph
            incoming_storage = StorageContext.from_defaults(persist_dir=incoming_persist_dir)
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
            incoming_storage = StorageContext.from_defaults(persist_dir=incoming_persist_dir)
            incoming_kg = load_index_from_storage(storage_context=incoming_storage)

            # Get the existing modular graph
            if graph_id not in self.kg_indices:
                raise ValueError(f"Modular graph {graph_id} not found for merging")

            existing_kg = self.kg_indices[graph_id]

            # Get NetworkX representations for merging
            existing_graph = self._safe_get_networkx_graph(existing_kg, existing_graph_id)
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

    async def _rebuild_modular_from_networkx(self, networkx_graph, graph_id: str):
        """
        Rebuild a modular knowledge graph from a NetworkX graph.

        Args:
            networkx_graph: NetworkX graph to rebuild from
            graph_id: ID of the modular graph
        """
        try:
            # Create new graph store and storage context
            from llama_index.core.graph_stores import SimpleGraphStore

            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

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
        from datetime import datetime

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
            from llama_index.core.graph_stores import SimpleGraphStore

            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

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

    def get_subgraph(self, query: str, max_hops: int = 2, max_nodes: int = 100) -> Dict[str, Any]:
        """
        Extract a relevant subgraph for GraphRAG queries.

        Args:
            query: Query string to find relevant nodes
            max_hops: Maximum hops from relevant nodes
            max_nodes: Maximum number of nodes in subgraph

        Returns:
            Dict containing subgraph data
        """
        try:
            import networkx as nx

            # Get all available graphs
            all_graphs = []
            graph_sources = []

            # Add modular graphs
            for graph_id, kg_index in self.kg_indices.items():
                G = self._safe_get_networkx_graph(kg_index, graph_id)
                if G is not None:
                    all_graphs.append(G)
                    graph_sources.append(graph_id)

            # Add dynamic graph
            if self.dynamic_kg:
                G = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
                if G is not None:
                    all_graphs.append(G)
                    graph_sources.append("dynamic")

            if not all_graphs:
                return {"nodes": [], "edges": [], "metadata": {"query": query, "sources": []}}

            # Find relevant nodes based on query
            relevant_nodes = set()
            query_lower = query.lower()

            for i, G in enumerate(all_graphs):
                for node, data in G.nodes(data=True):
                    node_text = str(node).lower()
                    # Simple text matching - could be enhanced with embeddings
                    if query_lower in node_text or any(
                        query_lower in str(v).lower() for v in data.values()
                    ):
                        relevant_nodes.add((node, i))  # Store node with graph index

            # If no relevant nodes found, return empty subgraph
            if not relevant_nodes:
                return {
                    "nodes": [],
                    "edges": [],
                    "metadata": {"query": query, "sources": graph_sources},
                }

            # Expand around relevant nodes
            subgraph_nodes = set()
            subgraph_edges = []

            for node, graph_idx in relevant_nodes:
                G = all_graphs[graph_idx]

                # Get neighborhood around this node
                try:
                    neighborhood = nx.single_source_shortest_path_length(G, node, cutoff=max_hops)
                    for neighbor in neighborhood.keys():
                        subgraph_nodes.add((neighbor, graph_idx, graph_sources[graph_idx]))

                        # Stop if we've reached max nodes
                        if len(subgraph_nodes) >= max_nodes:
                            break
                except nx.NetworkXError:
                    # Node might not exist in graph, skip
                    continue

                if len(subgraph_nodes) >= max_nodes:
                    break

            # Collect edges within the subgraph
            node_set = {(node, graph_idx) for node, graph_idx, _ in subgraph_nodes}

            for node, graph_idx, source in subgraph_nodes:
                G = all_graphs[graph_idx]
                for neighbor in G.neighbors(node):
                    if (neighbor, graph_idx) in node_set:
                        edge_data = G.edges[node, neighbor]
                        subgraph_edges.append(
                            {
                                "source": node,
                                "target": neighbor,
                                "data": edge_data,
                                "graph_source": source,
                            }
                        )

            # Format response
            nodes_data = []
            for node, graph_idx, source in subgraph_nodes:
                G = all_graphs[graph_idx]
                node_data = G.nodes.get(node, {})
                nodes_data.append({"id": node, "data": node_data, "graph_source": source})

            return {
                "nodes": nodes_data,
                "edges": subgraph_edges,
                "metadata": {
                    "query": query,
                    "max_hops": max_hops,
                    "max_nodes": max_nodes,
                    "total_nodes": len(nodes_data),
                    "total_edges": len(subgraph_edges),
                    "sources": graph_sources,
                    "relevant_nodes_found": len(relevant_nodes),
                },
            }

        except Exception as e:
            self.logger.error(f"Error extracting subgraph: {str(e)}")
            return {"nodes": [], "edges": [], "metadata": {"query": query, "error": str(e)}}

    def get_graph_traversal(self, source: str, target: str, max_hops: int = 3) -> List[List[str]]:
        """
        Perform multi-hop graph traversal using NetworkX.

        Args:
            source: Source node
            target: Target node
            max_hops: Maximum number of hops

        Returns:
            List[List[str]]: List of paths from source to target
        """
        try:
            # Try dynamic graph first
            if self.dynamic_kg:
                G = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
                if G is not None:
                    try:
                        paths = list(
                            nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops)
                        )
                        if paths:
                            return paths
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

            # Try all pre-built graphs
            all_paths = []
            for graph_id, kg_index in self.kg_indices.items():
                G = self._safe_get_networkx_graph(kg_index, graph_id)
                if G is not None:
                    try:
                        paths = list(
                            nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops)
                        )
                        if paths:
                            all_paths.extend(paths)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

            return all_paths

        except Exception as e:
            self.logger.error(f"Error in graph traversal: {str(e)}")
            return []

    def register_index(self, graph_id: str, kg_index, graph_type: str = "modular"):
        """
        Register a knowledge graph index with the manager.

        Args:
            graph_id: ID for the graph
            kg_index: LlamaIndex KnowledgeGraphIndex instance
            graph_type: Type of graph ("modular" or "dynamic")
        """
        try:
            if graph_type == "dynamic":
                self.dynamic_kg = kg_index
                self.logger.info(f"Registered dynamic graph: {graph_id}")
            else:
                self.kg_indices[graph_id] = kg_index
                self.metrics["graphs_loaded"] += 1
                self.logger.info(f"Registered modular graph: {graph_id}")

            # Update metrics
            try:
                G = kg_index.get_networkx_graph()
                self.metrics["total_nodes"] += G.number_of_nodes()
                self.metrics["total_edges"] += G.number_of_edges()
            except Exception as e:
                self.logger.error(f"Error updating graph statistics: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error registering graph {graph_id}: {str(e)}")
            raise

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
                self._dynamic_updates_count = 0
                self._last_persist_time = current_time
                self.logger.info("Persisted dynamic graph (periodic checkpoint)")

        except Exception as e:
            self.logger.error(f"Error in periodic persistence: {str(e)}")

    async def _persist_dynamic_graph(self):
        """
        Persist the dynamic graph to disk.
        """
        try:
            if not self.dynamic_kg:
                return

            persist_dir = os.path.join(self.dynamic_storage, "main")
            os.makedirs(persist_dir, exist_ok=True)

            # Run persistence in thread to avoid blocking
            await asyncio.to_thread(
                self.dynamic_kg.storage_context.persist, persist_dir=persist_dir
            )

        except Exception as e:
            self.logger.error(f"Error persisting dynamic graph: {str(e)}")
            raise

        except Exception as e:
            self.logger.error(f"Error in graph traversal: {str(e)}")
            return []

    def rebuild_graphs_with_new_embeddings(self) -> Dict[str, bool]:
        """
        Rebuild all knowledge graphs with the current embedding model.

        This is useful when changing embedding models to avoid mixed-dimensionality errors.

        Returns:
            Dict[str, bool]: Dictionary of graph IDs and whether they were successfully rebuilt
        """
        results = {}

        try:
            # Rebuild pre-built graphs
            for graph_id, kg_index in list(self.kg_indices.items()):
                try:
                    self.logger.info(f"Rebuilding graph {graph_id} with new embedding model")

                    # Get the original documents or data
                    # This is a simplified approach - in a real system, you might need to store
                    # the original documents or have a more sophisticated way to rebuild

                    # For this implementation, we'll extract the triplets from the existing graph
                    # and rebuild using those
                    G = self._safe_get_networkx_graph(kg_index, graph_id)
                    triplets = []

                    if G is not None:
                        for u, v, data in G.edges(data=True):
                            relation = data.get("relation", "related_to")
                            triplets.append((u, relation, v))

                    # Create a new graph store and storage context
                    graph_store = SimpleGraphStore()
                    storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

                    # Create a new knowledge graph index
                    new_kg_index = KnowledgeGraphIndex(storage_context=storage_ctx)

                    # Add all triplets to the new graph
                    for triplet in triplets:
                        new_kg_index.upsert_triplet_and_node(triplet)

                    # Replace the old index with the new one
                    self.kg_indices[graph_id] = new_kg_index

                    self.logger.info(f"Successfully rebuilt graph {graph_id}")
                    results[graph_id] = True

                except Exception as e:
                    self.logger.error(f"Error rebuilding graph {graph_id}: {str(e)}")
                    results[graph_id] = False

            # Rebuild dynamic graph if it exists
            if self.dynamic_kg:
                try:
                    self.logger.info("Rebuilding dynamic graph with new embedding model")

                    # Extract triplets from the existing dynamic graph
                    G = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
                    triplets = []

                    if G is not None:
                        for u, v, data in G.edges(data=True):
                            relation = data.get("relation", "related_to")
                            triplets.append((u, relation, v))

                    # Create a new graph store and storage context
                    graph_store = SimpleGraphStore()
                    storage_ctx = StorageContext.from_defaults(graph_store=graph_store)

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
