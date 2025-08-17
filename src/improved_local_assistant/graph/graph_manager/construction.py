"""
Construction module for KnowledgeGraphManager.

Handles graph creation and dynamic updates including document processing
and entity extraction.
"""

import asyncio
import json
import os
import time
from datetime import datetime

from llama_index.core import KnowledgeGraphIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import PropertyGraphIndex


class KnowledgeGraphConstruction:
    """Handles construction and dynamic updates of knowledge graphs."""

    def load_prebuilt_graphs(self, directory: str | None = None) -> list[str]:
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

                    # Skip test graphs that interfere with real content
                    if (
                        subdir in ["new_graph", "sample_graph"]
                        and os.environ.get("SKIP_TEST_GRAPHS") == "1"
                    ):
                        self.logger.info(
                            f"Skipping test graph {subdir} as requested via environment variable"
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
                        # Try standard loading first (should work with UTF-8 files)
                        try:
                            from ..utf8_import_helper import get_utf8_filesystem

                            storage_context = StorageContext.from_defaults(
                                persist_dir=graph_path, fs=get_utf8_filesystem()
                            )
                            # Handle multiple indices by loading all and selecting the knowledge graph
                            try:
                                from llama_index.core.indices.loading import (
                                    load_indices_from_storage,
                                )

                                indices = load_indices_from_storage(storage_context)

                                if len(indices) == 1:
                                    kg_index = indices[0]
                                    self.logger.info(
                                        f"Successfully loaded single index for {graph_id}"
                                    )
                                else:
                                    # Multiple indices - prefer KnowledgeGraphIndex
                                    kg_index = None
                                    for idx in indices:
                                        if hasattr(idx, "graph_store") or "KnowledgeGraph" in str(
                                            type(idx)
                                        ):
                                            kg_index = idx
                                            self.logger.info(
                                                f"Successfully loaded knowledge graph index from {len(indices)} indices for {graph_id}"
                                            )
                                            break

                                    if kg_index is None:
                                        # Use the first available index as fallback
                                        kg_index = indices[0]
                                        self.logger.info(
                                            f"Using first available index from {len(indices)} indices as fallback for {graph_id}"
                                        )

                            except Exception as multi_error:
                                self.logger.warning(
                                    f"Failed to load multiple indices for {graph_id}: {multi_error}"
                                )
                                # Final fallback: try single index load
                                try:
                                    kg_index = load_index_from_storage(
                                        storage_context=storage_context
                                    )
                                    self.logger.info(
                                        f"Successfully loaded using single index fallback for {graph_id}"
                                    )
                                except Exception as final_error:
                                    self.logger.error(
                                        f"All loading methods failed for {graph_id}: {final_error}"
                                    )
                                    raise final_error
                        except Exception as e:
                            self.logger.warning(
                                f"Standard loading failed: {str(e)}, trying encoding fix"
                            )
                            # Import and use the persistence method
                            from .persistence import KnowledgeGraphPersistence

                            persistence = KnowledgeGraphPersistence()
                            kg_index = persistence._load_from_persistence_with_encoding_fix(
                                graph_path
                            )
                            if kg_index is None:
                                continue
                    else:
                        try:
                            # Load documents and create graph
                            docs = SimpleDirectoryReader(graph_path).load_data()

                            # Check config for graph type
                            graph_type = (
                                getattr(self, "config", {}).get("graph", {}).get("type", "simple")
                            )

                            # Create graph store and storage context based on type
                            from ..utf8_import_helper import get_utf8_filesystem

                            if graph_type == "property":
                                from llama_index.core.indices.property_graph import (
                                    SchemaLLMPathExtractor,
                                )

                                # Define schema for chat-memory aware extraction
                                SCHEMA = {
                                    "entities": [
                                        "Person",
                                        "Utterance",
                                        "Preference",
                                        "Goal",
                                        "Task",
                                        "Fact",
                                        "Episode",
                                        "CommunitySummary",
                                        "Tool",
                                        "Doc",
                                        "Claim",
                                        "Topic",
                                    ],
                                    "relations": [
                                        "MENTIONS",
                                        "ASSERTS",
                                        "REFERS_TO",
                                        "PREFERS",
                                        "GOAL_OF",
                                        "RELATES_TO",
                                        "SUMMARIZES",
                                        "CITES",
                                        "DERIVED_FROM",
                                        "SAID_BY",
                                    ],
                                }

                                graph_store = SimplePropertyGraphStore()
                                storage_ctx = StorageContext.from_defaults(
                                    property_graph_store=graph_store, fs=get_utf8_filesystem()
                                )

                                # Use schema-guided extractor for stable types
                                extractor = SchemaLLMPathExtractor(schema=SCHEMA)

                                # Build property graph index with schema guidance
                                kg_index = PropertyGraphIndex.from_documents(
                                    docs,
                                    storage_context=storage_ctx,
                                    kg_extractors=[extractor],
                                    include_embeddings=True,
                                    show_progress=True,
                                )
                            else:
                                graph_store = SimpleGraphStore()
                                storage_ctx = StorageContext.from_defaults(
                                    graph_store=graph_store, fs=get_utf8_filesystem()
                                )

                                # Build knowledge graph index with improved settings
                                kg_index = KnowledgeGraphIndex.from_documents(
                                    docs,
                                    max_triplets_per_chunk=max(
                                        8, self.max_triplets_per_chunk
                                    ),  # Better for dense data
                                    include_embeddings=True,  # Enable vector search
                                    storage_context=storage_ctx,
                                    show_progress=True,  # Show progress for long operations
                                )
                        except Exception as e:
                            error_msg = str(e)
                            if "cudaMalloc failed: out of memory" in error_msg:
                                self.logger.error(
                                    f"CUDA out of memory error for {graph_path}. Consider using CPU-only mode or reducing model size."
                                )
                            elif "llama runner process has terminated" in error_msg:
                                self.logger.error(
                                    f"Ollama model crashed for {graph_path}. This may be due to insufficient GPU memory or model issues."
                                )
                            else:
                                self.logger.error(
                                    f"Error creating graph from documents: {error_msg}"
                                )
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
        self, docs_path: str, graph_id: str | None = None
    ) -> str | None:
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

            # Check config for graph type
            graph_type = getattr(self, "config", {}).get("graph", {}).get("type", "simple")

            # Create graph store and storage context based on type
            from ..utf8_import_helper import get_utf8_filesystem

            if graph_type == "property":
                from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

                # Define schema for chat-memory aware extraction
                SCHEMA = {
                    "entities": [
                        "Person",
                        "Utterance",
                        "Preference",
                        "Goal",
                        "Task",
                        "Fact",
                        "Episode",
                        "CommunitySummary",
                        "Tool",
                        "Doc",
                        "Claim",
                        "Topic",
                    ],
                    "relations": [
                        "MENTIONS",
                        "ASSERTS",
                        "REFERS_TO",
                        "PREFERS",
                        "GOAL_OF",
                        "RELATES_TO",
                        "SUMMARIZES",
                        "CITES",
                        "DERIVED_FROM",
                        "SAID_BY",
                    ],
                }

                graph_store = SimplePropertyGraphStore()
                storage_ctx = StorageContext.from_defaults(
                    property_graph_store=graph_store, fs=get_utf8_filesystem()
                )

                # Use schema-guided extractor for stable types
                extractor = SchemaLLMPathExtractor(schema=SCHEMA)

                # Build property graph index with schema guidance
                kg_index = PropertyGraphIndex.from_documents(
                    docs,
                    storage_context=storage_ctx,
                    kg_extractors=[extractor],
                    include_embeddings=True,
                )
            else:
                graph_store = SimpleGraphStore()
                storage_ctx = StorageContext.from_defaults(
                    graph_store=graph_store, fs=get_utf8_filesystem()
                )

                # Build knowledge graph index with improved settings
                kg_index = KnowledgeGraphIndex.from_documents(
                    docs,
                    max_triplets_per_chunk=max(
                        8, self.max_triplets_per_chunk
                    ),  # Better for dense data
                    include_embeddings=True,  # Enable vector search
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

            # Use centralized embedding model (guard against missing model)
            from llama_index.core import Settings

            if Settings.embed_model is None:
                self.logger.warning("Embed model not initialized; falling back to default")
                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                    EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
                    Settings.embed_model = HuggingFaceEmbedding(
                        model_name=EMBED_MODEL_NAME, device="cpu"
                    )
                    self.logger.info(f"Fallback: Configured embedding model: {EMBED_MODEL_NAME}")
                except Exception as e:
                    self.logger.warning(f"Could not configure fallback embedding model: {e}")
            else:
                self.logger.debug("Using centralized embedding model for dynamic graph")

            # Check config for graph type
            graph_type = getattr(self, "config", {}).get("graph", {}).get("type", "simple")

            # Create graph store and storage context based on type
            from ..utf8_import_helper import get_utf8_filesystem

            if graph_type == "property":
                graph_store = SimplePropertyGraphStore()
                storage_ctx = StorageContext.from_defaults(
                    property_graph_store=graph_store, fs=get_utf8_filesystem()
                )
            else:
                graph_store = SimpleGraphStore()
                storage_ctx = StorageContext.from_defaults(
                    graph_store=graph_store, fs=get_utf8_filesystem()
                )

            # Try to load existing dynamic graph first
            persist_dir = os.path.join(self.dynamic_storage, "main")
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                try:
                    self.logger.info("Loading existing dynamic graph from persistence")
                    from ..utf8_import_helper import get_utf8_filesystem

                    # Check if this is a property graph
                    graph_meta_path = os.path.join(persist_dir, "graph_meta.json")
                    existing_graph_type = "simple"  # default

                    if os.path.exists(graph_meta_path):
                        try:
                            with open(graph_meta_path, encoding="utf-8") as f:
                                meta = json.load(f)
                                existing_graph_type = meta.get("graph_type", "simple")
                        except Exception:
                            pass

                    # If graph types don't match, recreate the graph
                    if existing_graph_type != graph_type:
                        self.logger.info(
                            f"Graph type mismatch (existing: {existing_graph_type}, requested: {graph_type}), recreating..."
                        )
                        # Remove existing files to force recreation
                        import shutil

                        shutil.rmtree(persist_dir)
                        os.makedirs(persist_dir, exist_ok=True)
                    else:
                        # Load existing graph
                        storage_ctx = StorageContext.from_defaults(
                            persist_dir=persist_dir, fs=get_utf8_filesystem()
                        )

                        # Handle multiple indices by loading all and selecting the appropriate graph
                        try:
                            from llama_index.core.indices.loading import load_indices_from_storage

                            indices = load_indices_from_storage(storage_ctx)

                            if len(indices) == 1:
                                self.dynamic_kg = indices[0]
                                self.logger.info("Successfully loaded single dynamic graph index")
                            else:
                                # Multiple indices - prefer the right type
                                for idx in indices:
                                    if graph_type == "property" and isinstance(
                                        idx, PropertyGraphIndex  # noqa: F823
                                    ):
                                        self.dynamic_kg = idx
                                        self.logger.info(
                                            f"Successfully loaded dynamic PropertyGraphIndex from {len(indices)} indices"
                                        )
                                        break
                                    elif graph_type == "simple" and (
                                        hasattr(idx, "graph_store")
                                        or "KnowledgeGraph" in str(type(idx))
                                    ):
                                        self.dynamic_kg = idx
                                        self.logger.info(
                                            f"Successfully loaded dynamic KnowledgeGraphIndex from {len(indices)} indices"
                                        )
                                        break

                                if self.dynamic_kg is None:
                                    # Use the first available index as fallback
                                    self.dynamic_kg = indices[0]
                                    self.logger.info(
                                        f"Using first available dynamic index from {len(indices)} indices as fallback"
                                    )

                        except Exception as multi_error:
                            self.logger.warning(
                                f"Failed to load multiple dynamic indices: {multi_error}"
                            )
                            # Final fallback: try single index load
                            self.dynamic_kg = load_index_from_storage(storage_context=storage_ctx)
                            self.logger.info(
                                "Successfully loaded dynamic graph using single index fallback"
                            )

                        return True

                except Exception as e:
                    self.logger.warning(f"Failed to load existing dynamic graph: {e}")
                    # Continue with creating new graph

            # Create knowledge graph index with a system context document
            from llama_index.core import Document

            system_doc = Document(
                text=f"System: Dynamic knowledge graph initialized on {datetime.now().isoformat()}. "
                f"This graph will be updated with entities and relationships from conversations.",
                metadata={"source": "system", "type": "initialization"},
            )

            # Create index based on graph type
            if graph_type == "property":
                # For property graphs, create with schema-guided extractors for chat memory
                try:
                    from llama_index.core.indices.property_graph import PropertyGraphIndex
                    from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

                    # Define schema for chat-memory aware extraction
                    SCHEMA = {
                        "entities": [
                            "Person",
                            "Utterance",
                            "Preference",
                            "Goal",
                            "Task",
                            "Fact",
                            "Episode",
                            "CommunitySummary",
                            "Tool",
                            "Doc",
                            "Claim",
                            "Topic",
                        ],
                        "relations": [
                            "MENTIONS",
                            "ASSERTS",
                            "REFERS_TO",
                            "PREFERS",
                            "GOAL_OF",
                            "RELATES_TO",
                            "SUMMARIZES",
                            "CITES",
                            "DERIVED_FROM",
                            "SAID_BY",
                        ],
                    }

                    # Use schema-guided extractor for stable types with KG LLM
                    # Get the KG LLM from model manager if available
                    kg_llm = None
                    try:
                        # Try to get KG LLM from orchestrated model manager
                        if hasattr(self, "model_manager") and hasattr(self.model_manager, "kg_llm"):
                            kg_llm = self.model_manager.kg_llm
                        # Or check if it's set in Settings
                        elif hasattr(Settings, "llm") and Settings.llm is not None:
                            # Use global LLM but log that we're using it for KG
                            kg_llm = Settings.llm
                            self.logger.info("Using global Settings.llm for KG extraction")
                    except Exception as e:
                        self.logger.warning(f"Could not get KG LLM: {e}")

                    if kg_llm:
                        extractor = SchemaLLMPathExtractor(schema=SCHEMA, llm=kg_llm)
                        self.logger.info(f"Using {type(kg_llm).__name__} for KG extraction")
                    else:
                        extractor = SchemaLLMPathExtractor(schema=SCHEMA)
                        self.logger.warning("No specific KG LLM found, using default")

                    self.dynamic_kg = PropertyGraphIndex.from_documents(
                        [system_doc],
                        storage_context=storage_ctx,
                        kg_extractors=[extractor],
                        include_embeddings=True,
                        show_progress=False,
                    )
                    self.logger.info(
                        "Successfully created schema-guided PropertyGraphIndex for dynamic graph"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"PropertyGraphIndex creation failed: {e}, falling back to KnowledgeGraphIndex"
                    )
                    # Fallback to KnowledgeGraphIndex if PropertyGraphIndex fails
                    self.dynamic_kg = KnowledgeGraphIndex.from_documents(
                        [system_doc],
                        storage_context=storage_ctx,
                        max_triplets_per_chunk=self.max_triplets_per_chunk,
                        show_progress=False,
                    )
            else:
                self.dynamic_kg = KnowledgeGraphIndex.from_documents(
                    [system_doc],
                    storage_context=storage_ctx,
                    max_triplets_per_chunk=self.max_triplets_per_chunk,
                    show_progress=False,
                )

            # Persist the initial graph
            os.makedirs(persist_dir, exist_ok=True)
            self.dynamic_kg.storage_context.persist(persist_dir=persist_dir)

            # Create graph_meta.json to indicate graph type
            graph_meta = {
                "graph_type": graph_type,
                "created_at": datetime.now().isoformat(),
                "description": "Dynamic knowledge graph for conversation entities",
            }

            with open(os.path.join(persist_dir, "graph_meta.json"), "w", encoding="utf-8") as f:
                json.dump(graph_meta, f, ensure_ascii=False, indent=2)

            self.logger.info("Successfully initialized dynamic knowledge graph")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing dynamic graph: {str(e)}")
            return False

    async def update_dynamic_graph(self, conversation_text: str, speaker: str = "user") -> bool:
        """
        Update dynamic knowledge graph with new conversation data.

        Args:
            conversation_text: Text from conversation to extract entities from
            speaker: Who said this (user/assistant)

        Returns:
            bool: True if update was successful
        """
        try:
            # Initialize dynamic graph if it doesn't exist
            if not self.dynamic_kg and not self.initialize_dynamic_graph():
                return False

            # Skip if text is too short or doesn't contain useful content
            if len(conversation_text.strip()) < 20:
                self.logger.debug("Skipping entity extraction for short text")
                return True

            # Add utterance as a node first (for provenance)
            utt_id = self._add_utterance(conversation_text, speaker)

            # Extract entities in background using TinyLlama with optimizations
            if self.model_manager:
                self.logger.info("Extracting entities from conversation text")

                # Split text into smaller chunks for better performance
                chunks = self._split_text_into_chunks(conversation_text, max_tokens=256)

                # Process chunks in background thread to avoid blocking WebSocket
                asyncio.create_task(self._process_text_chunks_async(chunks, utt_id))

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

    def _add_utterance(self, text: str, speaker: str = "user"):
        """Add utterance as a node for provenance tracking."""
        if not hasattr(self, "_utt_seq"):
            self._utt_seq = 0
        self._utt_seq += 1
        utt_id = f"utt_{int(time.time())}_{self._utt_seq}"

        if hasattr(self.dynamic_kg, "property_graph_store"):
            self.dynamic_kg.property_graph_store.upsert_nodes(
                [
                    {
                        "id": utt_id,
                        "label": "Utterance",
                        "properties": {"text": text, "speaker": speaker, "ts": time.time()},
                    }
                ]
            )
            self.logger.debug(f"Added utterance node: {utt_id}")

            # Log to WAL for durability
            if hasattr(self, "_append_wal"):
                self._append_wal(
                    {
                        "type": "utterance_add",
                        "id": utt_id,
                        "text": text,
                        "speaker": speaker,
                        "ts": time.time(),
                    }
                )

        return utt_id

    def _split_text_into_chunks(self, text: str, max_tokens: int = 256) -> list[str]:
        """Split text into smaller chunks for more efficient processing."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            # Rough token estimation (1 word ≈ 1.3 tokens)
            word_tokens = len(word) // 4 + 1

            if current_length + word_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_tokens
            else:
                current_chunk.append(word)
                current_length += word_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _norm_key(self, name: str, etype: str) -> str:
        """Normalize entity name and type for catalog lookup."""
        return f"{etype.lower()}::{name.strip().lower()}"

    def _load_catalog(self):
        """Load entity catalog from disk."""
        try:
            with open(self._entity_catalog_path, encoding="utf-8") as f:
                self._entity_catalog = json.load(f)
        except Exception:
            self._entity_catalog = {}

    def _save_catalog(self):
        """Save entity catalog to disk."""
        os.makedirs(os.path.dirname(self._entity_catalog_path), exist_ok=True)
        with open(self._entity_catalog_path, "w", encoding="utf-8") as f:
            json.dump(self._entity_catalog, f, ensure_ascii=False, indent=2)

    def _canonical_entity(self, name: str, etype: str):
        """Get canonical entity ID, creating new entry if needed."""
        from llama_index.core import Settings

        self._load_catalog()
        key = self._norm_key(name, etype)

        # Hit - return existing ID
        if key in self._entity_catalog:
            return self._entity_catalog[key]["id"]

        # New entity - embed and create
        emb = Settings.embed_model.get_text_embedding(name) if Settings.embed_model else None
        ent_id = f"ent_{abs(hash((name, etype))) % 10**10}"
        self._entity_catalog[key] = {"id": ent_id, "name": name, "etype": etype, "embed": emb}
        self._save_catalog()
        return ent_id

    async def _process_text_chunks_async(self, chunks: list[str], utt_id: str = None) -> None:
        """Process text chunks asynchronously in background thread."""
        try:
            # Run in thread to avoid blocking the event loop
            await asyncio.to_thread(self._process_text_chunks, chunks, utt_id)
        except Exception as e:
            self.logger.error(f"Error in background chunk processing: {e}")

    def _process_text_chunks(self, chunks: list[str], utt_id: str = None) -> None:
        """Process text chunks synchronously in background thread."""
        all_triples = []

        for i, chunk in enumerate(chunks):
            try:
                self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")

                # Extract triples using LLM
                raw_triples = self._extract_triples_llm(chunk)
                triples = self._parse_triple_lines(raw_triples)
                all_triples.extend(triples)

            except Exception as e:
                self.logger.warning(f"Error processing chunk {i}: {e}")
                continue

        # Add triples to graph with utterance provenance
        if all_triples:
            asyncio.run(self._add_triples_with_provenance(all_triples, utt_id))

    def _extract_triples_llm(self, chunk: str) -> str:
        """Extract triples using local LLM."""
        prompt = f"""Extract key entities and relationships from this text.
Format as simple triplets: (subject, relation, object)
Limit to {self.max_triplets_per_chunk} most important triplets.

Text: {chunk}

Triplets:"""

        try:
            # Use the model manager to get completions
            if hasattr(self.model_manager, "complete_sync"):
                return self.model_manager.complete_sync(prompt, model="tinyllama")
            else:
                # Fallback - return empty for now
                self.logger.debug("Model manager doesn't support sync completion")
                return ""
        except Exception as e:
            self.logger.warning(f"LLM extraction failed: {e}")
            return ""

    def _parse_triple_lines(self, raw_text: str) -> list[tuple]:
        """Parse triple lines from LLM output."""
        triples = []
        lines = raw_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Triplets:"):
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
                    triples.append((subject, relation, obj))

        return triples

    async def _add_triples_with_provenance(self, triples: list[tuple], utt_id: str = None):
        """Add triples to graph with utterance provenance."""
        try:
            if not self.dynamic_kg:
                return

            is_property_graph = hasattr(self.dynamic_kg, "property_graph_store")

            for triplet in triples:
                subject, predicate, obj = triplet

                if is_property_graph:
                    # Use canonical entity IDs
                    src_id = self._canonical_entity(subject, "Entity")
                    tgt_id = self._canonical_entity(obj, "Entity")

                    graph_store = self.dynamic_kg.property_graph_store

                    # Add nodes with canonical IDs
                    graph_store.upsert_nodes(
                        [
                            {
                                "id": src_id,
                                "label": "ENTITY",
                                "properties": {"name": subject, "type": "conversation_entity"},
                            },
                            {
                                "id": tgt_id,
                                "label": "ENTITY",
                                "properties": {"name": obj, "type": "conversation_entity"},
                            },
                        ]
                    )

                    # Add main relation
                    graph_store.upsert_relations(
                        [
                            {
                                "source_id": src_id,
                                "target_id": tgt_id,
                                "label": predicate,
                                "properties": {"source": "conversation", "timestamp": time.time()},
                            }
                        ]
                    )

                    # Connect utterance to entities if we have utterance ID
                    if utt_id:
                        graph_store.upsert_relations(
                            [
                                {
                                    "source_id": utt_id,
                                    "target_id": src_id,
                                    "label": "MENTIONS",
                                    "properties": {"ts": time.time()},
                                },
                                {
                                    "source_id": utt_id,
                                    "target_id": tgt_id,
                                    "label": "MENTIONS",
                                    "properties": {"ts": time.time()},
                                },
                            ]
                        )

                else:
                    # Fallback for simple graphs
                    from llama_index.core.schema import TextNode

                    node_text = f"{subject} {predicate} {obj}"
                    node_id = f"conv_triplet_{hash(node_text)}_{int(time.time())}"
                    node = TextNode(text=node_text, id_=node_id)

                    try:
                        self.dynamic_kg.upsert_triplet_and_node(triplet, node)
                    except Exception:
                        if hasattr(self.dynamic_kg, "upsert_triplet"):
                            self.dynamic_kg.upsert_triplet(triplet)

                self.logger.debug(f"Added triple with provenance: {triplet}")

        except Exception as e:
            self.logger.error(f"Error adding triples with provenance: {e}")

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
                try:
                    # Create nodes for the triplet
                    from llama_index.core.schema import TextNode

                    subject, relation, obj = triplet

                    # Enhanced logging for triple addition
                    self.logger.info(f"🔗 ADDING TRIPLE: [{subject}] --[{relation}]--> [{obj}]")

                    # Create a node that represents this triplet
                    node_text = f"{subject} {relation} {obj}"
                    node_id = f"triplet_{hash(node_text)}"
                    node = TextNode(text=node_text, id_=node_id)

                    # Use the new API signature with both triplet and node
                    self.dynamic_kg.upsert_triplet_and_node(triplet, node)

                    # Log successful addition with more details
                    self.logger.info("✅ Successfully added triple to dynamic graph")
                    self.logger.info(f"   Subject: {subject}")
                    self.logger.info(f"   Relation: {relation}")
                    self.logger.info(f"   Object: {obj}")
                    self.logger.info(f"   Node ID: {node_id}")

                    # Log to WAL for durability
                    if hasattr(self, "_append_wal"):
                        self._append_wal(
                            {
                                "type": "triple_add",
                                "s": subject,
                                "p": relation,
                                "o": obj,
                                "ts": time.time(),
                            }
                        )

                    # Update graph statistics
                    if not hasattr(self, "_triple_count"):
                        self._triple_count = 0
                    self._triple_count += 1
                    self.logger.info(f"📊 Total triples in dynamic graph: {self._triple_count}")

                except Exception as e:
                    self.logger.warning(f"❌ Error adding triplet {triplet}: {e}")
                    # Fallback for older LlamaIndex versions
                    try:
                        if hasattr(self.dynamic_kg, "upsert_triplet"):
                            self.dynamic_kg.upsert_triplet(triplet)
                            self.logger.info(
                                f"✅ Successfully added triple using fallback method: {triplet}"
                            )
                    except Exception as e2:
                        self.logger.error(f"💥 Failed to add triplet with fallback: {e2}")

            self.logger.info(f"Added {len(triplets)} triplets to dynamic graph")

            # Update metrics
            if triplets:
                G = self._safe_get_networkx_graph(self.dynamic_kg, "dynamic")
                if G is not None:
                    self.metrics["total_nodes"] = G.number_of_nodes()
                    self.metrics["total_edges"] = G.number_of_edges()

        except Exception as e:
            self.logger.error(f"Error processing entities: {str(e)}")

    def add_new_graph(self, graph_path: str, graph_id: str | None = None) -> str | None:
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
            from ..utf8_import_helper import get_utf8_filesystem

            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(
                graph_store=graph_store, fs=get_utf8_filesystem()
            )

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

    async def add_triples_to_dynamic_graph(self, triples: list[tuple]) -> bool:
        """
        Add triples directly to the dynamic knowledge graph.

        Args:
            triples: List of (subject, predicate, object) tuples

        Returns:
            bool: True if successful
        """
        try:
            # Initialize dynamic graph if it doesn't exist
            if not self.dynamic_kg and not self.initialize_dynamic_graph():
                return False

            self.logger.info(f"🔗 Adding {len(triples)} triples to persistent dynamic graph")

            # Check if this is actually a property graph by inspecting the index type
            is_property_graph = hasattr(self.dynamic_kg, "property_graph_store")

            # Add each triple to the graph
            for i, triplet in enumerate(triples):
                try:
                    subject, predicate, obj = triplet

                    # Enhanced logging for triple addition
                    self.logger.info(
                        f"🔗 ADDING PERSISTENT TRIPLE {i+1}: [{subject}] --[{predicate}]--> [{obj}]"
                    )

                    if is_property_graph:
                        # For property graphs, add nodes and edges directly to the graph store
                        graph_store = self.dynamic_kg.property_graph_store

                        # Use canonical entity IDs to prevent drift
                        src_id = self._canonical_entity(subject, "Entity")
                        tgt_id = self._canonical_entity(obj, "Entity")

                        # Add nodes with canonical IDs
                        graph_store.upsert_nodes(
                            [
                                {
                                    "id": src_id,
                                    "label": "ENTITY",
                                    "properties": {"name": subject, "type": "conversation_entity"},
                                },
                                {
                                    "id": tgt_id,
                                    "label": "ENTITY",
                                    "properties": {"name": obj, "type": "conversation_entity"},
                                },
                            ]
                        )

                        # Add edge with properties
                        graph_store.upsert_relations(
                            [
                                {
                                    "source_id": src_id,
                                    "target_id": tgt_id,
                                    "label": predicate,
                                    "properties": {
                                        "source": "conversation",
                                        "timestamp": time.time(),
                                    },
                                }
                            ]
                        )
                    else:
                        # For simple graphs, use the traditional approach
                        from llama_index.core.schema import TextNode

                        node_text = f"{subject} {predicate} {obj}"
                        node_id = f"conv_triplet_{hash(node_text)}_{int(time.time())}"
                        node = TextNode(text=node_text, id_=node_id)

                        # Use the new API signature with both triplet and node
                        self.dynamic_kg.upsert_triplet_and_node(triplet, node)

                    # Log successful addition
                    self.logger.info("✅ Successfully added persistent triple to dynamic graph")

                except Exception as e:
                    self.logger.warning(f"❌ Error adding triplet {triplet}: {e}")
                    # Fallback for older LlamaIndex versions
                    try:
                        if hasattr(self.dynamic_kg, "upsert_triplet"):
                            self.dynamic_kg.upsert_triplet(triplet)
                            self.logger.info(
                                f"✅ Successfully added triple using fallback method: {triplet}"
                            )
                    except Exception as e2:
                        self.logger.error(f"💥 Failed to add triplet with fallback: {e2}")
                        continue

            # Trigger persistence after adding triples
            self._dynamic_updates_count += len(triples)
            await self._maybe_persist_dynamic_graph()

            self.logger.info(
                f"✅ Successfully added {len(triples)} triples to persistent dynamic graph"
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Error adding triples to dynamic graph: {e}")
            return False
