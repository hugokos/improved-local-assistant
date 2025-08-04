"""
Initialization and configuration module for KnowledgeGraphManager.

Handles setup and configuration logic including LlamaIndex configuration
and NetworkX graph utilities.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import networkx as nx
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import embedding model (will be used if available)
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    # This will be handled gracefully in the code
    HuggingFaceEmbedding = None


class KnowledgeGraphInitializer:
    """Handles initialization and configuration of knowledge graphs."""

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

        # Advanced GraphRouter
        self._graph_router = None

        # Configure LlamaIndex to use Ollama
        self._configure_llama_index()
        
        # Initialize advanced GraphRouter if available
        asyncio.create_task(self._initialize_graph_router())

    def _configure_llama_index(self):
        """Configure LlamaIndex to use Ollama models and centralized embedding model."""
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
                    temperature=0.7  # Set a reasonable temperature
                )
                self.logger.info(f"Configured LlamaIndex to use Ollama model: {model_name}")
            except Exception as e:
                self.logger.error(f"Error configuring Ollama LLM: {str(e)}")
                self.logger.warning("Knowledge graph operations may be limited without a working LLM")
                # Don't set a fallback - we'll handle operations without LLM if needed

            # Configure centralized embedding model to match prebuilt graphs
            self._configure_embedding_model()

        except Exception as e:
            self.logger.error(f"Error configuring LlamaIndex: {str(e)}")
            self.logger.warning("Knowledge graph operations may be limited due to configuration errors")

    def _configure_embedding_model(self):
        """Configure embedding model to match prebuilt graph metadata using singleton."""
        try:
            # Always use the embedding singleton to ensure consistency
            from services.embedding_singleton import get_embedding_model, configure_global_embedding
            
            # Read metadata from prebuilt graphs to get embedding model info
            embed_model_name = self._get_embedding_model_from_metadata()
            
            # Configure global embedding using singleton
            configure_global_embedding(embed_model_name)
            
            self.logger.info(f"Configured embedding model using singleton: {embed_model_name}")
            
        except ImportError as e:
            self.logger.warning(f"Could not import embedding singleton: {e}")
            # Try to set a basic local embedding
            try:
                from sentence_transformers import SentenceTransformer
                self.logger.info("Using SentenceTransformer directly as fallback")
            except ImportError:
                self.logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
        except Exception as e:
            self.logger.warning(f"Could not configure embedding model: {e}")
            self.logger.info("Continuing without embedding model - some features may be limited")

    def _get_embedding_model_from_metadata(self) -> str:
        """Get embedding model name from prebuilt graph metadata."""
        import json
        import os
        
        # Default embedding model
        default_model = "BAAI/bge-small-en-v1.5"
        
        try:
            # Check survivalist graph metadata first
            meta_path = os.path.join(self.prebuilt_directory, "survivalist", "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    embed_model = metadata.get("embed_model", default_model)
                    self.logger.info(f"Found embedding model in survivalist metadata: {embed_model}")
                    return embed_model
            
            # Check other graph directories for metadata
            if os.path.exists(self.prebuilt_directory):
                for graph_dir in os.listdir(self.prebuilt_directory):
                    meta_path = os.path.join(self.prebuilt_directory, graph_dir, "meta.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                embed_model = metadata.get("embed_model", default_model)
                                self.logger.info(f"Found embedding model in {graph_dir} metadata: {embed_model}")
                                return embed_model
                        except Exception as e:
                            self.logger.warning(f"Could not read metadata from {meta_path}: {e}")
                            continue
            
            self.logger.info(f"No metadata found, using default embedding model: {default_model}")
            return default_model
            
        except Exception as e:
            self.logger.warning(f"Error reading graph metadata: {e}")
            self.logger.info(f"Using default embedding model: {default_model}")
            return default_model
    
    async def _initialize_graph_router(self):
        """Initialize the advanced GraphRouter if registry is available."""
        try:
            # Check if registry exists
            registry_path = "data/graph_registry.json"
            if not os.path.exists(registry_path):
                self.logger.info("Graph registry not found, skipping advanced router initialization")
                return
            
            # Import and initialize GraphRouter with singleton embedder
            from services.graph_router import GraphRouter
            from services.embedding_singleton import get_embedding_model
            
            router_config = {
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "registry_path": registry_path,
                "data_path": self.prebuilt_directory,
                "max_cached_indices": 10,
                "hybrid_alpha": 0.6,
                "rerank_top_n": 10,
                "min_score_threshold": 0.4
            }
            
            # Get singleton embedder to avoid duplicate loading
            singleton_embedder = get_embedding_model("BAAI/bge-small-en-v1.5")
            self._graph_router = GraphRouter(router_config, embedder=singleton_embedder)
            
            if await self._graph_router.initialize():
                self.logger.info("Advanced GraphRouter initialized successfully")
            else:
                self.logger.warning("Failed to initialize advanced GraphRouter")
                self._graph_router = None
                
        except Exception as e:
            self.logger.warning(f"Could not initialize advanced GraphRouter: {e}")
            self._graph_router = None

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
            if hasattr(kg_index, 'get_networkx_graph'):
                return kg_index.get_networkx_graph()
            else:
                # For PropertyGraphIndex, we might need a different approach
                index_type = type(kg_index).__name__
                self.logger.debug(f"Graph {graph_id} ({index_type}) does not support get_networkx_graph()")
                return None
        except Exception as e:
            index_type = type(kg_index).__name__
            self.logger.error(f"Error getting NetworkX graph from {graph_id} ({index_type}): {str(e)}")
            return None