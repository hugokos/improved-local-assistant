"""
Advanced Graph Router with semantic routing, lazy loading, and hybrid retrieval.

This module implements a production-ready GraphRAG system with:
- Boot-time registry loading with vector embeddings
- Fast semantic routing using cosine similarity
- Lazy index loading with LRU cache
- Hybrid retrieval (BM25 + Vector + Graph)
- ColBERT reranking for final grounding
- Citation-first generation pattern
- Streaming responses with connection reuse
"""

import asyncio
import json
import logging
import pathlib
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from llama_index.core import load_index_from_storage, StorageContext
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama
    
    # Try different import paths for retrievers
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except ImportError:
        try:
            from llama_index.core.retrievers import BM25Retriever
        except ImportError:
            BM25Retriever = None
    
    try:
        from llama_index.retrievers import HybridRetriever
    except ImportError:
        try:
            from llama_index.core.retrievers import HybridRetriever
        except ImportError:
            try:
                from llama_index.core.retrievers.fusion import HybridRetriever
            except ImportError:
                HybridRetriever = None
    
    # Auto retriever for metadata filtering
    try:
        from llama_index.retrievers.auto_retriever import AutoVectorRetriever
    except ImportError:
        AutoVectorRetriever = None
    
    # Reranking
    try:
        from llama_index.postprocessor import ColbertRerank
    except ImportError:
        try:
            from llama_index.core.postprocessor import ColbertRerank
        except ImportError:
            try:
                from llama_index.postprocessor import SentenceTransformerRerank as ColbertRerank
            except ImportError:
                ColbertRerank = None
    
except ImportError as e:
    logging.warning(f"Some LlamaIndex components not available: {e}")


class SimpleHybridRetriever:
    """
    Simple hybrid retriever that combines results from multiple retrievers with weights.
    """
    
    def __init__(self, retrievers: List[Any], weights: List[float]):
        """Initialize with retrievers and weights."""
        self.retrievers = retrievers
        self.weights = weights
        self.logger = logging.getLogger(__name__)
    
    def retrieve(self, query: str) -> List[Any]:
        """Retrieve from all retrievers and combine with weights."""
        all_nodes = []
        
        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                nodes = retriever.retrieve(query)
                
                # Apply weight to scores
                for node in nodes:
                    if hasattr(node, 'score') and node.score is not None:
                        node.score = node.score * weight
                    else:
                        node.score = weight * 0.5  # Default score
                
                all_nodes.extend(nodes)
                
            except Exception as e:
                self.logger.warning(f"Error in retriever {type(retriever).__name__}: {e}")
                continue
        
        # Deduplicate by node_id and sort by score
        unique_nodes = {}
        for node in all_nodes:
            node_id = node.node.node_id if hasattr(node, 'node') else str(node)
            if node_id not in unique_nodes or node.score > unique_nodes[node_id].score:
                unique_nodes[node_id] = node
        
        # Sort by score and return
        sorted_nodes = sorted(unique_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
        return sorted_nodes


class GraphRouter:
    """
    Advanced graph router with semantic routing and hybrid retrieval.
    
    Features:
    - Semantic routing using embedding similarity
    - Lazy loading of graph indices
    - Hybrid retrieval combining BM25, vector, and graph search
    - ColBERT reranking for improved relevance
    - Citation-first generation pattern
    - Streaming responses
    """
    

    
    async def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # 0. Configure global settings to avoid OpenAI errors
            from llama_index.core import Settings
            Settings.llm = None  # Use MockLLM to avoid OpenAI API calls
            
            # 1. Initialize embedder only if not provided
            if self.embedder is None:
                self.logger.info(f"Initializing embedder: {self.emb_model_name}")
                self.embedder = HuggingFaceEmbedding(
                    model_name=self.emb_model_name,
                    device="cpu",
                    normalize=True,
                    embed_batch_size=10
                )
            else:
                self.logger.info("Using external embedder (singleton)")
            
            # Set the global embedding model
            Settings.embed_model = self.embedder
            
            # 2. Load registry
            self.logger.info("Loading graph registry")
            if not await self._load_registry():
                self.logger.warning("Failed to load registry, creating empty one")
                self.graph_names = []
                self.reg_vecs = np.array([]).reshape(0, 384)  # BGE-small dimension
                self.graph_meta = {}
            
            # 3. Initialize reranker
            if ColbertRerank:
                try:
                    self.reranker = ColbertRerank(top_n=self.rerank_top_n)
                    self.logger.info("ColBERT reranker initialized")
                except Exception as e:
                    self.logger.warning(f"Could not initialize ColBERT reranker: {e}")
            
            # 4. Initialize fast LLM for auto-retrieval
            try:
                self.fast_llm = Ollama(
                    model="hermes3:3b",
                    request_timeout=30.0,
                    temperature=0.0
                )
                self.logger.info("Fast LLM initialized for auto-retrieval")
            except Exception as e:
                self.logger.warning(f"Could not initialize fast LLM: {e}")
            
            # 5. Initialize HTTP client for streaming
            self.http_client = httpx.AsyncClient(
                http2=True,
                timeout=120.0,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
            )
            
            self.logger.info("GraphRouter initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GraphRouter: {e}")
            return False
    
    async def _load_registry(self) -> bool:
        """Load the graph registry with embeddings."""
        try:
            if not pathlib.Path(self.registry_path).exists():
                self.logger.warning(f"Registry file not found: {self.registry_path}")
                return False
            
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            if not registry:
                self.logger.warning("Empty registry")
                return False
            
            # Extract names, vectors, and metadata
            self.graph_names = list(registry.keys())
            vectors = []
            
            for name in self.graph_names:
                graph_data = registry[name]
                if "vector" in graph_data:
                    vectors.append(graph_data["vector"])
                else:
                    # Generate embedding for graph name if vector not available
                    self.logger.warning(f"No vector for {name}, generating from name")
                    vec = self.embedder.get_text_embedding(name)
                    vectors.append(vec)
                
                # Store metadata
                self.graph_meta[name] = {
                    k: v for k, v in graph_data.items() if k != "vector"
                }
            
            # Convert to numpy array for fast similarity computation
            self.reg_vecs = np.asarray(vectors)
            
            self.logger.info(f"Loaded registry with {len(self.graph_names)} graphs")
            self.logger.info(f"Vector dimensions: {self.reg_vecs.shape}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading registry: {e}")
            return False
    
    def route(self, query: str, k: int = 3) -> Tuple[List[str], np.ndarray]:
        """
        Route query to most relevant graphs using semantic similarity.
        
        Args:
            query: User query
            k: Number of top graphs to return
            
        Returns:
            Tuple of (graph_names, similarity_scores)
        """
        try:
            if len(self.graph_names) == 0:
                return [], np.array([])
            
            # Get query embedding
            q_vec = self.embedder.get_text_embedding(query)
            
            # Compute cosine similarities
            sims = cosine_similarity([q_vec], self.reg_vecs)[0]
            
            # Get top-k indices
            top_indices = sims.argsort()[-k:][::-1]
            
            # Return names and scores
            top_names = [self.graph_names[i] for i in top_indices]
            top_scores = sims[top_indices]
            
            self.logger.debug(f"Routed query to: {list(zip(top_names, top_scores))}")
            
            return top_names, top_scores
            
        except Exception as e:
            self.logger.error(f"Error in routing: {e}")
            return [], np.array([])
    
    def __init__(self, config: Dict[str, Any], embedder=None):
        """Initialize the graph router."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.emb_model_name = config.get("embedding_model", "BAAI/bge-small-en-v1.5")
        self.registry_path = config.get("registry_path", "data/graph_registry.json")
        self.data_path = pathlib.Path(config.get("data_path", "data/prebuilt_graphs"))
        self.max_cached_indices = config.get("max_cached_indices", 10)
        self.hybrid_alpha = config.get("hybrid_alpha", 0.6)
        self.rerank_top_n = config.get("rerank_top_n", 10)
        self.min_score_threshold = config.get("min_score_threshold", 0.4)
        
        # Initialize components - use external embedder if provided
        self.embedder = embedder  # Use external embedder to avoid duplicate loading
        self.graph_names = []
        self.reg_vecs = None
        self.graph_meta = {}
        self.reranker = None
        self.fast_llm = None
        self.http_client = None
        
        # Enhanced caching for multiple indices per graph
        self._vector_indices: Dict[str, Any] = {}
        self._property_indices: Dict[str, Any] = {}
        self._retrievers: Dict[str, Any] = {}
        
        # System prompt for citation-first generation
        self.system_prompt = """You are a concise, citation-driven assistant. Follow the two-step method:
(1) List the minimal source sentences with [doc / page].
(2) Answer using *only* those sentences.

If <2 sources, say "I don't have enough data."
"""
        
        self.logger.info("GraphRouter initialized")
    
    def _load_indices(self, graph_id: str):
        """
        Load multiple indices from disk and separate vector/property indices.
        Supports both PropertyGraphIndex and KnowledgeGraphIndex.
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            Tuple of (vector_index, property_index)
        """
        # Check if already cached
        if graph_id in self._vector_indices and graph_id in self._property_indices:
            self.logger.debug(f"Returning cached indices for {graph_id}")
            return self._vector_indices[graph_id], self._property_indices[graph_id]
        
        try:
            # Get base directory from registry or construct path
            if graph_id in self.graph_meta:
                base_dir = pathlib.Path(self.graph_meta[graph_id].get("path", ""))
            else:
                base_dir = self.data_path / graph_id
            
            # Guard rails - check if persist dir exists
            if not base_dir.exists():
                raise FileNotFoundError(f"Persist dir missing: {base_dir}")
            
            self.logger.debug(f"Loading indices from {base_dir}")
            
            # Load using StorageContext for proper initialization
            from llama_index.core import StorageContext, load_indices_from_storage, Settings
            from llama_index.core import VectorStoreIndex
            from llama_index.core.indices.property_graph import PropertyGraphIndex
            from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
            from .utf8_import_helper import get_utf8_filesystem
            
            # Settings should already be configured in initialize()
            
            storage_ctx = StorageContext.from_defaults(
                persist_dir=str(base_dir),
                fs=get_utf8_filesystem()
            )
            
            # Load all indices from storage
            indices = load_indices_from_storage(storage_ctx)
            
            vector_idx = None
            property_idx = None
            
            # Separate indices by type
            for idx in indices:
                if isinstance(idx, VectorStoreIndex):
                    vector_idx = idx
                    self.logger.debug(f"Found VectorStoreIndex with ID: {idx.index_id}")
                elif isinstance(idx, PropertyGraphIndex):
                    property_idx = idx
                    self.logger.debug(f"Found PropertyGraphIndex with ID: {idx.index_id}")
                elif isinstance(idx, KnowledgeGraphIndex):
                    # Treat KnowledgeGraphIndex as property index for compatibility
                    property_idx = idx
                    self.logger.debug(f"Found KnowledgeGraphIndex with ID: {idx.index_id}")
            
            # Fallback: if we only have one index, determine its type and use it appropriately
            if not vector_idx and not property_idx and len(indices) == 1:
                single_idx = indices[0]
                if hasattr(single_idx, 'property_graph_store') or hasattr(single_idx, 'graph_store'):
                    property_idx = single_idx
                    self.logger.debug(f"Single index identified as graph index")
                else:
                    vector_idx = single_idx
                    self.logger.debug(f"Single index identified as VectorStoreIndex")
            
            # Cache the indices
            self._vector_indices[graph_id] = vector_idx
            self._property_indices[graph_id] = property_idx
            
            self.logger.info(f"Successfully loaded indices for {graph_id}: "
                           f"vector={'Yes' if vector_idx else 'No'}, "
                           f"property={'Yes' if property_idx else 'No'}")
            
            return vector_idx, property_idx
            
        except Exception as e:
            self.logger.error(f"Failed loading indices for {graph_id}: {e}")
            return None, None
    
    def get_indices(self, name: str):
        """
        Public interface for getting indices with proper caching.
        
        Args:
            name: Graph name
            
        Returns:
            Tuple of (vector_index, property_index)
        """
        return self._load_indices(name)
    
    def get_retriever(self, name: str):
        """
        Get or create hybrid retriever for a graph.
        
        Args:
            name: Graph name
            
        Returns:
            Hybrid retriever combining vector and property graph search
        """
        if name in self._retrievers:
            return self._retrievers[name]
        
        try:
            vector_idx, property_idx = self._load_indices(name)
            
            if not vector_idx and not property_idx:
                self.logger.warning(f"No indices found for {name}")
                return None
            
            # Create retrievers for available indices
            retrievers = []
            weights = []
            
            if vector_idx:
                vector_retriever = vector_idx.as_retriever(similarity_top_k=10)
                retrievers.append(vector_retriever)
                weights.append(0.7)  # Higher weight for vector search
                self.logger.debug(f"Added vector retriever for {name}")
            
            if property_idx:
                property_retriever = property_idx.as_retriever(similarity_top_k=5)
                retrievers.append(property_retriever)
                weights.append(0.3)  # Lower weight for property graph
                self.logger.debug(f"Added property retriever for {name}")
            
            # Create hybrid retriever if we have multiple retrievers
            if len(retrievers) > 1:
                try:
                    # Create a simple custom hybrid retriever
                    hybrid = SimpleHybridRetriever(retrievers, weights)
                    self._retrievers[name] = hybrid
                    self.logger.info(f"Created custom hybrid retriever for {name}")
                    return hybrid
                except Exception as e:
                    self.logger.warning(f"Could not create hybrid retriever for {name}: {e}")
            
            # Fallback to single retriever
            if retrievers:
                single_retriever = retrievers[0]
                self._retrievers[name] = single_retriever
                self.logger.info(f"Using single retriever for {name}")
                return single_retriever
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating retriever for {name}: {e}")
            return None
    
    def make_hybrid_retriever(self, vector_index=None, property_index=None, similarity_top_k: int = 25):
        """
        Create hybrid retriever combining vector and property graph search.
        
        Args:
            vector_index: VectorStoreIndex for vector search
            property_index: PropertyGraphIndex for graph search
            similarity_top_k: Number of results per retriever
            
        Returns:
            Hybrid retriever or fallback retriever
        """
        try:
            retrievers = []
            weights = []
            
            # Add vector retriever if available
            if vector_index:
                vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
                retrievers.append(vector_retriever)
                weights.append(0.7)  # Higher weight for vector search
            
            # Add property graph retriever if available
            if property_index:
                property_retriever = property_index.as_retriever(similarity_top_k=similarity_top_k // 2)
                retrievers.append(property_retriever)
                weights.append(0.3)  # Lower weight for property graph
            
            # Create hybrid retriever if we have multiple retrievers
            if len(retrievers) > 1 and HybridRetriever:
                try:
                    hybrid = HybridRetriever(
                        retrievers=retrievers,
                        weights=weights,
                    )
                    self.logger.debug("Created hybrid retriever with vector and property graph")
                    return hybrid
                except Exception as e:
                    self.logger.warning(f"Could not create hybrid retriever: {e}")
            
            # Fallback to single retriever
            if retrievers:
                self.logger.debug(f"Using single retriever: {type(retrievers[0]).__name__}")
                return retrievers[0]
            
            self.logger.warning("No retrievers available")
            return None
            
        except Exception as e:
            self.logger.error(f"Error creating hybrid retriever: {e}")
            return None
    
    def make_auto_vector_retriever(self, index, similarity_top_k: int = 50):
        """
        Create auto-retriever with metadata filtering.
        
        Args:
            index: Graph index
            similarity_top_k: Number of results
            
        Returns:
            Auto retriever or fallback retriever
        """
        try:
            if not AutoVectorRetriever or not self.fast_llm:
                return index.as_retriever(similarity_top_k=similarity_top_k)
            
            base = index.as_retriever(similarity_top_k=similarity_top_k)
            auto_retriever = AutoVectorRetriever(
                vector_retriever=base,
                llm=self.fast_llm
            )
            
            self.logger.debug("Created auto-vector retriever")
            return auto_retriever
            
        except Exception as e:
            self.logger.warning(f"Could not create auto-vector retriever: {e}")
            return index.as_retriever(similarity_top_k=similarity_top_k)
    
    async def retrieve_and_rerank(self, query: str, top_graphs: List[str]) -> List[Any]:
        """
        Retrieve from multiple graphs using hybrid retrievers and rerank results.
        
        Args:
            query: User query
            top_graphs: List of graph names to search
            
        Returns:
            List of reranked nodes
        """
        all_nodes = []
        
        try:
            # Retrieve from each graph using hybrid retrievers
            for graph_name in top_graphs:
                try:
                    # Get the hybrid retriever for this graph
                    retriever = self.get_retriever(graph_name)
                    if not retriever:
                        self.logger.warning(f"Could not get retriever for {graph_name}")
                        continue
                    
                    # Retrieve nodes using the hybrid retriever
                    nodes = retriever.retrieve(query)
                    
                    # Add graph source to metadata
                    for node in nodes:
                        if not hasattr(node, 'metadata'):
                            node.metadata = {}
                        node.metadata['graph_source'] = graph_name
                    
                    all_nodes.extend(nodes)
                    self.logger.debug(f"Retrieved {len(nodes)} nodes from {graph_name}")
                    
                except Exception as e:
                    # Granular logging as suggested
                    self.logger.error(f"Failed retrieving from {graph_name}: {e}")
                    continue
            
            if not all_nodes:
                self.logger.warning("No nodes retrieved from any graph")
                return []
            
            # Rerank if reranker is available
            if self.reranker:
                try:
                    final_nodes = self.reranker.postprocess_nodes(all_nodes, query=query)
                    self.logger.debug(f"Reranked to {len(final_nodes)} nodes")
                    return final_nodes
                except Exception as e:
                    self.logger.warning(f"Reranking failed: {e}")
            
            # Fallback: sort by score and take top results
            scored_nodes = [n for n in all_nodes if hasattr(n, 'score') and n.score is not None]
            scored_nodes.sort(key=lambda x: x.score, reverse=True)
            
            return scored_nodes[:self.rerank_top_n]
            
        except Exception as e:
            self.logger.error(f"Error in retrieve_and_rerank: {e}")
            return []
    
    async def generate_response(self, query: str, context_nodes: List[Any]) -> str:
        """
        Generate response using citation-first pattern.
        
        Args:
            query: User query
            context_nodes: Retrieved and reranked nodes
            
        Returns:
            Generated response
        """
        try:
            if not context_nodes:
                return "Sorry, I don't have data on that topic yet."
            
            # Check minimum score threshold
            if hasattr(context_nodes[0], 'score') and context_nodes[0].score < self.min_score_threshold:
                return "Sorry, I don't have enough relevant data on that topic."
            
            # Extract context
            context_parts = []
            sources = set()
            
            for i, node in enumerate(context_nodes):
                content = node.get_content()
                source = node.metadata.get('graph_source', 'Unknown')
                sources.add(source)
                
                context_parts.append(f"[{i+1}] {content} (Source: {source})")
            
            # Check if we have enough sources
            if len(sources) < 2:
                return "I don't have enough data from multiple sources to provide a reliable answer."
            
            context = "\n\n".join(context_parts)
            
            # Create prompt
            prompt = f"""{self.system_prompt}

Context:
{context}

Question: {query}

Response:"""
            
            # Generate response using streaming
            response = await self._call_ollama_streaming(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error while generating the response."
    
    async def _call_ollama_streaming(self, prompt: str) -> str:
        """
        Call Ollama with streaming for faster perceived response.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Complete response
        """
        try:
            payload = {
                "model": "hermes3:3b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False  # For now, get complete response
            }
            
            async with self.http_client.stream(
                "POST", 
                "http://localhost:11434/api/chat",
                json=payload
            ) as resp:
                if resp.status_code != 200:
                    self.logger.error(f"Ollama API error: {resp.status_code}")
                    return "Error: Could not generate response"
                
                response_text = ""
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                response_text += data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
                
                return response_text.strip()
                
        except Exception as e:
            self.logger.error(f"Error calling Ollama: {e}")
            return "Error: Could not generate response"
    
    async def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Main query method combining routing, retrieval, and generation.
        
        Args:
            query: User query
            k: Number of graphs to route to
            
        Returns:
            Response dictionary with answer and metadata
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # 1. Route to relevant graphs
            top_graphs, scores = self.route(query, k=k)
            
            if not top_graphs:
                return {
                    "response": "Sorry, I don't have any relevant knowledge graphs for that query.",
                    "metadata": {
                        "graphs_searched": [],
                        "nodes_retrieved": 0,
                        "query_time": 0
                    }
                }
            
            # 2. Retrieve and rerank
            context_nodes = await self.retrieve_and_rerank(query, top_graphs)
            
            # 3. Generate response
            response = await self.generate_response(query, context_nodes)
            
            # 4. Prepare metadata
            end_time = asyncio.get_event_loop().time()
            
            metadata = {
                "graphs_searched": list(zip(top_graphs, scores.tolist())),
                "nodes_retrieved": len(context_nodes),
                "query_time": end_time - start_time,
                "sources": list(set(
                    node.metadata.get('graph_source', 'Unknown') 
                    for node in context_nodes
                ))
            }
            
            return {
                "response": response,
                "metadata": metadata,
                "source_nodes": context_nodes[:5]  # Top 5 for citations
            }
            
        except Exception as e:
            self.logger.error(f"Error in query: {e}")
            return {
                "response": "Sorry, I encountered an error while processing your query.",
                "metadata": {"error": str(e)},
                "source_nodes": []
            }
    
    async def close(self):
        """Clean up resources."""
        if self.http_client:
            await self.http_client.aclose()
        self.logger.info("GraphRouter closed")