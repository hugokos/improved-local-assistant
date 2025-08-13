"""
Hybrid Ensemble Retriever for multi-modal knowledge retrieval on edge devices.

This module provides the HybridEnsembleRetriever class that combines:
• Property-graph retrieval (KnowledgeGraphRAGRetriever if present)
• Optional vector-store retrieval  
• Optional BM25 / keyword retrieval
into a single RankedNode list.

Designed to work with the graph built by graph_builder_v6 (EntityNode + ImplicitPathExtractor, no external LLM).
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

# REQUIRED symbols – fail loudly if they're missing
from llama_index.core.retrievers import BaseRetriever          # new path ✔
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import Settings

# OPTIONAL symbols – keep the fallback logic
try:
    from llama_index.core.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None

try:
    from llama_index.core.retrievers import QueryFusionRetriever
except ImportError:
    QueryFusionRetriever = None

# Fail fast on truly missing essentials
if BaseRetriever is None:
    raise ImportError(
        "Cannot find BaseRetriever in LlamaIndex ≥0.12; "
        "check your installation."
    )
    QueryBundle = None

from services.working_set_cache import WorkingSetCache


# ---------------------------------------------------------------------------
# graph retriever factory                                                     
# ---------------------------------------------------------------------------

def _make_graph_retriever(graph_index, *, depth: int, top_k: int) -> Optional[BaseRetriever]:
    """
    Create appropriate retriever for PropertyGraphIndex or KnowledgeGraphIndex.
    Prefer KnowledgeGraphRAGRetriever for KnowledgeGraphIndex.
    Use as_retriever() for PropertyGraphIndex with include_text=True.
    Returns None if graph_index is None.
    """
    if graph_index is None:
        return None
    
    # Check if this is a PropertyGraphIndex
    from llama_index.core.indices.property_graph import PropertyGraphIndex
    if isinstance(graph_index, PropertyGraphIndex):
        # For PropertyGraphIndex, use as_retriever with include_text
        return graph_index.as_retriever(
            include_text=True,
            similarity_top_k=top_k,
        )
        
    # For KnowledgeGraphIndex, try KnowledgeGraphRAGRetriever first
    try:
        from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
        
        return KnowledgeGraphRAGRetriever(
            index=graph_index,
            depth=depth,
            similarity_top_k=top_k,
        )
    except ImportError:
        # Fallback to as_retriever for older versions
        retr = graph_index.as_retriever(
            include_text=True,          # ← keeps node text so similarity > 0
            similarity_top_k=top_k,
            retriever_mode="embedding",  # semantic search replacement
        )
        return retr


class RetrievedChunk:
    """Represents a retrieved chunk with source and score information."""
    
    def __init__(self, content: str, source: str, score: float, chunk_type: str):
        self.content = content
        self.source = source
        self.score = score
        self.chunk_type = chunk_type  # "graph", "bm25", "vector"
        self.metadata = {}


class HybridEnsembleRetriever(BaseRetriever):
    """
    Combines graph, BM25, and vector retrieval with budget enforcement.
    
    Uses the improved graph retriever factory that prefers KnowledgeGraphRAGRetriever
    for proper entity extraction and sub-graph traversal, with automatic fallback
    to include_text=True for compatibility.
    """
    
    def __init__(
        self,
        *,
        graph_index,
        vector_index=None,
        bm25_index=None,
        graph_depth: int = 2,
        graph_top_k: int = 4,
        bm25_top_k: int = 4,
        vector_top_k: int = 4,
        weights=(0.6, 0.25, 0.15),  # graph, vector, bm25
        working_set_cache: Optional[WorkingSetCache] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Hybrid Ensemble Retriever with improved graph handling."""
        super().__init__()
        
        self.config = config or {}
        self.working_set_cache = working_set_cache
        self.logger = logging.getLogger(__name__)
        
        # Create retrievers using the improved factory
        self.graph = _make_graph_retriever(
            graph_index, depth=graph_depth, top_k=graph_top_k
        )
        self.vector = (
            vector_index.as_retriever(similarity_top_k=vector_top_k)
            if vector_index
            else None
        )
        self.bm25 = (
            bm25_index.as_retriever(similarity_top_k=bm25_top_k)
            if bm25_index
            else None
        )
        
        # Initialize configuration first
        retriever_config = self.config.get("hybrid_retriever", {})
        self.use_rrf = retriever_config.get("use_rrf", True)
        
        # Normalise weights over RETRIEVERS THAT EXIST
        g_w, v_w, b_w = weights
        live = [self.graph, self.vector, self.bm25]
        live_weights = [
            w
            for w, r in zip((g_w, v_w, b_w), live)
            if r is not None
        ]
        norm = sum(live_weights) if live_weights else 1.0
        self.weights = [w / norm for w in live_weights]
        self.live_retrievers: Sequence[BaseRetriever] = [
            r for r in live if r is not None
        ]
        
        # Initialize RRF fusion retriever if available and requested
        self.fuser = None
        if self.use_rrf and QueryFusionRetriever and len(self.live_retrievers) > 1:
            try:
                self.fuser = QueryFusionRetriever(
                    retrievers=self.live_retrievers,
                    mode="reciprocal_rerank_fusion",
                    num_queries=1,  # Can be raised to 3-4 for query variants
                )
                self.logger.info("Initialized RRF fusion retriever")
            except Exception as e:
                self.logger.warning(f"Could not initialize RRF fusion: {e}")
                self.fuser = None
        
        # Budget configuration from config
        budget_config = retriever_config.get("budget", {})
        self.max_chunks = budget_config.get("max_chunks", 12)
        
        # Scoring configuration
        self.half_life_secs = retriever_config.get("half_life_secs", 7*24*3600)  # 1 week
        self.rerank_top_n = retriever_config.get("rerank_top_n", 10)
        
        # Metrics
        self.metrics = {
            "queries_processed": 0,
            "graph_retrievals": 0,
            "bm25_retrievals": 0,
            "vector_retrievals": 0,
            "budget_exceeded": 0,
            "avg_retrieval_time": 0.0,
            "chunks_returned": 0,
        }
        
        self.logger.info(f"Initialized HybridEnsembleRetriever with {len(self.live_retrievers)} active retrievers")
        self.logger.info(f"Retriever weights: {self.weights}")
        self.logger.info(f"Graph retriever type: {type(self.graph).__name__}")
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Override BaseRetriever interface."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Processing query: {query_bundle.query_str[:100]}...")
            
            # Use RRF fusion if available
            if self.fuser:
                all_nodes = self.fuser.retrieve(query_bundle.query_str)
                self.logger.debug(f"RRF fusion returned {len(all_nodes)} nodes")
            else:
                # Fallback to weighted combination
                all_nodes = self._weighted_retrieve(query_bundle)
            
            # Apply time-decay and confidence boosts
            all_nodes = self._apply_scoring_boosts(all_nodes)
            
            # Deduplicate on node_id
            unique: dict[str, NodeWithScore] = {}
            for n in all_nodes:
                node_id = n.node.node_id
                if node_id not in unique or n.score > unique[node_id].score:
                    unique[node_id] = n
            
            # Sort by score and apply budget
            ranked_nodes = sorted(unique.values(), key=lambda n: n.score or 0.0, reverse=True)
            final_nodes = ranked_nodes[:self.max_chunks]
            
            # Apply ColBERT reranking if available
            final_nodes = self._apply_colbert_rerank(final_nodes, query_bundle.query_str)
            
            if len(ranked_nodes) > self.max_chunks:
                self.metrics["budget_exceeded"] += 1
                self.logger.debug(f"Budget exceeded: {len(ranked_nodes)} nodes reduced to {self.max_chunks}")
            
            # Update metrics
            elapsed_time = time.time() - start_time
            self.metrics["queries_processed"] += 1
            self.metrics["chunks_returned"] += len(final_nodes)
            self._update_avg_metric("avg_retrieval_time", elapsed_time)
            
            self.logger.debug(f"Retrieved {len(final_nodes)} nodes in {elapsed_time:.3f}s")
            
            return final_nodes
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {e}")
            return []
    
    # Helper so you can call retriever(query_text) directly
    def __call__(self, query: str) -> List[NodeWithScore]:
        return self.retrieve(query)
    
    def _update_avg_metric(self, metric_name: str, new_value: float) -> None:
        """Update rolling average metric."""
        current_avg = self.metrics.get(metric_name, 0.0)
        count = self.metrics.get("queries_processed", 1)
        
        if count > 1:
            self.metrics[metric_name] = (current_avg * (count - 1) + new_value) / count
        else:
            self.metrics[metric_name] = new_value

    def _weighted_retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Fallback weighted retrieval when RRF is not available."""
        all_nodes: List[NodeWithScore] = []
        
        for i, (retriever, w) in enumerate(zip(self.live_retrievers, self.weights)):
            try:
                nodes = retriever.retrieve(query_bundle.query_str)
                
                # Apply weight to scores
                for n in nodes:
                    if n.score is not None:
                        n.score = n.score * w
                    else:
                        n.score = w * 0.5  # Default score if None
                
                all_nodes.extend(nodes)
                
                # Update metrics based on retriever type
                self._update_retriever_metrics(retriever)
                self.logger.debug(f"Retriever {i} returned {len(nodes)} nodes")
                
            except Exception as e:
                self.logger.error(f"Error in retriever {i}: {e}")
                continue
        
        return all_nodes

    def _update_retriever_metrics(self, retriever):
        """Update metrics based on retriever type."""
        from llama_index.core.retrievers import VectorIndexRetriever
        try:
            from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
        except ImportError:
            KnowledgeGraphRAGRetriever = None
        
        if KnowledgeGraphRAGRetriever and isinstance(retriever, KnowledgeGraphRAGRetriever):
            self.metrics["graph_retrievals"] += 1
        elif isinstance(retriever, VectorIndexRetriever):
            self.metrics["vector_retrievals"] += 1
        elif BM25Retriever and isinstance(retriever, BM25Retriever):
            self.metrics["bm25_retrievals"] += 1
        else:
            # Fallback for unknown retriever types
            if retriever == self.graph:
                self.metrics["graph_retrievals"] += 1
            else:
                self.metrics["vector_retrievals"] += 1

    def _apply_scoring_boosts(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Apply time-decay and confidence boosts to node scores."""
        import math
        
        def time_boost(ts):
            if not ts:
                return 0.0
            return math.exp(-(time.time() - float(ts)) / self.half_life_secs)
        
        for n in nodes:
            meta = n.node.metadata or {}
            ts = meta.get("timestamp") or meta.get("ts")
            conf = float(meta.get("confidence", 0.5))
            
            # Apply boosts
            time_bonus = 0.15 * time_boost(ts) if ts else 0.0
            conf_bonus = 0.10 * conf
            
            n.score = (n.score or 0.0) + time_bonus + conf_bonus
        
        return nodes

    def _apply_colbert_rerank(self, nodes: List[NodeWithScore], query: str) -> List[NodeWithScore]:
        """Apply ColBERT reranking to final results."""
        try:
            from llama_index.postprocessor.colbert_rerank import ColbertRerank
            
            if len(nodes) <= self.rerank_top_n:
                return nodes  # No need to rerank if we have few nodes
            
            reranker = ColbertRerank(
                model="colbert-ir/colbertv2.0", 
                top_n=self.rerank_top_n
            )
            
            reranked = reranker.postprocess_nodes(nodes, query_str=query)
            self.logger.debug(f"ColBERT reranked {len(nodes)} -> {len(reranked)} nodes")
            return reranked
            
        except Exception as e:
            self.logger.debug(f"ColBERT rerank skipped: {e}")
            return nodes[:self.rerank_top_n]
    
    @classmethod
    async def create(
        cls,
        graph_index,
        document_nodes=None,
        config: Optional[Dict[str, Any]] = None,
        working_set_cache: Optional[WorkingSetCache] = None
    ) -> "HybridEnsembleRetriever":
        """
        Factory method to create HybridEnsembleRetriever with proper initialization.
        
        Args:
            graph_index: PropertyGraphIndex for graph retrieval
            document_nodes: Document nodes for BM25 and vector retrieval
            config: Configuration dictionary
            working_set_cache: Working set cache instance
            
        Returns:
            Initialized HybridEnsembleRetriever
        """
        try:
            # Extract configuration
            config = config or {}
            retriever_config = config.get("hybrid_retriever", {})
            budget_config = retriever_config.get("budget", {})
            weights_config = retriever_config.get("weights", {})
            
            # Get parameters
            graph_depth = budget_config.get("graph_depth", 2)
            graph_top_k = budget_config.get("max_chunks", 12) // 3  # Distribute budget
            bm25_top_k = budget_config.get("bm25_top_k", 3)
            vector_top_k = budget_config.get("vector_top_k", 3)
            
            weights = (
                weights_config.get("graph", 0.6),
                weights_config.get("vector", 0.2),
                weights_config.get("bm25", 0.2)
            )
            
            # Create vector index if we have document nodes
            vector_index = None
            if document_nodes:
                try:
                    from llama_index.core import VectorStoreIndex
                    vector_index = VectorStoreIndex.from_documents(document_nodes)
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Could not create vector index: {e}")
            
            # Create BM25 index if we have document nodes and BM25 is available
            bm25_index = None
            if document_nodes and BM25Retriever:
                try:
                    # BM25Retriever expects nodes, not documents
                    if hasattr(document_nodes[0], 'node_id'):
                        # Already nodes
                        bm25_nodes = document_nodes
                    else:
                        # Convert documents to nodes
                        from llama_index.core.schema import TextNode
                        bm25_nodes = [
                            TextNode(text=doc.text, metadata=doc.metadata)
                            for doc in document_nodes
                        ]
                    
                    bm25_index = BM25Retriever.from_defaults(
                        nodes=bm25_nodes,
                        similarity_top_k=bm25_top_k
                    )
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Could not create BM25 index: {e}")
            
            # Create the retriever
            retriever = cls(
                graph_index=graph_index,
                vector_index=vector_index,
                bm25_index=bm25_index,
                graph_depth=graph_depth,
                graph_top_k=graph_top_k,
                bm25_top_k=bm25_top_k,
                vector_top_k=vector_top_k,
                weights=weights,
                working_set_cache=working_set_cache,
                config=config
            )
            
            return retriever
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to create HybridEnsembleRetriever: {e}")
            raise
    
    async def retrieve_chunks(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        budget: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks using hybrid ensemble approach.
        
        Args:
            query: Query string
            session_id: Session ID for working set context
            budget: Maximum chunks to return (defaults to config)
            
        Returns:
            List[RetrievedChunk]: Retrieved and ranked chunks
        """
        if budget is None:
            budget = self.max_chunks
        
        try:
            # Use the new _retrieve method
            query_bundle = QueryBundle(query_str=query)
            nodes_with_scores = self._retrieve(query_bundle)
            
            # Apply working-set boost if we have session context
            if session_id and self.working_set_cache:
                nodes_with_scores = await self._apply_working_set_boost(nodes_with_scores, session_id)
            
            # Convert to RetrievedChunk format for backward compatibility
            chunks = []
            for node_with_score in nodes_with_scores[:budget]:
                chunk = RetrievedChunk(
                    content=node_with_score.node.get_content(),
                    source=node_with_score.node.node_id,
                    score=node_with_score.score or 0.0,
                    chunk_type="hybrid"
                )
                chunk.metadata = node_with_score.node.metadata or {}
                chunks.append(chunk)
            
            # Update working set cache with retrieved nodes
            if session_id and self.working_set_cache and chunks:
                retrieved_node_ids = [chunk.source for chunk in chunks]
                if retrieved_node_ids:
                    await self.working_set_cache.update_working_set(session_id, retrieved_node_ids)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error in hybrid retrieval: {e}")
            return []

    async def _apply_working_set_boost(self, nodes: List[NodeWithScore], session_id: str) -> List[NodeWithScore]:
        """Apply boost to nodes that are in the working set (recent conversational focus)."""
        try:
            recent = set(await self.working_set_cache.get_working_set(session_id) or [])
            
            for n in nodes:
                if n.node.node_id in recent:
                    n.score = (n.score or 0.0) + 0.05  # Small boost for working set items
                    self.logger.debug(f"Applied working set boost to node: {n.node.node_id}")
            
            return nodes
            
        except Exception as e:
            self.logger.warning(f"Error applying working set boost: {e}")
            return nodes
    
    async def query_with_context(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Query with context assembly using the retriever.
        
        Args:
            query: Query string
            session_id: Session ID for working set context
            
        Returns:
            str: Assembled context response
        """
        try:
            # Use the new retrieve method
            chunks = await self.retrieve_chunks(query, session_id)
            return "\n\n".join([chunk.content for chunk in chunks])
            
        except Exception as e:
            self.logger.error(f"Error in query with context: {e}")
            return ""
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retriever metrics."""
        return self.metrics.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get retriever status."""
        return {
            "graph_retriever_available": self.graph is not None,
            "bm25_retriever_available": self.bm25 is not None,
            "vector_retriever_available": self.vector is not None,
            "active_retrievers": len(self.live_retrievers),
            "graph_retriever_type": type(self.graph).__name__ if self.graph else None,
            "configuration": {
                "max_chunks": self.max_chunks,
                "weights": self.weights,
                "retriever_types": [type(r).__name__ for r in self.live_retrievers]
            },
            "metrics": self.get_metrics()
        } 
   # ---------------------------------------------------------------------
    # Compatibility shim – OrchestratedModelManager expects .initialize()
    # ---------------------------------------------------------------------
    @classmethod
    def initialize(
        cls,
        graph_index=None,
        document_nodes=None,
        config: Optional[Dict[str, Any]] = None,
        working_set_cache: Optional[WorkingSetCache] = None,
    ):
        """
        Synchronous wrapper that simply awaits .create() if called
        from a non-async context.  Keeps legacy code paths working.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're already in an event loop – call create() directly
            return cls.create(                       # returns a coroutine
                graph_index=graph_index,
                document_nodes=document_nodes,
                config=config,
                working_set_cache=working_set_cache,
            )
        else:
            # No loop – run the coroutine synchronously
            return asyncio.run(
                cls.create(
                    graph_index=graph_index,
                    document_nodes=document_nodes,
                    config=config,
                    working_set_cache=working_set_cache,
                )
            )