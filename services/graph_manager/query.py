"""
Query module for KnowledgeGraphManager.

Handles querying the graphs including ensemble queries, subgraph extraction,
and graph traversal operations.
"""

import time
from typing import Any, Dict, List, Optional

import networkx as nx


class KnowledgeGraphQuery:
    """Handles querying operations for knowledge graphs."""

    async def query_graphs(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Query all knowledge graphs using advanced routing and hybrid retrieval.

        Args:
            query: Query string
            context: Optional context for the query

        Returns:
            Dict[str, Any]: Query results
        """
        start_time = time.time()

        # Try to use the advanced GraphRouter first
        try:
            if hasattr(self, '_graph_router') and self._graph_router:
                self.logger.info("Using advanced GraphRouter for query")
                result = await self._graph_router.query(query, k=3)
                
                if result and result.get("response"):
                    # Convert to expected format
                    query_time = time.time() - start_time
                    
                    # Update metrics
                    self.metrics["queries_processed"] += 1
                    self.metrics["last_query_time"] = query_time
                    
                    if self.metrics["queries_processed"] > 1:
                        self.metrics["avg_query_time"] = (
                            self.metrics["avg_query_time"] * (self.metrics["queries_processed"] - 1) + query_time
                        ) / self.metrics["queries_processed"]
                    else:
                        self.metrics["avg_query_time"] = query_time
                    
                    return {
                        "response": result["response"],
                        "source_nodes": result.get("source_nodes", []),
                        "metadata": {
                            **result.get("metadata", {}),
                            "method": "advanced_router",
                            "query_time": query_time
                        }
                    }
        except Exception as e:
            self.logger.warning(f"Advanced GraphRouter failed, falling back to legacy method: {e}")

        # Fallback to legacy query method
        return await self._legacy_query_graphs(query, context)
    
    async def _legacy_query_graphs(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Legacy query method for backward compatibility.
        """
        start_time = time.time()

        # Try to use optimizer if available
        cached_result = None
        optimized_query = query
        
        try:
            # Import optimizer here to avoid circular imports
            from services.kg_optimizer import (
                cache_query_result,
                get_cached_query,
                optimize_query,
                optimizer,
            )

            # Only use optimizer if it's initialized
            if optimizer is not None:
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
            else:
                self.logger.debug("Knowledge graph optimizer not initialized, skipping optimization")
                
        except Exception as e:
            self.logger.warning(f"Knowledge graph optimizer not available: {str(e)}")
            # Continue without optimization

        try:
            query_engines = []
            graph_sources = []

            # Add all knowledge graph query engines with hybrid retrieval
            for graph_id, kg_index in self.kg_indices.items():
                qe = kg_index.as_query_engine(
                    include_text=True,                   # Enables summarization
                    response_mode="tree_summarize",      # Synthesizes node info
                    embedding_mode="hybrid",             # Activates vector + keyword search
                    similarity_top_k=5                   # Adjust for corpus density
                )
                query_engines.append(qe)
                graph_sources.append(graph_id)

            # Add dynamic graph if available with hybrid retrieval
            if self.dynamic_kg:
                dynamic_qe = self.dynamic_kg.as_query_engine(
                    include_text=True,
                    response_mode="tree_summarize",
                    embedding_mode="hybrid",
                    similarity_top_k=5
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

            # Use HybridEnsembleRetriever as primary approach
            try:
                from services.hybrid_retriever import HybridEnsembleRetriever
                from llama_index.core.schema import QueryBundle
                
                # Build hybrid retriever once and cache
                if not hasattr(self, '_cached_hybrid_retriever'):
                    graph_index = self.dynamic_kg or next(iter(self.kg_indices.values()), None)
                    doc_nodes = []  # Could collect doc nodes from indices for BM25/vector
                    
                    self._cached_hybrid_retriever = await HybridEnsembleRetriever.create(
                        graph_index=graph_index,
                        document_nodes=doc_nodes,
                        config=getattr(self, "config", {}),
                        working_set_cache=getattr(self, "working_set_cache", None),
                    )
                    self.logger.info("Created cached HybridEnsembleRetriever")
                
                # Use hybrid retriever
                query_bundle = QueryBundle(query_str=optimized_query)
                nodes = self._cached_hybrid_retriever._retrieve(query_bundle)
                
                if nodes:
                    # Create a mock result object with the nodes
                    class MockResult:
                        def __init__(self, nodes):
                            self.response = self._synthesize_response_from_nodes(nodes, optimized_query)
                            self.source_nodes = nodes
                    
                    result = MockResult(nodes)
                    self.logger.info(f"HybridEnsembleRetriever returned {len(nodes)} nodes")
                else:
                    # Fallback to legacy approach
                    raise Exception("No nodes from hybrid retriever")
                    
            except Exception as e:
                self.logger.warning(f"HybridEnsembleRetriever failed, using legacy approach: {e}")
                
                # Legacy fallback
                if len(query_engines) > 1:
                    self.logger.info(f"Querying {len(query_engines)} knowledge graphs")
                    
                    # Try each query engine and use the first one that returns good results
                    best_result = None
                    best_source = None
                    
                    for i, qe in enumerate(query_engines):
                        try:
                            result = qe.query(optimized_query)
                            source_nodes = getattr(result, "source_nodes", [])
                            
                            # If we get good results (with source nodes), use this result
                            if source_nodes and len(source_nodes) > 0:
                                self.logger.info(f"Got {len(source_nodes)} results from {graph_sources[i]}")
                                best_result = result
                                best_source = graph_sources[i]
                                break
                            else:
                                self.logger.debug(f"No results from {graph_sources[i]}")
                                
                        except Exception as e:
                            self.logger.warning(f"Error querying {graph_sources[i]}: {str(e)}")
                            continue
                    
                    # Use the best result, or fall back to first engine if none had good results
                    if best_result:
                        result = best_result
                        self.logger.info(f"Using results from {best_source}")
                    else:
                        self.logger.info("No engines returned good results, using first engine")
                        result = query_engines[0].query(optimized_query)
                        
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
                    self.metrics["avg_query_time"] * (self.metrics["queries_processed"] - 1) + query_time
                ) / self.metrics["queries_processed"]
            else:
                self.metrics["avg_query_time"] = query_time

            # Check if we got good results, otherwise use vector fallback
            source_nodes = getattr(result, "source_nodes", [])
            
            # If no good results from KG engines, try vector store fallback
            if not source_nodes or len(source_nodes) == 0:
                self.logger.info("No results from KG engines, trying vector store fallback")
                try:
                    fallback_result = self._try_vector_fallback(optimized_query, graph_sources)
                    if fallback_result:
                        result = fallback_result
                        source_nodes = getattr(result, "source_nodes", [])
                        graph_sources.append("vector_fallback")
                except Exception as e:
                    self.logger.warning(f"Vector fallback failed: {e}")

            # Prepare response
            query_result = {
                "response": result.response,
                "source_nodes": source_nodes,
                "metadata": {
                    "graph_sources": graph_sources,
                    "query_time": query_time,
                    "optimized_query": optimized_query if optimized_query != query else None,
                    "used_fallback": "vector_fallback" in graph_sources,
                },
            }

            # Cache the result if optimizer is available
            try:
                from services.kg_optimizer import optimizer
                if optimizer is not None:
                    cache_query_result(query, query_result)
            except Exception as e:
                self.logger.debug(f"Could not cache query result: {str(e)}")

            return query_result

        except Exception as e:
            self.logger.error(f"Error querying graphs: {str(e)}")
            return {
                "response": f"Error querying knowledge graphs: {str(e)}",
                "source_nodes": [],
                "metadata": {"graph_sources": graph_sources, "query_time": time.time() - start_time, "error": str(e)},
            }

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
                    if query_lower in node_text or any(query_lower in str(v).lower() for v in data.values()):
                        relevant_nodes.add((node, i))  # Store node with graph index

            # If no relevant nodes found, return empty subgraph
            if not relevant_nodes:
                return {"nodes": [], "edges": [], "metadata": {"query": query, "sources": graph_sources}}

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
                            {"source": node, "target": neighbor, "data": edge_data, "graph_source": source}
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
                        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops))
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
                        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops))
                        if paths:
                            all_paths.extend(paths)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

            return all_paths

        except Exception as e:
            self.logger.error(f"Error in graph traversal: {str(e)}")
            return []

    def _synthesize_response_from_nodes(self, nodes, query: str) -> str:
        """Synthesize a response from retrieved nodes."""
        try:
            if not nodes:
                return "No relevant information found."
            
            # Extract content from nodes
            contexts = []
            for node in nodes[:5]:  # Use top 5 nodes
                content = node.node.get_content()
                if content:
                    contexts.append(content)
            
            if not contexts:
                return "No relevant information found."
            
            # Simple synthesis - in production you'd use an LLM
            combined_context = "\n\n".join(contexts)
            
            # Truncate if too long
            if len(combined_context) > 1000:
                combined_context = combined_context[:1000] + "..."
            
            return f"Based on the available information:\n\n{combined_context}"
            
        except Exception as e:
            self.logger.error(f"Error synthesizing response: {e}")
            return "Error processing the retrieved information."

    def _try_vector_fallback(self, query: str, graph_sources: List[str]):
        """
        Try vector store fallback when KG engines return no results.
        
        Args:
            query: Query string
            graph_sources: List of graph sources that were tried
            
        Returns:
            Query result from vector store or None
        """
        try:
            # Check if we have any vector indices available
            vector_engines = []
            
            # Try to create vector engines from the same documents used in KG
            for graph_id, kg_index in self.kg_indices.items():
                try:
                    # Try to get the underlying documents from the KG index
                    if hasattr(kg_index, '_docstore') and kg_index._docstore:
                        # Create a vector index from the same documents
                        from llama_index.core import VectorStoreIndex
                        
                        # Get documents from docstore
                        docs = list(kg_index._docstore.docs.values())
                        if docs:
                            vector_index = VectorStoreIndex.from_documents(docs)
                            vector_qe = vector_index.as_query_engine(
                                similarity_top_k=5,
                                response_mode="tree_summarize"
                            )
                            vector_engines.append(vector_qe)
                            self.logger.debug(f"Created vector fallback for {graph_id}")
                            
                except Exception as e:
                    self.logger.debug(f"Could not create vector fallback for {graph_id}: {e}")
                    continue
            
            # If we have vector engines, try them
            if vector_engines:
                # Use the first available vector engine
                result = vector_engines[0].query(query)
                self.logger.info("Vector fallback provided results")
                return result
            else:
                self.logger.debug("No vector fallback engines available")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in vector fallback: {e}")
            return None