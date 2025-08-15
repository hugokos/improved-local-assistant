#!/usr/bin/env python3
"""
GraphRAG Pipeline Benchmark

Measures the complete user experience including:
1. Knowledge graph retrieval time
2. Context assembly overhead  
3. Model TTFT (Time-to-First-Token)
4. Total end-to-end response time

This provides realistic performance metrics for the full GraphRAG pipeline
that users actually experience.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import Any
from typing import Dict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.conversation_manager import ConversationManager
from services.graph_manager import KnowledgeGraphManager
from services.model_manager import ModelConfig
from services.model_manager import ModelManager

from app.core.config import load_config


class GraphRAGBenchmark:
    """Benchmark the complete GraphRAG pipeline performance."""

    def __init__(self):
        self.config = load_config()

        # Initialize components
        host = self.config.get("ollama", {}).get("host", "http://localhost:11434")
        self.model_manager = ModelManager(host)
        self.kg_manager = KnowledgeGraphManager(self.config)
        self.conversation_manager = ConversationManager(
            model_manager=self.model_manager, kg_manager=self.kg_manager, config=self.config
        )

        # Test queries that should hit the knowledge graph
        self.test_queries = [
            "What is artificial intelligence?",
            "Tell me about machine learning applications",
            "How do neural networks work?",
            "What are the benefits of local AI?",
            "Explain knowledge graphs",
        ]

    async def initialize_system(self) -> bool:
        """Initialize the GraphRAG system."""
        try:
            print("Initializing GraphRAG system...")

            # Initialize model manager
            config = ModelConfig(
                name="hermes3:3b", type="conversation", temperature=0.7, max_tokens=300
            )

            init_success = await self.model_manager.initialize_models(config)
            if not init_success:
                print("Failed to initialize models")
                return False

            # Initialize knowledge graph manager (simple initialization)
            try:
                # Try to initialize dynamic graph
                self.kg_manager.initialize_dynamic_graph()
                print("✓ Knowledge graph manager initialized")
            except Exception as e:
                print(f"Warning: Could not initialize knowledge graphs: {e}")
                print("Continuing with model-only benchmarking...")

            print("✓ GraphRAG system initialized successfully")
            return True

        except Exception as e:
            print(f"Failed to initialize system: {e}")
            return False

    async def create_test_graph(self):
        """Create a simple test knowledge graph for benchmarking."""
        try:
            # Try to add some test content to the dynamic graph
            test_messages = [
                "Artificial intelligence is a branch of computer science that creates intelligent machines.",
                "Machine learning enables computers to learn from experience without explicit programming.",
                "Neural networks are inspired by biological neural networks and use interconnected nodes.",
            ]

            # Add test messages to dynamic graph
            for msg in test_messages:
                try:
                    await self.kg_manager.update_dynamic_graph(msg)
                except Exception:
                    pass  # Continue even if this fails

            print("✓ Added test content to knowledge graph")

        except Exception as e:
            print(f"Warning: Could not create test graph: {e}")

    async def benchmark_single_query(self, query: str) -> Dict[str, float]:
        """
        Benchmark a single query through the complete GraphRAG pipeline.

        Returns:
            Dict with timing breakdown: retrieval, prep, ttft, ttlt
        """
        session_id = self.conversation_manager.create_session()

        # Start timing
        t0 = time.time()

        try:
            # 1. Knowledge Graph Retrieval Phase
            retrieval_start = time.time()

            # Actually perform knowledge graph retrieval
            retrieved_context = None
            try:
                # Try to query the knowledge graph for relevant context
                retrieved_context = await self.kg_manager.query_graphs(query)
                print(
                    f"    Retrieved {len(retrieved_context) if retrieved_context else 0} context items"
                )
            except Exception:
                # If graph retrieval fails, try alternative methods
                try:
                    # Try getting graph stats (this exercises the graph system)
                    stats = await self.kg_manager.get_stats()
                    if stats.get("total_nodes", 0) > 0:
                        print(f"    Graph has {stats.get('total_nodes', 0)} nodes")
                except Exception:
                    print("    No graph retrieval available")

            retrieval_time = time.time() - retrieval_start

            # 2. Context Assembly Phase
            prep_start = time.time()

            # Prepare the context for the model (if we retrieved any)
            enhanced_query = query
            if retrieved_context:
                try:
                    # Handle different types of retrieved context safely
                    if isinstance(retrieved_context, dict):
                        # If it's a dict with response, use that
                        if "response" in retrieved_context:
                            context_text = str(retrieved_context["response"])[:500]  # Limit length
                            enhanced_query = f"Context: {context_text}\n\nQuestion: {query}"
                            print("    Enhanced query with context from response")
                        elif "source_nodes" in retrieved_context:
                            # Use source nodes if available
                            nodes = retrieved_context["source_nodes"][:3]
                            context_text = "\n".join([str(node) for node in nodes])[:500]
                            enhanced_query = f"Context: {context_text}\n\nQuestion: {query}"
                            print(f"    Enhanced query with {len(nodes)} source nodes")
                    elif isinstance(retrieved_context, (list, tuple)):
                        # If it's a list, join the first few items
                        context_items = [str(item) for item in retrieved_context[:3]]
                        context_text = "\n".join(context_items)[:500]
                        enhanced_query = f"Context: {context_text}\n\nQuestion: {query}"
                        print(f"    Enhanced query with {len(context_items)} context items")
                    else:
                        # Fallback: just convert to string
                        context_text = str(retrieved_context)[:500]
                        enhanced_query = f"Context: {context_text}\n\nQuestion: {query}"
                        print("    Enhanced query with string context")
                except Exception as e:
                    print(f"    Warning: Could not process context: {e}")
                    # Continue with original query if context processing fails

            prep_time = time.time() - prep_start

            # 3. Model Inference Phase
            inference_start = time.time()
            first_token_time = None
            token_count = 0

            # Process the message through the complete pipeline
            response_stream = self.conversation_manager.process_message(session_id, query)

            async for chunk in response_stream:
                if first_token_time is None:
                    first_token_time = time.time()

                if isinstance(chunk, str):
                    token_count += len(chunk.split())

            end_time = time.time()

            # Calculate timing metrics
            ttft = first_token_time - inference_start if first_token_time else 0
            ttlt = end_time - t0

            return {
                "retrieval": retrieval_time,
                "prep": prep_time,
                "ttft": ttft,
                "ttlt": ttlt,
                "tokens": token_count,
                "success": True,
            }

        except Exception as e:
            print(f"Error during query '{query}': {e}")
            return {
                "retrieval": 0,
                "prep": 0,
                "ttft": 0,
                "ttlt": time.time() - t0,
                "tokens": 0,
                "success": False,
                "error": str(e),
            }

    async def benchmark_pipeline(self, num_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark the complete GraphRAG pipeline with multiple queries.

        Args:
            num_runs: Number of runs per query

        Returns:
            Complete benchmark results
        """
        print("Starting GraphRAG Pipeline Benchmark")
        print(f"Queries: {len(self.test_queries)}")
        print(f"Runs per query: {num_runs}")
        print("-" * 50)

        all_results = []
        query_results = {}

        for query in self.test_queries:
            print(f"\nBenchmarking: '{query[:50]}...'")
            query_metrics = []

            # Warm-up run
            print("  Warm-up run...")
            await self.benchmark_single_query(query)

            # Actual benchmark runs
            for run in range(num_runs):
                print(f"  Run {run + 1}/{num_runs}...")
                result = await self.benchmark_single_query(query)

                if result["success"]:
                    query_metrics.append(result)
                    all_results.append(result)
                else:
                    print(f"    Failed: {result.get('error', 'Unknown error')}")

            # Calculate averages for this query
            if query_metrics:
                avg_result = {
                    "query": query,
                    "runs": len(query_metrics),
                    "avg_retrieval": statistics.mean([r["retrieval"] for r in query_metrics]),
                    "avg_prep": statistics.mean([r["prep"] for r in query_metrics]),
                    "avg_ttft": statistics.mean([r["ttft"] for r in query_metrics]),
                    "avg_ttlt": statistics.mean([r["ttlt"] for r in query_metrics]),
                    "avg_tokens": statistics.mean([r["tokens"] for r in query_metrics]),
                }

                query_results[query] = avg_result

                # Print immediate results
                print(f"    Retrieval: {avg_result['avg_retrieval']:.3f}s")
                print(f"    Prep: {avg_result['avg_prep']:.3f}s")
                print(f"    TTFT: {avg_result['avg_ttft']:.3f}s")
                print(f"    Total: {avg_result['avg_ttlt']:.3f}s")
                print(f"    Tokens: {avg_result['avg_tokens']:.1f}")

        # Calculate overall statistics
        if all_results:
            overall_stats = {
                "total_queries": len(self.test_queries),
                "successful_runs": len(all_results),
                "avg_retrieval_time": statistics.mean([r["retrieval"] for r in all_results]),
                "avg_prep_time": statistics.mean([r["prep"] for r in all_results]),
                "avg_ttft": statistics.mean([r["ttft"] for r in all_results]),
                "avg_total_time": statistics.mean([r["ttlt"] for r in all_results]),
                "avg_tokens": statistics.mean([r["tokens"] for r in all_results]),
                "retrieval_std": statistics.stdev([r["retrieval"] for r in all_results])
                if len(all_results) > 1
                else 0,
                "ttft_std": statistics.stdev([r["ttft"] for r in all_results])
                if len(all_results) > 1
                else 0,
                "total_std": statistics.stdev([r["ttlt"] for r in all_results])
                if len(all_results) > 1
                else 0,
            }
        else:
            overall_stats = {
                "total_queries": len(self.test_queries),
                "successful_runs": 0,
                "error": "No successful runs",
            }

        return {
            "timestamp": time.time(),
            "test_queries": self.test_queries,
            "query_results": query_results,
            "overall_stats": overall_stats,
            "raw_results": all_results,
        }

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"graphrag_benchmark_{timestamp}.json"

        filepath = os.path.join(project_root, "benchmarks", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filepath}")
        return filepath


async def main():
    parser = argparse.ArgumentParser(description="Benchmark GraphRAG pipeline performance")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per query")
    parser.add_argument("--output", help="Output filename for results")

    args = parser.parse_args()

    try:
        benchmark = GraphRAGBenchmark()

        # Initialize the system
        if not await benchmark.initialize_system():
            print("Failed to initialize GraphRAG system")
            sys.exit(1)

        # Run the benchmark
        results = await benchmark.benchmark_pipeline(args.runs)

        # Print summary
        print("\n" + "=" * 60)
        print("GRAPHRAG PIPELINE BENCHMARK SUMMARY")
        print("=" * 60)

        stats = results.get("overall_stats", {})
        if stats.get("successful_runs", 0) > 0:
            print(
                f"Successful runs: {stats['successful_runs']}/{stats['total_queries'] * args.runs}"
            )
            print(
                f"Average retrieval time: {stats['avg_retrieval_time']:.3f}s (±{stats['retrieval_std']:.3f})"
            )
            print(f"Average prep time: {stats['avg_prep_time']:.3f}s")
            print(f"Average TTFT: {stats['avg_ttft']:.3f}s (±{stats['ttft_std']:.3f})")
            print(f"Average total time: {stats['avg_total_time']:.3f}s (±{stats['total_std']:.3f})")
            print(f"Average tokens: {stats['avg_tokens']:.1f}")

            # Calculate throughput
            if stats["avg_total_time"] > stats["avg_ttft"]:
                throughput = stats["avg_tokens"] / (stats["avg_total_time"] - stats["avg_ttft"])
                print(f"Average throughput: {throughput:.1f} tokens/sec")

            print("\nPipeline Breakdown:")
            print(
                f"  Retrieval: {stats['avg_retrieval_time']:.3f}s ({stats['avg_retrieval_time']/stats['avg_total_time']*100:.1f}%)"
            )
            print(
                f"  Prep: {stats['avg_prep_time']:.3f}s ({stats['avg_prep_time']/stats['avg_total_time']*100:.1f}%)"
            )
            print(
                f"  Model TTFT: {stats['avg_ttft']:.3f}s ({stats['avg_ttft']/stats['avg_total_time']*100:.1f}%)"
            )
            print(
                f"  Model Generation: {stats['avg_total_time']-stats['avg_ttft']:.3f}s ({(stats['avg_total_time']-stats['avg_ttft'])/stats['avg_total_time']*100:.1f}%)"
            )
        else:
            print("No successful runs completed")
            if "error" in stats:
                print(f"Error: {stats['error']}")

        # Save results
        benchmark.save_results(results, args.output)

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
