#!/usr/bin/env python3
"""
Simple RAG Query Script - Fire Entries from Survivalist Graph

This script demonstrates how to query the survivalist knowledge graph
for entries related to "fire" with detailed logging and multiple retrieval methods.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("fire_query.log")],
)

logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment for querying."""
    logger.info("🔧 Setting up environment...")

    # Configure LlamaIndex settings
    from llama_index.core import Settings
    from services.embedding_singleton import get_embedding_model

    # Set up local embedding model
    logger.info("📊 Initializing embedding model...")
    embed_model = get_embedding_model("BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    # Disable LLM for pure retrieval
    Settings.llm = None

    logger.info("✅ Environment setup complete")


def load_survivalist_graph():
    """Load the survivalist knowledge graph."""
    logger.info("📚 Loading survivalist knowledge graph...")

    from llama_index.core import StorageContext
    from llama_index.core import load_index_from_storage
    from services.utf8_import_helper import get_utf8_filesystem

    # Path to the survivalist graph
    graph_path = project_root / "data" / "prebuilt_graphs" / "survivalist"

    if not graph_path.exists():
        logger.error(f"❌ Graph path not found: {graph_path}")
        return None

    try:
        # Load with UTF-8 filesystem support
        logger.info(f"🔍 Loading graph from: {graph_path}")
        storage_context = StorageContext.from_defaults(
            persist_dir=str(graph_path), fs=get_utf8_filesystem()
        )

        index = load_index_from_storage(storage_context=storage_context)
        logger.info("✅ Successfully loaded survivalist graph")

        # Log graph statistics
        if hasattr(index, "storage_context"):
            storage_ctx = index.storage_context

            # Document count
            if hasattr(storage_ctx, "docstore") and hasattr(storage_ctx.docstore, "docs"):
                doc_count = len(storage_ctx.docstore.docs)
                logger.info(f"📊 Graph contains {doc_count} documents")

            # Vector count
            if hasattr(storage_ctx, "vector_store"):
                try:
                    vector_data = storage_ctx.vector_store.to_dict()
                    if "embedding_dict" in vector_data:
                        vector_count = len(vector_data["embedding_dict"])
                        logger.info(f"🔢 Graph contains {vector_count} vectors")
                except Exception as e:
                    logger.debug(f"Could not get vector count: {e}")

        return index

    except Exception as e:
        logger.error(f"❌ Failed to load graph: {e}")
        return None


def query_with_vector_retriever(index, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Query using vector similarity retrieval."""
    logger.info(f"🔍 Querying with vector retriever: '{query}' (top_k={top_k})")

    try:
        # Create vector retriever
        retriever = index.as_retriever(similarity_top_k=top_k)

        # Perform retrieval
        nodes = retriever.retrieve(query)

        results = []
        for i, node in enumerate(nodes):
            result = {
                "rank": i + 1,
                "score": getattr(node, "score", "N/A"),
                "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "full_text": node.text,
                "metadata": getattr(node, "metadata", {}),
                "node_id": getattr(node, "node_id", "N/A"),
            }
            results.append(result)

        logger.info(f"✅ Vector retrieval found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"❌ Vector retrieval failed: {e}")
        return []


def query_with_keyword_search(index, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Query using keyword-based search through document store."""
    logger.info(f"🔍 Performing keyword search: '{query}' (top_k={top_k})")

    try:
        results = []

        if hasattr(index, "storage_context") and hasattr(index.storage_context, "docstore"):
            docstore = index.storage_context.docstore

            # Search through all documents
            query_lower = query.lower()
            matches = []

            for doc_id, doc in docstore.docs.items():
                text = doc.text.lower()
                if query_lower in text:
                    # Count occurrences for scoring
                    count = text.count(query_lower)
                    matches.append({"doc_id": doc_id, "doc": doc, "count": count, "text": doc.text})

            # Sort by relevance (occurrence count)
            matches.sort(key=lambda x: x["count"], reverse=True)

            # Format results
            for i, match in enumerate(matches[:top_k]):
                result = {
                    "rank": i + 1,
                    "score": f"keyword_count:{match['count']}",
                    "text": match["text"][:200] + "..."
                    if len(match["text"]) > 200
                    else match["text"],
                    "full_text": match["text"],
                    "metadata": getattr(match["doc"], "metadata", {}),
                    "node_id": match["doc_id"],
                }
                results.append(result)

        logger.info(f"✅ Keyword search found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"❌ Keyword search failed: {e}")
        return []


def query_graph_relationships(index, query: str) -> list[dict[str, Any]]:
    """Query graph relationships if available."""
    logger.info(f"🔍 Searching graph relationships for: '{query}'")

    try:
        results = []

        if hasattr(index, "storage_context") and hasattr(index.storage_context, "graph_store"):

            # Check if it's a property graph store
            if hasattr(index.storage_context, "property_graph_store"):
                prop_graph_store = index.storage_context.property_graph_store

                if hasattr(prop_graph_store, "graph"):
                    graph = prop_graph_store.graph
                    query_lower = query.lower()

                    # Search nodes
                    if hasattr(graph, "nodes"):
                        for node in graph.nodes():
                            node_str = str(node).lower()
                            if query_lower in node_str:
                                results.append(
                                    {
                                        "type": "node",
                                        "content": str(node),
                                        "data": graph.nodes[node]
                                        if hasattr(graph.nodes[node], "__dict__")
                                        else {},
                                    }
                                )

                    # Search edges
                    if hasattr(graph, "edges"):
                        for source, target, data in graph.edges(data=True):
                            edge_text = f"{source} -> {target}".lower()
                            if query_lower in edge_text or query_lower in str(data).lower():
                                results.append(
                                    {
                                        "type": "edge",
                                        "content": f"{source} -> {target}",
                                        "data": data,
                                    }
                                )

        logger.info(f"✅ Graph relationship search found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"❌ Graph relationship search failed: {e}")
        return []


def display_results(results: list[dict[str, Any]], title: str):
    """Display query results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"🔥 {title}")
    print(f"{'='*60}")

    if not results:
        print("❌ No results found")
        return

    for result in results:
        print(f"\n📄 Rank {result['rank']} (Score: {result['score']})")
        print(f"📝 Text: {result['text']}")

        if result.get("metadata"):
            print(f"📊 Metadata: {result['metadata']}")

        print(f"🆔 Node ID: {result['node_id']}")
        print("-" * 40)


def display_graph_results(results: list[dict[str, Any]], title: str):
    """Display graph relationship results."""
    print(f"\n{'='*60}")
    print(f"🔗 {title}")
    print(f"{'='*60}")

    if not results:
        print("❌ No graph relationships found")
        return

    for i, result in enumerate(results, 1):
        print(f"\n🔗 {i}. {result['type'].upper()}: {result['content']}")
        if result.get("data"):
            print(f"📊 Data: {result['data']}")
        print("-" * 40)


def save_results_to_file(all_results: dict[str, list], query: str):
    """Save all results to a JSON file."""
    output_file = f"fire_query_results_{query.replace(' ', '_')}.json"

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to: {output_file}")
        print(f"\n💾 Detailed results saved to: {output_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main function to run the fire query script."""
    print("🔥 Fire Entries Query Script")
    print("=" * 50)

    # Setup environment
    setup_environment()

    # Load the survivalist graph
    index = load_survivalist_graph()
    if not index:
        logger.error("❌ Failed to load graph. Exiting.")
        return

    # Define fire-related queries
    fire_queries = [
        "fire",
        "fire starting",
        "fire making",
        "ignition",
        "flame",
        "burning",
        "tinder",
        "kindling",
        "matches",
        "lighter",
        "flint",
    ]

    all_results = {}

    for query in fire_queries:
        logger.info(f"\n🔍 Processing query: '{query}'")
        print(f"\n🔍 Processing query: '{query}'")

        # Vector-based retrieval
        vector_results = query_with_vector_retriever(index, query, top_k=5)

        # Keyword-based search
        keyword_results = query_with_keyword_search(index, query, top_k=5)

        # Graph relationship search
        graph_results = query_graph_relationships(index, query)

        # Store results
        all_results[query] = {
            "vector_results": vector_results,
            "keyword_results": keyword_results,
            "graph_results": graph_results,
        }

        # Display results
        if vector_results:
            display_results(vector_results, f"Vector Results for '{query}'")

        if keyword_results:
            display_results(keyword_results, f"Keyword Results for '{query}'")

        if graph_results:
            display_graph_results(graph_results, f"Graph Relationships for '{query}'")

    # Save all results to file
    save_results_to_file(all_results, "fire_comprehensive")

    # Summary
    total_results = sum(
        len(results["vector_results"])
        + len(results["keyword_results"])
        + len(results["graph_results"])
        for results in all_results.values()
    )

    print("\n🎉 Query completed!")
    print(f"📊 Total queries processed: {len(fire_queries)}")
    print(f"📊 Total results found: {total_results}")
    print("📝 Check fire_query.log for detailed logs")


if __name__ == "__main__":
    print("🚀 Starting fire query script...")
    try:
        main()
    except Exception as e:
        print(f"❌ Script failed: {e}")
        import traceback

        traceback.print_exc()
    print("🏁 Script execution completed.")
