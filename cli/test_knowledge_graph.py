#!/usr/bin/env python
"""
Test script for knowledge graph functionality.

This script provides command-line testing for the KnowledgeGraphManager,
allowing testing of graph creation, loading, querying, and visualization.
"""

import argparse
import asyncio
import logging
import sys

import yaml

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import time
from pathlib import Path

# Add parent directory to path to import from services
sys.path.append(str(Path(__file__).parent.parent))

from improved_local_assistant.services.graph_manager import KnowledgeGraphManager
from improved_local_assistant.services.model_mgr import ModelConfig
from improved_local_assistant.services.model_mgr import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


async def initialize_managers():
    """Initialize model and graph managers."""
    config = load_config()

    # Initialize model manager
    model_manager = ModelManager(
        host=config.get("ollama", {}).get("host", "http://localhost:11434")
    )

    # Get model configuration
    model_config = ModelConfig(
        name=config.get("models", {}).get("conversation", {}).get("name", "hermes3:3b"),
        type="conversation",
        context_window=config.get("models", {}).get("conversation", {}).get("context_window", 8000),
        temperature=config.get("models", {}).get("conversation", {}).get("temperature", 0.7),
        max_tokens=config.get("models", {}).get("conversation", {}).get("max_tokens", 2048),
        timeout=config.get("ollama", {}).get("timeout", 120),
        max_parallel=config.get("ollama", {}).get("max_parallel", 2),
        max_loaded=config.get("ollama", {}).get("max_loaded", 2),
    )

    # Initialize model manager
    await model_manager.initialize_models(model_config)

    # Initialize graph manager
    graph_manager = KnowledgeGraphManager(model_manager, config)

    # Initialize the knowledge graph optimizer
    try:
        from improved_local_assistant.services.kg_optimizer import initialize_optimizer

        initialize_optimizer(graph_manager)
        logger.info("Knowledge graph optimizer initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize knowledge graph optimizer: {e}")
        logger.info("Continuing without optimizer - basic functionality will still work")

    return model_manager, graph_manager


async def test_create_graph(graph_manager, docs_path=None):
    """Test creating a knowledge graph from documents."""
    logger.info("Testing knowledge graph creation")

    # Use default test documents path if not provided
    if not docs_path:
        docs_path = Path(__file__).parent.parent / "data" / "test_docs"

        # Create test documents directory if it doesn't exist
        if not docs_path.exists():
            docs_path.mkdir(parents=True, exist_ok=True)

            # Create a sample document
            sample_doc = docs_path / "sample.txt"
            with open(sample_doc, "w") as f:
                f.write(
                    """
                Knowledge Graph Test Document

                Hermes 3:3B is a language model trained by Nous Research.
                TinyLlama is a smaller language model designed for efficiency.
                LlamaIndex is a framework for building RAG applications.

                Knowledge graphs store information as entities and relationships.
                NetworkX is a Python library for graph analysis.
                PyVis can visualize NetworkX graphs in HTML.

                Local AI assistants run entirely on the user's device.
                Ollama is a tool for running language models locally.
                """
                )

    # Create graph from documents
    logger.info(f"Creating graph from documents in {docs_path}")
    graph_id = graph_manager.create_graph_from_documents(docs_path, "test_graph")

    if graph_id:
        logger.info(f"Successfully created graph with ID: {graph_id}")

        # Test querying the graph
        logger.info("Testing query on newly created graph")
        result = graph_manager.query_graphs("What is Hermes 3:3B?")
        logger.info(f"Query result: {result['response']}")

        # Test visualization
        logger.info("Testing graph visualization")
        html = graph_manager.visualize_graph(graph_id)

        # Save visualization to file
        vis_path = Path(__file__).parent.parent / "data" / "graph_visualization.html"
        with open(vis_path, "w") as f:
            f.write(html)
        logger.info(f"Saved graph visualization to {vis_path}")

        return True
    else:
        logger.error("Failed to create graph")
        return False


async def test_load_prebuilt(graph_manager, prebuilt_path=None):
    """Test loading pre-built knowledge graphs."""
    logger.info("Testing loading pre-built knowledge graphs")

    # Use default prebuilt graphs path if not provided
    if not prebuilt_path:
        prebuilt_path = Path(__file__).parent.parent / "data" / "prebuilt_graphs"

        # Create prebuilt graphs directory if it doesn't exist
        if not prebuilt_path.exists():
            prebuilt_path.mkdir(parents=True, exist_ok=True)

            # Create a sample graph directory
            sample_graph_dir = prebuilt_path / "sample_graph"
            sample_graph_dir.mkdir(exist_ok=True)

            # Create a sample document
            sample_doc = sample_graph_dir / "sample.txt"
            with open(sample_doc, "w") as f:
                f.write(
                    """
                Pre-built Knowledge Graph Test Document

                Python is a programming language created by Guido van Rossum.
                FastAPI is a web framework for building APIs with Python.
                Uvicorn is an ASGI server that can run FastAPI applications.

                Knowledge graphs represent information as a graph structure.
                Nodes represent entities and edges represent relationships.

                LlamaIndex provides tools for building knowledge graphs.
                SimpleGraphStore is an in-memory graph store provided by LlamaIndex.
                """
                )

    # Load pre-built graphs
    logger.info(f"Loading pre-built graphs from {prebuilt_path}")
    loaded_graphs = graph_manager.load_prebuilt_graphs(prebuilt_path)

    if loaded_graphs:
        logger.info(f"Successfully loaded {len(loaded_graphs)} graphs: {loaded_graphs}")

        # Test querying the loaded graphs
        logger.info("Testing query on loaded graphs")
        result = graph_manager.query_graphs("What is Python?")
        logger.info(f"Query result: {result['response']}")

        # Test visualization of first loaded graph
        logger.info("Testing visualization of loaded graph")
        html = graph_manager.visualize_graph(loaded_graphs[0])

        # Save visualization to file
        vis_path = Path(__file__).parent.parent / "data" / "prebuilt_visualization.html"
        with open(vis_path, "w") as f:
            f.write(html)
        logger.info(f"Saved graph visualization to {vis_path}")

        # Test hot-loading a new graph
        logger.info("Testing hot-loading a new graph")

        # Create a new graph directory
        new_graph_dir = prebuilt_path / "new_graph"
        new_graph_dir.mkdir(exist_ok=True)

        # Create a sample document
        new_doc = new_graph_dir / "new.txt"
        with open(new_doc, "w") as f:
            f.write(
                """
            Hot-loaded Knowledge Graph Test Document

            NetworkX is a Python package for complex networks.
            PyVis is a visualization library that can render NetworkX graphs.
            HTML is used to display PyVis visualizations in web browsers.

            Knowledge graph visualization helps users understand relationships.
            Interactive visualizations allow exploration of complex data.
            """
            )

        # Add the new graph
        new_graph_id = graph_manager.add_new_graph(str(new_graph_dir), "hot_loaded_graph")

        if new_graph_id:
            logger.info(f"Successfully hot-loaded new graph with ID: {new_graph_id}")

            # Test querying the new graph
            logger.info("Testing query on hot-loaded graph")
            result = graph_manager.query_graphs("What is NetworkX?")
            logger.info(f"Query result: {result['response']}")

            return True
        else:
            logger.error("Failed to hot-load new graph")
            return False
    else:
        logger.error("Failed to load pre-built graphs")
        return False


async def test_updates(graph_manager, model_manager):
    """Test dynamic knowledge graph updates."""
    logger.info("Testing dynamic knowledge graph updates")

    # Initialize dynamic graph
    success = graph_manager.initialize_dynamic_graph()
    if not success:
        logger.error("Failed to initialize dynamic graph")
        return False

    # Test updating the graph with conversation text
    logger.info("Testing graph update with conversation text")
    conversation_text = """
    The Raspberry Pi 5 is a single-board computer with 8GB of RAM.
    It uses an ARM processor and can run Linux operating systems.
    The Raspberry Pi Foundation created it for educational purposes.
    """

    success = await graph_manager.update_dynamic_graph(conversation_text)
    if not success:
        logger.error("Failed to update dynamic graph")
        return False

    # Wait for background processing to complete
    logger.info("Waiting for background processing to complete...")
    await asyncio.sleep(5)

    # Test querying the dynamic graph
    logger.info("Testing query on dynamic graph")
    result = graph_manager.query_graphs("What is Raspberry Pi 5?")
    logger.info(f"Query result: {result['response']}")

    # Test visualization of dynamic graph
    logger.info("Testing visualization of dynamic graph")
    html = graph_manager.visualize_graph()

    # Save visualization to file
    vis_path = Path(__file__).parent.parent / "data" / "dynamic_visualization.html"
    with open(vis_path, "w") as f:
        f.write(html)
    logger.info(f"Saved dynamic graph visualization to {vis_path}")

    # Test multiple updates
    logger.info("Testing multiple graph updates")

    # Update with more conversation text
    conversation_text2 = """
    Python is a programming language that supports multiple paradigms.
    It was created by Guido van Rossum and first released in 1991.
    Python emphasizes code readability and simplicity.
    """

    await graph_manager.update_dynamic_graph(conversation_text2)

    # Wait for background processing to complete
    logger.info("Waiting for background processing to complete...")
    await asyncio.sleep(5)

    # Test querying with updated information
    logger.info("Testing query on updated dynamic graph")
    result = graph_manager.query_graphs("Who created Python?")
    logger.info(f"Query result: {result['response']}")

    # Get graph statistics
    stats = graph_manager.get_graph_statistics()
    logger.info(f"Graph statistics: {stats}")

    return True


async def test_queries(graph_manager):
    """Test graph-based retrieval and query system."""
    logger.info("Testing graph-based retrieval and query system")

    # Ensure we have at least one graph
    if not graph_manager.kg_indices and not graph_manager.dynamic_kg:
        logger.info("No graphs available, creating a test graph")
        docs_path = Path(__file__).parent.parent / "data" / "test_docs"

        # Create test documents directory if it doesn't exist
        if not docs_path.exists():
            docs_path.mkdir(parents=True, exist_ok=True)

            # Create a sample document
            sample_doc = docs_path / "sample.txt"
            with open(sample_doc, "w") as f:
                f.write(
                    """
                Knowledge Graph Query Test Document

                Albert Einstein was a physicist who developed the theory of relativity.
                Marie Curie was a physicist and chemist who conducted research on radioactivity.
                Isaac Newton developed the laws of motion and universal gravitation.

                The theory of relativity was published by Einstein in 1915.
                Radioactivity was discovered by Henri Becquerel in 1896.
                Newton's laws of motion were published in Principia Mathematica in 1687.

                Physics is the study of matter, energy, and the interactions between them.
                Chemistry is the study of substances, their properties, and reactions.
                Mathematics is the study of numbers, quantities, and shapes.
                """
                )

        graph_id = graph_manager.create_graph_from_documents(docs_path, "query_test_graph")
        if not graph_id:
            logger.error("Failed to create test graph for queries")
            return False

    # Test basic query
    logger.info("Testing basic query")
    result = graph_manager.query_graphs("Who developed the theory of relativity?")
    logger.info(f"Basic query result: {result['response']}")

    # Test query with context
    logger.info("Testing query with context")
    context = ["We're discussing famous physicists and their contributions."]
    result = graph_manager.query_graphs("What did Marie Curie research?", context)
    logger.info(f"Query with context result: {result['response']}")

    # Test multi-graph querying
    if len(graph_manager.kg_indices) > 1 or (graph_manager.kg_indices and graph_manager.dynamic_kg):
        logger.info("Testing multi-graph querying")
        result = graph_manager.query_graphs("What is physics?")
        logger.info(f"Multi-graph query result: {result['response']}")
        logger.info(f"Graph sources: {result['metadata'].get('graph_sources', [])}")

    # Test graph traversal
    logger.info("Testing graph traversal")
    paths = graph_manager.get_graph_traversal("Albert Einstein", "physics", max_hops=2)
    logger.info(f"Graph traversal paths: {paths}")

    # Test query performance
    logger.info("Testing query performance")
    start_time = time.time()
    result = graph_manager.query_graphs("What are Newton's laws?")
    query_time = time.time() - start_time
    logger.info(f"Query time: {query_time:.4f} seconds")
    logger.info(f"Query result: {result['response']}")

    # Get graph statistics
    stats = graph_manager.get_graph_statistics()
    logger.info(f"Graph statistics: {stats}")

    return True


async def test_visualize(graph_manager):
    """Test knowledge graph visualization capabilities."""
    logger.info("Testing knowledge graph visualization capabilities")

    # Ensure we have at least one graph
    if not graph_manager.kg_indices and not graph_manager.dynamic_kg:
        logger.info("No graphs available, creating a test graph")
        docs_path = Path(__file__).parent.parent / "data" / "test_docs"

        # Create test documents directory if it doesn't exist
        if not docs_path.exists():
            docs_path.mkdir(parents=True, exist_ok=True)

            # Create a sample document
            sample_doc = docs_path / "sample.txt"
            with open(sample_doc, "w") as f:
                f.write(
                    """
                Knowledge Graph Visualization Test Document

                The solar system contains eight planets.
                Mercury is the closest planet to the Sun.
                Venus is the second planet from the Sun.
                Earth is the third planet from the Sun.
                Mars is the fourth planet from the Sun.
                Jupiter is the fifth planet from the Sun.
                Saturn is the sixth planet from the Sun.
                Uranus is the seventh planet from the Sun.
                Neptune is the eighth planet from the Sun.

                The Sun is a star at the center of the solar system.
                Planets orbit around the Sun.
                Earth has one natural satellite called the Moon.
                Mars has two moons: Phobos and Deimos.
                Jupiter has many moons, including Europa and Ganymede.
                """
                )

        graph_id = graph_manager.create_graph_from_documents(docs_path, "visualization_test_graph")
        if not graph_id:
            logger.error("Failed to create test graph for visualization")
            return False

    # Get all graph IDs
    graph_ids = list(graph_manager.kg_indices.keys())
    if graph_manager.dynamic_kg:
        graph_ids.append("dynamic")

    # Create visualizations directory
    vis_dir = Path(__file__).parent.parent / "data" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations for all graphs
    for graph_id in graph_ids:
        logger.info(f"Generating visualization for graph: {graph_id}")

        # Generate visualization
        html = graph_manager.visualize_graph(graph_id if graph_id != "dynamic" else None)

        # Save visualization to file
        vis_path = vis_dir / f"{graph_id}_visualization.html"
        with open(vis_path, "w") as f:
            f.write(html)
        logger.info(f"Saved visualization to {vis_path}")

    # Generate combined visualization
    logger.info("Generating combined visualization")

    # This is a simplified implementation - in a real system,
    # you would create a more sophisticated combined visualization
    combined_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knowledge Graph Visualizations</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            .graph-container { margin-bottom: 30px; }
            iframe { border: 1px solid #ddd; width: 100%; height: 600px; }
        </style>
    </head>
    <body>
        <h1>Knowledge Graph Visualizations</h1>
    """

    for graph_id in graph_ids:
        vis_path = f"{graph_id}_visualization.html"
        combined_html += f"""
        <div class="graph-container">
            <h2>Graph: {graph_id}</h2>
            <iframe src="{vis_path}"></iframe>
        </div>
        """

    combined_html += """
    </body>
    </html>
    """

    # Save combined visualization
    combined_path = vis_dir / "combined_visualization.html"
    with open(combined_path, "w") as f:
        f.write(combined_html)
    logger.info(f"Saved combined visualization to {combined_path}")

    return True


async def main():
    """Main function to run tests based on command-line arguments."""
    parser = argparse.ArgumentParser(description="Test knowledge graph functionality")
    parser.add_argument(
        "--create-graph", action="store_true", help="Test creating a knowledge graph"
    )
    parser.add_argument(
        "--load-prebuilt", action="store_true", help="Test loading pre-built knowledge graphs"
    )
    parser.add_argument(
        "--test-updates", action="store_true", help="Test dynamic knowledge graph updates"
    )
    parser.add_argument(
        "--test-queries", action="store_true", help="Test graph-based retrieval and queries"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Test knowledge graph visualization"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--docs-path", help="Path to documents for graph creation")
    parser.add_argument("--prebuilt-path", help="Path to pre-built knowledge graphs")

    args = parser.parse_args()

    # Initialize managers
    model_manager, graph_manager = await initialize_managers()

    # Run tests based on arguments
    if args.all or args.create_graph:
        await test_create_graph(graph_manager, args.docs_path)

    if args.all or args.load_prebuilt:
        await test_load_prebuilt(graph_manager, args.prebuilt_path)

    if args.all or args.test_updates:
        await test_updates(graph_manager, model_manager)

    if args.all or args.test_queries:
        await test_queries(graph_manager)

    if args.all or args.visualize:
        await test_visualize(graph_manager)

    # If no specific test was requested, show help
    if not (
        args.all
        or args.create_graph
        or args.load_prebuilt
        or args.test_updates
        or args.test_queries
        or args.visualize
    ):
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
