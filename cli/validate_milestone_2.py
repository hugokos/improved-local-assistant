#!/usr/bin/env python
"""
Interactive validation CLI for Knowledge Graph milestone.

This script provides an interactive interface for testing and validating
the knowledge graph functionality, including graph creation, loading,
querying, visualization, and performance benchmarking.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import psutil
import yaml

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services import KnowledgeGraphManager
from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager

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

    return model_manager, graph_manager


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_result(success, message):
    """Print a formatted result."""
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")


async def interactive_graph_creation(graph_manager):
    """Interactive graph creation testing."""
    print_header("Knowledge Graph Creation")

    # Ask for document path
    print("Enter the path to a directory containing documents for graph creation.")
    print("Leave blank to use the default test documents.")
    docs_path = input("Document path: ").strip()

    if not docs_path:
        docs_path = Path(__file__).parent.parent / "data" / "test_docs"
        print(f"Using default path: {docs_path}")

    # Check if path exists
    if not os.path.exists(docs_path):
        print(f"Path {docs_path} does not exist. Creating directory...")
        try:
            os.makedirs(docs_path, exist_ok=True)

            # Create a sample document
            sample_doc = Path(docs_path) / "sample.txt"
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
            print(f"Created sample document at {sample_doc}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return

    # Ask for graph ID
    print("\nEnter an ID for the graph (leave blank for auto-generated ID):")
    graph_id = input("Graph ID: ").strip()

    # Create graph
    print("\nCreating knowledge graph...")
    start_time = time.time()
    created_graph_id = graph_manager.create_graph_from_documents(
        docs_path, graph_id if graph_id else None
    )
    creation_time = time.time() - start_time

    if created_graph_id:
        print_result(True, f"Graph created with ID: {created_graph_id}")
        print(f"Creation time: {creation_time:.2f} seconds")

        # Get graph statistics
        stats = graph_manager.get_graph_statistics()
        if "graphs" in stats and created_graph_id in stats["graphs"]:
            graph_stats = stats["graphs"][created_graph_id]
            print("\nGraph statistics:")
            print(f"  Nodes: {graph_stats.get('nodes', 'N/A')}")
            print(f"  Edges: {graph_stats.get('edges', 'N/A')}")
            print(f"  Density: {graph_stats.get('density', 'N/A')}")

        # Ask if user wants to visualize the graph
        print("\nDo you want to visualize the graph? (y/n)")
        if input().lower().startswith("y"):
            print("Generating visualization...")
            html = graph_manager.visualize_graph(created_graph_id)

            # Save visualization to file
            vis_path = (
                Path(__file__).parent.parent
                / "data"
                / "visualizations"
                / f"{created_graph_id}_visualization.html"
            )
            os.makedirs(vis_path.parent, exist_ok=True)
            with open(vis_path, "w") as f:
                f.write(html)
            print(f"Visualization saved to {vis_path}")

            # Try to open the visualization in a browser
            try:
                import webbrowser

                webbrowser.open(f"file://{vis_path.absolute()}")
                print("Opened visualization in browser")
            except Exception as e:
                print(f"Could not open browser: {e}")
                print(f"Please open {vis_path} manually")
    else:
        print_result(False, "Failed to create graph")


async def interactive_prebuilt_loading(graph_manager):
    """Interactive pre-built graph loading testing."""
    print_header("Pre-built Knowledge Graph Loading")

    # Ask for pre-built graphs path
    print("Enter the path to a directory containing pre-built knowledge graphs.")
    print("Leave blank to use the default pre-built graphs directory.")
    prebuilt_path = input("Pre-built graphs path: ").strip()

    if not prebuilt_path:
        prebuilt_path = Path(__file__).parent.parent / "data" / "prebuilt_graphs"
        print(f"Using default path: {prebuilt_path}")

    # Check if path exists
    if not os.path.exists(prebuilt_path):
        print(f"Path {prebuilt_path} does not exist. Creating directory...")
        try:
            os.makedirs(prebuilt_path, exist_ok=True)

            # Create a sample graph directory
            sample_graph_dir = Path(prebuilt_path) / "sample_graph"
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
            print(f"Created sample graph at {sample_graph_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return

    # Load pre-built graphs
    print("\nLoading pre-built graphs...")
    start_time = time.time()
    loaded_graphs = graph_manager.load_prebuilt_graphs(prebuilt_path)
    loading_time = time.time() - start_time

    if loaded_graphs:
        print_result(True, f"Loaded {len(loaded_graphs)} graphs: {loaded_graphs}")
        print(f"Loading time: {loading_time:.2f} seconds")

        # Get graph statistics
        stats = graph_manager.get_graph_statistics()
        print("\nGraph statistics:")
        for graph_id in loaded_graphs:
            if graph_id in stats["graphs"]:
                graph_stats = stats["graphs"][graph_id]
                print(f"\n  Graph: {graph_id}")
                print(f"    Nodes: {graph_stats.get('nodes', 'N/A')}")
                print(f"    Edges: {graph_stats.get('edges', 'N/A')}")
                print(f"    Density: {graph_stats.get('density', 'N/A')}")

        # Ask if user wants to test hot-loading
        print("\nDo you want to test hot-loading a new graph? (y/n)")
        if input().lower().startswith("y"):
            # Ask for new graph path
            print("\nEnter the path to a directory for the new graph.")
            print("Leave blank to create a new sample graph.")
            new_graph_path = input("New graph path: ").strip()

            if not new_graph_path:
                new_graph_path = Path(prebuilt_path) / "new_graph"
                os.makedirs(new_graph_path, exist_ok=True)

                # Create a sample document
                new_doc = Path(new_graph_path) / "new.txt"
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
                print(f"Created new graph at {new_graph_path}")

            # Ask for graph ID
            print("\nEnter an ID for the new graph (leave blank for auto-generated ID):")
            new_graph_id = input("Graph ID: ").strip()

            # Add the new graph
            print("\nHot-loading new graph...")
            start_time = time.time()
            added_graph_id = graph_manager.add_new_graph(
                new_graph_path, new_graph_id if new_graph_id else None
            )
            hot_loading_time = time.time() - start_time

            if added_graph_id:
                print_result(True, f"Hot-loaded graph with ID: {added_graph_id}")
                print(f"Hot-loading time: {hot_loading_time:.2f} seconds")

                # Get updated graph statistics
                stats = graph_manager.get_graph_statistics()
                if "graphs" in stats and added_graph_id in stats["graphs"]:
                    graph_stats = stats["graphs"][added_graph_id]
                    print("\nNew graph statistics:")
                    print(f"  Nodes: {graph_stats.get('nodes', 'N/A')}")
                    print(f"  Edges: {graph_stats.get('edges', 'N/A')}")
                    print(f"  Density: {graph_stats.get('density', 'N/A')}")
            else:
                print_result(False, "Failed to hot-load new graph")
    else:
        print_result(False, "Failed to load pre-built graphs")


async def interactive_dynamic_updates(graph_manager, model_manager):
    """Interactive dynamic knowledge graph updates testing."""
    print_header("Dynamic Knowledge Graph Updates")

    # Initialize dynamic graph
    print("Initializing dynamic knowledge graph...")
    success = graph_manager.initialize_dynamic_graph()

    if success:
        print_result(True, "Dynamic graph initialized")

        # Ask for conversation text
        print("\nEnter some text to extract entities from (or leave blank for sample text):")
        conversation_text = input().strip()

        if not conversation_text:
            conversation_text = """
            The Raspberry Pi 5 is a single-board computer with 8GB of RAM.
            It uses an ARM processor and can run Linux operating systems.
            The Raspberry Pi Foundation created it for educational purposes.
            """
            print(f"Using sample text:\n{conversation_text}")

        # Update dynamic graph
        print("\nUpdating dynamic graph...")
        start_time = time.time()
        update_success = await graph_manager.update_dynamic_graph(conversation_text)
        update_time = time.time() - start_time

        if update_success:
            print_result(True, "Dynamic graph updated")
            print(f"Update time: {update_time:.2f} seconds")

            # Wait for background processing to complete
            print("Waiting for background processing to complete...")
            await asyncio.sleep(5)

            # Get graph statistics
            stats = graph_manager.get_graph_statistics()
            if "graphs" in stats and "dynamic" in stats["graphs"]:
                graph_stats = stats["graphs"]["dynamic"]
                print("\nDynamic graph statistics:")
                print(f"  Nodes: {graph_stats.get('nodes', 'N/A')}")
                print(f"  Edges: {graph_stats.get('edges', 'N/A')}")
                print(f"  Density: {graph_stats.get('density', 'N/A')}")

            # Ask if user wants to visualize the graph
            print("\nDo you want to visualize the dynamic graph? (y/n)")
            if input().lower().startswith("y"):
                print("Generating visualization...")
                html = graph_manager.visualize_graph()

                # Save visualization to file
                vis_path = (
                    Path(__file__).parent.parent
                    / "data"
                    / "visualizations"
                    / "dynamic_visualization.html"
                )
                os.makedirs(vis_path.parent, exist_ok=True)
                with open(vis_path, "w") as f:
                    f.write(html)
                print(f"Visualization saved to {vis_path}")

                # Try to open the visualization in a browser
                try:
                    import webbrowser

                    webbrowser.open(f"file://{vis_path.absolute()}")
                    print("Opened visualization in browser")
                except Exception as e:
                    print(f"Could not open browser: {e}")
                    print(f"Please open {vis_path} manually")

            # Ask if user wants to test multiple updates
            print("\nDo you want to test multiple graph updates? (y/n)")
            if input().lower().startswith("y"):
                # Ask for number of updates
                print("\nEnter the number of updates to perform:")
                try:
                    num_updates = int(input().strip())
                except ValueError:
                    num_updates = 3
                    print(f"Using default: {num_updates} updates")

                # Perform multiple updates
                print(f"\nPerforming {num_updates} updates...")
                for i in range(num_updates):
                    update_text = f"""
                    Update {i+1}: Python is a programming language that supports multiple paradigms.
                    It was created by Guido van Rossum and first released in 1991.
                    Python emphasizes code readability and simplicity.
                    """
                    print(f"\nUpdate {i+1}:")
                    print(update_text)

                    start_time = time.time()
                    await graph_manager.update_dynamic_graph(update_text)
                    update_time = time.time() - start_time
                    print(f"Update time: {update_time:.2f} seconds")

                # Wait for background processing to complete
                print("\nWaiting for background processing to complete...")
                await asyncio.sleep(5)

                # Get updated graph statistics
                stats = graph_manager.get_graph_statistics()
                if "graphs" in stats and "dynamic" in stats["graphs"]:
                    graph_stats = stats["graphs"]["dynamic"]
                    print("\nUpdated dynamic graph statistics:")
                    print(f"  Nodes: {graph_stats.get('nodes', 'N/A')}")
                    print(f"  Edges: {graph_stats.get('edges', 'N/A')}")
                    print(f"  Density: {graph_stats.get('density', 'N/A')}")

                # Ask if user wants to visualize the updated graph
                print("\nDo you want to visualize the updated dynamic graph? (y/n)")
                if input().lower().startswith("y"):
                    print("Generating visualization...")
                    html = graph_manager.visualize_graph()

                    # Save visualization to file
                    vis_path = (
                        Path(__file__).parent.parent
                        / "data"
                        / "visualizations"
                        / "dynamic_updated_visualization.html"
                    )
                    with open(vis_path, "w") as f:
                        f.write(html)
                    print(f"Visualization saved to {vis_path}")

                    # Try to open the visualization in a browser
                    try:
                        import webbrowser

                        webbrowser.open(f"file://{vis_path.absolute()}")
                        print("Opened visualization in browser")
                    except Exception as e:
                        print(f"Could not open browser: {e}")
                        print(f"Please open {vis_path} manually")
        else:
            print_result(False, "Failed to update dynamic graph")
    else:
        print_result(False, "Failed to initialize dynamic graph")


async def interactive_queries(graph_manager):
    """Interactive graph-based retrieval and query testing."""
    print_header("Graph-based Retrieval and Query System")

    # Check if we have any graphs
    stats = graph_manager.get_graph_statistics()
    if stats["total_graphs"] == 0:
        print("No graphs available. Please create or load graphs first.")
        return

    # Show available graphs
    print("Available graphs:")
    for graph_id in stats["graphs"]:
        graph_stats = stats["graphs"][graph_id]
        print(
            f"  {graph_id}: {graph_stats.get('nodes', 'N/A')} nodes, {graph_stats.get('edges', 'N/A')} edges"
        )

    # Interactive query loop
    while True:
        print("\nEnter a query (or 'exit' to return to main menu):")
        query = input().strip()

        if query.lower() == "exit":
            break

        # Process query
        print("\nProcessing query...")
        start_time = time.time()
        result = graph_manager.query_graphs(query)
        query_time = time.time() - start_time

        # Display results
        print(f"\nQuery time: {query_time:.2f} seconds")
        print("\nResponse:")
        print(f"{result['response']}")

        # Display metadata
        print(f"\nSource graphs: {result['metadata'].get('graph_sources', [])}")

        # Ask if user wants to try graph traversal
        print("\nDo you want to try graph traversal? (y/n)")
        if input().lower().startswith("y"):
            print("\nEnter source node:")
            source = input().strip()

            print("Enter target node:")
            target = input().strip()

            print("Enter maximum hops (default: 3):")
            try:
                max_hops = int(input().strip())
            except ValueError:
                max_hops = 3
                print(f"Using default: {max_hops} hops")

            # Perform graph traversal
            print("\nPerforming graph traversal...")
            start_time = time.time()
            paths = graph_manager.get_graph_traversal(source, target, max_hops)
            traversal_time = time.time() - start_time

            # Display results
            print(f"\nTraversal time: {traversal_time:.2f} seconds")
            print(f"\nPaths from '{source}' to '{target}' (max {max_hops} hops):")

            if paths:
                for i, path in enumerate(paths):
                    print(f"  Path {i+1}: {' -> '.join(path)}")
            else:
                print("  No paths found")


async def interactive_visualization(graph_manager):
    """Interactive knowledge graph visualization testing."""
    print_header("Knowledge Graph Visualization")

    # Check if we have any graphs
    stats = graph_manager.get_graph_statistics()
    if stats["total_graphs"] == 0:
        print("No graphs available. Please create or load graphs first.")
        return

    # Show available graphs
    print("Available graphs:")
    for graph_id in stats["graphs"]:
        graph_stats = stats["graphs"][graph_id]
        print(
            f"  {graph_id}: {graph_stats.get('nodes', 'N/A')} nodes, {graph_stats.get('edges', 'N/A')} edges"
        )

    # Ask which graph to visualize
    print("\nEnter the ID of the graph to visualize (or 'all' for all graphs):")
    graph_id = input().strip()

    if graph_id.lower() == "all":
        # Create visualizations directory
        vis_dir = Path(__file__).parent.parent / "data" / "visualizations"
        os.makedirs(vis_dir, exist_ok=True)

        # Generate visualizations for all graphs
        print("\nGenerating visualizations for all graphs...")
        for gid in stats["graphs"]:
            print(f"\nVisualizing graph: {gid}")

            # Generate visualization
            start_time = time.time()
            html = graph_manager.visualize_graph(gid if gid != "dynamic" else None)
            vis_time = time.time() - start_time

            # Save visualization to file
            vis_path = vis_dir / f"{gid}_visualization.html"
            with open(vis_path, "w") as f:
                f.write(html)
            print(f"Visualization saved to {vis_path}")
            print(f"Visualization time: {vis_time:.2f} seconds")

        # Generate combined visualization
        print("\nGenerating combined visualization...")

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

        for gid in stats["graphs"]:
            vis_path = f"{gid}_visualization.html"
            combined_html += f"""
            <div class="graph-container">
                <h2>Graph: {gid}</h2>
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
        print(f"Combined visualization saved to {combined_path}")

        # Try to open the combined visualization in a browser
        try:
            import webbrowser

            webbrowser.open(f"file://{combined_path.absolute()}")
            print("Opened combined visualization in browser")
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"Please open {combined_path} manually")

    elif graph_id in stats["graphs"]:
        # Generate visualization for the selected graph
        print(f"\nVisualizing graph: {graph_id}")

        # Generate visualization
        start_time = time.time()
        html = graph_manager.visualize_graph(graph_id if graph_id != "dynamic" else None)
        vis_time = time.time() - start_time

        # Save visualization to file
        vis_dir = Path(__file__).parent.parent / "data" / "visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = vis_dir / f"{graph_id}_visualization.html"
        with open(vis_path, "w") as f:
            f.write(html)
        print(f"Visualization saved to {vis_path}")
        print(f"Visualization time: {vis_time:.2f} seconds")

        # Try to open the visualization in a browser
        try:
            import webbrowser

            webbrowser.open(f"file://{vis_path.absolute()}")
            print("Opened visualization in browser")
        except Exception as e:
            print(f"Could not open browser: {e}")
            print(f"Please open {vis_path} manually")
    else:
        print(f"Graph '{graph_id}' not found")


async def interactive_performance_testing(graph_manager):
    """Interactive performance testing."""
    print_header("Performance Testing")

    # Check if we have any graphs
    stats = graph_manager.get_graph_statistics()
    if stats["total_graphs"] == 0:
        print("No graphs available. Please create or load graphs first.")
        return

    # Show available graphs
    print("Available graphs:")
    for graph_id in stats["graphs"]:
        graph_stats = stats["graphs"][graph_id]
        print(
            f"  {graph_id}: {graph_stats.get('nodes', 'N/A')} nodes, {graph_stats.get('edges', 'N/A')} edges"
        )

    # Memory usage monitoring
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)  # MB

    print(f"\nCurrent memory usage: {memory_mb:.2f} MB")

    # Query performance testing
    print("\nDo you want to run query performance tests? (y/n)")
    if input().lower().startswith("y"):
        # Ask for number of queries
        print("\nEnter the number of queries to run:")
        try:
            num_queries = int(input().strip())
        except ValueError:
            num_queries = 5
            print(f"Using default: {num_queries} queries")

        # Ask for query
        print("\nEnter a query to test (or leave blank for default):")
        query = input().strip()
        if not query:
            query = "What is a knowledge graph?"
            print(f"Using default query: '{query}'")

        # Run query performance tests
        print(f"\nRunning {num_queries} queries...")
        query_times = []

        for i in range(num_queries):
            print(f"\nQuery {i+1}/{num_queries}:")
            start_time = time.time()
            result = graph_manager.query_graphs(query)
            query_time = time.time() - start_time
            query_times.append(query_time)

            print(f"Query time: {query_time:.2f} seconds")
            print(f"Response: {result['response'][:100]}...")

        # Calculate statistics
        avg_time = sum(query_times) / len(query_times)
        min_time = min(query_times)
        max_time = max(query_times)

        print("\nQuery performance statistics:")
        print(f"  Average time: {avg_time:.2f} seconds")
        print(f"  Minimum time: {min_time:.2f} seconds")
        print(f"  Maximum time: {max_time:.2f} seconds")

    # Visualization performance testing
    print("\nDo you want to run visualization performance tests? (y/n)")
    if input().lower().startswith("y"):
        # Ask for number of visualizations
        print("\nEnter the number of visualizations to generate:")
        try:
            num_vis = int(input().strip())
        except ValueError:
            num_vis = 3
            print(f"Using default: {num_vis} visualizations")

        # Ask for graph ID
        print("\nEnter the ID of the graph to visualize (or leave blank for first available):")
        graph_id = input().strip()

        if not graph_id:
            graph_id = next(iter(stats["graphs"]))
            print(f"Using graph: {graph_id}")

        # Run visualization performance tests
        print(f"\nGenerating {num_vis} visualizations...")
        vis_times = []

        for i in range(num_vis):
            print(f"\nVisualization {i+1}/{num_vis}:")
            start_time = time.time()
            html = graph_manager.visualize_graph(graph_id if graph_id != "dynamic" else None)
            vis_time = time.time() - start_time
            vis_times.append(vis_time)

            print(f"Visualization time: {vis_time:.2f} seconds")
            print(f"HTML size: {len(html)} bytes")

        # Calculate statistics
        avg_time = sum(vis_times) / len(vis_times)
        min_time = min(vis_times)
        max_time = max(vis_times)

        print("\nVisualization performance statistics:")
        print(f"  Average time: {avg_time:.2f} seconds")
        print(f"  Minimum time: {min_time:.2f} seconds")
        print(f"  Maximum time: {max_time:.2f} seconds")

    # Memory usage after tests
    memory_mb_after = process.memory_info().rss / (1024 * 1024)  # MB
    memory_diff = memory_mb_after - memory_mb

    print(f"\nMemory usage after tests: {memory_mb_after:.2f} MB")
    print(f"Memory increase: {memory_diff:.2f} MB")


async def interactive_menu(model_manager, graph_manager):
    """Interactive menu for knowledge graph validation."""
    while True:
        print_header("Knowledge Graph Validation")
        print("1. Create Knowledge Graph from Documents")
        print("2. Load Pre-built Knowledge Graphs")
        print("3. Test Dynamic Knowledge Graph Updates")
        print("4. Test Graph-based Retrieval and Queries")
        print("5. Test Knowledge Graph Visualization")
        print("6. Run Performance Tests")
        print("7. View Graph Statistics")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "1":
            await interactive_graph_creation(graph_manager)
        elif choice == "2":
            await interactive_prebuilt_loading(graph_manager)
        elif choice == "3":
            await interactive_dynamic_updates(graph_manager, model_manager)
        elif choice == "4":
            await interactive_queries(graph_manager)
        elif choice == "5":
            await interactive_visualization(graph_manager)
        elif choice == "6":
            await interactive_performance_testing(graph_manager)
        elif choice == "7":
            stats = graph_manager.get_graph_statistics()
            print_header("Graph Statistics")
            print(f"Total graphs: {stats['total_graphs']}")
            print(f"Total nodes: {stats['total_nodes']}")
            print(f"Total edges: {stats['total_edges']}")
            print("\nIndividual graphs:")
            for graph_id, graph_stats in stats["graphs"].items():
                print(f"\n  Graph: {graph_id}")
                print(f"    Nodes: {graph_stats.get('nodes', 'N/A')}")
                print(f"    Edges: {graph_stats.get('edges', 'N/A')}")
                print(f"    Density: {graph_stats.get('density', 'N/A')}")

            print("\nPerformance metrics:")
            print(f"  Graphs loaded: {stats['metrics']['graphs_loaded']}")
            print(f"  Queries processed: {stats['metrics']['queries_processed']}")
            print(f"  Average query time: {stats['metrics']['avg_query_time']:.4f} seconds")
            print(f"  Last query time: {stats['metrics']['last_query_time']:.4f} seconds")

            input("\nPress Enter to continue...")
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please try again.")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Interactive validation CLI for Knowledge Graph milestone"
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    # Initialize managers
    model_manager, graph_manager = await initialize_managers()

    if args.interactive:
        await interactive_menu(model_manager, graph_manager)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
