"""
Comprehensive tests for Knowledge Graph milestone.

This module contains tests to verify all graph operations work correctly,
test performance under various graph sizes, validate memory usage stays
within limits, and ensure graph updates don't block other operations.
"""

import asyncio
import os
import sys
import time
import unittest

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import logging
from pathlib import Path

import psutil
import yaml

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services import KnowledgeGraphManager  # noqa: E402
from services.model_mgr import ModelConfig  # noqa: E402
from services.model_mgr import ModelManager  # noqa: E402

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


class TestMilestone2(unittest.TestCase):
    """Test cases for Knowledge Graph milestone."""

    @classmethod
    async def setUpClass(cls):
        """Set up test environment."""
        cls.config = load_config()

        # Initialize model manager
        cls.model_manager = ModelManager(
            host=cls.config.get("ollama", {}).get("host", "http://localhost:11434")
        )

        # Get model configuration
        model_config = ModelConfig(
            name=cls.config.get("models", {}).get("conversation", {}).get("name", "hermes3:3b"),
            type="conversation",
            context_window=cls.config.get("models", {})
            .get("conversation", {})
            .get("context_window", 8000),
            temperature=cls.config.get("models", {})
            .get("conversation", {})
            .get("temperature", 0.7),
            max_tokens=cls.config.get("models", {}).get("conversation", {}).get("max_tokens", 2048),
            timeout=cls.config.get("ollama", {}).get("timeout", 120),
            max_parallel=cls.config.get("ollama", {}).get("max_parallel", 2),
            max_loaded=cls.config.get("ollama", {}).get("max_loaded", 2),
        )

        # Initialize model manager
        await cls.model_manager.initialize_models(model_config)

        # Initialize graph manager
        cls.graph_manager = KnowledgeGraphManager(cls.model_manager, cls.config)

        # Create test data directory
        cls.test_data_dir = Path(__file__).parent / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)

        # Create test graphs directory
        cls.test_graphs_dir = cls.test_data_dir / "test_graphs"
        cls.test_graphs_dir.mkdir(exist_ok=True)

        # Create test documents
        cls.create_test_documents()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up test data
        import shutil

        if cls.test_data_dir.exists():
            shutil.rmtree(cls.test_data_dir)

    @classmethod
    def create_test_documents(cls):
        """Create test documents for graph creation."""
        # Create small graph documents
        small_graph_dir = cls.test_graphs_dir / "small_graph"
        small_graph_dir.mkdir(exist_ok=True)

        with open(small_graph_dir / "small.txt", "w") as f:
            f.write(
                """
            Small Knowledge Graph Test Document

            Python is a programming language created by Guido van Rossum.
            FastAPI is a web framework for building APIs with Python.
            Uvicorn is an ASGI server that can run FastAPI applications.
            """
            )

        # Create medium graph documents
        medium_graph_dir = cls.test_graphs_dir / "medium_graph"
        medium_graph_dir.mkdir(exist_ok=True)

        with open(medium_graph_dir / "medium.txt", "w") as f:
            f.write(
                """
            Medium Knowledge Graph Test Document

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

        # Create large graph documents
        large_graph_dir = cls.test_graphs_dir / "large_graph"
        large_graph_dir.mkdir(exist_ok=True)

        with open(large_graph_dir / "large.txt", "w") as f:
            f.write(
                """
            Large Knowledge Graph Test Document

            Albert Einstein was a physicist who developed the theory of relativity.
            Marie Curie was a physicist and chemist who conducted research on radioactivity.
            Isaac Newton developed the laws of motion and universal gravitation.
            Galileo Galilei made significant contributions to astronomy and physics.
            Nikola Tesla was an inventor who contributed to the design of the modern AC electricity supply system.
            Thomas Edison was an inventor who developed the phonograph and the electric light bulb.
            Ada Lovelace was a mathematician who wrote the first algorithm for a computing machine.
            Alan Turing was a mathematician and computer scientist who formalized the concepts of algorithm and computation.
            Grace Hopper was a computer scientist who developed the first compiler for a programming language.
            Richard Feynman was a physicist who worked on quantum mechanics and particle physics.

            The theory of relativity was published by Einstein in 1915.
            Radioactivity was discovered by Henri Becquerel in 1896.
            Newton's laws of motion were published in Principia Mathematica in 1687.
            Galileo's observations of Jupiter's moons were published in 1610.
            Tesla's AC motor design was patented in 1888.
            Edison's phonograph was patented in 1878.
            Lovelace's algorithm was published in 1843.
            Turing's paper on computability was published in 1936.
            Hopper's compiler was developed in 1952.
            Feynman's diagrams were introduced in 1948.

            Physics is the study of matter, energy, and the interactions between them.
            Chemistry is the study of substances, their properties, and reactions.
            Mathematics is the study of numbers, quantities, and shapes.
            Astronomy is the study of celestial objects and phenomena.
            Computer Science is the study of computation and information processing.
            Engineering is the application of scientific knowledge to design and build systems.

            The Nobel Prize in Physics was awarded to Einstein in 1921.
            The Nobel Prize in Physics and Chemistry was awarded to Marie Curie in 1903 and 1911.
            The Fields Medal is considered the highest honor in mathematics.
            The Turing Award is considered the highest honor in computer science.
            """
            )

    async def test_graph_operations(self):
        """Test that all graph operations work correctly."""
        # Test creating a graph
        graph_id = self.graph_manager.create_graph_from_documents(
            str(self.test_graphs_dir / "small_graph"), "test_small_graph"
        )
        self.assertIsNotNone(graph_id)
        self.assertEqual(graph_id, "test_small_graph")

        # Test loading pre-built graphs
        loaded_graphs = self.graph_manager.load_prebuilt_graphs(str(self.test_graphs_dir))
        self.assertGreater(len(loaded_graphs), 0)

        # Test initializing dynamic graph
        success = self.graph_manager.initialize_dynamic_graph()
        self.assertTrue(success)

        # Test updating dynamic graph
        conversation_text = "Python is a programming language created by Guido van Rossum."
        success = await self.graph_manager.update_dynamic_graph(conversation_text)
        self.assertTrue(success)

        # Wait for background processing to complete
        await asyncio.sleep(5)

        # Test querying graphs
        result = self.graph_manager.query_graphs("Who created Python?")
        self.assertIsNotNone(result)
        self.assertIn("response", result)
        self.assertIn("Guido", result["response"])

        # Test graph traversal
        paths = self.graph_manager.get_graph_traversal("Python", "programming language", max_hops=2)
        self.assertIsInstance(paths, list)

        # Test visualization
        html = self.graph_manager.visualize_graph(graph_id)
        self.assertIsInstance(html, str)
        self.assertIn("<!DOCTYPE html>", html)

        # Test adding a new graph
        new_graph_id = self.graph_manager.add_new_graph(
            str(self.test_graphs_dir / "medium_graph"), "test_medium_graph"
        )
        self.assertIsNotNone(new_graph_id)
        self.assertEqual(new_graph_id, "test_medium_graph")

        # Test getting graph statistics
        stats = self.graph_manager.get_graph_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("graphs", stats)
        self.assertIn("total_graphs", stats)
        self.assertIn("total_nodes", stats)
        self.assertIn("total_edges", stats)

    async def test_performance_under_various_sizes(self):
        """Test performance under various graph sizes."""
        # Create graphs of different sizes
        small_graph_id = self.graph_manager.create_graph_from_documents(
            str(self.test_graphs_dir / "small_graph"), "perf_small_graph"
        )
        medium_graph_id = self.graph_manager.create_graph_from_documents(
            str(self.test_graphs_dir / "medium_graph"), "perf_medium_graph"
        )
        large_graph_id = self.graph_manager.create_graph_from_documents(
            str(self.test_graphs_dir / "large_graph"), "perf_large_graph"
        )

        # Test query performance on small graph
        start_time = time.time()
        self.graph_manager.query_graphs("What is Python?")
        small_query_time = time.time() - start_time

        # Test query performance on medium graph
        start_time = time.time()
        self.graph_manager.query_graphs("What is the solar system?")
        medium_query_time = time.time() - start_time

        # Test query performance on large graph
        start_time = time.time()
        self.graph_manager.query_graphs("Who was Albert Einstein?")
        large_query_time = time.time() - start_time

        # Log performance results
        logger.info(f"Small graph query time: {small_query_time:.4f} seconds")
        logger.info(f"Medium graph query time: {medium_query_time:.4f} seconds")
        logger.info(f"Large graph query time: {large_query_time:.4f} seconds")

        # Verify that query times are reasonable
        # These thresholds may need adjustment based on the actual hardware
        self.assertLess(small_query_time, 5.0)
        self.assertLess(medium_query_time, 10.0)
        self.assertLess(large_query_time, 15.0)

        # Test visualization performance
        start_time = time.time()
        self.graph_manager.visualize_graph(small_graph_id)
        small_vis_time = time.time() - start_time

        start_time = time.time()
        self.graph_manager.visualize_graph(medium_graph_id)
        medium_vis_time = time.time() - start_time

        start_time = time.time()
        self.graph_manager.visualize_graph(large_graph_id)
        large_vis_time = time.time() - start_time

        # Log visualization performance results
        logger.info(f"Small graph visualization time: {small_vis_time:.4f} seconds")
        logger.info(f"Medium graph visualization time: {medium_vis_time:.4f} seconds")
        logger.info(f"Large graph visualization time: {large_vis_time:.4f} seconds")

        # Verify that visualization times are reasonable
        self.assertLess(small_vis_time, 1.0)
        self.assertLess(medium_vis_time, 2.0)
        self.assertLess(large_vis_time, 5.0)

    async def test_memory_usage(self):
        """Test that memory usage stays within limits."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Create a large graph
        self.graph_manager.create_graph_from_documents(
            str(self.test_graphs_dir / "large_graph"), "memory_large_graph"
        )

        # Get memory usage after creating large graph
        after_creation_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = after_creation_memory - initial_memory

        # Log memory usage
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        logger.info(f"Memory usage after creating large graph: {after_creation_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")

        # Verify that memory increase is reasonable
        # This threshold may need adjustment based on the actual hardware
        self.assertLess(memory_increase, 500)  # Less than 500 MB increase

        # Get system memory limit from config
        max_memory_gb = self.config.get("system", {}).get("max_memory_gb", 12)
        max_memory_mb = max_memory_gb * 1024

        # Verify that total memory usage is within limits
        self.assertLess(
            after_creation_memory, max_memory_mb * 0.5
        )  # Using less than 50% of max memory

        # Test memory usage during multiple operations
        # Create dynamic graph
        self.graph_manager.initialize_dynamic_graph()

        # Update dynamic graph multiple times
        for i in range(5):
            conversation_text = f"Test conversation {i}: Python is a programming language. Albert Einstein was a physicist."
            await self.graph_manager.update_dynamic_graph(conversation_text)

        # Wait for background processing to complete
        await asyncio.sleep(5)

        # Get memory usage after multiple operations
        after_operations_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Log memory usage
        logger.info(f"Memory usage after multiple operations: {after_operations_memory:.2f} MB")

        # Verify that memory usage is still within limits
        self.assertLess(
            after_operations_memory, max_memory_mb * 0.7
        )  # Using less than 70% of max memory

    async def test_non_blocking_updates(self):
        """Test that graph updates don't block other operations."""
        # Initialize dynamic graph
        self.graph_manager.initialize_dynamic_graph()

        # Start a long update operation
        long_text = """
        This is a long conversation text that will trigger a background update.
        Albert Einstein was a physicist who developed the theory of relativity.
        Marie Curie was a physicist and chemist who conducted research on radioactivity.
        Isaac Newton developed the laws of motion and universal gravitation.
        Galileo Galilei made significant contributions to astronomy and physics.
        Nikola Tesla was an inventor who contributed to the design of the modern AC electricity supply system.
        Thomas Edison was an inventor who developed the phonograph and the electric light bulb.
        Ada Lovelace was a mathematician who wrote the first algorithm for a computing machine.
        Alan Turing was a mathematician and computer scientist who formalized the concepts of algorithm and computation.
        Grace Hopper was a computer scientist who developed the first compiler for a programming language.
        Richard Feynman was a physicist who worked on quantum mechanics and particle physics.
        """

        # Start update
        start_time = time.time()
        update_task = asyncio.create_task(self.graph_manager.update_dynamic_graph(long_text))

        # Immediately perform a query operation
        query_start_time = time.time()
        self.graph_manager.query_graphs("What is Python?")
        query_time = time.time() - query_start_time

        # Log timing
        logger.info(f"Query time during background update: {query_time:.4f} seconds")

        # Verify that query was not blocked by the update
        self.assertLess(query_time, 5.0)  # Query should complete quickly

        # Wait for update to complete
        await update_task
        update_time = time.time() - start_time

        # Log timing
        logger.info(f"Total update time: {update_time:.4f} seconds")

        # Verify that update completed
        self.assertTrue(update_task.done())

        # Wait for background processing to complete
        await asyncio.sleep(5)

        # Perform another query to verify graph was updated
        result = self.graph_manager.query_graphs("Who was Albert Einstein?")
        self.assertIn("Einstein", result["response"])


def run_tests():
    """Run the tests."""
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestMilestone2("test_graph_operations"))
    suite.addTest(TestMilestone2("test_performance_under_various_sizes"))
    suite.addTest(TestMilestone2("test_memory_usage"))
    suite.addTest(TestMilestone2("test_non_blocking_updates"))

    # Run tests
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    # Set up asyncio event loop
    loop = asyncio.get_event_loop()

    # Set up test class
    test_class = TestMilestone2()
    loop.run_until_complete(test_class.setUpClass())

    # Run tests
    run_tests()

    # Clean up
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
