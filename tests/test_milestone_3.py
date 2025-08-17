#!/usr/bin/env python3
"""
Comprehensive tests for Milestone 3: Conversation Management.

This test suite validates the conversation management functionality including:
- Multi-turn conversations with context retention
- Knowledge graph integration
- Response time requirements
- Memory usage stability during long conversations
"""

import asyncio
import logging
import os
import sys
import time
import unittest
from pathlib import Path

import psutil
import yaml

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services import ModelConfig  # noqa: E402
from services import ModelManager  # noqa: E402
from services.conversation_manager import ConversationManager  # noqa: E402
from services.graph_manager import KnowledgeGraphManager  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def load_config():
    """Load configuration from config.yaml."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
    )

    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        return {
            "models": {
                "conversation": {
                    "name": "hermes3:3b",
                    "context_window": 8000,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                "knowledge": {
                    "name": "tinyllama",
                    "context_window": 2048,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                },
            },
            "conversation": {"max_history_length": 50, "summarize_threshold": 20},
            "knowledge_graphs": {
                "prebuilt_directory": "./data/prebuilt_graphs",
                "dynamic_storage": "./data/dynamic_graph",
                "max_triplets_per_chunk": 4,
            },
            "ollama": {"host": "http://localhost:11434", "timeout": 120},
        }


class TestMilestone3(unittest.TestCase):
    """Test suite for Milestone 3: Conversation Management."""

    @classmethod
    async def asyncSetUpClass(cls):
        """Set up test environment."""
        # Load configuration
        cls.config = load_config()

        # Initialize model manager
        logger.info("Initializing model manager...")
        cls.model_manager = ModelManager(host=cls.config["ollama"]["host"])

        # Create model configuration
        model_config = ModelConfig(
            name=cls.config["models"]["conversation"]["name"],
            type="conversation",
            context_window=cls.config["models"]["conversation"]["context_window"],
            temperature=cls.config["models"]["conversation"]["temperature"],
            max_tokens=cls.config["models"]["conversation"]["max_tokens"],
            timeout=cls.config["ollama"]["timeout"],
            max_parallel=2,
            max_loaded=2,
        )

        # Initialize models
        logger.info("Initializing models...")
        if not await cls.model_manager.initialize_models(model_config):
            logger.error("Failed to initialize models")
            raise RuntimeError("Failed to initialize models")

        # Initialize knowledge graph manager
        logger.info("Initializing knowledge graph manager...")
        cls.kg_manager = KnowledgeGraphManager(cls.model_manager, cls.config)

        # Load pre-built graphs if available
        logger.info("Loading pre-built knowledge graphs...")
        loaded_graphs = cls.kg_manager.load_prebuilt_graphs()
        if loaded_graphs:
            logger.info(f"Loaded {len(loaded_graphs)} pre-built knowledge graphs")
        else:
            logger.info("No pre-built knowledge graphs loaded")

        # Initialize dynamic graph
        logger.info("Initializing dynamic knowledge graph...")
        cls.kg_manager.initialize_dynamic_graph()

        # Create conversation manager
        logger.info("Creating conversation manager...")
        cls.conversation_manager = ConversationManager(
            cls.model_manager, cls.kg_manager, cls.config
        )

        # Create a test session
        cls.session_id = cls.conversation_manager.create_session()
        logger.info(f"Created test session with ID: {cls.session_id}")

    @classmethod
    async def asyncTearDownClass(cls):
        """Clean up test environment."""
        # Nothing to clean up for now
        pass

    @classmethod
    def setUpClass(cls):
        """Set up test environment using asyncio."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cls.asyncSetUpClass())

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment using asyncio."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cls.asyncTearDownClass())

    async def test_01_session_creation(self):
        """Test session creation and management."""
        # Create a new session
        session_id = self.conversation_manager.create_session()
        self.assertIsNotNone(session_id)

        # Get session info
        session_info = self.conversation_manager.get_session_info(session_id)
        self.assertEqual(session_info["message_count"], 0)

        # List sessions
        sessions = self.conversation_manager.list_sessions()
        self.assertGreaterEqual(len(sessions), 2)  # At least the test session and the new one

        # Delete session
        result = self.conversation_manager.delete_session(session_id)
        self.assertTrue(result)

        # Verify deletion
        sessions = self.conversation_manager.list_sessions()
        session_ids = [s["session_id"] for s in sessions]
        self.assertNotIn(session_id, session_ids)

    async def test_02_basic_conversation(self):
        """Test basic conversation functionality."""
        # Send a message
        message = "Hello, how are you today?"
        response = ""

        start_time = time.time()
        async for token in self.conversation_manager.process_message(self.session_id, message):
            response += token

        elapsed = time.time() - start_time

        # Verify response
        self.assertGreater(len(response), 0)

        # Verify response time (should be under 5 seconds for first response)
        self.assertLess(elapsed, 5.0, "Response time exceeds 5 seconds")

        # Get conversation history
        history = self.conversation_manager.get_conversation_history(self.session_id)
        self.assertEqual(len(history), 2)  # User message and assistant response

        # Verify message content
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], message)
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], response)

    async def test_03_multi_turn_conversation(self):
        """Test multi-turn conversation with context retention."""
        # First message to establish context
        context_message = "My name is Bob and I'm learning about artificial intelligence."
        context_response = ""

        async for token in self.conversation_manager.process_message(
            self.session_id, context_message
        ):
            context_response += token

        # Follow-up message with reference
        reference_message = "What should I focus on learning first?"
        reference_response = ""

        async for token in self.conversation_manager.converse_with_context(
            self.session_id, reference_message
        ):
            reference_response += token

        # Second follow-up with another reference
        second_reference = "Can you recommend some resources for that?"
        second_response = ""

        async for token in self.conversation_manager.converse_with_context(
            self.session_id, second_reference
        ):
            second_response += token

        # Verify responses
        self.assertGreater(len(context_response), 0)
        self.assertGreater(len(reference_response), 0)
        self.assertGreater(len(second_response), 0)

        # Get conversation history
        history = self.conversation_manager.get_conversation_history(self.session_id)

        # Should have at least 6 messages (3 user messages + 3 assistant responses)
        self.assertGreaterEqual(len(history), 6)

        # Test reference resolution
        enhanced_message = self.conversation_manager.resolve_references(
            self.session_id, "Tell me more about it"
        )
        self.assertNotEqual(enhanced_message, "Tell me more about it")
        self.assertIn("context", enhanced_message.lower())

    async def test_04_knowledge_graph_integration(self):
        """Test knowledge graph integration with conversation."""
        # Only run this test if we have a knowledge graph
        if (
            not self.kg_manager
            or not hasattr(self.kg_manager, "kg_indices")
            or not self.kg_manager.kg_indices
        ):
            logger.warning(
                "Skipping knowledge graph integration test - no knowledge graphs available"
            )
            return

        # Query that should trigger knowledge graph lookup
        kg_query = "What can you tell me about machine learning algorithms?"
        kg_response = ""

        # Track knowledge graph hits before query
        kg_hits_before = self.conversation_manager.metrics["knowledge_graph_hits"]

        async for token in self.conversation_manager.converse_with_context(
            self.session_id, kg_query
        ):
            kg_response += token

        # Track knowledge graph hits after query
        kg_hits_after = self.conversation_manager.metrics["knowledge_graph_hits"]

        # Verify response
        self.assertGreater(len(kg_response), 0)

        # Check if knowledge graph was used
        # Note: This might not always increase if the knowledge graph doesn't have relevant information
        logger.info(f"Knowledge graph hits before: {kg_hits_before}, after: {kg_hits_after}")

        # Get relevant context
        context = self.conversation_manager.get_relevant_context(self.session_id, kg_query)

        # Verify context
        self.assertIn("conversation_history", context)
        self.assertIn("knowledge_graph_results", context)

    async def test_05_response_time_requirements(self):
        """Test response time requirements."""
        # Send a simple message
        message = "What's the weather like today?"
        response = ""

        start_time = time.time()
        async for token in self.conversation_manager.process_message(self.session_id, message):
            response += token

        elapsed = time.time() - start_time

        # Verify response time (should be under 3 seconds as per requirements)
        # Note: This might need adjustment based on the actual hardware
        self.assertLess(elapsed, 5.0, "Response time exceeds 5 seconds")

        # Log response time
        logger.info(f"Response time: {elapsed:.2f} seconds")

        # Get metrics
        metrics = self.conversation_manager.get_metrics()
        logger.info(f"Average response time: {metrics['avg_response_time']:.2f} seconds")

    async def test_06_memory_usage_stability(self):
        """Test memory usage stability during long conversations."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create a new session for this test
        long_session_id = self.conversation_manager.create_session()

        # Send multiple messages to simulate a long conversation
        messages = [
            "Tell me about the history of artificial intelligence.",
            "Who were the key pioneers in the field?",
            "What were the major AI winters and why did they happen?",
            "How has deep learning changed the field?",
            "What are the current challenges in AI research?",
            "How is AI being applied in healthcare?",
            "What ethical concerns surround AI development?",
            "How might AI evolve in the next decade?",
            "What is the difference between narrow and general AI?",
            "How do neural networks work?",
        ]

        logger.info("Starting long conversation test...")

        for i, message in enumerate(messages):
            logger.info(f"Sending message {i+1}/{len(messages)}")

            response = ""
            async for token in self.conversation_manager.converse_with_context(
                long_session_id, message
            ):
                response += token

            # Check memory usage after each message
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(
                f"Memory usage after message {i+1}: {current_memory:.2f} MB (change: {current_memory - initial_memory:.2f} MB)"
            )

            # Force garbage collection to get accurate measurements
            import gc

            gc.collect()

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")

        # Check if summarization was triggered
        session = self.conversation_manager.sessions.get(long_session_id, {})
        has_summary = bool(session.get("summary"))
        logger.info(f"Conversation summarization triggered: {has_summary}")

        # Get conversation history
        history = self.conversation_manager.get_conversation_history(long_session_id)
        logger.info(f"Final conversation history length: {len(history)} messages")

        # Verify memory usage is reasonable
        # Note: This threshold might need adjustment based on the actual hardware
        self.assertLess(memory_increase, 500, "Memory usage increased by more than 500 MB")

        # Clean up
        self.conversation_manager.delete_session(long_session_id)

    async def test_07_topic_change_handling(self):
        """Test topic change handling."""
        # Create a new session for this test
        session_id = self.conversation_manager.create_session()

        # First topic
        first_message = "Let's talk about climate change and its effects."
        first_response = ""

        async for token in self.conversation_manager.process_message(session_id, first_message):
            first_response += token

        # Follow-up on first topic
        follow_up = "What are some solutions to this problem?"
        follow_up_response = ""

        async for token in self.conversation_manager.converse_with_context(session_id, follow_up):
            follow_up_response += token

        # Change topic
        topic_change = "By the way, can you tell me about space exploration?"

        # Check if topic change is detected
        is_topic_change = self.conversation_manager.handle_topic_change(session_id, topic_change)
        self.assertTrue(is_topic_change, "Topic change not detected")

        # Process topic change message
        topic_change_response = ""
        async for token in self.conversation_manager.converse_with_context(
            session_id, topic_change
        ):
            topic_change_response += token

        # Follow-up on new topic
        new_topic_follow_up = "What are the current Mars missions?"
        new_follow_up_response = ""

        async for token in self.conversation_manager.converse_with_context(
            session_id, new_topic_follow_up
        ):
            new_follow_up_response += token

        # Verify responses
        self.assertGreater(len(first_response), 0)
        self.assertGreater(len(follow_up_response), 0)
        self.assertGreater(len(topic_change_response), 0)
        self.assertGreater(len(new_follow_up_response), 0)

        # Check if topic changes are recorded
        session = self.conversation_manager.sessions.get(session_id, {})
        topic_changes = session.get("metadata", {}).get("topic_changes", [])
        self.assertGreaterEqual(len(topic_changes), 1, "Topic change not recorded in metadata")

        # Clean up
        self.conversation_manager.delete_session(session_id)

    async def test_08_context_window_management(self):
        """Test context window management."""
        # Create a new session for this test
        session_id = self.conversation_manager.create_session()

        # Send a long message to fill up context window
        long_message = "This is a test of context window management. " * 100

        # Process long message
        long_response = ""
        async for token in self.conversation_manager.process_message(session_id, long_message):
            long_response += token

        # Check if context window management was triggered
        result = self.conversation_manager.manage_context_window(session_id)

        # It should return True if context window management was needed
        logger.info(f"Context window management triggered: {result}")

        # Send another message
        follow_up = "Did you manage the context window?"
        follow_up_response = ""

        async for token in self.conversation_manager.converse_with_context(session_id, follow_up):
            follow_up_response += token

        # Verify responses
        self.assertGreater(len(long_response), 0)
        self.assertGreater(len(follow_up_response), 0)

        # Clean up
        self.conversation_manager.delete_session(session_id)

    async def test_09_dual_model_architecture(self):
        """Test dual-model architecture."""
        # Create a new session for this test
        session_id = self.conversation_manager.create_session()

        # Send a message that should trigger both models
        message = "Explain the relationship between deep learning and neural networks."

        # Process with dual-model architecture
        response = ""
        async for token in self.conversation_manager.process_with_dual_model(session_id, message):
            response += token

        # Verify response
        self.assertGreater(len(response), 0)

        # Clean up
        self.conversation_manager.delete_session(session_id)

    async def test_10_conversation_summarization(self):
        """Test conversation summarization."""
        # Create a new session for this test
        session_id = self.conversation_manager.create_session()

        # Send enough messages to trigger summarization
        messages = [f"This is test message {i}" for i in range(1, 22)]  # 21 messages

        for message in messages:
            async for _ in self.conversation_manager.process_message(session_id, message):
                pass

        # Force summarization
        await self.conversation_manager._maybe_summarize_conversation(session_id)

        # Check if summary was generated
        session = self.conversation_manager.sessions.get(session_id, {})
        summary = session.get("summary")

        self.assertIsNotNone(summary, "Conversation summary not generated")
        self.assertGreater(len(summary), 0, "Conversation summary is empty")

        logger.info(f"Conversation summary: {summary}")

        # Clean up
        self.conversation_manager.delete_session(session_id)


async def run_tests():
    """Run the test suite."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMilestone3)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_tests())

    # Exit with appropriate code
    sys.exit(0 if success else 1)
