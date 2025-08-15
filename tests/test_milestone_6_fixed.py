"""
Comprehensive system testing for Milestone 6.

This module provides comprehensive testing for the complete integrated system,
verifying all components work together correctly under various conditions.
"""

import asyncio
import logging
import os
import sys
import unittest

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import services
from services.conversation_manager import ConversationManager
from services.graph_manager import KnowledgeGraphManager
from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager
from services.system_monitor import SystemMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/test_milestone_6.log", mode="w")],
)
logger = logging.getLogger(__name__)


class TestMilestone6(unittest.IsolatedAsyncioTestCase):
    """Test cases for Milestone 6: System Integration and Final Testing."""

    async def asyncSetUp(self):
        """Set up test environment."""
        # Load configuration
        import yaml

        try:
            with open("config.yaml") as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}

        # Initialize services
        try:
            # Initialize model manager
            logger.info("Initializing ModelManager...")
            ollama_config = self.config.get("ollama", {})
            host = ollama_config.get("host", "http://localhost:11434")
            self.model_manager = ModelManager(host=host)

            # Create model config
            model_config = ModelConfig(
                name=self.config.get("models", {})
                .get("conversation", {})
                .get("name", "hermes3:3b"),
                type="conversation",
                context_window=self.config.get("models", {})
                .get("conversation", {})
                .get("context_window", 8000),
                temperature=self.config.get("models", {})
                .get("conversation", {})
                .get("temperature", 0.7),
                max_tokens=self.config.get("models", {})
                .get("conversation", {})
                .get("max_tokens", 2048),
                timeout=ollama_config.get("timeout", 120),
                max_parallel=ollama_config.get("max_parallel", 2),
                max_loaded=ollama_config.get("max_loaded_models", 2),
            )

            # Initialize models
            await self.model_manager.initialize_models(model_config)

            # Initialize knowledge graph manager
            logger.info("Initializing KnowledgeGraphManager...")
            self.kg_manager = KnowledgeGraphManager(
                model_manager=self.model_manager, config=self.config
            )

            # Load pre-built knowledge graphs
            kg_dir = self.config.get("knowledge_graphs", {}).get(
                "prebuilt_directory", "./data/prebuilt_graphs"
            )
            self.kg_manager.load_prebuilt_graphs(kg_dir)

            # Initialize conversation manager
            logger.info("Initializing ConversationManager...")
            self.conversation_manager = ConversationManager(
                model_manager=self.model_manager, kg_manager=self.kg_manager, config=self.config
            )

            # Initialize system monitor
            logger.info("Initializing SystemMonitor...")
            self.system_monitor = SystemMonitor(config=self.config)
            await self.system_monitor.start_monitoring()

            logger.info("All services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            raise

    async def asyncTearDown(self):
        """Clean up after tests."""
        # Stop system monitoring
        if hasattr(self, "system_monitor"):
            await self.system_monitor.stop_monitoring()

        # Additional cleanup if needed
        logger.info("Test cleanup completed")

    async def test_01_service_initialization(self):
        """Test that all services initialize correctly."""
        # Check that all services are initialized
        self.assertIsNotNone(self.model_manager, "Model manager should be initialized")
        self.assertIsNotNone(self.kg_manager, "Knowledge graph manager should be initialized")
        self.assertIsNotNone(
            self.conversation_manager, "Conversation manager should be initialized"
        )
        self.assertIsNotNone(self.system_monitor, "System monitor should be initialized")

        # Check model status
        model_status = await self.model_manager.get_model_status()
        self.assertIn(
            "conversation_model", model_status, "Model status should include conversation model"
        )
        self.assertIn(
            "knowledge_model", model_status, "Model status should include knowledge model"
        )

        # Check system monitor
        resource_usage = self.system_monitor.get_resource_usage()
        self.assertIn("cpu_percent", resource_usage, "Resource usage should include CPU percent")
        self.assertIn(
            "memory_percent", resource_usage, "Resource usage should include memory percent"
        )

    async def test_02_conversation_flow(self):
        """Test end-to-end conversation flow."""
        # Create a session
        session_id = self.conversation_manager.create_session()
        self.assertIsNotNone(session_id, "Session ID should not be None")

        # Send a message and get response
        message = "Hello, can you tell me about knowledge graphs?"
        response_text = ""
        async for token in self.conversation_manager.converse_with_context(session_id, message):
            response_text += token

        # Check response
        self.assertGreater(len(response_text), 0, "Response should not be empty")
        logger.info(f"Response: {response_text[:100]}...")

        # Check that the message was added to the session
        session = self.conversation_manager.sessions[session_id]
        self.assertEqual(len(session["messages"]), 2, "Session should have 2 messages")
        self.assertEqual(
            session["messages"][0]["role"], "user", "First message should be from user"
        )
        self.assertEqual(
            session["messages"][1]["role"], "assistant", "Second message should be from assistant"
        )


def run_tests():
    """Run the tests."""
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMilestone6)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success/failure
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
