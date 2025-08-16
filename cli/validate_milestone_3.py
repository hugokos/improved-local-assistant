#!/usr/bin/env python3
"""
Interactive validation CLI for Milestone 3: Conversation Management.

This script provides a comprehensive interactive testing interface for:
- Multi-turn conversation testing with context validation
- Knowledge graph integration verification
- Conversation analysis and performance metrics
"""

import argparse
import asyncio
import logging
import os
import sys
import time

import yaml

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from services import ModelConfig
from services import ModelManager
from services.conversation_manager import ConversationManager
from services.graph_manager import KnowledgeGraphManager

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


class ValidationCLI:
    """Interactive CLI for validating Milestone 3: Conversation Management."""

    def __init__(self):
        """Initialize the validation CLI."""
        self.config = load_config()
        self.model_manager = None
        self.kg_manager = None
        self.conversation_manager = None
        self.current_session_id = None
        self.sessions = {}
        self.debug_mode = False
        self.memory_tracking = False
        self.memory_samples = []
        self.response_times = []
        self.test_results = {
            "basic_conversation": False,
            "multi_turn_conversation": False,
            "reference_resolution": False,
            "knowledge_graph_integration": False,
            "topic_change_handling": False,
            "context_window_management": False,
            "dual_model_architecture": False,
            "conversation_summarization": False,
            "response_time": False,
            "memory_usage": False,
        }

    async def initialize(self):
        """Initialize components."""
        try:
            # Initialize model manager
            logger.info("Initializing model manager...")
            self.model_manager = ModelManager(host=self.config["ollama"]["host"])

            # Create model configuration
            model_config = ModelConfig(
                name=self.config["models"]["conversation"]["name"],
                type="conversation",
                context_window=self.config["models"]["conversation"]["context_window"],
                temperature=self.config["models"]["conversation"]["temperature"],
                max_tokens=self.config["models"]["conversation"]["max_tokens"],
                timeout=self.config["ollama"]["timeout"],
                max_parallel=2,
                max_loaded=2,
            )

            # Initialize models
            logger.info("Initializing models...")
            if not await self.model_manager.initialize_models(model_config):
                logger.error("Failed to initialize models")
                return False

            # Initialize knowledge graph manager
            logger.info("Initializing knowledge graph manager...")
            self.kg_manager = KnowledgeGraphManager(self.model_manager, self.config)

            # Load pre-built graphs if available
            logger.info("Loading pre-built knowledge graphs...")
            loaded_graphs = self.kg_manager.load_prebuilt_graphs()
            if loaded_graphs:
                logger.info(f"Loaded {len(loaded_graphs)} pre-built knowledge graphs")
            else:
                logger.info("No pre-built knowledge graphs loaded")

            # Initialize dynamic graph
            logger.info("Initializing dynamic knowledge graph...")
            self.kg_manager.initialize_dynamic_graph()

            # Create conversation manager
            logger.info("Creating conversation manager...")
            self.conversation_manager = ConversationManager(
                self.model_manager, self.kg_manager, self.config
            )

            # Create a default session
            self.current_session_id = self.conversation_manager.create_session()
            logger.info(f"Created default session with ID: {self.current_session_id}")

            return True

        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def print_header(self):
        """Print the validation CLI header."""
        print("\n" + "=" * 80)
        print("MILESTONE 3 VALIDATION: CONVERSATION MANAGEMENT".center(80))
        print("=" * 80)
        print(
            "\nThis interactive CLI allows you to validate the conversation management functionality."
        )
        print(
            "You can test multi-turn conversations, context handling, knowledge graph integration,"
        )
        print("and performance metrics.")
        print("\nType '/help' to see available commands.")
        print("=" * 80 + "\n")

    def print_help(self):
        """Print help information."""
        print("\nAvailable Commands:")
        print("  /help                - Show this help message")
        print("  /exit                - Exit the validation CLI")
        print("  /test                - Run automated validation tests")
        print("  /status              - Show test status and results")
        print("  /debug <on|off>      - Toggle debug mode")
        print("  /memory <on|off>     - Toggle memory tracking")
        print("  /new                 - Create a new session")
        print("  /sessions            - List all active sessions")
        print("  /switch <id>         - Switch to another session")
        print("  /history             - Show conversation history")
        print("  /info                - Show session information")
        print("  /metrics             - Show performance metrics")
        print("  /analyze             - Analyze conversation context and references")
        print("  /summarize           - Force conversation summarization")
        print("  /kg                  - Show knowledge graph information")
        print("  /dual                - Test dual-model architecture")
        print("  /topic <message>     - Test topic change detection")
        print("  /context             - Test context window management")
        print("  /reference <message> - Test reference resolution")
        print("  /report              - Generate validation report")
        print("\nAny other input will be treated as a message to the assistant.")

    async def run_automated_tests(self):
        """Run automated validation tests."""
        print("\nRunning automated validation tests...")

        # Test 1: Basic conversation
        print("\n[TEST 1/10] Basic conversation...")
        try:
            session_id = self.conversation_manager.create_session()

            message = "Hello, how are you today?"
            response = ""

            start_time = time.time()
            async for token in self.conversation_manager.process_message(session_id, message):
                response += token

            elapsed = time.time() - start_time

            if len(response) > 0:
                print("✓ Received valid response")
                self.test_results["basic_conversation"] = True
            else:
                print("✗ No response received")

            print(f"Response time: {elapsed:.2f} seconds")
            self.response_times.append(elapsed)

            # Clean up
            self.conversation_manager.delete_session(session_id)

        except Exception as e:
            print(f"✗ Test failed: {str(e)}")

        # Test 2: Multi-turn conversation
        print("\n[TEST 2/10] Multi-turn conversation...")
        try:
            session_id = self.conversation_manager.create_session()

            # First message
            message1 = "My name is Alice and I'm a software engineer."
            response1 = ""
            async for token in self.conversation_manager.process_message(session_id, message1):
                response1 += token

            # Second message
            message2 = "I work on machine learning projects."
            response2 = ""
            async for token in self.conversation_manager.process_message(session_id, message2):
                response2 += token

            # Third message
            message3 = "What kind of skills should I develop next?"
            response3 = ""
            async for token in self.conversation_manager.process_message(session_id, message3):
                response3 += token

            # Check history
            history = self.conversation_manager.get_conversation_history(session_id)

            if len(history) == 6:  # 3 user messages + 3 assistant responses
                print("✓ Conversation history maintained correctly")
                self.test_results["multi_turn_conversation"] = True
            else:
                print(f"✗ Conversation history incorrect: {len(history)} messages")

            # Clean up
            self.conversation_manager.delete_session(session_id)

        except Exception as e:
            print(f"✗ Test failed: {str(e)}")

        # Test 3: Reference resolution
        print("\n[TEST 3/10] Reference resolution...")
        try:
            session_id = self.conversation_manager.create_session()

            # First message
            message1 = "The Large Language Model Transformer architecture was developed by Google."
            async for _ in self.conversation_manager.process_message(session_id, message1):
                pass

            # Reference message
            reference_message = "When was it developed?"

            # Resolve references
            enhanced_message = self.conversation_manager.resolve_references(
                session_id, reference_message
            )

            if enhanced_message != reference_message and "context" in enhanced_message.lower():
                print("✓ Reference resolution working correctly")
                print(f"Original: '{reference_message}'")
                print(f"Enhanced: '{enhanced_message[:100]}...'")
                self.test_results["reference_resolution"] = True
            else:
                print("✗ Reference resolution not working")

            # Clean up
            self.conversation_manager.delete_session(session_id)

        except Exception as e:
            print(f"✗ Test failed: {str(e)}")

        # Print summary
        print("\nTest Summary:")
        passed = sum(1 for result in self.test_results.values() if result)
        print(f"Passed: {passed}/{len(self.test_results)} tests")

        for test, result in self.test_results.items():
            status = "✓" if result else "✗"
            print(f"{status} {test}")

    def print_status(self):
        """Print test status and results."""
        print("\nTest Status:")
        passed = sum(1 for result in self.test_results.values() if result)
        print(f"Passed: {passed}/{len(self.test_results)} tests")

        for test, result in self.test_results.items():
            status = "✓" if result else "✗"
            print(f"{status} {test}")

    async def handle_command(self, command, args):
        """Handle a command."""
        if command == "/help":
            self.print_help()
            return True

        elif command == "/exit":
            return False

        elif command == "/test":
            await self.run_automated_tests()
            return True

        elif command == "/status":
            self.print_status()
            return True

        elif command == "/debug":
            if args and args[0].lower() in ["on", "true", "yes", "1"]:
                self.debug_mode = True
                print("Debug mode enabled")
            elif args and args[0].lower() in ["off", "false", "no", "0"]:
                self.debug_mode = False
                print("Debug mode disabled")
            else:
                self.debug_mode = not self.debug_mode
                print(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
            return True

        elif command == "/new":
            session_id = self.conversation_manager.create_session()
            self.current_session_id = session_id
            print(f"Created and switched to new session: {session_id}")
            return True

        elif command == "/sessions":
            sessions = self.conversation_manager.list_sessions()
            print("\nActive Sessions:")
            for i, session in enumerate(sessions):
                current_marker = (
                    " (current)" if session["session_id"] == self.current_session_id else ""
                )
                print(
                    f"{i+1}. {session['session_id']}{current_marker} - {session['message_count']} messages"
                )
            return True

        elif command == "/history":
            history = self.conversation_manager.get_conversation_history(self.current_session_id)
            print("\nConversation History:")
            for i, msg in enumerate(history):
                role = msg["role"]
                content = msg["content"]
                timestamp = msg.get("timestamp", "")
                print(
                    f"{i+1}. [{role.upper()}] {timestamp}: {content[:100]}{'...' if len(content) > 100 else ''}"
                )
            return True

        elif command == "/summarize":
            print("Forcing conversation summarization...")
            await self.conversation_manager._maybe_summarize_conversation(self.current_session_id)
            summary = self.conversation_manager.sessions.get(self.current_session_id, {}).get(
                "summary"
            )
            if summary:
                print(f"\nSummary: {summary}")
            else:
                print("No summary generated")
            return True

        return True

    async def process_message(self, message):
        """Process a user message."""
        start_time = time.time()

        # Detect topic changes
        is_topic_change = self.conversation_manager.handle_topic_change(
            self.current_session_id, message
        )
        if self.debug_mode and is_topic_change:
            print("[DEBUG] Topic change detected")

        # Manage context window
        if self.debug_mode:
            context_managed = self.conversation_manager.manage_context_window(
                self.current_session_id
            )
            if context_managed:
                print("[DEBUG] Context window managed")

        # Process message
        print("Assistant: ", end="", flush=True)
        response = ""
        async for token in self.conversation_manager.converse_with_context(
            self.current_session_id, message
        ):
            response += token
            print(token, end="", flush=True)
        print()

        # Show performance metrics in debug mode
        elapsed = time.time() - start_time
        self.response_times.append(elapsed)

        if self.debug_mode:
            print(f"[DEBUG] Response time: {elapsed:.2f}s")

            # Get session info
            session = self.conversation_manager.sessions.get(self.current_session_id, {})
            message_count = len(session.get("messages", []))
            print(f"[DEBUG] Total messages: {message_count}")

        # Update test results
        if len(response) > 0:
            self.test_results["basic_conversation"] = True

        if elapsed < 5.0:  # Adjust threshold as needed
            self.test_results["response_time"] = True

    async def run(self):
        """Run the validation CLI."""
        # Initialize components
        if not await self.initialize():
            logger.error("Failed to initialize components")
            return

        # Print header
        self.print_header()

        # Main loop
        running = True
        while running:
            try:
                user_input = input("\nYou: ")

                # Check if this is a command
                if user_input.startswith("/"):
                    parts = user_input.split()
                    command = parts[0].lower()
                    args = parts[1:] if len(parts) > 1 else []

                    running = await self.handle_command(command, args)
                else:
                    # Process as a regular message
                    await self.process_message(user_input)

            except KeyboardInterrupt:
                print("\nExiting...")
                running = False
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                import traceback

                traceback.print_exc()

        print("\nValidation CLI exited.")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Interactive validation CLI for Milestone 3: Conversation Management"
    )
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    parser.parse_args()

    # Always run in interactive mode for now
    cli = ValidationCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
