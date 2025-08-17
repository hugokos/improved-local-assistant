#!/usr/bin/env python3
"""
Test script for the ConversationManager component.

This script provides command-line testing functionality for the ConversationManager,
allowing testing of session creation, streaming responses, and context handling.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import yaml

# Add parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_local_assistant.services.conversation_manager import ConversationManager
from improved_local_assistant.services.graph_manager import KnowledgeGraphManager
from improved_local_assistant.services.model_mgr import ModelConfig
from improved_local_assistant.services.model_mgr import ModelManager

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


async def create_session(model_manager, kg_manager, config):
    """Test creating a conversation session."""
    logger.info("Creating conversation manager...")
    conversation_manager = ConversationManager(model_manager, kg_manager, config)

    logger.info("Creating a new session...")
    session_id = conversation_manager.create_session()
    logger.info(f"Created session with ID: {session_id}")

    # Get session info
    session_info = conversation_manager.get_session_info(session_id)
    logger.info(f"Session info: {json.dumps(session_info, indent=2)}")

    return conversation_manager, session_id


async def test_streaming(conversation_manager, session_id):
    """Test streaming conversation responses with dual-model architecture."""
    logger.info("Testing streaming conversation responses with dual-model architecture...")

    # Test basic streaming
    test_message = "Hello, can you tell me about yourself?"
    logger.info(f"Sending test message: '{test_message}'")

    # Process message with streaming
    logger.info("Response (basic streaming):")
    async for token in conversation_manager.process_message(session_id, test_message):
        print(token, end="", flush=True)
    print()  # Add newline after response

    # Test dual-model functionality
    test_message_2 = "Tell me about artificial intelligence and machine learning."
    logger.info(f"Sending test message for dual-model processing: '{test_message_2}'")

    # Process message with dual-model architecture
    logger.info("Response (dual-model processing):")
    async for token in conversation_manager.process_with_dual_model(session_id, test_message_2):
        print(token, end="", flush=True)
    print()  # Add newline after response

    # Test context-aware response generation
    test_message_3 = "What are the key differences between supervised and unsupervised learning?"
    logger.info(f"Sending test message for context-aware response: '{test_message_3}'")

    # Process message with context-aware response generation
    logger.info("Response (context-aware):")
    async for token in conversation_manager.converse_with_context(session_id, test_message_3):
        print(token, end="", flush=True)
    print()  # Add newline after response

    # Get conversation history
    history = conversation_manager.get_conversation_history(session_id)
    logger.info(f"Conversation history has {len(history)} messages")

    return history


async def test_context(conversation_manager, session_id):
    """Test conversation context and reference resolution."""
    logger.info("Testing conversation context handling...")

    # First message to establish context
    context_message = (
        "My name is Alice and I work as a software engineer at a company called TechCorp."
    )
    logger.info(f"Sending context message: '{context_message}'")

    # Process first message
    logger.info("Response to context message:")
    async for token in conversation_manager.process_message(session_id, context_message):
        print(token, end="", flush=True)
    print()

    # Follow-up message with reference
    reference_message = "What do I do for a living?"
    logger.info(f"Sending reference message: '{reference_message}'")

    # Resolve references
    enhanced_message = conversation_manager.resolve_references(session_id, reference_message)
    logger.info(f"Enhanced message with context: '{enhanced_message}'")

    # Process with context
    logger.info("Response to reference message:")
    async for token in conversation_manager.converse_with_context(session_id, reference_message):
        print(token, end="", flush=True)
    print()

    # Test context window management
    logger.info("Testing context window management...")
    conversation_manager.manage_context_window(session_id)

    # Add more context to test topic change detection
    topic_message = "By the way, can you tell me about machine learning algorithms?"
    logger.info(f"Sending topic change message: '{topic_message}'")

    # Detect topic change
    is_topic_change = conversation_manager.handle_topic_change(session_id, topic_message)
    logger.info(f"Topic change detected: {is_topic_change}")

    # Process topic change message
    logger.info("Response to topic change message:")
    async for token in conversation_manager.converse_with_context(session_id, topic_message):
        print(token, end="", flush=True)
    print()

    # Test another reference after topic change
    after_change_message = "Which one is best for image recognition?"
    logger.info(f"Sending follow-up after topic change: '{after_change_message}'")

    # Process follow-up message
    logger.info("Response to follow-up after topic change:")
    async for token in conversation_manager.converse_with_context(session_id, after_change_message):
        print(token, end="", flush=True)
    print()

    # Get relevant context
    context = conversation_manager.get_relevant_context(session_id, after_change_message)
    logger.info(f"Retrieved {len(context['conversation_history'])} messages from history")

    if context.get("knowledge_graph_results"):
        logger.info("Knowledge graph results retrieved")

    if context.get("session_summary"):
        logger.info(f"Session summary: {context['session_summary']}")

    # Get session info
    session_info = conversation_manager.get_session_info(session_id)
    logger.info(f"Session info: {json.dumps(session_info, indent=2)}")

    return context


async def interactive_session(conversation_manager, session_id):
    """
    Run an interactive conversation session with debugging features.

    This interactive tool provides:
    - Real-time conversation with the assistant
    - Session management and history viewing
    - Conversation analysis and debugging features
    - Performance metrics for conversation processing
    """
    logger.info("Starting interactive conversation testing and debugging session...")
    logger.info(f"Session ID: {session_id}")
    logger.info("Available commands:")
    logger.info("  /help - Show available commands")
    logger.info("  /exit - End the session")
    logger.info("  /history - Show conversation history")
    logger.info("  /info - Show session information")
    logger.info("  /metrics - Show performance metrics")
    logger.info("  /analyze - Analyze conversation context and references")
    logger.info("  /summarize - Force conversation summarization")
    logger.info("  /new - Create a new session")
    logger.info("  /sessions - List all active sessions")
    logger.info("  /switch <id> - Switch to another session")
    logger.info("  /debug <on|off> - Toggle debug mode")

    # Debug mode settings
    debug_mode = False
    current_session_id = session_id

    while True:
        try:
            user_input = input("\nYou: ")

            # Process commands
            if user_input.startswith("/"):
                command = user_input.split()[0].lower()
                args = user_input.split()[1:] if len(user_input.split()) > 1 else []

                if command == "/exit":
                    break

                elif command == "/help":
                    logger.info("Available commands:")
                    logger.info("  /help - Show available commands")
                    logger.info("  /exit - End the session")
                    logger.info("  /history - Show conversation history")
                    logger.info("  /info - Show session information")
                    logger.info("  /metrics - Show performance metrics")
                    logger.info("  /analyze - Analyze conversation context and references")
                    logger.info("  /summarize - Force conversation summarization")
                    logger.info("  /new - Create a new session")
                    logger.info("  /sessions - List all active sessions")
                    logger.info("  /switch <id> - Switch to another session")
                    logger.info("  /debug <on|off> - Toggle debug mode")
                    continue

                elif command == "/history":
                    history = conversation_manager.get_conversation_history(current_session_id)
                    print("\nConversation History:")
                    for i, msg in enumerate(history):
                        role = msg["role"]
                        content = msg["content"]
                        timestamp = msg.get("timestamp", "")
                        print(
                            f"{i+1}. [{role.upper()}] {timestamp}: {content[:100]}{'...' if len(content) > 100 else ''}"
                        )
                    continue

                elif command == "/info":
                    info = conversation_manager.get_session_info(current_session_id)
                    print("\nSession Information:")
                    print(json.dumps(info, indent=2))
                    continue

                elif command == "/metrics":
                    metrics = conversation_manager.get_metrics()
                    print("\nPerformance Metrics:")
                    print(json.dumps(metrics, indent=2))
                    continue

                elif command == "/analyze":
                    history = conversation_manager.get_conversation_history(current_session_id)
                    if not history:
                        print("No conversation history to analyze")
                        continue

                    print("\nConversation Analysis:")

                    # Analyze message patterns
                    user_messages = [msg for msg in history if msg["role"] == "user"]
                    assistant_messages = [msg for msg in history if msg["role"] == "assistant"]

                    print(f"Total messages: {len(history)}")
                    print(f"User messages: {len(user_messages)}")
                    print(f"Assistant messages: {len(assistant_messages)}")

                    # Analyze token usage (rough estimate)
                    total_tokens = 0
                    for msg in history:
                        content = msg.get("content", "")
                        total_tokens += (
                            len(content.split()) * 4
                        )  # Rough estimate: 4 tokens per word

                    print(f"Estimated token usage: {total_tokens}")
                    print(
                        f"Context window utilization: {total_tokens / conversation_manager.context_window_tokens * 100:.1f}%"
                    )

                    # Check for reference patterns
                    reference_count = 0
                    for msg in user_messages:
                        content = msg.get("content", "")
                        enhanced = conversation_manager.resolve_references(
                            current_session_id, content
                        )
                        if enhanced != content:
                            reference_count += 1

                    print(f"Messages with references: {reference_count}")

                    # Get topic changes
                    session = conversation_manager.sessions.get(current_session_id, {})
                    topic_changes = session.get("metadata", {}).get("topic_changes", [])
                    print(f"Topic changes: {len(topic_changes)}")

                    continue

                elif command == "/summarize":
                    print("Forcing conversation summarization...")
                    await conversation_manager._maybe_summarize_conversation(current_session_id)
                    summary = conversation_manager.sessions.get(current_session_id, {}).get(
                        "summary"
                    )
                    if summary:
                        print(f"\nSummary: {summary}")
                    else:
                        print("No summary generated")
                    continue

                elif command == "/new":
                    new_session_id = conversation_manager.create_session()
                    current_session_id = new_session_id
                    print(f"Created and switched to new session: {new_session_id}")
                    continue

                elif command == "/sessions":
                    sessions = conversation_manager.list_sessions()
                    print("\nActive Sessions:")
                    for i, session in enumerate(sessions):
                        current_marker = (
                            " (current)" if session["session_id"] == current_session_id else ""
                        )
                        print(
                            f"{i+1}. {session['session_id']}{current_marker} - {session['message_count']} messages"
                        )
                    continue

                elif command == "/switch":
                    if not args:
                        print("Please provide a session ID")
                        continue

                    new_session_id = args[0]
                    if new_session_id in [
                        s["session_id"] for s in conversation_manager.list_sessions()
                    ]:
                        current_session_id = new_session_id
                        print(f"Switched to session: {current_session_id}")
                    else:
                        print(f"Session {new_session_id} not found")
                    continue

                elif command == "/debug":
                    if args and args[0].lower() in ["on", "true", "yes", "1"]:
                        debug_mode = True
                        print("Debug mode enabled")
                    elif args and args[0].lower() in ["off", "false", "no", "0"]:
                        debug_mode = False
                        print("Debug mode disabled")
                    else:
                        debug_mode = not debug_mode
                        print(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
                    continue

                else:
                    print(f"Unknown command: {command}")
                    continue

            # Process regular message
            start_time = time.time()  # noqa: F821

            # Detect topic changes
            is_topic_change = conversation_manager.handle_topic_change(
                current_session_id, user_input
            )
            if debug_mode and is_topic_change:
                print("[DEBUG] Topic change detected")

            # Manage context window
            if debug_mode:
                context_managed = conversation_manager.manage_context_window(current_session_id)
                if context_managed:
                    print("[DEBUG] Context window managed")

            # Resolve references if in debug mode
            if debug_mode:
                enhanced_message = conversation_manager.resolve_references(
                    current_session_id, user_input
                )
                if enhanced_message != user_input:
                    print(f"[DEBUG] Enhanced message: {enhanced_message[:100]}...")

            # Process message
            print("Assistant: ", end="", flush=True)
            async for token in conversation_manager.converse_with_context(
                current_session_id, user_input
            ):
                print(token, end="", flush=True)
            print()

            # Show performance metrics in debug mode
            if debug_mode:
                elapsed = time.time() - start_time  # noqa: F821
                print(f"[DEBUG] Response time: {elapsed:.2f}s")

                # Get session info
                session = conversation_manager.sessions.get(current_session_id, {})
                message_count = len(session.get("messages", []))
                print(f"[DEBUG] Total messages: {message_count}")

                # Estimate token usage
                if "messages" in session:
                    total_tokens = 0
                    for msg in session["messages"]:
                        content = msg.get("content", "")
                        total_tokens += (
                            len(content.split()) * 4
                        )  # Rough estimate: 4 tokens per word

                    print(f"[DEBUG] Estimated token usage: {total_tokens}")
                    print(
                        f"[DEBUG] Context window utilization: {total_tokens / conversation_manager.context_window_tokens * 100:.1f}%"
                    )

        except KeyboardInterrupt:
            logger.info("Session interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in interactive session: {str(e)}")
            import traceback

            traceback.print_exc()

    # Get session info at the end
    session_info = conversation_manager.get_session_info(current_session_id)
    logger.info(f"Session ended with {session_info['message_count']} messages")

    # Get metrics
    metrics = conversation_manager.get_metrics()
    logger.info(f"Conversation metrics: {json.dumps(metrics, indent=2)}")


async def main():
    """Main function to run the conversation tests."""
    parser = argparse.ArgumentParser(description="Test the ConversationManager component")
    parser.add_argument(
        "--create-session", action="store_true", help="Test creating a conversation session"
    )
    parser.add_argument(
        "--test-streaming", action="store_true", help="Test streaming conversation responses"
    )
    parser.add_argument(
        "--test-context", action="store_true", help="Test conversation context handling"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run an interactive conversation session"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Initialize model manager
    logger.info("Initializing model manager...")
    model_manager = ModelManager(host=config["ollama"]["host"])

    # Create model configuration
    model_config = ModelConfig(
        name=config["models"]["conversation"]["name"],
        type="conversation",
        context_window=config["models"]["conversation"]["context_window"],
        temperature=config["models"]["conversation"]["temperature"],
        max_tokens=config["models"]["conversation"]["max_tokens"],
        timeout=config["ollama"]["timeout"],
        max_parallel=2,
        max_loaded=2,
    )

    # Initialize models
    logger.info("Initializing models...")
    if not await model_manager.initialize_models(model_config):
        logger.error("Failed to initialize models")
        return

    # Initialize knowledge graph manager
    logger.info("Initializing knowledge graph manager...")
    kg_manager = KnowledgeGraphManager(model_manager, config)

    # Load pre-built graphs if available
    logger.info("Loading pre-built knowledge graphs...")
    loaded_graphs = kg_manager.load_prebuilt_graphs()
    if loaded_graphs:
        logger.info(f"Loaded {len(loaded_graphs)} pre-built knowledge graphs")
    else:
        logger.info("No pre-built knowledge graphs loaded")

    # Initialize dynamic graph
    logger.info("Initializing dynamic knowledge graph...")
    kg_manager.initialize_dynamic_graph()

    # Create conversation manager and session
    conversation_manager, session_id = await create_session(model_manager, kg_manager, config)

    # Run requested tests
    if args.create_session:
        # Already created session above
        logger.info("Session creation test completed successfully")

    if args.test_streaming:
        await test_streaming(conversation_manager, session_id)
        logger.info("Streaming test completed successfully")

    if args.test_context:
        await test_context(conversation_manager, session_id)
        logger.info("Context handling test completed successfully")

    if args.interactive:
        await interactive_session(conversation_manager, session_id)

    # If no specific test was requested, run all tests
    if not any([args.create_session, args.test_streaming, args.test_context, args.interactive]):
        logger.info("No specific test requested, running all tests...")
        await test_streaming(conversation_manager, session_id)
        await test_context(conversation_manager, session_id)
        await interactive_session(conversation_manager, session_id)

    logger.info("All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
