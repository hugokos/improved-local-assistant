#!/usr/bin/env python
"""
Test script for GraphRAG with Ollama Hermes 3 model.

This script creates a simple chatbot that connects to the survivalist prebuilt graph
and uses the Ollama Hermes 3 model for answering questions.
"""

import argparse
import asyncio
import logging
import signal
import sys
import time

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import yaml

# Import services from the improved-local-assistant project
from services.graph_manager import KnowledgeGraphManager
from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("graphrag_test.log", mode="w"),  # Log to file as well
    ],
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


class OllamaGraphRAGBot:
    """
    Simple chatbot that uses Ollama Hermes 3 model with the survivalist prebuilt graph.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the chatbot with configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.model_manager = None
        self.graph_manager = None
        self.conversation_history = []

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Use default configuration
            logger.info("Using default configuration")
            return {
                "ollama": {"host": "http://localhost:11434", "timeout": 120},
                "models": {
                    "conversation": {
                        "name": "hermes3:3b",
                        "context_window": 8000,
                        "temperature": 0.7,
                        "max_tokens": 2048,
                    }
                },
                "knowledge_graphs": {
                    "prebuilt_directory": "./data/prebuilt_graphs",
                    "max_triplets_per_chunk": 4,
                },
            }

    async def initialize(self) -> bool:
        """
        Initialize the model and graph managers.

        Returns:
            bool: True if initialization was successful
        """
        try:
            # Initialize model manager
            ollama_host = self.config.get("ollama", {}).get("host", "http://localhost:11434")
            self.model_manager = ModelManager(host=ollama_host)

            # Configure model
            model_config = self.config.get("models", {}).get("conversation", {})
            model_name = model_config.get("name", "hermes3:3b")
            context_window = model_config.get("context_window", 8000)
            temperature = model_config.get("temperature", 0.7)
            max_tokens = model_config.get("max_tokens", 2048)

            # Create model configuration
            config = ModelConfig(
                name=model_name,
                type="conversation",
                context_window=context_window,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.config.get("ollama", {}).get("timeout", 120),
            )

            # Initialize models
            await self.model_manager.initialize_models(config)

            # Initialize graph manager
            self.graph_manager = KnowledgeGraphManager(
                model_manager=self.model_manager, config=self.config
            )

            # Initialize the knowledge graph optimizer
            from services.kg_optimizer import initialize_optimizer

            initialize_optimizer(self.graph_manager)

            # Load prebuilt graphs (specifically the survivalist graph)
            loaded_graphs = self.graph_manager.load_prebuilt_graphs()

            if not loaded_graphs:
                logger.warning(
                    "No graphs were loaded. Make sure the survivalist graph exists in the prebuilt directory."
                )
                return False

            logger.info(f"Loaded graphs: {loaded_graphs}")

            # Check if survivalist graph was loaded
            if "prebuilt_survivalist" not in loaded_graphs:
                logger.warning(
                    "Survivalist graph was not loaded. Make sure it exists in the prebuilt directory."
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error initializing: {str(e)}")
            return False

    async def process_query(self, query: str, timeout: int = 120) -> str:
        """
        Process a user query using the GraphRAG approach with timeout and detailed logging.

        Args:
            query: User query string
            timeout: Timeout in seconds for the entire query process

        Returns:
            str: Response from the model
        """
        start_time = time.time()

        try:
            logger.info(f"Starting query processing: '{query}'")

            # Step 1: Query the knowledge graph
            logger.info("Step 1: Querying knowledge graph...")
            kg_start = time.time()

            try:
                graph_result = await asyncio.wait_for(
                    asyncio.to_thread(self.graph_manager.query_graphs, query),
                    timeout=30,  # 30 second timeout for graph query
                )
                kg_time = time.time() - kg_start
                logger.info(f"Knowledge graph query completed in {kg_time:.2f}s")
            except asyncio.TimeoutError:
                logger.error("Knowledge graph query timed out after 30 seconds")
                return (
                    "I'm sorry, the knowledge graph query timed out. Please try a simpler question."
                )
            except Exception as e:
                logger.error(f"Knowledge graph query failed: {str(e)}")
                return f"I'm sorry, there was an error querying the knowledge graph: {str(e)}"

            # Extract the response and source nodes
            graph_response = graph_result.get("response", "")
            source_nodes = graph_result.get("source_nodes", [])
            metadata = graph_result.get("metadata", {})

            logger.info("Graph query results:")
            logger.info(f"  - Response length: {len(graph_response)} characters")
            logger.info(f"  - Source nodes: {len(source_nodes)}")
            logger.info(f"  - Query time: {metadata.get('query_time', 'unknown')}")
            logger.info(f"  - Graph sources: {metadata.get('graph_sources', [])}")

            if graph_response:
                logger.debug(f"Graph response preview: {graph_response[:200]}...")

            # Step 2: Prepare context from source nodes
            logger.info("Step 2: Preparing context from source nodes...")
            context = ""
            if source_nodes:
                context = "Information from knowledge graph:\n"
                for i, node in enumerate(source_nodes, 1):
                    if hasattr(node, "get_text"):
                        node_text = node.get_text()[:300]
                        context += f"{i}. {node_text}...\n"
                        logger.debug(f"  - Node {i}: {node_text[:100]}...")
            else:
                logger.warning("No source nodes found in knowledge graph response")

            # Step 3: Prepare messages for the conversation model
            logger.info("Step 3: Preparing conversation messages...")
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with knowledge about survivalist topics. "
                    "Use the provided information from the knowledge graph to answer questions accurately.",
                },
            ]

            # Add conversation history (last 5 messages)
            history_count = len(self.conversation_history[-10:])  # Last 5 pairs
            if history_count > 0:
                logger.info(f"Adding {history_count} previous messages for context")
                for msg in self.conversation_history[-10:]:
                    messages.append(msg)

            # Add the current query with context
            if context:
                user_content = (
                    f"Question: {query}\n\nRelevant information from knowledge base:\n{context}"
                )
                logger.info(f"Using context ({len(context)} characters)")
            else:
                user_content = query
                logger.warning("No context available from knowledge graph")

            messages.append({"role": "user", "content": user_content})
            logger.debug(f"Final message count: {len(messages)}")

            # Step 4: Query the conversation model
            logger.info("Step 4: Querying Ollama conversation model...")
            llm_start = time.time()

            response_text = ""
            token_count = 0

            try:

                async def query_with_timeout():
                    nonlocal response_text, token_count
                    async for token in self.model_manager.query_conversation_model(messages):
                        if shutdown_requested:
                            logger.info("Shutdown requested, stopping token generation")
                            break
                        response_text += token
                        token_count += 1

                        # Log progress every 50 tokens
                        if token_count % 50 == 0:
                            logger.debug(f"Generated {token_count} tokens...")

                await asyncio.wait_for(
                    query_with_timeout(), timeout=timeout - 30
                )  # Reserve 30s for other operations

                llm_time = time.time() - llm_start
                logger.info(f"Conversation model completed in {llm_time:.2f}s")
                logger.info(f"Generated {token_count} tokens, {len(response_text)} characters")

            except asyncio.TimeoutError:
                logger.error(f"Conversation model timed out after {timeout-30} seconds")
                if response_text:
                    logger.info(f"Partial response available ({len(response_text)} characters)")
                    response_text += "\n\n[Response was cut short due to timeout]"
                else:
                    return "I'm sorry, the AI model took too long to respond. Please try a simpler question."
            except Exception as e:
                logger.error(f"Conversation model failed: {str(e)}")
                return f"I'm sorry, there was an error with the AI model: {str(e)}"

            # Step 5: Update conversation history
            logger.info("Step 5: Updating conversation history...")
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": response_text})

            # Keep history manageable
            if len(self.conversation_history) > 20:  # Keep last 10 exchanges
                self.conversation_history = self.conversation_history[-20:]
                logger.debug("Trimmed conversation history")

            total_time = time.time() - start_time
            logger.info(f"Query processing completed in {total_time:.2f}s total")

            return response_text

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Error processing query after {total_time:.2f}s: {str(e)}")
            logger.exception("Full error traceback:")
            return f"I'm sorry, I encountered an unexpected error: {str(e)}"

    async def run_interactive(self):
        """
        Run the chatbot in interactive mode with improved error handling and interruption support.
        """
        global shutdown_requested

        print("GraphRAG Ollama Bot initialized with survivalist knowledge graph.")
        print("Type 'exit' or 'quit' to end the session.")
        print("Press Ctrl+C at any time to interrupt.")
        print("Logs are being written to 'graphrag_test.log'")
        print("-" * 60)

        query_count = 0

        while not shutdown_requested:
            try:
                # Get user input with timeout to allow for interruption
                print(f"\n[Query #{query_count + 1}]")
                user_input = await asyncio.wait_for(
                    asyncio.to_thread(input, "You: "),
                    timeout=300,  # 5 minute timeout for user input
                )
                user_input = user_input.strip()

                if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
                    print("Goodbye!")
                    break

                if not user_input:
                    print("Please enter a question or type 'exit' to quit.")
                    continue

                query_count += 1
                logger.info(f"Processing query #{query_count}: '{user_input}'")

                # Show a progress indicator
                print("Processing your question...")

                # Process the query with timeout
                try:
                    response = await asyncio.wait_for(
                        self.process_query(user_input),
                        timeout=180,  # 3 minute timeout for entire query
                    )

                    print(f"\nBot: {response}")
                    logger.info(f"Successfully completed query #{query_count}")

                except asyncio.TimeoutError:
                    print("\nSorry, your question took too long to process (>3 minutes).")
                    print("Try asking a simpler question or check if Ollama is responding.")
                    logger.error(f"Query #{query_count} timed out after 3 minutes")

                except Exception as e:
                    print(f"\nError processing your question: {str(e)}")
                    logger.error(f"Query #{query_count} failed: {str(e)}")
                    logger.exception("Full error traceback:")

            except asyncio.TimeoutError:
                print("\nNo input received for 5 minutes. Exiting...")
                logger.info("Session timed out due to inactivity")
                break

            except KeyboardInterrupt:
                print("\n\nInterrupt received! Shutting down gracefully...")
                logger.info("KeyboardInterrupt received")
                shutdown_requested = True
                break

            except EOFError:
                print("\nInput stream closed. Goodbye!")
                logger.info("EOFError - input stream closed")
                break

            except Exception as e:
                print(f"\nUnexpected error in interactive mode: {str(e)}")
                logger.error(f"Unexpected error in interactive loop: {str(e)}")
                logger.exception("Full error traceback:")

                # Ask if user wants to continue
                try:
                    continue_choice = input("\nDo you want to continue? (y/n): ").strip().lower()
                    if continue_choice not in ("y", "yes"):
                        break
                except:
                    break

        print("\nSession summary:")
        print(f"  - Queries processed: {query_count}")
        print("  - Log file: graphrag_test.log")
        logger.info(f"Interactive session ended. Total queries: {query_count}")


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    logger.info(f"Received signal {signum}")
    shutdown_requested = True


async def main():
    """
    Main function to run the chatbot with proper signal handling.
    """
    global shutdown_requested

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="GraphRAG Ollama Bot with Enhanced Logging")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--timeout", type=int, default=120, help="Query timeout in seconds")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Adjust logging level if debug is requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")

    logger.info("Starting GraphRAG Ollama Bot...")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Query timeout: {args.timeout}s")

    try:
        bot = OllamaGraphRAGBot(config_path=args.config)

        # Initialize the bot
        logger.info("Initializing bot components...")
        success = await bot.initialize()

        if not success:
            logger.error("Failed to initialize the bot. Exiting.")
            print("Bot initialization failed. Check the logs for details.")
            return 1

        logger.info("Bot initialization completed successfully")

        # Run the interactive session
        if not shutdown_requested:
            await bot.run_interactive()

        logger.info("Bot session completed")
        return 0

    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt in main")
        print("\nInterrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        logger.exception("Full error traceback:")
        print(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nFinal interrupt caught")
        sys.exit(0)
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
        sys.exit(1)
