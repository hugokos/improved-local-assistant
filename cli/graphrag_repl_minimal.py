#!/usr/bin/env python3
"""
GraphRAG REPL - Minimal CLI tool for exercising the conversation-plus-GraphRAG stack.

This is a minimal version that focuses on core functionality without resource monitoring
to avoid high memory usage issues. Uses synchronous input for reliability.

Usage:
    python cli/graphrag_repl_minimal.py
    python cli/graphrag_repl_minimal.py --no-kg
    python cli/graphrag_repl_minimal.py --max-triple-per-chunk 6
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

# Add parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure minimal logging
logging.basicConfig(
    level=logging.WARNING,  # Reduced logging to minimize memory usage
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Mute resource manager warnings to avoid interrupting chat
resource_logger = logging.getLogger("services.resource_manager")
resource_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Also mute system monitor warnings
system_monitor_logger = logging.getLogger("services.system_monitor")
system_monitor_logger.setLevel(logging.ERROR)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import services
# Import conversation extensions to add get_response method
from services.conversation_manager import ConversationManager
from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager


class MinimalGraphRAGREPL:
    """
    Minimal GraphRAG REPL that focuses on core conversation functionality.
    """

    def __init__(self, use_kg: bool = True, max_triple_per_chunk: int = 4):
        """
        Initialize the Minimal GraphRAG REPL.

        Args:
            use_kg: Whether to use knowledge graph retrieval and extraction
            max_triple_per_chunk: Maximum triples per chunk for KG extraction
        """
        self.use_kg = use_kg
        self.max_triple_per_chunk = max_triple_per_chunk

        # Initialize components
        self.model_manager: ModelManager | None = None
        self.kg_manager: Optional = None  # Will be None if --no-kg
        self.conversation_manager: ConversationManager | None = None
        self.session_id: str | None = None

        # Simple metrics
        self.queries_processed = 0

        print(f"Minimal GraphRAG REPL initialized (KG: {'ON' if use_kg else 'OFF'})")

    async def initialize_model_manager(self) -> bool:
        """Initialize ModelManager with the specified models."""
        try:
            print("Initializing models...")

            # Initialize ModelManager
            self.model_manager = ModelManager(
                host="http://localhost:11434", healthcheck_mode="version"
            )

            # Set model names
            self.model_manager.conversation_model = "hermes3:3b"
            self.model_manager.knowledge_model = "tinyllama:latest"

            # Create model configuration
            model_config = ModelConfig(
                name="hermes3:3b",
                type="conversation",
                context_window=8000,
                temperature=0.7,
                max_tokens=2048,
                timeout=120,
                max_parallel=1,  # Reduced for memory efficiency
                max_loaded=1,  # Reduced for memory efficiency
            )

            # Initialize models
            success = await self.model_manager.initialize_models(model_config)
            if success:
                print("✅ Models initialized successfully")
                return True
            else:
                print("❌ Failed to initialize models")
                return False

        except Exception as e:
            print(f"❌ Error initializing models: {str(e)}")
            return False

    async def initialize_kg_manager(self) -> bool:
        """Initialize KnowledgeGraphManager if needed."""
        try:
            if not self.use_kg:
                print("Skipping knowledge graph (--no-kg)")
                return True

            print("Initializing knowledge graph...")

            # Import KG manager only if needed
            from services.graph_manager import KnowledgeGraphManager

            # Minimal KG configuration
            kg_config = {
                "knowledge_graphs": {
                    "max_triplets_per_chunk": self.max_triple_per_chunk,
                    "enable_visualization": False,
                    "enable_caching": False,  # Disabled to save memory
                }
            }

            self.kg_manager = KnowledgeGraphManager(self.model_manager, kg_config)

            # Try to initialize dynamic graph, but don't fail if it doesn't work
            try:
                self.kg_manager.initialize_dynamic_graph()

                # Load prebuilt knowledge graphs
                loaded_graphs = self.kg_manager.load_prebuilt_graphs()
                if loaded_graphs:
                    print(
                        f"✅ Knowledge graph initialized with {len(loaded_graphs)} prebuilt graphs"
                    )
                else:
                    print("✅ Knowledge graph initialized (dynamic only)")
            except Exception as e:
                print(f"⚠️  Knowledge graph initialization failed, continuing without: {str(e)}")
                self.kg_manager = None

            return True

        except Exception as e:
            print(f"❌ Error initializing knowledge graph: {str(e)}")
            self.kg_manager = None
            return True  # Continue without KG

    async def initialize_conversation_manager(self) -> bool:
        """Initialize ConversationManager and create a session."""
        try:
            print("Initializing conversation manager...")

            # Minimal configuration
            config = {
                "conversation": {
                    "max_history_length": 20,  # Reduced for memory efficiency
                    "summarize_threshold": 10,  # Reduced for memory efficiency
                    "context_window_tokens": 4000,  # Reduced for memory efficiency
                }
            }

            self.conversation_manager = ConversationManager(
                self.model_manager, self.kg_manager, config
            )

            # Create a session
            self.session_id = self.conversation_manager.create_session()

            print("✅ Conversation manager initialized")
            return True

        except Exception as e:
            print(f"❌ Error initializing conversation manager: {str(e)}")
            return False

    async def initialize_all_components(self) -> bool:
        """Initialize all components."""
        try:
            print("Starting initialization...")

            # Initialize model manager (required)
            if not await self.initialize_model_manager():
                return False

            # Initialize KG manager (optional)
            if not await self.initialize_kg_manager():
                return False

            # Initialize conversation manager (required)
            if not await self.initialize_conversation_manager():
                return False

            print("✅ All components initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize components: {str(e)}")
            return False

    def get_user_input(self) -> str | None:
        """Get user input using simple synchronous input."""
        try:
            return input("You: ")
        except (KeyboardInterrupt, EOFError):
            return None

    async def process_user_query(self, user_input: str) -> None:
        """Process user query through the conversation stack."""
        try:
            # Use the get_response method with use_kg parameter
            response_stream = self.conversation_manager.get_response(
                self.session_id, user_input, use_kg=self.use_kg
            )

            # Stream response
            print("Assistant: ", end="", flush=True)

            async for token in response_stream:
                print(token, end="", flush=True)

            print()  # Add newline

            # Enhanced citation display
            if self.use_kg and self.kg_manager:
                try:
                    citations_data = self.conversation_manager.get_citations(self.session_id)
                    if citations_data and citations_data.get("citations"):
                        citations = citations_data["citations"]
                        if citations:
                            print("\nKnowledge Sources:")
                            for citation in citations:
                                citation_id = citation.get("id", "?")
                                source = citation.get("source", "Unknown")
                                score = citation.get("score", 0.0)
                                text = citation.get("text", "")

                                print(f"[{citation_id}] {source} (relevance: {score:.2f})")

                                # Show a brief text snippet if available
                                if text:
                                    snippet = text[:100] + "..." if len(text) > 100 else text
                                    print(f"    Content: {snippet}")

                                # Show key metadata
                                metadata = citation.get("metadata", {})
                                if metadata:
                                    file_info = (
                                        metadata.get("file_name")
                                        or metadata.get("file_path")
                                        or metadata.get("document_id")
                                    )
                                    if file_info:
                                        print(f"    Source: {file_info}")
                except Exception:
                    # Ignore citation errors
                    pass

            self.queries_processed += 1

        except Exception as e:
            print(f"\nError: {str(e)}")

    async def run_repl_loop(self) -> None:
        """Run the main REPL loop."""
        try:
            print("\n" + "=" * 60)
            print("MINIMAL GRAPHRAG REPL")
            print("=" * 60)
            print(f"Knowledge Graph: {'Enabled' if self.use_kg else 'Disabled'}")
            print("Models: hermes3:3b (conversation), tinyllama:latest (knowledge)")
            print("Type 'exit', 'quit', or press Ctrl+C to exit")
            print("=" * 60)

            while True:
                try:
                    # Get user input
                    user_input = self.get_user_input()

                    if user_input is None:
                        break

                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    # Check for exit commands
                    if user_input.lower() in ["exit", "quit", "bye"]:
                        break

                    # Process the query
                    await self.process_user_query(user_input)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

        except Exception as e:
            print(f"Fatal error: {str(e)}")
        finally:
            print(f"\nSession ended. Processed {self.queries_processed} queries.")

    async def run(self) -> None:
        """Main entry point."""
        try:
            # Initialize components
            if not await self.initialize_all_components():
                print("❌ Initialization failed, exiting")
                return

            # Run the REPL
            await self.run_repl_loop()

        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"Fatal error: {str(e)}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Minimal GraphRAG REPL - Memory-efficient CLI tool"
    )

    parser.add_argument(
        "--no-kg", action="store_true", help="Disable knowledge graph for pure chat mode"
    )

    parser.add_argument(
        "--max-triple-per-chunk",
        type=int,
        default=4,
        help="Maximum triples per chunk for KG extraction (default: 4)",
    )

    args = parser.parse_args()

    # Create and run the REPL
    repl = MinimalGraphRAGREPL(
        use_kg=not args.no_kg, max_triple_per_chunk=args.max_triple_per_chunk
    )

    await repl.run()


if __name__ == "__main__":
    asyncio.run(main())
