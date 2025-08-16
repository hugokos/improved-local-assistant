#!/usr/bin/env python3
"""
GraphRAG REPL - Simple and Robust CLI tool for exercising the full conversation-plus-GraphRAG stack.

This is a simplified version that focuses on reliability and handles high memory situations better.
Uses synchronous input to avoid hanging issues with prompt_toolkit.

Usage:
    python cli/graphrag_repl_simple.py
    python cli/graphrag_repl_simple.py --no-kg
    python cli/graphrag_repl_simple.py --max-triple-per-chunk 6
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import TYPE_CHECKING

# Add parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("graphrag_repl.log", mode="a"),
    ],
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
from services.graph_manager import KnowledgeGraphManager
from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager

if TYPE_CHECKING:
    from services.resource_manager import ResourceManager

# Import embedding model singleton
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    EMBEDDING_AVAILABLE = True
except ImportError:
    logger.warning("HuggingFaceEmbedding not available, will use default embeddings")
    EMBEDDING_AVAILABLE = False


class SimpleGraphRAGREPL:
    """
    Simple and robust GraphRAG REPL that exercises the full conversation-plus-GraphRAG stack.
    """

    def __init__(
        self,
        use_kg: bool = True,
        max_triple_per_chunk: int = 4,
        citation_detail: str = "detailed",
        prebuilt_dir: str = "./data/prebuilt_graphs",
    ):
        """
        Initialize the GraphRAG REPL.

        Args:
            use_kg: Whether to use knowledge graph retrieval and extraction
            max_triple_per_chunk: Maximum triples per chunk for KG extraction
            citation_detail: Level of citation detail ("minimal", "compact", "detailed")
            prebuilt_dir: Directory containing pre-built knowledge graphs
        """
        self.use_kg = use_kg
        self.max_triple_per_chunk = max_triple_per_chunk
        self.citation_detail = citation_detail
        self.prebuilt_dir = prebuilt_dir

        # Initialize components
        self.model_manager: ModelManager | None = None
        self.kg_manager: KnowledgeGraphManager | None = None
        self.conversation_manager: ConversationManager | None = None
        self.resource_manager: ResourceManager | None = None
        self.session_id: str | None = None

        # Embedding model singleton
        self.embedding_model = None

        # Performance tracking
        self.metrics = {
            "queries_processed": 0,
            "total_response_time": 0.0,
            "kg_citations_returned": 0,
            "tokens_generated": 0,
        }

        logger.info(
            f"Simple GraphRAG REPL initialized with use_kg={use_kg}, max_triple_per_chunk={max_triple_per_chunk}, citation_detail={citation_detail}"
        )

    async def initialize_embedding_model(self) -> bool:
        """
        Initialize the singleton embedding model that loads BAAI/bge-small-en-v1.5 int8.

        Returns:
            bool: True if initialization was successful
        """
        try:
            if not EMBEDDING_AVAILABLE:
                logger.warning(
                    "HuggingFaceEmbedding not available, skipping embedding model initialization"
                )
                return False

            logger.info("Initializing singleton embedding model: BAAI/bge-small-en-v1.5 int8")

            # Initialize the embedding model with int8 quantization
            self.embedding_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                device="cpu",  # Force CPU to avoid GPU issues
                trust_remote_code=False,  # Security best practice
            )

            # Configure LlamaIndex to use this embedding model globally
            from llama_index.core import Settings

            Settings.embed_model = self.embedding_model

            logger.info("Successfully initialized singleton embedding model")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            return False

    async def initialize_model_manager(self) -> bool:
        """
        Initialize ModelManager with conversation_model="hermes3:3b" and knowledge_model="tinyllama:latest".

        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing ModelManager with hermes3:3b and tinyllama:latest")

            # Initialize ModelManager with Ollama daemon on localhost:11434
            self.model_manager = ModelManager(
                host="http://localhost:11434",
                healthcheck_mode="version",  # Use light healthcheck for faster startup
            )

            # Override model names to ensure we use the specified models
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
                max_parallel=2,
                max_loaded=2,
            )

            # Initialize models
            success = await self.model_manager.initialize_models(model_config)
            if not success:
                logger.error("Failed to initialize models")
                return False

            logger.info("Successfully initialized ModelManager")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ModelManager: {str(e)}")
            return False

    async def initialize_kg_manager(self) -> bool:
        """
        Initialize KnowledgeGraphManager in lazy-load mode.

        Returns:
            bool: True if initialization was successful
        """
        try:
            if not self.use_kg:
                logger.info("Skipping KnowledgeGraphManager initialization (--no-kg flag)")
                return True

            logger.info("Initializing KnowledgeGraphManager in lazy-load mode")

            # Create minimal configuration for KG manager
            kg_config = {
                "knowledge_graphs": {
                    "prebuilt_directory": getattr(self, "prebuilt_dir", "./data/prebuilt_graphs"),
                    "dynamic_storage": "./data/dynamic_graph",
                    "max_triplets_per_chunk": self.max_triple_per_chunk,
                    "enable_visualization": False,
                    "enable_caching": True,
                },
                "models": {
                    "conversation": {"name": "hermes3:3b"},
                    "knowledge": {"name": "tinyllama:latest"},
                },
                "ollama": {"host": "http://localhost:11434", "timeout": 120},
            }

            # Initialize KG manager
            self.kg_manager = KnowledgeGraphManager(self.model_manager, kg_config)

            # Initialize dynamic graph with simple bootstrap
            await self.initialize_simple_dynamic_graph()

            # Load prebuilt knowledge graphs
            logger.info("Loading prebuilt knowledge graphs...")
            loaded_graphs = self.kg_manager.load_prebuilt_graphs()
            if loaded_graphs:
                logger.info(
                    f"Successfully loaded {len(loaded_graphs)} prebuilt graphs: {loaded_graphs}"
                )
            else:
                logger.warning("No prebuilt graphs were loaded")

            logger.info("Successfully initialized KnowledgeGraphManager")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphManager: {str(e)}")
            return False

    async def initialize_simple_dynamic_graph(self) -> bool:
        """
        Initialize dynamic graph with simple bootstrap to ensure kg_index is never None.

        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing simple dynamic graph")

            # Try to initialize the dynamic graph
            success = self.kg_manager.initialize_dynamic_graph()
            if success:
                logger.info("Successfully initialized simple dynamic graph")
                return True
            else:
                logger.warning("Dynamic graph initialization failed, but continuing")
                return True  # Continue even if KG fails

        except Exception as e:
            logger.error(f"Failed to initialize dynamic graph: {str(e)}")
            return True  # Continue even if KG fails

    async def initialize_conversation_manager(self) -> bool:
        """
        Initialize ConversationManager and create a session.

        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing ConversationManager")

            # Create minimal configuration
            conv_config = {
                "conversation": {
                    "max_history_length": 50,
                    "summarize_threshold": 20,
                    "context_window_tokens": 8000,
                }
            }

            # Initialize conversation manager
            self.conversation_manager = ConversationManager(
                self.model_manager, self.kg_manager, conv_config
            )

            # Create a session
            self.session_id = self.conversation_manager.create_session()

            logger.info(
                f"Successfully initialized ConversationManager with session {self.session_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ConversationManager: {str(e)}")
            return False

    async def initialize_all_components(self) -> bool:
        """
        Initialize all components in the correct order.

        Returns:
            bool: True if all components were initialized successfully
        """
        try:
            logger.info("Starting component initialization...")

            # Initialize embedding model singleton first (optional)
            await self.initialize_embedding_model()

            # Initialize model manager (required)
            if not await self.initialize_model_manager():
                return False

            # Initialize KG manager (optional if --no-kg)
            if not await self.initialize_kg_manager():
                return False

            # Initialize conversation manager (required)
            if not await self.initialize_conversation_manager():
                return False

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            return False

    def get_user_input(self, prompt: str = "You: ") -> str | None:
        """
        Get user input using simple synchronous input().

        Args:
            prompt: Prompt string to display

        Returns:
            Optional[str]: User input or None if interrupted
        """
        try:
            return input(prompt)
        except (KeyboardInterrupt, EOFError):
            return None

    async def process_user_query(self, user_input: str) -> None:
        """
        Process user query through the full conversation-plus-GraphRAG stack.

        Args:
            user_input: User's question/input
        """
        start_time = time.time()

        try:
            logger.debug(f"Processing user query: {user_input[:100]}...")

            # Step 1: Pass raw text to ConversationManager.get_response() with use_kg parameter
            response_stream = self.conversation_manager.get_response(
                self.session_id, user_input, use_kg=self.use_kg
            )

            # Step 2: Stream the assistant's response token by token to stdout
            print("Assistant: ", end="", flush=True)
            token_count = 0

            async for token in response_stream:
                print(token, end="", flush=True)
                token_count += 1

            print()  # Add newline after response

            # Step 3: After final token, print compact reference list of KG citations
            if self.use_kg:
                await self.print_kg_citations()

            # Update metrics
            elapsed_time = time.time() - start_time
            self.metrics["queries_processed"] += 1
            self.metrics["total_response_time"] += elapsed_time
            self.metrics["tokens_generated"] += token_count

            logger.debug(f"Query processed in {elapsed_time:.2f}s, {token_count} tokens generated")

        except Exception as e:
            logger.error(f"Error processing user query: {str(e)}")
            print(f"\nError: {str(e)}")

    async def print_kg_citations(self) -> None:
        """
        Print KG citations with configurable detail level showing exactly what knowledge is being used.
        """
        try:
            if not self.use_kg or not self.conversation_manager:
                return

            # Get citations from the last response
            citations_data = self.conversation_manager.get_citations(self.session_id)

            if not citations_data or not citations_data.get("citations"):
                return

            citations = citations_data["citations"]
            if not citations:
                return

            # Print citations based on detail level
            if self.citation_detail == "minimal":
                await self._print_minimal_citations(citations, citations_data)
            elif self.citation_detail == "compact":
                await self._print_compact_citations(citations, citations_data)
            else:  # detailed
                await self._print_detailed_citations(citations, citations_data)

            # Update metrics
            self.metrics["kg_citations_returned"] += len(citations)

        except Exception as e:
            logger.error(f"Error printing KG citations: {str(e)}")
            # Fallback to simple citation display
            await self._print_fallback_citations()

    async def _print_minimal_citations(self, citations, citations_data):
        """Print minimal citation format with first 25 characters of content."""
        print("\nSources:")
        for citation in citations:
            citation_id = citation.get("id", "?")
            source = citation.get("source", "Unknown")
            score = citation.get("score", 0.0)
            text = citation.get("text", "")

            # Get first 25 characters of content
            content_preview = text[:25] + "..." if len(text) > 25 else text
            content_preview = content_preview.replace("\n", " ").replace(
                "\r", " "
            )  # Clean up newlines

            print(f'[{citation_id}] {source} ({score:.2f}) "{content_preview}"')

    async def _print_compact_citations(self, citations, citations_data):
        """Print compact citation format with key details and first 25 characters."""
        print("\nKnowledge Sources:")
        print("-" * 50)

        for citation in citations:
            citation_id = citation.get("id", "?")
            source = citation.get("source", "Unknown")
            score = citation.get("score", 0.0)
            text = citation.get("text", "")
            metadata = citation.get("metadata", {})

            # Get first 25 characters for quick preview
            content_preview = text[:25] + "..." if len(text) > 25 else text
            content_preview = content_preview.replace("\n", " ").replace("\r", " ")

            print(f"[{citation_id}] {source} (relevance: {score:.3f})")
            print(f'    Preview: "{content_preview}"')

            # Show longer content snippet (120 chars)
            if text and len(text) > 25:
                snippet = text[:120] + "..." if len(text) > 120 else text
                snippet = snippet.replace("\n", " ").replace("\r", " ")
                print(f'    Content: "{snippet}"')

            # Show key metadata
            file_info = (
                metadata.get("file_name")
                or metadata.get("document_title")
                or metadata.get("file_path", "").split("/")[-1]
                if metadata.get("file_path")
                else None
            )
            if file_info:
                print(f"    From: {file_info}")

            print()

    async def _print_detailed_citations(self, citations, citations_data):
        """Print detailed citation format with comprehensive information."""
        print("\n" + "=" * 80)
        print("KNOWLEDGE GRAPH CITATIONS - DETAILED VIEW")
        print("=" * 80)

        # Show query context
        query = citations_data.get("query", "")
        if query:
            print(f'Query: "{query}"')
            print("-" * 80)

        for i, citation in enumerate(citations, 1):
            citation_id = citation.get("id", i)
            source = citation.get("source", "Unknown Source")
            score = citation.get("score", 0.0)
            text = citation.get("text", "")
            metadata = citation.get("metadata", {})

            # Get first 25 characters for quick identification
            content_preview = text[:25] + "..." if len(text) > 25 else text
            content_preview = content_preview.replace("\n", " ").replace("\r", " ")

            # Determine graph source from metadata
            graph_source = "unknown"
            if metadata:
                if "graph_source" in metadata:
                    graph_source = metadata["graph_source"]
                elif "source" in metadata and metadata["source"] in [
                    "survivalist",
                    "dynamic",
                    "prebuilt_survivalist",
                ]:
                    graph_source = metadata["source"]

            print(f'\n[{citation_id}] {source} "{content_preview}" (from: {graph_source})')
            print(f"Relevance Score: {score:.3f}")

            # Show text content with smart truncation
            if text:
                if len(text) > 400:
                    # Show beginning and end for very long text
                    text_preview = (
                        text[:200] + "\n    [...content truncated...]\n    " + text[-200:]
                    )
                elif len(text) > 200:
                    # Show beginning with ellipsis for medium text
                    text_preview = text[:200] + "..."
                else:
                    text_preview = text

                print("Full Content:")
                # Indent the content for better readability
                for line in text_preview.split("\n"):
                    print(f"    {line}")

            # Show comprehensive metadata
            if metadata:
                print("Source Details:")

                # File information
                file_fields = ["file_name", "file_path", "document_title", "document_id"]
                for field in file_fields:
                    if field in metadata and metadata[field]:
                        print(f"  {field.replace('_', ' ').title()}: {metadata[field]}")

                # Location information
                location_fields = ["page_number", "section", "chunk_id", "paragraph"]
                for field in location_fields:
                    if field in metadata and metadata[field]:
                        print(f"  {field.replace('_', ' ').title()}: {metadata[field]}")

                # Timestamps
                time_fields = ["creation_date", "last_modified", "timestamp"]
                for field in time_fields:
                    if field in metadata and metadata[field]:
                        print(f"  {field.replace('_', ' ').title()}: {metadata[field]}")

                # Content statistics
                if metadata.get("text_length"):
                    print(f"  Text Length: {metadata['text_length']} characters")

                # Node type information
                if metadata.get("node_type"):
                    print(f"  Node Type: {metadata['node_type']}")

                # Other relevant metadata
                other_fields = [
                    k
                    for k in metadata
                    if k
                    not in file_fields
                    + location_fields
                    + time_fields
                    + ["text_length", "node_type", "has_content"]
                    and isinstance(metadata[k], str | int | float)
                    and len(str(metadata[k])) < 100
                ]

                for field in other_fields:
                    print(f"  {field.replace('_', ' ').title()}: {metadata[field]}")

            # Show relationships if available (knowledge graph specific)
            relationships = citation.get("relationships", metadata.get("relationships", []))
            if relationships:
                print("Knowledge Graph Relationships:")
                for j, rel in enumerate(relationships[:5]):  # Show first 5 relationships
                    if isinstance(rel, dict):
                        subj = rel.get("subject", "")
                        pred = rel.get("predicate", "related_to")
                        obj = rel.get("object", "")
                        print(f"  {j+1}. {subj} → {pred} → {obj}")
                    elif isinstance(rel, list | tuple) and len(rel) >= 3:
                        print(f"  {j+1}. {rel[0]} → {rel[1]} → {rel[2]}")
                    elif isinstance(rel, str):
                        print(f"  {j+1}. {rel}")

                if len(relationships) > 5:
                    print(f"  ... and {len(relationships) - 5} more relationships")

            if i < len(citations):
                print("-" * 60)

        # Show summary statistics
        print("\nCitation Summary:")
        print(f"  Total Sources: {len(citations)}")
        avg_score = sum(c.get("score", 0.0) for c in citations) / len(citations) if citations else 0
        print(f"  Average Relevance: {avg_score:.3f}")

        # Show content statistics
        total_chars = sum(len(c.get("text", "")) for c in citations)
        print(f"  Total Content: {total_chars} characters")

        # Show metadata summary
        metadata_summary = citations_data.get("metadata", {})
        if metadata_summary.get("has_relationships"):
            print("  Contains Knowledge Graph Relationships: Yes")

        # Show timestamp if available
        timestamp = citations_data.get("timestamp", "")
        if timestamp:
            print(f"  Retrieved: {timestamp}")

        print("=" * 80)

    async def _print_fallback_citations(self):
        """Fallback citation display if main method fails."""
        try:
            citations_data = self.conversation_manager.get_citations(self.session_id)
            if citations_data and citations_data.get("citations"):
                citations = citations_data["citations"]
                print("\nReferences (fallback):")
                for citation in citations:
                    citation_id = citation.get("id", "?")
                    source = citation.get("source", "Unknown")
                    score = citation.get("score", 0.0)
                    text = citation.get("text", "")

                    # Get first 25 characters even in fallback
                    content_preview = text[:25] + "..." if len(text) > 25 else text
                    content_preview = content_preview.replace("\n", " ").replace("\r", " ")

                    print(f'[{citation_id}] {source} (score: {score:.2f}) "{content_preview}"')
        except Exception as e:
            logger.error(f"Fallback citation display failed: {str(e)}")

    async def handle_keyboard_interrupt(self) -> None:
        """
        Handle KeyboardInterrupt gracefully by persisting updated graph_store and conversation sessions.
        """
        try:
            logger.info("Handling keyboard interrupt, persisting data...")

            # Persist updated graph store
            if self.use_kg and self.kg_manager and self.kg_manager.dynamic_kg:
                try:
                    persist_dir = os.path.join("./data/dynamic_graph", "main")
                    os.makedirs(persist_dir, exist_ok=True)
                    self.kg_manager.dynamic_kg.storage_context.persist(persist_dir=persist_dir)
                    logger.info("Successfully persisted dynamic knowledge graph")
                except Exception as e:
                    logger.error(f"Failed to persist knowledge graph: {str(e)}")

            # Print final metrics
            self.print_final_metrics()

            logger.info("Graceful shutdown completed")

        except Exception as e:
            logger.error(f"Error during graceful shutdown: {str(e)}")

    def print_final_metrics(self) -> None:
        """Print final performance metrics."""
        try:
            print("\n" + "=" * 50)
            print("SESSION METRICS")
            print("=" * 50)

            queries = self.metrics["queries_processed"]
            total_time = self.metrics["total_response_time"]
            citations = self.metrics["kg_citations_returned"]
            tokens = self.metrics["tokens_generated"]

            print(f"Queries processed: {queries}")
            print(f"Total response time: {total_time:.2f}s")
            print(
                f"Average response time: {total_time/queries:.2f}s"
                if queries > 0
                else "Average response time: N/A"
            )
            print(f"KG citations returned: {citations}")
            print(f"Tokens generated: {tokens}")
            print(f"Knowledge graph enabled: {self.use_kg}")
            print(f"Max triples per chunk: {self.max_triple_per_chunk}")

        except Exception as e:
            logger.error(f"Error printing final metrics: {str(e)}")

    async def run_repl_loop(self) -> None:
        """
        Run the main REPL loop with simple synchronous input.
        """
        try:
            print("GraphRAG REPL - Conversation + Knowledge Graph Assistant")
            print("=" * 60)
            print(f"Knowledge Graph: {'Enabled' if self.use_kg else 'Disabled (--no-kg)'}")
            print(f"Max Triples per Chunk: {self.max_triple_per_chunk}")
            print("Models: hermes3:3b (conversation), tinyllama:latest (knowledge)")
            print("Embedding: BAAI/bge-small-en-v1.5 int8")
            print("Type 'exit', 'quit', or press Ctrl+C to exit")
            print("=" * 60)

            while True:
                try:
                    # Get user input using simple synchronous input
                    user_input = self.get_user_input()

                    if user_input is None:
                        # Handle Ctrl+C or EOF
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
                    logger.error(f"Error in REPL loop: {str(e)}")
                    print(f"Error: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Fatal error in REPL loop: {str(e)}")
            raise
        finally:
            # Always handle cleanup
            await self.handle_keyboard_interrupt()

    async def run(self) -> None:
        """
        Main entry point for the Simple GraphRAG REPL.
        """
        try:
            # Initialize all components
            if not await self.initialize_all_components():
                logger.error("Failed to initialize components, exiting")
                return

            # Run the REPL loop
            await self.run_repl_loop()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            await self.handle_keyboard_interrupt()
        except Exception as e:
            logger.error(f"Fatal error in Simple GraphRAG REPL: {str(e)}")
            raise


async def main():
    """
    Main function to run the Simple GraphRAG REPL.
    """
    parser = argparse.ArgumentParser(
        description="Simple GraphRAG REPL - Robust CLI tool for conversation-plus-GraphRAG stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli/graphrag_repl_simple.py
    python cli/graphrag_repl_simple.py --no-kg
    python cli/graphrag_repl_simple.py --max-triple-per-chunk 6
    python cli/graphrag_repl_simple.py --citation-detail compact
    python cli/graphrag_repl_simple.py --citation-detail detailed --log-level DEBUG

Citation Detail Levels:
    minimal  - Just source names and scores: [1] document.txt (0.85)
    compact  - Source, score, and brief content snippet
    detailed - Full content, metadata, relationships, and statistics
        """,
    )

    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Bypass KG retrieval and extraction for benchmarking pure chat vs GraphRAG",
    )

    parser.add_argument(
        "--max-triple-per-chunk",
        type=int,
        default=4,
        help="Maximum triples per chunk for KG extraction (default: 4)",
    )

    parser.add_argument(
        "--prebuilt-dir",
        type=str,
        default="./data/prebuilt_graphs",
        help="Directory containing pre-built knowledge graphs (default: ./data/prebuilt_graphs)",
    )

    parser.add_argument(
        "--citation-detail",
        choices=["minimal", "compact", "detailed"],
        default="detailed",
        help="Level of detail for knowledge graph citations (default: detailed)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create and run the REPL
    repl = SimpleGraphRAGREPL(
        use_kg=not args.no_kg,
        max_triple_per_chunk=args.max_triple_per_chunk,
        citation_detail=args.citation_detail,
        prebuilt_dir=args.prebuilt_dir,
    )

    await repl.run()


if __name__ == "__main__":
    # Run with proper asyncio setup
    asyncio.run(main())
