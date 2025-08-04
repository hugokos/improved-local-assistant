#!/usr/bin/env python3
"""
GraphRAG REPL - Comprehensive CLI tool for exercising the full conversation-plus-GraphRAG stack.

This tool provides an asynchronous REPL loop using asyncio and prompt_toolkit that:
- Instantiates ModelManager with conversation_model="hermes3:3b" and knowledge_model="tinyllama:latest"
- Instantiates KnowledgeGraphManager in lazy-load mode with system-context bootstrap
- Streams assistant responses token by token to stdout
- Prints compact reference lists of KG citations
- Supports --no-kg flag to bypass KG retrieval for benchmarking
- Handles KeyboardInterrupt gracefully with proper persistence
- Supports --max-triple-per-chunk CLI option for tuning extraction load
- Includes robust error logging for debugging

Usage:
    python -m asyncio cli.graphrag_repl
    python -m asyncio cli.graphrag_repl --no-kg
    python -m asyncio cli.graphrag_repl --max-triple-per-chunk 6
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path to import from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('graphrag_repl.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Mute resource manager warnings to avoid interrupting chat
resource_logger = logging.getLogger('services.resource_manager')
resource_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Also mute system monitor warnings
system_monitor_logger = logging.getLogger('services.system_monitor')
system_monitor_logger.setLevel(logging.ERROR)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Import prompt_toolkit with graceful fallback
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    logger.warning("prompt_toolkit not available, falling back to input()")
    PROMPT_TOOLKIT_AVAILABLE = False

# Import services
from services.model_mgr import ModelManager, ModelConfig
from services.graph_manager import KnowledgeGraphManager
from services.conversation_manager import ConversationManager
from services.resource_manager import ResourceManager

# Import conversation extensions to add get_response method
import services.conversation_extensions

# Import embedding model singleton
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    EMBEDDING_AVAILABLE = True
except ImportError:
    logger.warning("HuggingFaceEmbedding not available, will use default embeddings")
    EMBEDDING_AVAILABLE = False


class GraphRAGREPL:
    """
    Comprehensive GraphRAG REPL that exercises the full conversation-plus-GraphRAG stack.
    """
    
    def __init__(self, use_kg: bool = True, max_triple_per_chunk: int = 4):
        """
        Initialize the GraphRAG REPL.
        
        Args:
            use_kg: Whether to use knowledge graph retrieval and extraction
            max_triple_per_chunk: Maximum triples per chunk for KG extraction
        """
        self.use_kg = use_kg
        self.max_triple_per_chunk = max_triple_per_chunk
        
        # Initialize components
        self.model_manager: Optional[ModelManager] = None
        self.kg_manager: Optional[KnowledgeGraphManager] = None
        self.conversation_manager: Optional[ConversationManager] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.session_id: Optional[str] = None
        
        # Embedding model singleton
        self.embedding_model = None
        
        # Performance tracking
        self.metrics = {
            "queries_processed": 0,
            "total_response_time": 0.0,
            "kg_citations_returned": 0,
            "tokens_generated": 0
        }
        
        # Setup prompt session if available
        if PROMPT_TOOLKIT_AVAILABLE:
            self.prompt_session = PromptSession()
        else:
            self.prompt_session = None
            
        logger.info(f"GraphRAG REPL initialized with use_kg={use_kg}, max_triple_per_chunk={max_triple_per_chunk}")

    async def initialize_embedding_model(self) -> bool:
        """
        Initialize the singleton embedding model that loads BAAI/bge-small-en-v1.5.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing singleton embedding model: BAAI/bge-small-en-v1.5")
            
            # Use the embedding singleton to ensure consistency
            from services.embedding_singleton import get_embedding_model, configure_global_embedding
            
            # Configure global embedding using singleton
            configure_global_embedding("BAAI/bge-small-en-v1.5")
            
            # Get the singleton instance
            self.embedding_model = get_embedding_model("BAAI/bge-small-en-v1.5")
            
            logger.info("Successfully initialized singleton embedding model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            # Try fallback initialization
            try:
                if EMBEDDING_AVAILABLE:
                    self.embedding_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-small-en-v1.5",
                        device="cpu",
                        trust_remote_code=False
                    )
                    
                    from llama_index.core import Settings
                    Settings.embed_model = self.embedding_model
                    
                    logger.info("Fallback embedding initialization successful")
                    return True
            except Exception as fallback_error:
                logger.error(f"Fallback embedding initialization failed: {fallback_error}")
            
            return False

    async def initialize_model_manager(self) -> bool:
        """
        Initialize ModelManager with conversation_model="hermes3:3b" and knowledge_model="tinyllama:latest".
        Ensure it calls the Ollama daemon running on localhost:11434.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing ModelManager with hermes3:3b and tinyllama:latest")
            
            # Initialize ModelManager with Ollama daemon on localhost:11434
            self.model_manager = ModelManager(
                host="http://localhost:11434",
                healthcheck_mode="version"  # Use light healthcheck for faster startup
            )
            
            # Override model names to ensure we use the specified models
            self.model_manager.conversation_model = "hermes3:3b"
            self.model_manager.knowledge_model = "tinyllama:latest"
            
            # Create model configuration
            model_config = ModelConfig(
                name="hermes3:3b",  # This will be overridden by the manager
                type="conversation",
                context_window=8000,
                temperature=0.7,
                max_tokens=2048,
                timeout=120,
                max_parallel=2,
                max_loaded=2
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
        Initialize KnowledgeGraphManager in lazy-load mode so it boots even if no graphs are on disk.
        Seed it with the system-context bootstrap document so kg_index is never None.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            if not self.use_kg:
                logger.info("Skipping KnowledgeGraphManager initialization (--no-kg flag)")
                return True
                
            logger.info("Initializing KnowledgeGraphManager in lazy-load mode")
            
            # Create configuration for KG manager
            kg_config = {
                "knowledge_graphs": {
                    "prebuilt_directory": "./data/prebuilt_graphs",
                    "dynamic_storage": "./data/dynamic_graph",
                    "max_triplets_per_chunk": self.max_triple_per_chunk,
                    "enable_visualization": False,  # Disabled for CLI performance
                    "enable_caching": True
                },
                "models": {
                    "conversation": {
                        "name": "hermes3:3b",
                        "context_window": 8000,
                        "temperature": 0.7,
                        "max_tokens": 2048
                    },
                    "knowledge": {
                        "name": "tinyllama:latest",
                        "context_window": 2048,
                        "temperature": 0.2,
                        "max_tokens": 1024
                    }
                },
                "ollama": {
                    "host": "http://localhost:11434",
                    "timeout": 120
                }
            }
            
            # Initialize KG manager
            self.kg_manager = KnowledgeGraphManager(self.model_manager, kg_config)
            
            # Initialize dynamic graph with system-context bootstrap document
            await self.initialize_dynamic_graph_with_bootstrap()
            
            # Load pre-built graphs in lazy-load mode (don't fail if none exist)
            try:
                loaded_graphs = self.kg_manager.load_prebuilt_graphs()
                if loaded_graphs:
                    logger.info(f"Loaded {len(loaded_graphs)} pre-built knowledge graphs")
                    
                    # Add smoke test guards
                    self._validate_loaded_graphs(loaded_graphs)
                else:
                    logger.info("No pre-built knowledge graphs found (lazy-load mode)")
            except Exception as e:
                logger.warning(f"Failed to load pre-built graphs (continuing in lazy-load mode): {str(e)}")
            
            logger.info("Successfully initialized KnowledgeGraphManager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphManager: {str(e)}")
            return False

    async def initialize_dynamic_graph_with_bootstrap(self) -> bool:
        """
        Initialize dynamic graph with system-context bootstrap document so kg_index is never None.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing dynamic graph with system-context bootstrap document")
            
            # Create system-context bootstrap document
            bootstrap_text = f"""
            System Context Bootstrap Document
            
            This is a system-generated bootstrap document for the dynamic knowledge graph.
            It ensures that the knowledge graph index (kg_index) is never None and provides
            initial context for the GraphRAG system.
            
            Initialization Details:
            - Timestamp: {datetime.now().isoformat()}
            - System: GraphRAG REPL CLI Tool
            - Models: hermes3:3b (conversation), tinyllama:latest (knowledge extraction)
            - Embedding Model: BAAI/bge-small-en-v1.5 int8
            - Max Triples Per Chunk: {self.max_triple_per_chunk}
            - Knowledge Graph Enabled: {self.use_kg}
            
            This document serves as the foundation for dynamic knowledge graph construction
            and ensures that the system can always perform knowledge graph operations
            even when starting with an empty graph store.
            
            Key Concepts:
            - GraphRAG: Graph-based Retrieval Augmented Generation
            - Knowledge Extraction: Process of extracting entities and relationships from text
            - Dynamic Graph: Knowledge graph that grows and evolves with conversation
            - System Bootstrap: Initial seeding of the knowledge graph with system context
            """
            
            # Initialize dynamic graph with bootstrap document
            from llama_index.core import Document
            bootstrap_doc = Document(
                text=bootstrap_text,
                metadata={
                    "source": "system_bootstrap",
                    "type": "initialization",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0"
                }
            )
            
            # Create the dynamic graph with bootstrap document
            from llama_index.core import KnowledgeGraphIndex, StorageContext
            from llama_index.core.graph_stores import SimpleGraphStore
            
            # Use SimpleGraphStore without custom filesystem for compatibility
            graph_store = SimpleGraphStore()
            storage_ctx = StorageContext.from_defaults(graph_store=graph_store)
            
            self.kg_manager.dynamic_kg = KnowledgeGraphIndex.from_documents(
                [bootstrap_doc],
                storage_context=storage_ctx,
                max_triplets_per_chunk=self.max_triple_per_chunk,
                show_progress=False
            )
            
            # Persist the bootstrap graph
            persist_dir = os.path.join("./data/dynamic_graph", "main")
            os.makedirs(persist_dir, exist_ok=True)
            self.kg_manager.dynamic_kg.storage_context.persist(persist_dir=persist_dir)
            
            logger.info("Successfully initialized dynamic graph with system-context bootstrap")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dynamic graph with bootstrap: {str(e)}")
            # Try fallback initialization
            try:
                success = self.kg_manager.initialize_dynamic_graph()
                if success:
                    logger.info("Fallback dynamic graph initialization successful")
                    return True
            except Exception as fallback_error:
                logger.error(f"Fallback initialization also failed: {str(fallback_error)}")
            return False

    async def initialize_conversation_manager(self) -> bool:
        """
        Initialize ConversationManager and create a session.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing ConversationManager")
            
            # Create configuration for conversation manager
            conv_config = {
                "conversation": {
                    "max_history_length": 50,
                    "summarize_threshold": 20,
                    "context_window_tokens": 8000
                }
            }
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager(
                self.model_manager,
                self.kg_manager,
                conv_config
            )
            
            # Create a session
            self.session_id = self.conversation_manager.create_session()
            
            logger.info(f"Successfully initialized ConversationManager with session {self.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ConversationManager: {str(e)}")
            return False

    async def initialize_resource_manager(self) -> bool:
        """
        Initialize ResourceManager for performance monitoring.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            logger.info("Initializing ResourceManager")
            
            # Create configuration for resource manager
            resource_config = {
                "system": {
                    "max_memory_gb": 12,
                    "cpu_cores": 4,
                    "memory_threshold_percent": 80,
                    "cpu_threshold_percent": 80
                }
            }
            
            # Initialize resource manager
            self.resource_manager = ResourceManager(resource_config)
            
            # Start monitoring
            await self.resource_manager.start_monitoring()
            
            logger.info("Successfully initialized ResourceManager")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ResourceManager: {str(e)}")
            return False

    async def initialize_all_components(self) -> bool:
        """
        Initialize all components in the correct order.
        
        Returns:
            bool: True if all components were initialized successfully
        """
        try:
            logger.info("Starting component initialization...")
            
            # Initialize embedding model singleton first
            await self.initialize_embedding_model()
            
            # Initialize model manager
            if not await self.initialize_model_manager():
                return False
            
            # Initialize KG manager
            if not await self.initialize_kg_manager():
                return False
            
            # Initialize conversation manager
            if not await self.initialize_conversation_manager():
                return False
            
            # Initialize resource manager
            if not await self.initialize_resource_manager():
                return False
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            return False

    async def get_user_input(self, prompt: str = "You: ") -> Optional[str]:
        """
        Get user input using prompt_toolkit if available, otherwise fallback to input().
        
        Args:
            prompt: Prompt string to display
            
        Returns:
            Optional[str]: User input or None if interrupted
        """
        try:
            if self.prompt_session:
                # Use prompt_toolkit for better experience
                with patch_stdout():
                    return await self.prompt_session.prompt_async(prompt)
            else:
                # Fallback to regular input
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
                self.session_id, 
                user_input, 
                use_kg=self.use_kg
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
        Print a compact reference list of any KG citations the pipeline returned.
        Format: [1] source_title_or_id
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
                
            print("\nReferences:")
            for citation in citations:
                citation_id = citation.get("id", "?")
                source = citation.get("source", "Unknown")
                score = citation.get("score", 0.0)
                
                # Create compact reference format
                print(f"[{citation_id}] {source} (score: {score:.2f})")
            
            # Update metrics
            self.metrics["kg_citations_returned"] += len(citations)
            
        except Exception as e:
            logger.error(f"Error printing KG citations: {str(e)}")

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
            
            # Persist conversation sessions (if there's a persistence mechanism)
            if self.conversation_manager:
                try:
                    # Note: The current ConversationManager doesn't have built-in persistence
                    # In a production system, you would implement session persistence here
                    session_count = len(self.conversation_manager.sessions)
                    logger.info(f"Conversation sessions in memory: {session_count}")
                except Exception as e:
                    logger.error(f"Error handling conversation sessions: {str(e)}")
            
            # Stop resource monitoring
            if self.resource_manager:
                try:
                    await self.resource_manager.stop_monitoring()
                    logger.info("Stopped resource monitoring")
                except Exception as e:
                    logger.error(f"Error stopping resource monitoring: {str(e)}")
            
            # Print final metrics
            self.print_final_metrics()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {str(e)}")

    def _validate_loaded_graphs(self, loaded_graphs: list) -> None:
        """Add smoke test guards to validate graph quality."""
        try:
            for graph_id in loaded_graphs:
                if graph_id in self.kg_manager.kg_indices:
                    kg_index = self.kg_manager.kg_indices[graph_id]
                    
                    # Get graph statistics
                    try:
                        # Check if we can get some basic stats
                        if hasattr(kg_index, 'storage_context'):
                            storage_ctx = kg_index.storage_context
                            
                            # Check document store
                            if hasattr(storage_ctx, 'docstore') and hasattr(storage_ctx.docstore, 'docs'):
                                num_docs = len(storage_ctx.docstore.docs)
                                assert num_docs > 0, f"Graph {graph_id} has no documents"
                                logger.info(f"Graph {graph_id}: {num_docs} documents")
                            
                            # Check graph store
                            if hasattr(storage_ctx, 'graph_store') and hasattr(storage_ctx.graph_store, 'rel_map'):
                                num_relations = sum(len(v) for v in storage_ctx.graph_store.rel_map.values())
                                assert num_relations > 10, f"Graph {graph_id} too small: {num_relations} relations"
                                logger.info(f"Graph {graph_id}: {num_relations} relations")
                            
                            # Check vector store with smoke test guard
                            if hasattr(storage_ctx, 'vector_store'):
                                try:
                                    # Try to get vector count (different methods for different stores)
                                    vector_count = 0
                                    if hasattr(storage_ctx.vector_store, '_collection'):
                                        if hasattr(storage_ctx.vector_store._collection, 'count'):
                                            vector_count = storage_ctx.vector_store._collection.count()
                                    elif hasattr(storage_ctx.vector_store, 'get_all'):
                                        vector_count = len(storage_ctx.vector_store.get_all())
                                    
                                    if vector_count > 0:
                                        logger.info(f"Graph {graph_id}: {vector_count} vectors")
                                        
                                        # Smoke test guard: check if embeddings match graph nodes
                                        if hasattr(storage_ctx, 'graph_store') and hasattr(storage_ctx.graph_store, 'rel_map'):
                                            num_relations = sum(len(v) for v in storage_ctx.graph_store.rel_map.values())
                                            if vector_count == 1 and num_relations > 100:
                                                logger.warning(f"Graph {graph_id}: Vector/node mismatch - {vector_count} vectors vs {num_relations} relations")
                                                logger.warning("This suggests embeddings weren't properly generated during graph build")
                                    else:
                                        logger.warning(f"Graph {graph_id}: No vectors found - semantic search may not work")
                                        
                                except Exception as ve:
                                    logger.debug(f"Could not check vector count for {graph_id}: {ve}")
                    
                    except Exception as e:
                        logger.warning(f"Could not validate graph {graph_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during graph validation: {e}")

    def print_final_metrics(self) -> None:
        """Print final performance metrics."""
        try:
            print("\n" + "="*50)
            print("SESSION METRICS")
            print("="*50)
            
            queries = self.metrics["queries_processed"]
            total_time = self.metrics["total_response_time"]
            citations = self.metrics["kg_citations_returned"]
            tokens = self.metrics["tokens_generated"]
            
            print(f"Queries processed: {queries}")
            print(f"Total response time: {total_time:.2f}s")
            print(f"Average response time: {total_time/queries:.2f}s" if queries > 0 else "Average response time: N/A")
            print(f"KG citations returned: {citations}")
            print(f"Tokens generated: {tokens}")
            print(f"Knowledge graph enabled: {self.use_kg}")
            print(f"Max triples per chunk: {self.max_triple_per_chunk}")
            
            # Resource usage if available
            if self.resource_manager:
                try:
                    resource_usage = self.resource_manager.get_resource_usage()
                    print(f"Final CPU usage: {resource_usage['cpu_percent']:.1f}%")
                    print(f"Final memory usage: {resource_usage['memory_percent']:.1f}%")
                except Exception as e:
                    logger.debug(f"Could not get final resource usage: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error printing final metrics: {str(e)}")

    async def run_repl_loop(self) -> None:
        """
        Run the main asynchronous REPL loop using asyncio and prompt_toolkit.
        """
        try:
            print("GraphRAG REPL - Conversation + Knowledge Graph Assistant")
            print("="*60)
            print(f"Knowledge Graph: {'Enabled' if self.use_kg else 'Disabled (--no-kg)'}")
            print(f"Max Triples per Chunk: {self.max_triple_per_chunk}")
            print(f"Models: hermes3:3b (conversation), tinyllama:latest (knowledge)")
            print(f"Embedding: BAAI/bge-small-en-v1.5 int8")
            print("Type 'exit', 'quit', or press Ctrl+C to exit")
            print("="*60)
            
            while True:
                try:
                    # Get user input
                    user_input = await self.get_user_input()
                    
                    if user_input is None:
                        # Handle Ctrl+C or EOF
                        break
                    
                    user_input = user_input.strip()
                    
                    if not user_input:
                        continue
                    
                    # Check for exit commands
                    if user_input.lower() in ['exit', 'quit', 'bye']:
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
        Main entry point for the GraphRAG REPL.
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
            logger.error(f"Fatal error in GraphRAG REPL: {str(e)}")
            raise


async def main():
    """
    Main function to run the GraphRAG REPL.
    Target Python 3.11, rely only on packages already in requirements.txt.
    """
    parser = argparse.ArgumentParser(
        description="GraphRAG REPL - Comprehensive CLI tool for conversation-plus-GraphRAG stack",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m asyncio cli.graphrag_repl
    python -m asyncio cli.graphrag_repl --no-kg
    python -m asyncio cli.graphrag_repl --max-triple-per-chunk 6
        """
    )
    
    parser.add_argument(
        "--no-kg",
        action="store_true",
        help="Bypass KG retrieval and extraction for benchmarking pure chat vs GraphRAG"
    )
    
    parser.add_argument(
        "--max-triple-per-chunk",
        type=int,
        default=4,
        help="Maximum triples per chunk for KG extraction (default: 4)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run the REPL
    repl = GraphRAGREPL(
        use_kg=not args.no_kg,
        max_triple_per_chunk=args.max_triple_per_chunk
    )
    
    await repl.run()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())