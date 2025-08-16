#!/usr/bin/env python3
"""
Test script for long conversation persistence and dynamic graph storage.

This script tests:
1. Dynamic graph updates during extended conversations
2. Persistence of extracted entities and relationships
3. Storage structure and data integrity
4. Graph growth over time
5. Query performance with accumulated knowledge

Usage:
    python scripts/test_long_conversation_persistence.py --messages 50 --inspect
    python scripts/test_long_conversation_persistence.py --interactive
    python scripts/test_long_conversation_persistence.py --analyze-storage
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from services.conversation_manager import ConversationManager
from services.embedding_singleton import configure_global_embedding
from services.graph_manager import KnowledgeGraphManager
from services.model_mgr import ModelConfig
from services.model_mgr import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ConversationPersistenceTester:
    """Tests dynamic graph persistence during long conversations."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_manager = None
        self.kg_manager = None
        self.conversation_manager = None
        self.session_id = None
        self.test_messages = []
        self.storage_snapshots = []

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing components for persistence testing...")

        # Configure embedding model
        configure_global_embedding("BAAI/bge-small-en-v1.5")

        # Initialize model manager
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
        if not await self.model_manager.initialize_models(model_config):
            raise Exception("Failed to initialize models")

        # Initialize knowledge graph manager
        self.kg_manager = KnowledgeGraphManager(self.model_manager, self.config)

        # Load pre-built graphs
        loaded_graphs = self.kg_manager.load_prebuilt_graphs()
        logger.info(f"Loaded {len(loaded_graphs)} pre-built knowledge graphs")

        # Initialize dynamic graph
        if not self.kg_manager.initialize_dynamic_graph():
            raise Exception("Failed to initialize dynamic graph")

        # Create conversation manager
        self.conversation_manager = ConversationManager(
            self.model_manager, self.kg_manager, self.config
        )

        # Create session
        self.session_id = self.conversation_manager.create_session()
        logger.info(f"Created test session: {self.session_id}")

    def generate_test_messages(self, count: int = 50) -> list[str]:
        """Generate diverse test messages for conversation."""
        messages = [
            # Basic introductions and context setting
            "Hello, my name is Alex and I'm interested in learning about survival skills.",
            "I'm planning a camping trip next month in the mountains.",
            "What are the most important things to pack for wilderness survival?",
            # Fire and shelter topics
            "How do I start a fire without matches?",
            "What materials make the best kindling?",
            "Can you explain the bow drill method for fire starting?",
            "What types of shelter work best in cold weather?",
            "How do I build a lean-to shelter?",
            "What's the difference between a debris hut and a lean-to?",
            # Water and food topics
            "How can I find clean water in the wilderness?",
            "What are the best water purification methods?",
            "Which plants are safe to eat in North America?",
            "How do I identify edible berries?",
            "What are some basic hunting techniques for small game?",
            "How do I set up simple traps for rabbits?",
            # Navigation and signaling
            "How do I navigate without a compass?",
            "What are the best ways to signal for rescue?",
            "How can I use the stars for navigation?",
            "What should I do if I get lost in the woods?",
            # First aid and health
            "What are the most important first aid skills for wilderness survival?",
            "How do I treat cuts and wounds in the field?",
            "What plants can be used for natural medicine?",
            "How do I recognize and treat hypothermia?",
            # Weather and environmental awareness
            "How can I predict weather changes in the wilderness?",
            "What are the signs of an approaching storm?",
            "How do I stay warm in freezing temperatures?",
            "What's the best way to stay cool in hot desert conditions?",
            # Tool making and crafting
            "How do I make tools from natural materials?",
            "What's the best way to sharpen a knife in the field?",
            "How can I make rope from plant fibers?",
            "What are some basic knots every survivalist should know?",
            # Psychological aspects
            "How do I stay mentally strong in survival situations?",
            "What's the survival rule of threes?",
            "How do I prioritize tasks in an emergency?",
            "What are common mistakes people make in survival situations?",
            # Advanced topics
            "How do I preserve meat without refrigeration?",
            "What are some advanced fire techniques?",
            "How do I make charcoal for water filtration?",
            "What's the best way to smoke fish for preservation?",
            # Seasonal considerations
            "How does survival strategy change in winter?",
            "What are the unique challenges of summer survival?",
            "How do I prepare for survival in different climates?",
            "What gear is essential for each season?",
            # References to previous topics
            "Going back to fire starting, what if the bow drill method doesn't work?",
            "You mentioned water purification earlier - can you elaborate on boiling techniques?",
            "Regarding the shelter we discussed, how do I insulate it properly?",
            "About those edible plants you mentioned, which ones are available in winter?",
            # Complex scenarios
            "If I'm lost in a snowstorm with limited supplies, what's my priority order?",
            "How would survival strategy differ if I'm injured and alone?",
            "What if I need to survive for weeks instead of just days?",
            "How do I balance energy conservation with the need to find resources?",
        ]

        # Extend with additional messages if needed
        while len(messages) < count:
            messages.extend(
                [
                    f"Can you tell me more about the topic we discussed {len(messages) // 10} messages ago?",
                    f"I have a follow-up question about survival technique number {len(messages)}.",
                    "What's your opinion on the survival method we talked about earlier?",
                    "How does this relate to what we discussed about wilderness survival?",
                ]
            )

        return messages[:count]

    async def take_storage_snapshot(self, label: str) -> dict[str, Any]:
        """Take a snapshot of current storage state."""
        logger.info(f"Taking storage snapshot: {label}")

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "dynamic_graph": {},
            "prebuilt_graphs": {},
            "session_data": {},
            "file_sizes": {},
        }

        # Dynamic graph storage analysis
        dynamic_path = Path("data/dynamic_graph/main")
        if dynamic_path.exists():
            snapshot["dynamic_graph"] = await self._analyze_graph_storage(dynamic_path)

        # Prebuilt graphs analysis
        prebuilt_path = Path("data/prebuilt_graphs")
        if prebuilt_path.exists():
            for graph_dir in prebuilt_path.iterdir():
                if graph_dir.is_dir():
                    snapshot["prebuilt_graphs"][graph_dir.name] = await self._analyze_graph_storage(
                        graph_dir
                    )

        # Session data
        if self.session_id and self.conversation_manager:
            session = self.conversation_manager.sessions.get(self.session_id, {})
            snapshot["session_data"] = {
                "message_count": len(session.get("messages", [])),
                "metadata": session.get("metadata", {}),
                "has_summary": bool(session.get("summary")),
            }

        # KG manager metrics
        if self.kg_manager:
            snapshot["kg_metrics"] = self.kg_manager.metrics.copy()

        self.storage_snapshots.append(snapshot)
        return snapshot

    async def _analyze_graph_storage(self, graph_path: Path) -> dict[str, Any]:
        """Analyze storage files in a graph directory."""
        analysis = {
            "path": str(graph_path),
            "files": {},
            "total_size": 0,
            "node_count": 0,
            "edge_count": 0,
            "document_count": 0,
        }

        # Analyze each storage file
        storage_files = [
            "docstore.json",
            "graph_store.json",
            "property_graph_store.json",
            "default__vector_store.json",
            "image__vector_store.json",
            "index_store.json",
        ]

        for filename in storage_files:
            file_path = graph_path / filename
            if file_path.exists():
                try:
                    size = file_path.stat().st_size
                    analysis["files"][filename] = {"size": size, "exists": True}
                    analysis["total_size"] += size

                    # Try to extract counts from specific files
                    if filename == "docstore.json":
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                docstore = json.load(f)
                            analysis["document_count"] = len(docstore.get("docstore/data", {}))
                        except Exception as e:
                            logger.debug(f"Could not parse {filename}: {e}")

                    elif filename == "graph_store.json":
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                graph_store = json.load(f)
                            # Count relationships
                            rel_map = graph_store.get("graph_store/data", {}).get("rel_map", {})
                            analysis["edge_count"] = sum(len(rels) for rels in rel_map.values())
                        except Exception as e:
                            logger.debug(f"Could not parse {filename}: {e}")

                    elif filename == "property_graph_store.json":
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                prop_store = json.load(f)
                            nodes = prop_store.get("nodes", [])
                            edges = prop_store.get("edges", [])
                            analysis["node_count"] = len(nodes)
                            analysis["edge_count"] = len(edges)
                        except Exception as e:
                            logger.debug(f"Could not parse {filename}: {e}")

                except Exception as e:
                    analysis["files"][filename] = {"size": 0, "exists": True, "error": str(e)}
            else:
                analysis["files"][filename] = {"size": 0, "exists": False}

        return analysis

    async def run_long_conversation_test(self, message_count: int = 50, inspect_interval: int = 10):
        """Run a long conversation test with periodic storage inspection."""
        logger.info(f"Starting long conversation test with {message_count} messages")

        # Generate test messages
        self.test_messages = self.generate_test_messages(message_count)

        # Take initial snapshot
        await self.take_storage_snapshot("initial")

        # Process messages with periodic snapshots
        for i, message in enumerate(self.test_messages):
            logger.info(f"Processing message {i+1}/{message_count}: {message[:50]}...")

            try:
                # Process the message
                response_tokens = []
                async for token in self.conversation_manager.converse_with_context(
                    self.session_id, message
                ):
                    response_tokens.append(token)

                response = "".join(response_tokens)
                logger.info(f"Response length: {len(response)} characters")

                # Take snapshot at intervals
                if (i + 1) % inspect_interval == 0:
                    await self.take_storage_snapshot(f"after_{i+1}_messages")

                    # Force persistence of dynamic graph
                    if self.kg_manager.dynamic_kg:
                        await self.kg_manager._persist_dynamic_graph()
                        logger.info("Forced dynamic graph persistence")

                # Small delay to allow background processing
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing message {i+1}: {e}")
                continue

        # Take final snapshot
        await self.take_storage_snapshot("final")

        # Force final persistence
        if self.kg_manager.dynamic_kg:
            await self.kg_manager._persist_dynamic_graph()

        logger.info("Long conversation test completed")

    def analyze_storage_growth(self):
        """Analyze how storage grew during the conversation."""
        logger.info("Analyzing storage growth patterns...")

        if len(self.storage_snapshots) < 2:
            logger.warning("Need at least 2 snapshots to analyze growth")
            return

        print("\n=== STORAGE GROWTH ANALYSIS ===")

        # Compare snapshots
        initial = self.storage_snapshots[0]
        final = self.storage_snapshots[-1]

        print(f"Time span: {initial['timestamp']} to {final['timestamp']}")

        # Dynamic graph growth
        if "dynamic_graph" in initial and "dynamic_graph" in final:
            initial_dg = initial["dynamic_graph"]
            final_dg = final["dynamic_graph"]

            print("\nDynamic Graph Growth:")
            print(
                f"  Total size: {initial_dg.get('total_size', 0):,} → {final_dg.get('total_size', 0):,} bytes"
            )
            print(
                f"  Node count: {initial_dg.get('node_count', 0)} → {final_dg.get('node_count', 0)}"
            )
            print(
                f"  Edge count: {initial_dg.get('edge_count', 0)} → {final_dg.get('edge_count', 0)}"
            )
            print(
                f"  Document count: {initial_dg.get('document_count', 0)} → {final_dg.get('document_count', 0)}"
            )

        # Session data growth
        if "session_data" in initial and "session_data" in final:
            initial_session = initial["session_data"]
            final_session = final["session_data"]

            print("\nSession Data Growth:")
            print(
                f"  Message count: {initial_session.get('message_count', 0)} → {final_session.get('message_count', 0)}"
            )
            print(
                f"  Has summary: {initial_session.get('has_summary', False)} → {final_session.get('has_summary', False)}"
            )

        # KG metrics comparison
        if "kg_metrics" in initial and "kg_metrics" in final:
            initial_metrics = initial["kg_metrics"]
            final_metrics = final["kg_metrics"]

            print("\nKnowledge Graph Metrics:")
            print(
                f"  Queries processed: {initial_metrics.get('queries_processed', 0)} → {final_metrics.get('queries_processed', 0)}"
            )
            print(
                f"  Total nodes: {initial_metrics.get('total_nodes', 0)} → {final_metrics.get('total_nodes', 0)}"
            )
            print(
                f"  Total edges: {initial_metrics.get('total_edges', 0)} → {final_metrics.get('total_edges', 0)}"
            )

        # File-by-file analysis
        print("\nFile Growth Analysis:")
        for snapshot in self.storage_snapshots:
            label = snapshot["label"]
            dg = snapshot.get("dynamic_graph", {})
            total_size = dg.get("total_size", 0)
            print(f"  {label}: {total_size:,} bytes")

        # Growth rate calculation
        if len(self.storage_snapshots) > 2:
            print("\nGrowth Rate Analysis:")
            for i in range(1, len(self.storage_snapshots)):
                prev = self.storage_snapshots[i - 1]
                curr = self.storage_snapshots[i]

                prev_size = prev.get("dynamic_graph", {}).get("total_size", 0)
                curr_size = curr.get("dynamic_graph", {}).get("total_size", 0)

                growth = curr_size - prev_size
                print(f"  {prev['label']} → {curr['label']}: +{growth:,} bytes")

    def inspect_storage_files(self):
        """Inspect the actual storage files in detail."""
        logger.info("Inspecting storage files in detail...")

        print("\n=== DETAILED STORAGE INSPECTION ===")

        # Dynamic graph files
        dynamic_path = Path("data/dynamic_graph/main")
        if dynamic_path.exists():
            print(f"\nDynamic Graph Storage ({dynamic_path}):")
            self._inspect_directory(dynamic_path)

        # Prebuilt graph files
        prebuilt_path = Path("data/prebuilt_graphs")
        if prebuilt_path.exists():
            print(f"\nPrebuilt Graphs Storage ({prebuilt_path}):")
            for graph_dir in prebuilt_path.iterdir():
                if graph_dir.is_dir():
                    print(f"\n  {graph_dir.name}:")
                    self._inspect_directory(graph_dir, indent="    ")

        # Session storage
        sessions_path = Path("data/sessions")
        if sessions_path.exists():
            print(f"\nSession Storage ({sessions_path}):")
            self._inspect_directory(sessions_path)

    def _inspect_directory(self, directory: Path, indent: str = "  "):
        """Inspect files in a directory."""
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    print(f"{indent}{file_path.name}: {size:,} bytes (modified: {modified})")

                    # Show sample content for JSON files
                    if file_path.suffix == ".json" and size < 10000:  # Only small files
                        try:
                            with open(file_path, encoding="utf-8") as f:
                                data = json.load(f)

                            if isinstance(data, dict):
                                keys = list(data.keys())[:5]  # First 5 keys
                                print(f"{indent}  Keys: {keys}")
                            elif isinstance(data, list):
                                print(f"{indent}  Items: {len(data)}")

                        except Exception as e:
                            print(f"{indent}  (Could not parse JSON: {e})")

                except Exception as e:
                    print(f"{indent}{file_path.name}: Error reading file - {e}")

    async def test_query_performance(self):
        """Test query performance with accumulated knowledge."""
        logger.info("Testing query performance with accumulated knowledge...")

        if not self.kg_manager:
            logger.error("KG manager not initialized")
            return

        test_queries = [
            "How do I start a fire?",
            "What are the best water purification methods?",
            "How do I build a shelter?",
            "What plants are safe to eat?",
            "How do I navigate without a compass?",
        ]

        print("\n=== QUERY PERFORMANCE TEST ===")

        for query in test_queries:
            print(f"\nQuery: {query}")

            start_time = time.time()
            try:
                result = await self.kg_manager.query_graphs(query)
                elapsed = time.time() - start_time

                response = result.get("response", "")
                source_nodes = result.get("source_nodes", [])
                metadata = result.get("metadata", {})

                print(f"  Response time: {elapsed:.3f}s")
                print(f"  Response length: {len(response)} characters")
                print(f"  Source nodes: {len(source_nodes)}")
                print(f"  Graph sources: {metadata.get('graph_sources', [])}")
                print(f"  Response preview: {response[:100]}...")

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  Error after {elapsed:.3f}s: {e}")

    async def interactive_inspection(self):
        """Interactive mode for inspecting storage and testing queries."""
        logger.info("Starting interactive inspection mode...")

        print("\n=== INTERACTIVE STORAGE INSPECTION ===")
        print("Available commands:")
        print("  /snapshot - Take a storage snapshot")
        print("  /analyze - Analyze storage growth")
        print("  /inspect - Inspect storage files")
        print("  /query <text> - Test a query")
        print("  /stats - Show current statistics")
        print("  /persist - Force persistence")
        print("  /help - Show this help")
        print("  /exit - Exit interactive mode")

        while True:
            try:
                user_input = input("\nInspection> ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    command_parts = user_input.split(None, 1)
                    command = command_parts[0].lower()
                    args = command_parts[1] if len(command_parts) > 1 else ""

                    if command == "/exit":
                        break
                    elif command == "/help":
                        print("Available commands:")
                        print("  /snapshot - Take a storage snapshot")
                        print("  /analyze - Analyze storage growth")
                        print("  /inspect - Inspect storage files")
                        print("  /query <text> - Test a query")
                        print("  /stats - Show current statistics")
                        print("  /persist - Force persistence")
                        print("  /help - Show this help")
                        print("  /exit - Exit interactive mode")
                    elif command == "/snapshot":
                        await self.take_storage_snapshot(
                            f"interactive_{len(self.storage_snapshots)}"
                        )
                        print("Storage snapshot taken")
                    elif command == "/analyze":
                        self.analyze_storage_growth()
                    elif command == "/inspect":
                        self.inspect_storage_files()
                    elif command == "/query":
                        if not args:
                            print("Please provide a query text")
                            continue

                        print(f"Querying: {args}")
                        start_time = time.time()
                        try:
                            result = await self.kg_manager.query_graphs(args)
                            elapsed = time.time() - start_time

                            response = result.get("response", "")
                            print(f"Response ({elapsed:.3f}s): {response}")
                        except Exception as e:
                            elapsed = time.time() - start_time
                            print(f"Error ({elapsed:.3f}s): {e}")
                    elif command == "/stats":
                        if self.kg_manager:
                            print("KG Manager Metrics:")
                            print(json.dumps(self.kg_manager.metrics, indent=2))
                        if self.conversation_manager:
                            print("Conversation Manager Metrics:")
                            print(json.dumps(self.conversation_manager.metrics, indent=2))
                    elif command == "/persist":
                        if self.kg_manager and self.kg_manager.dynamic_kg:
                            await self.kg_manager._persist_dynamic_graph()
                            print("Dynamic graph persisted")
                        else:
                            print("No dynamic graph to persist")
                    else:
                        print(f"Unknown command: {command}")
                else:
                    # Treat as a conversation message
                    if self.conversation_manager and self.session_id:
                        print("Assistant: ", end="", flush=True)
                        async for token in self.conversation_manager.converse_with_context(
                            self.session_id, user_input
                        ):
                            print(token, end="", flush=True)
                        print()
                    else:
                        print("Conversation manager not available")

            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
            except Exception as e:
                print(f"Error: {e}")


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    else:
        logger.warning("Config file not found, using defaults")
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


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test long conversation persistence")
    parser.add_argument("--messages", type=int, default=50, help="Number of test messages")
    parser.add_argument("--inspect-interval", type=int, default=10, help="Snapshot interval")
    parser.add_argument("--interactive", action="store_true", help="Interactive inspection mode")
    parser.add_argument("--analyze-storage", action="store_true", help="Analyze existing storage")
    parser.add_argument("--query-test", action="store_true", help="Test query performance")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Create tester
    tester = ConversationPersistenceTester(config)

    if args.analyze_storage:
        # Just analyze existing storage without running conversation
        tester.inspect_storage_files()
        return

    # Initialize components
    await tester.initialize()

    if args.interactive:
        # Interactive mode
        await tester.interactive_inspection()
    else:
        # Run automated test
        await tester.run_long_conversation_test(args.messages, args.inspect_interval)

        # Analyze results
        tester.analyze_storage_growth()
        tester.inspect_storage_files()

        if args.query_test:
            await tester.test_query_performance()

    logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(main())
