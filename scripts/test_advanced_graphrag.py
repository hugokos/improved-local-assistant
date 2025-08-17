#!/usr/bin/env python3
"""
Test the advanced GraphRAG system with semantic routing and hybrid retrieval.
"""

import asyncio
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_advanced_graphrag():
    """Test the advanced GraphRAG system."""

    print("🚀 Advanced GraphRAG Test")
    print("=" * 50)

    try:
        # Initialize the system
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from improved_local_assistant.services.conversation_manager import ConversationManager
        from improved_local_assistant.services.graph_manager import KnowledgeGraphManager
        from improved_local_assistant.services.model_mgr import ModelConfig
        from improved_local_assistant.services.model_mgr import ModelManager

        # Configure embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5", device="cpu", normalize=True
        )

        # Initialize model manager
        model_manager = ModelManager(host="http://localhost:11434")
        model_manager.conversation_model = "hermes3:3b"
        model_manager.knowledge_model = "tinyllama:latest"

        model_config = ModelConfig(
            name="hermes3:3b",
            type="conversation",
            context_window=8000,
            temperature=0.7,
            max_tokens=2048,
            timeout=60,
            max_parallel=1,
            max_loaded=1,
        )

        success = await model_manager.initialize_models(model_config)
        if not success:
            print("❌ Model manager initialization failed")
            return False

        # Initialize KG manager with advanced router
        kg_config = {
            "knowledge_graphs": {
                "prebuilt_directory": "./data/prebuilt_graphs",
                "dynamic_storage": "./data/dynamic_graph",
                "max_triplets_per_chunk": 4,
                "enable_visualization": False,
                "enable_caching": True,
            }
        }

        kg_manager = KnowledgeGraphManager(model_manager, kg_config)
        loaded_graphs = kg_manager.load_prebuilt_graphs()
        print(f"📊 Loaded graphs: {loaded_graphs}")

        # Wait a moment for the GraphRouter to initialize
        await asyncio.sleep(2)

        # Initialize conversation manager
        conv_manager = ConversationManager(model_manager, kg_manager, {"conversation": {}})
        session_id = conv_manager.create_session()

        print("✅ System initialized with advanced GraphRouter")

        # Test queries that should benefit from advanced routing
        test_queries = [
            "What are the most important survival priorities?",
            "How do you find and purify water in the wilderness?",
            "What edible wild plants can I find?",
            "How do you build an emergency shelter?",
            "What are the best fire starting techniques?",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            print("💬 Response: ", end="", flush=True)

            response_tokens = []
            async for token in conv_manager.get_response(session_id, query, use_kg=True):
                print(token, end="", flush=True)
                response_tokens.append(token)

            response = "".join(response_tokens)
            print(f"\n📊 Response length: {len(response)} characters")

            # Check citations
            citations = conv_manager.get_citations(session_id)
            if citations and citations.get("citations"):
                print(f"📚 Citations: {len(citations['citations'])} found")

                # Check if we're getting better quality citations
                for j, citation in enumerate(citations["citations"][:2], 1):
                    text = citation.get("text", "")[:80]
                    score = citation.get("score", 0.0)
                    print(f"  {j}. {text}... (score: {score:.3f})")
            else:
                print("📚 No citations found")

            # Small delay between queries
            await asyncio.sleep(1)

        print("\n✅ Advanced GraphRAG test completed!")
        return True

    except Exception as e:
        print(f"❌ Error in advanced GraphRAG test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_direct_router():
    """Test the GraphRouter directly."""

    print("\n🧪 Direct Router Test")
    print("=" * 30)

    try:
        from improved_local_assistant.services.graph_router import GraphRouter

        config = {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "registry_path": "data/graph_registry.json",
            "data_path": "data/prebuilt_graphs",
            "max_cached_indices": 10,
            "hybrid_alpha": 0.6,
            "rerank_top_n": 10,
            "min_score_threshold": 0.3,
        }

        router = GraphRouter(config)

        if not await router.initialize():
            print("❌ Failed to initialize router")
            return False

        # Test direct queries
        test_queries = [
            "How do you purify water?",
            "What plants are safe to eat?",
            "Building emergency shelter",
        ]

        for query in test_queries:
            print(f"\n🔍 Direct query: {query}")

            result = await router.query(query, k=2)

            if result:
                response = result.get("response", "")
                metadata = result.get("metadata", {})

                print(f"💬 Response: {response[:200]}...")
                print(f"📊 Metadata: {metadata}")
            else:
                print("❌ No result")

        await router.close()

        print("✅ Direct router test completed")
        return True

    except Exception as e:
        print(f"❌ Error in direct router test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""

    # Test the complete system
    system_success = await test_advanced_graphrag()

    # Test the router directly
    router_success = await test_direct_router()

    if system_success and router_success:
        print("\n🎉 ALL ADVANCED GRAPHRAG TESTS PASSED!")
        print("The advanced routing and hybrid retrieval system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

    return system_success and router_success


if __name__ == "__main__":
    asyncio.run(main())
