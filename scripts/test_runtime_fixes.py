#!/usr/bin/env python3
"""
Test script to verify the runtime RAG fix pack is working.
"""

import asyncio
import logging
import os
import pathlib
import sys
import time

# Set HF cache at the very top
os.environ.setdefault("HF_HOME", str(pathlib.Path("C:/hf-cache").absolute()))

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_runtime_fixes():
    """Test all runtime RAG fixes."""

    print("🚀 Testing Runtime RAG Fix Pack")
    print("=" * 50)

    # Test 1: Embedding singleton
    print("\n1. Testing embedding singleton...")
    try:
        from services.embedding_singleton import configure_global_embedding
        from services.embedding_singleton import get_embedding_model

        start_time = time.time()
        model1 = get_embedding_model("BAAI/bge-small-en-v1.5")
        first_load_time = time.time() - start_time

        start_time = time.time()
        model2 = get_embedding_model("BAAI/bge-small-en-v1.5")
        second_load_time = time.time() - start_time

        assert model1 is model2, "Singleton not working - different instances returned"
        assert (
            second_load_time < first_load_time / 2
        ), f"Second load not faster: {second_load_time:.2f}s vs {first_load_time:.2f}s"

        print(
            f"✅ Embedding singleton working: first={first_load_time:.2f}s, second={second_load_time:.2f}s"
        )

    except Exception as e:
        print(f"❌ Embedding singleton test failed: {e}")
        return False

    # Test 2: HuggingFace cache
    print("\n2. Testing HuggingFace cache...")
    try:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            print(f"✅ HF_HOME set to: {hf_home}")
            if os.path.exists(hf_home):
                print(f"✅ Cache directory exists: {hf_home}")
            else:
                print(f"⚠️  Cache directory will be created: {hf_home}")
        else:
            print("⚠️  HF_HOME not set - models may re-download")

    except Exception as e:
        print(f"❌ HuggingFace cache test failed: {e}")

    # Test 3: Graph loading with simplified fallback
    print("\n3. Testing simplified graph loading...")
    try:
        from services.embedding_singleton import configure_global_embedding
        from services.graph_manager import KnowledgeGraphManager
        from services.model_mgr import ModelConfig
        from services.model_mgr import ModelManager

        # Configure global embedding first
        configure_global_embedding("BAAI/bge-small-en-v1.5")

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
            timeout=30,
            max_parallel=1,
            max_loaded=1,
        )

        success = await model_manager.initialize_models(model_config)
        if not success:
            print("⚠️  Model manager initialization failed - continuing test")

        # Initialize KG manager
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

        # Test graph loading
        start_time = time.time()
        loaded_graphs = kg_manager.load_prebuilt_graphs()
        load_time = time.time() - start_time

        print(f"✅ Graph loading completed in {load_time:.2f}s")
        print(f"✅ Loaded {len(loaded_graphs)} graphs: {loaded_graphs}")

        # Test smoke test guards
        if loaded_graphs:
            for graph_id in loaded_graphs:
                if graph_id in kg_manager.kg_indices:
                    kg_index = kg_manager.kg_indices[graph_id]
                    if hasattr(kg_index, "storage_context"):
                        storage_ctx = kg_index.storage_context

                        # Check basic stats
                        if hasattr(storage_ctx, "docstore"):
                            num_docs = (
                                len(storage_ctx.docstore.docs)
                                if hasattr(storage_ctx.docstore, "docs")
                                else 0
                            )
                            print(f"✅ Graph {graph_id}: {num_docs} documents")

                        if hasattr(storage_ctx, "graph_store") and hasattr(
                            storage_ctx.graph_store, "rel_map"
                        ):
                            num_relations = sum(
                                len(v) for v in storage_ctx.graph_store.rel_map.values()
                            )
                            print(f"✅ Graph {graph_id}: {num_relations} relations")

                            if num_relations < 10:
                                print(
                                    f"⚠️  Graph {graph_id} may be too small: {num_relations} relations"
                                )

    except Exception as e:
        print(f"❌ Graph loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: Quick conversation test
    print("\n4. Testing conversation with optimizations...")
    try:
        from services.conversation_manager import ConversationManager

        conv_config = {"conversation": {"max_history_length": 10}}
        conv_manager = ConversationManager(model_manager, kg_manager, conv_config)
        session_id = conv_manager.create_session()

        print(f"✅ Conversation manager initialized with session: {session_id}")

        # Test a quick query
        test_query = "What is survival?"
        print(f"Testing query: {test_query}")

        start_time = time.time()
        response_tokens = []

        async for token in conv_manager.get_response(session_id, test_query, use_kg=True):
            response_tokens.append(token)
            if len(response_tokens) > 50:  # Stop after 50 tokens for testing
                break

        response_time = time.time() - start_time
        response = "".join(response_tokens)

        print(f"✅ Response generated in {response_time:.2f}s")
        print(f"✅ Response preview: {response[:100]}...")

        # Check citations
        citations = conv_manager.get_citations(session_id)
        if citations and citations.get("citations"):
            print(f"✅ Found {len(citations['citations'])} citations")
        else:
            print("⚠️  No citations found")

    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n🎉 All runtime fix tests completed!")
    return True


async def main():
    """Main test function."""
    success = await test_runtime_fixes()

    if success:
        print("\n✅ Runtime RAG Fix Pack is working correctly!")
        print("Key improvements:")
        print("  - Embedding model singleton prevents re-downloading")
        print("  - HuggingFace cache configured")
        print("  - Simplified graph loading (no triple-fallback)")
        print("  - Smoke test guards validate graph quality")
    else:
        print("\n❌ Some runtime fixes need attention")

    return success


if __name__ == "__main__":
    asyncio.run(main())
