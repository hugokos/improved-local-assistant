#!/usr/bin/env python3
"""
One-minute smoke test for GraphRAG functionality.
"""

import asyncio
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def smoke_test():
    """One-minute smoke test as specified."""

    print("🚀 GraphRAG Smoke Test")
    print("=" * 40)

    try:
        # Initialize components
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from services.conversation_manager import ConversationManager
        from services.graph_manager import KnowledgeGraphManager
        from services.model_mgr import ModelConfig
        from services.model_mgr import ModelManager

        # Configure embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            trust_remote_code=False,
            device="cpu",
            normalize=True,
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
            return

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
        loaded_graphs = kg_manager.load_prebuilt_graphs()
        print(f"📊 Loaded {len(loaded_graphs)} graphs: {loaded_graphs}")

        # Initialize conversation manager
        cm = ConversationManager(model_manager, kg_manager, {"conversation": {}})
        sid = cm.create_session()

        print(f"✅ System initialized. Session: {sid}")
        print()

        # Test query
        test_query = "What edible wild plants grow in Northern California?"
        print(f"🔍 Query: {test_query}")
        print("Response: ", end="", flush=True)

        # Stream response
        async for t in cm.get_response(sid, test_query, use_kg=True):
            print(t, end="", flush=True)

        print("\n")

        # Check citations
        citations = cm.get_citations(sid)
        print(f"📚 CITATIONS ➜ {citations}")

        if citations.get("citations"):
            print(f"✅ Found {len(citations['citations'])} citations!")
            for i, citation in enumerate(citations["citations"][:2], 1):
                print(
                    f"  {i}. {citation.get('source', 'Unknown')} - {citation.get('text', '')[:100]}..."
                )
        else:
            print("⚠️  No citations found")

        print("\n🎉 Smoke test completed!")

    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(smoke_test())
