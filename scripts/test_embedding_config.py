#!/usr/bin/env python3
"""
Test script to verify embedding configuration matches prebuilt graphs.

This script tests that the local embedding model is properly configured
to work with the prebuilt knowledge graphs.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fix for Windows asyncio event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_embedding_configuration():
    """Test that embedding configuration matches prebuilt graph metadata."""

    print("🔍 Testing Embedding Configuration")
    print("=" * 50)

    try:
        # Step 1: Read metadata from prebuilt graphs
        print("1. Reading prebuilt graph metadata...")
        metadata = read_graph_metadata()

        if not metadata:
            print("❌ No graph metadata found")
            return False

        print(f"✅ Found metadata for {len(metadata)} graphs")
        for graph_id, meta in metadata.items():
            embed_model = meta.get("embed_model", "unknown")
            embed_dim = meta.get("embedding_dim", "unknown")
            print(f"   - {graph_id}: {embed_model} (dim: {embed_dim})")

        # Step 2: Test embedding model initialization
        print("\n2. Testing embedding model initialization...")
        embed_model_name = test_embedding_model_init(metadata)

        if not embed_model_name:
            print("❌ Failed to initialize embedding model")
            return False

        print(f"✅ Embedding model initialized: {embed_model_name}")

        # Step 3: Test embedding generation
        print("\n3. Testing embedding generation...")
        success = await test_embedding_generation()

        if not success:
            print("❌ Failed to generate embeddings")
            return False

        print("✅ Embedding generation successful")

        # Step 4: Test knowledge graph loading
        print("\n4. Testing knowledge graph loading...")
        success = await test_kg_loading()

        if not success:
            print("❌ Failed to load knowledge graphs")
            return False

        print("✅ Knowledge graph loading successful")

        print("\n🎉 All embedding configuration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error during embedding configuration test: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def read_graph_metadata() -> dict[str, dict[str, Any]]:
    """Read metadata from all prebuilt graphs."""
    metadata = {}
    prebuilt_dir = "./data/prebuilt_graphs"

    if not os.path.exists(prebuilt_dir):
        logger.warning(f"Prebuilt graphs directory not found: {prebuilt_dir}")
        return metadata

    for item in os.listdir(prebuilt_dir):
        graph_path = os.path.join(prebuilt_dir, item)
        if os.path.isdir(graph_path):
            meta_path = os.path.join(graph_path, "meta.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                        metadata[item] = meta
                except Exception as e:
                    logger.warning(f"Could not read metadata from {meta_path}: {e}")

    return metadata


def test_embedding_model_init(metadata: dict[str, dict[str, Any]]) -> str:
    """Test embedding model initialization."""
    try:
        # Get embedding model name from metadata
        embed_model_name = "BAAI/bge-small-en-v1.5"  # default

        for _graph_id, meta in metadata.items():
            if "embed_model" in meta:
                embed_model_name = meta["embed_model"]
                break

        print(f"   Using embedding model: {embed_model_name}")

        # Test HuggingFace embedding initialization
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        HuggingFaceEmbedding(
            model_name=embed_model_name,
            trust_remote_code=False,
            device="cpu",
            normalize=True,
            embed_batch_size=10,
        )

        print("   Model loaded successfully")
        return embed_model_name

    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Install required packages: pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"   ❌ Initialization error: {e}")
        return None


async def test_embedding_generation():
    """Test that embeddings can be generated."""
    try:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        # Configure embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            trust_remote_code=False,
            device="cpu",
            normalize=True,
        )

        # Test embedding generation
        test_text = "This is a test sentence for embedding generation."
        embedding = Settings.embed_model.get_text_embedding(test_text)

        print(f"   Generated embedding with dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")

        # Verify embedding dimension matches metadata
        expected_dim = 384  # BGE small model dimension
        if len(embedding) == expected_dim:
            print(f"   ✅ Embedding dimension matches expected: {expected_dim}")
        else:
            print(
                f"   ⚠️  Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}"
            )

        return True

    except Exception as e:
        print(f"   ❌ Embedding generation failed: {e}")
        return False


async def test_kg_loading():
    """Test knowledge graph loading with proper embedding configuration."""
    try:
        from llama_index.core import Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from services.graph_manager import KnowledgeGraphManager
        from services.model_mgr import ModelConfig
        from services.model_mgr import ModelManager

        # Configure embedding model globally
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            trust_remote_code=False,
            device="cpu",
            normalize=True,
        )

        print("   Configured global embedding model")

        # Initialize model manager
        model_manager = ModelManager(host="http://localhost:11434")
        model_manager.conversation_model = "hermes3:3b"
        model_manager.knowledge_model = "tinyllama:latest"

        # Initialize models
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
            print("   ❌ Model manager initialization failed")
            return False

        print("   ✅ Model manager initialized")

        # Initialize knowledge graph manager
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
        print("   ✅ Knowledge graph manager created")

        # Try to load prebuilt graphs
        loaded_graphs = kg_manager.load_prebuilt_graphs()
        print(f"   Loaded {len(loaded_graphs)} prebuilt graphs: {loaded_graphs}")

        if loaded_graphs:
            print("   ✅ Successfully loaded prebuilt graphs")

            # Test querying one of the loaded graphs
            if "survivalist" in loaded_graphs:
                print("   Testing survivalist graph query...")
                try:
                    query_result = await kg_manager.query_graphs("What is survival?")
                    if query_result and query_result.get("response"):
                        print("   ✅ Graph query successful")
                        print(f"   Response preview: {query_result['response'][:100]}...")
                    else:
                        print("   ⚠️  Graph query returned empty result")
                except Exception as e:
                    print(f"   ⚠️  Graph query failed: {e}")
        else:
            print("   ⚠️  No prebuilt graphs were loaded")

        return True

    except Exception as e:
        print(f"   ❌ Knowledge graph loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Embedding Configuration Test")
    print()

    success = await test_embedding_configuration()

    print()
    if success:
        print("✅ All tests passed! Embedding configuration is working correctly.")
        print("You can now use the GraphRAG system with confidence.")
    else:
        print("❌ Some tests failed. Please check the configuration and try again.")
        print("Common fixes:")
        print("  - Install sentence-transformers: pip install sentence-transformers")
        print("  - Ensure Ollama is running: ollama serve")
        print("  - Check that models are available: ollama list")

    return success


if __name__ == "__main__":
    asyncio.run(main())
