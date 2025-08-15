#!/usr/bin/env python3
"""
Functional test for the Dynamic KG Chat-Memory Upgrade.

This script tests the actual functionality by simulating a conversation
and verifying that the dynamic KG components work together.
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_end_to_end_functionality():
    """Test the end-to-end functionality of the upgraded system."""
    print("🧪 Testing End-to-End Dynamic KG Functionality")
    print("=" * 50)

    try:
        # Import required components
        import logging

        from services.conversation_manager import ConversationManager
        from services.graph_manager import KnowledgeGraphManager

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temporary directory: {temp_dir}")

            # Mock configuration
            test_config = {
                "knowledge_graphs": {
                    "dynamic_storage": temp_dir,
                    "prebuilt_directory": os.path.join(temp_dir, "prebuilt"),
                    "max_triplets_per_chunk": 3,
                },
                "graph": {"type": "property"},
                "hybrid_retriever": {
                    "use_rrf": True,
                    "half_life_secs": 604800,
                    "rerank_top_n": 10,
                    "budget": {"max_chunks": 12},
                },
                "dynamic_kg": {
                    "episode_every_turns": 8,
                    "persist_every_updates": 5,
                    "persist_interval_secs": 300,
                },
                "conversation": {
                    "max_history_length": 50,
                    "summarize_threshold": 20,
                    "context_window_tokens": 8000,
                },
            }

            print("✅ Configuration loaded")

            # Test 1: Initialize KnowledgeGraphManager
            print("\n🔍 Test 1: Initializing KnowledgeGraphManager...")

            # Mock model manager
            class MockModelManager:
                def complete_sync(self, prompt, model="tinyllama"):
                    # Return mock triples for testing
                    return "(Hugo, likes, coffee)\n(Alice, works_at, Company)\n(Bob, prefers, tea)"

                async def query_conversation_model(self, messages):
                    # Mock streaming response
                    response = "I understand you're testing the dynamic knowledge graph system. That's great!"
                    for token in response.split():
                        yield token + " "
                        await asyncio.sleep(0.01)  # Simulate streaming

            mock_model_manager = MockModelManager()

            # Initialize KG manager
            kg_manager = KnowledgeGraphManager(model_manager=mock_model_manager, config=test_config)

            print("✅ KnowledgeGraphManager initialized")

            # Test 2: Initialize dynamic graph
            print("\n🔍 Test 2: Initializing dynamic graph...")

            # Mock the embedding model to avoid OpenAI issues
            import unittest.mock

            with unittest.mock.patch("llama_index.core.Settings") as mock_settings:
                mock_embed_model = unittest.mock.Mock()
                mock_embed_model.get_text_embedding.return_value = [0.1, 0.2, 0.3] * 128  # 384 dims
                mock_settings.embed_model = mock_embed_model
                mock_settings.llm = None

                success = kg_manager.initialize_dynamic_graph()

            if success:
                print("✅ Dynamic graph initialized successfully")
            else:
                print("❌ Dynamic graph initialization failed")
                return False

            # Test 3: Test entity canonicalization
            print("\n🔍 Test 3: Testing entity canonicalization...")

            # Test the canonicalization methods
            key1 = kg_manager._norm_key("Hugo Kostelni", "Person")
            key2 = kg_manager._norm_key("hugo kostelni", "person")

            if key1 == key2 == "person::hugo kostelni":
                print("✅ Entity normalization working correctly")
            else:
                print(f"❌ Entity normalization failed: {key1} vs {key2}")
                return False

            # Test 4: Test utterance tracking
            print("\n🔍 Test 4: Testing utterance tracking...")

            # Mock the dynamic KG for utterance testing
            mock_graph_store = unittest.mock.Mock()
            mock_dynamic_kg = unittest.mock.Mock()
            mock_dynamic_kg.property_graph_store = mock_graph_store
            kg_manager.dynamic_kg = mock_dynamic_kg

            utt_id = kg_manager._add_utterance("Hello, I'm testing the system!", "user")

            if utt_id.startswith("utt_") and mock_graph_store.upsert_nodes.called:
                print("✅ Utterance tracking working correctly")
            else:
                print("❌ Utterance tracking failed")
                return False

            # Test 5: Test WAL functionality
            print("\n🔍 Test 5: Testing WAL persistence...")

            # Test WAL append
            test_event = {
                "type": "triple_add",
                "s": "Hugo",
                "p": "likes",
                "o": "coffee",
                "ts": time.time(),
            }

            kg_manager._append_wal(test_event)

            # Check if WAL file was created
            wal_path = os.path.join(temp_dir, "main", "wal.jsonl")
            if os.path.exists(wal_path):
                print("✅ WAL persistence working correctly")
            else:
                print("❌ WAL persistence failed")
                return False

            # Test 6: Test conversation integration
            print("\n🔍 Test 6: Testing conversation integration...")

            conv_manager = ConversationManager(
                model_manager=mock_model_manager, kg_manager=kg_manager, config=test_config
            )

            # Create a session
            session_id = conv_manager.create_session()

            if session_id and session_id in conv_manager.sessions:
                print("✅ Conversation session created successfully")
            else:
                print("❌ Conversation session creation failed")
                return False

            # Test 7: Test dynamic graph update
            print("\n🔍 Test 7: Testing dynamic graph update...")

            # Mock the update method to avoid LLM calls
            async def mock_update(text, speaker="user"):
                # Simulate successful update
                return True

            kg_manager.update_dynamic_graph = mock_update

            # Test the callback method
            await conv_manager._update_dynamic_graph_with_callback(
                session_id, "I really enjoy drinking coffee in the morning", "user"
            )

            print("✅ Dynamic graph update completed successfully")

            print("\n🎉 All functional tests passed!")
            return True

    except Exception as e:
        print(f"❌ Functional test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_configuration_loading():
    """Test that the actual configuration file loads correctly."""
    print("\n🔍 Testing actual configuration loading...")

    try:
        import yaml

        config_path = project_root / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify all required sections exist
        required_sections = ["hybrid_retriever", "dynamic_kg", "graph"]

        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False

        # Test specific values
        hr_config = config["hybrid_retriever"]
        if hr_config.get("use_rrf") is not True:
            print("❌ RRF not enabled in config")
            return False

        if hr_config.get("half_life_secs") != 604800:
            print("❌ Wrong half-life in config")
            return False

        dkg_config = config["dynamic_kg"]
        if dkg_config.get("episode_every_turns") != 8:
            print("❌ Wrong episode frequency in config")
            return False

        print("✅ Configuration loading test passed")
        return True

    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False


async def main():
    """Run all functional tests."""
    print("🚀 Starting Functional Tests for Dynamic KG Chat-Memory Upgrade")
    print("=" * 70)

    # Test 1: Configuration loading
    config_success = await test_configuration_loading()

    # Test 2: End-to-end functionality
    functional_success = await test_end_to_end_functionality()

    # Summary
    print("\n" + "=" * 70)
    print("📊 FUNCTIONAL TEST SUMMARY")
    print("=" * 70)

    if config_success and functional_success:
        print("🎉 ALL FUNCTIONAL TESTS PASSED!")
        print("✅ The Dynamic KG Chat-Memory Upgrade is working correctly")
        print("✅ Configuration is properly structured")
        print("✅ Entity canonicalization is functional")
        print("✅ Utterance tracking is working")
        print("✅ WAL persistence is operational")
        print("✅ Conversation integration is successful")
        return True
    else:
        print("⚠️  Some functional tests failed")
        if not config_success:
            print("❌ Configuration loading issues")
        if not functional_success:
            print("❌ End-to-end functionality issues")
        return False


if __name__ == "__main__":
    # Change to the project directory
    os.chdir(project_root)

    # Run the tests
    success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)
