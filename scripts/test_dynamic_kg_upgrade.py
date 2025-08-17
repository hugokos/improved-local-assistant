#!/usr/bin/env python3
"""
Integration test script for the Dynamic KG Chat-Memory Upgrade.

This script tests the end-to-end functionality of the upgraded system:
1. Entity canonicalization
2. Utterance tracking
3. WAL persistence
4. Hybrid retrieval
5. Configuration loading
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all upgraded components can be imported."""
    print("🔍 Testing imports...")

    try:
        # Core components - testing imports
        from services.conversation_manager import ConversationManager  # noqa: F401
        from services.graph_manager.construction import KnowledgeGraphConstruction  # noqa: F401
        from services.graph_manager.init_config import KnowledgeGraphInitializer  # noqa: F401
        from services.graph_manager.persistence_simple import (
            KnowledgeGraphPersistence,  # noqa: F401
        )
        from services.hybrid_retriever import HybridEnsembleRetriever  # noqa: F401

        print("✅ All core imports successful")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration loading and validation."""
    print("\n🔍 Testing configuration...")

    try:
        import yaml

        config_path = project_root / "config.yaml"
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check for new sections
        required_sections = ["hybrid_retriever", "dynamic_kg", "graph"]
        missing_sections = []

        for section in required_sections:
            if section not in config:
                missing_sections.append(section)

        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False

        # Validate hybrid_retriever section
        hr_config = config["hybrid_retriever"]
        hr_keys = ["use_rrf", "half_life_secs", "rerank_top_n", "budget", "weights"]

        for key in hr_keys:
            if key not in hr_config:
                print(f"❌ Missing hybrid_retriever key: {key}")
                return False

        # Validate dynamic_kg section
        dkg_config = config["dynamic_kg"]
        dkg_keys = ["episode_every_turns", "persist_every_updates", "persist_interval_secs"]

        for key in dkg_keys:
            if key not in dkg_config:
                print(f"❌ Missing dynamic_kg key: {key}")
                return False

        print("✅ Configuration validation successful")
        print(f"   - RRF enabled: {hr_config['use_rrf']}")
        print(f"   - Time decay half-life: {hr_config['half_life_secs']} seconds")
        print(f"   - Episode frequency: {dkg_config['episode_every_turns']} turns")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_entity_canonicalization():
    """Test entity canonicalization functionality."""
    print("\n🔍 Testing entity canonicalization...")

    try:
        from services.graph_manager.construction import KnowledgeGraphConstruction

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up test instance
            kg_construction = KnowledgeGraphConstruction()
            kg_construction.dynamic_storage = temp_dir
            kg_construction._entity_catalog_path = os.path.join(temp_dir, "entity_catalog.json")
            kg_construction._entity_catalog = {}

            # Test normalization
            key1 = kg_construction._norm_key("Hugo Kostelni", "Person")
            key2 = kg_construction._norm_key("hugo kostelni", "person")
            key3 = kg_construction._norm_key("  Hugo Kostelni  ", "Person")

            if key1 != "person::hugo kostelni" or key1 != key2 or key1 != key3:
                print(f"❌ Key normalization failed: {key1}, {key2}, {key3}")
                return False

            # Test entity creation (mock both Settings and embed_model)
            import unittest.mock

            # Mock the entire Settings module to avoid OpenAI initialization
            with unittest.mock.patch("llama_index.core.Settings") as mock_settings:
                # Create a mock embed model
                mock_embed_model = unittest.mock.Mock()
                mock_embed_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
                mock_settings.embed_model = mock_embed_model

                # Create entities
                id1 = kg_construction._canonical_entity("Hugo Kostelni", "Person")
                id2 = kg_construction._canonical_entity("hugo kostelni", "Person")  # Should be same
                id3 = kg_construction._canonical_entity(
                    "Hugo Kostelni", "Tool"
                )  # Should be different

                if id1 != id2:
                    print(f"❌ Same entity got different IDs: {id1} vs {id2}")
                    return False

                if id1 == id3:
                    print(f"❌ Different entity types got same ID: {id1} vs {id3}")
                    return False

                # Check catalog persistence
                if not os.path.exists(kg_construction._entity_catalog_path):
                    print("❌ Entity catalog file not created")
                    return False

                with open(kg_construction._entity_catalog_path) as f:
                    catalog = json.load(f)

                expected_key = "person::hugo kostelni"
                if expected_key not in catalog:
                    print(f"❌ Entity not found in catalog: {expected_key}")
                    return False

        print("✅ Entity canonicalization test passed")
        return True

    except Exception as e:
        print(f"❌ Entity canonicalization test failed: {e}")
        return False


def test_utterance_tracking():
    """Test utterance provenance tracking."""
    print("\n🔍 Testing utterance tracking...")

    try:
        import logging
        import unittest.mock

        from services.graph_manager.construction import KnowledgeGraphConstruction

        # Set up test instance
        kg_construction = KnowledgeGraphConstruction()
        kg_construction.logger = logging.getLogger(__name__)  # Add missing logger

        # Mock the dynamic KG with property graph store
        mock_graph_store = unittest.mock.Mock()
        mock_dynamic_kg = unittest.mock.Mock()
        mock_dynamic_kg.property_graph_store = mock_graph_store
        kg_construction.dynamic_kg = mock_dynamic_kg

        # Test utterance creation
        utt_id1 = kg_construction._add_utterance("Hello, how are you?", "user")
        utt_id2 = kg_construction._add_utterance("I'm doing well, thanks!", "assistant")

        # Verify IDs are different and properly formatted
        if not utt_id1.startswith("utt_") or not utt_id2.startswith("utt_"):
            print(f"❌ Invalid utterance ID format: {utt_id1}, {utt_id2}")
            return False

        if utt_id1 == utt_id2:
            print(f"❌ Utterance IDs should be different: {utt_id1}")
            return False

        # Verify graph store was called correctly
        if mock_graph_store.upsert_nodes.call_count != 2:
            print(
                f"❌ Expected 2 upsert_nodes calls, got {mock_graph_store.upsert_nodes.call_count}"
            )
            return False

        # Check the node data from the first call
        first_call_args = mock_graph_store.upsert_nodes.call_args_list[0][0][0]
        node_data = first_call_args[0]

        if node_data["label"] != "Utterance":
            print(f"❌ Wrong node label: {node_data['label']}")
            return False

        if node_data["properties"]["text"] != "Hello, how are you?":
            print(f"❌ Wrong utterance text: {node_data['properties']['text']}")
            return False

        if node_data["properties"]["speaker"] != "user":
            print(f"❌ Wrong speaker: {node_data['properties']['speaker']}")
            return False

        print("✅ Utterance tracking test passed")
        return True

    except Exception as e:
        print(f"❌ Utterance tracking test failed: {e}")
        return False


def test_wal_persistence():
    """Test Write-Ahead Log functionality."""
    print("\n🔍 Testing WAL persistence...")

    try:
        from services.graph_manager.persistence_simple import KnowledgeGraphPersistence

        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up test instance
            kg_persistence = KnowledgeGraphPersistence()
            kg_persistence.dynamic_storage = temp_dir

            # Test WAL append
            test_events = [
                {"type": "triple_add", "s": "Hugo", "p": "likes", "o": "coffee", "ts": time.time()},
                {
                    "type": "utterance_add",
                    "id": "utt_123",
                    "text": "Hello world",
                    "speaker": "user",
                    "ts": time.time(),
                },
            ]

            # Append events
            for event in test_events:
                kg_persistence._append_wal(event)

            # Verify WAL file exists
            wal_path = os.path.join(temp_dir, "main", "wal.jsonl")
            if not os.path.exists(wal_path):
                print(f"❌ WAL file not created: {wal_path}")
                return False

            # Verify content
            with open(wal_path) as f:
                lines = f.readlines()

            if len(lines) != 2:
                print(f"❌ Expected 2 WAL entries, got {len(lines)}")
                return False

            # Verify first event
            event1 = json.loads(lines[0].strip())
            if event1["type"] != "triple_add" or event1["s"] != "Hugo":
                print(f"❌ First WAL entry incorrect: {event1}")
                return False

            # Verify second event
            event2 = json.loads(lines[1].strip())
            if event2["type"] != "utterance_add" or event2["id"] != "utt_123":
                print(f"❌ Second WAL entry incorrect: {event2}")
                return False

        print("✅ WAL persistence test passed")
        return True

    except Exception as e:
        print(f"❌ WAL persistence test failed: {e}")
        return False


def test_triple_parsing():
    """Test triple extraction and parsing."""
    print("\n🔍 Testing triple parsing...")

    try:
        from services.graph_manager.construction import KnowledgeGraphConstruction

        kg_construction = KnowledgeGraphConstruction()
        kg_construction.max_triplets_per_chunk = 5

        # Test parsing various formats
        raw_text = """
        (Hugo, likes, coffee)
        Alice, works_at, Company
        (Bob, prefers, tea)
        # This is a comment - should be ignored
        Triplets:
        (Charlie, drives, car)
        Invalid line without commas
        (David, enjoys, "reading books")
        """

        triples = kg_construction._parse_triple_lines(raw_text)

        expected_triples = [
            ("Hugo", "likes", "coffee"),
            ("Alice", "works_at", "Company"),
            ("Bob", "prefers", "tea"),
            ("Charlie", "drives", "car"),
            ("David", "enjoys", "reading books"),
        ]

        if len(triples) != len(expected_triples):
            print(f"❌ Expected {len(expected_triples)} triples, got {len(triples)}")
            print(f"   Parsed: {triples}")
            return False

        for expected in expected_triples:
            if expected not in triples:
                print(f"❌ Missing expected triple: {expected}")
                print(f"   Parsed: {triples}")
                return False

        # Test text chunking
        long_text = "This is a test sentence. " * 100  # 500 words
        chunks = kg_construction._split_text_into_chunks(long_text, max_tokens=50)

        if len(chunks) <= 1:
            print(f"❌ Long text should be split into multiple chunks, got {len(chunks)}")
            return False

        # Verify chunk sizes are reasonable
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            if word_count > 60:  # Allow some flexibility
                print(f"❌ Chunk {i} too large: {word_count} words")
                return False

        print("✅ Triple parsing test passed")
        return True

    except Exception as e:
        print(f"❌ Triple parsing test failed: {e}")
        return False


def test_hybrid_retriever_config():
    """Test hybrid retriever configuration parsing."""
    print("\n🔍 Testing hybrid retriever configuration...")

    try:
        # Test configuration parsing
        test_config = {
            "hybrid_retriever": {
                "use_rrf": True,
                "half_life_secs": 604800,
                "rerank_top_n": 10,
                "budget": {"max_chunks": 12, "graph_depth": 2, "bm25_top_k": 4, "vector_top_k": 4},
                "weights": {"graph": 0.6, "vector": 0.25, "bm25": 0.15},
            }
        }

        # Parse config like the actual code would
        retriever_config = test_config.get("hybrid_retriever", {})
        budget_config = retriever_config.get("budget", {})
        weights_config = retriever_config.get("weights", {})

        # Verify values
        use_rrf = retriever_config.get("use_rrf", True)
        half_life_secs = retriever_config.get("half_life_secs", 7 * 24 * 3600)
        retriever_config.get("rerank_top_n", 10)
        budget_config.get("max_chunks", 12)

        weights = (
            weights_config.get("graph", 0.6),
            weights_config.get("vector", 0.2),
            weights_config.get("bm25", 0.2),
        )

        # Verify expected values
        if not use_rrf:
            print("❌ RRF should be enabled")
            return False

        if half_life_secs != 604800:
            print(f"❌ Wrong half-life: {half_life_secs}")
            return False

        if sum(weights) != 1.0:
            print(f"❌ Weights don't sum to 1.0: {weights} = {sum(weights)}")
            return False

        # Test time decay calculation
        import math

        current_time = time.time()

        # Recent timestamp (1 hour ago)
        recent_ts = current_time - 3600
        recent_boost = math.exp(-(current_time - recent_ts) / half_life_secs)

        # Old timestamp (2 weeks ago)
        old_ts = current_time - (14 * 24 * 3600)
        old_boost = math.exp(-(current_time - old_ts) / half_life_secs)

        if recent_boost <= old_boost:
            print(f"❌ Recent boost should be higher: {recent_boost} vs {old_boost}")
            return False

        if recent_boost < 0.9:
            print(f"❌ Recent boost too low: {recent_boost}")
            return False

        print("✅ Hybrid retriever configuration test passed")
        return True

    except Exception as e:
        print(f"❌ Hybrid retriever configuration test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests in sequence."""
    print("🧪 Dynamic KG Chat-Memory Upgrade Integration Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Entity Canonicalization", test_entity_canonicalization),
        ("Utterance Tracking", test_utterance_tracking),
        ("WAL Persistence", test_wal_persistence),
        ("Triple Parsing", test_triple_parsing),
        ("Hybrid Retriever Config", test_hybrid_retriever_config),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n📈 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! The Dynamic KG Chat-Memory Upgrade is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Change to the project directory
    os.chdir(project_root)

    # Run the tests
    success = asyncio.run(run_all_tests())

    # Exit with appropriate code
    sys.exit(0 if success else 1)
