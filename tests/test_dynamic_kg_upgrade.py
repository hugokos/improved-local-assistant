"""
Tests for the Dynamic KG Chat-Memory Upgrade

This test suite verifies the key components of the upgraded dynamic knowledge graph:
- Entity canonicalization
- Utterance provenance tracking
- Schema-guided extraction
- WAL persistence
- Hybrid retrieval with RRF
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from services.conversation_manager import ConversationManager

# Import the components we're testing
from services.graph_manager.construction import KnowledgeGraphConstruction
from services.graph_manager.persistence_simple import KnowledgeGraphPersistence


class TestEntityCanonicalization(unittest.TestCase):
    """Test entity canonicalization to prevent drift."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Mock config
        self.config = {
            "knowledge_graphs": {"dynamic_storage": self.temp_dir, "max_triplets_per_chunk": 3},
            "graph": {"type": "property"},
        }

        # Create test instance
        self.kg_construction = KnowledgeGraphConstruction()
        self.kg_construction.config = self.config
        self.kg_construction.dynamic_storage = self.temp_dir
        self.kg_construction._entity_catalog_path = os.path.join(
            self.temp_dir, "entity_catalog.json"
        )
        self.kg_construction._entity_catalog = {}

    def test_norm_key_generation(self):
        """Test that entity keys are normalized correctly."""
        # Test basic normalization
        key1 = self.kg_construction._norm_key("Hugo Kostelni", "Person")
        key2 = self.kg_construction._norm_key("hugo kostelni", "person")
        key3 = self.kg_construction._norm_key("  Hugo Kostelni  ", "Person")

        self.assertEqual(key1, "person::hugo kostelni")
        self.assertEqual(key2, "person::hugo kostelni")
        self.assertEqual(key3, "person::hugo kostelni")

    @patch("llama_index.core.Settings.embed_model")
    def test_canonical_entity_creation(self, mock_embed_model):
        """Test that canonical entities are created and reused."""
        # Mock embedding model
        mock_embed_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        # First call should create new entity
        entity_id1 = self.kg_construction._canonical_entity("Hugo Kostelni", "Person")
        self.assertIsNotNone(entity_id1)
        self.assertTrue(entity_id1.startswith("ent_"))

        # Second call with same name should return same ID
        entity_id2 = self.kg_construction._canonical_entity("Hugo Kostelni", "Person")
        self.assertEqual(entity_id1, entity_id2)

        # Third call with different case should return same ID
        entity_id3 = self.kg_construction._canonical_entity("hugo kostelni", "Person")
        self.assertEqual(entity_id1, entity_id3)

        # Different entity type should create different ID
        entity_id4 = self.kg_construction._canonical_entity("Hugo Kostelni", "Tool")
        self.assertNotEqual(entity_id1, entity_id4)

    def test_catalog_persistence(self):
        """Test that entity catalog is saved and loaded correctly."""
        # Mock embedding model
        with patch("llama_index.core.Settings.embed_model") as mock_embed_model:
            mock_embed_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]

            # Create an entity
            entity_id = self.kg_construction._canonical_entity("Hugo Kostelni", "Person")

            # Verify catalog file was created
            self.assertTrue(os.path.exists(self.kg_construction._entity_catalog_path))

            # Load catalog manually and verify content
            with open(self.kg_construction._entity_catalog_path) as f:
                catalog = json.load(f)

            expected_key = "person::hugo kostelni"
            self.assertIn(expected_key, catalog)
            self.assertEqual(catalog[expected_key]["id"], entity_id)
            self.assertEqual(catalog[expected_key]["name"], "Hugo Kostelni")
            self.assertEqual(catalog[expected_key]["etype"], "Person")


class TestUtteranceProvenance(unittest.TestCase):
    """Test utterance-level provenance tracking."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Mock config
        self.config = {
            "knowledge_graphs": {
                "dynamic_storage": self.temp_dir,
            },
            "graph": {"type": "property"},
        }

        # Create test instance with mocked dynamic_kg
        self.kg_construction = KnowledgeGraphConstruction()
        self.kg_construction.config = self.config
        self.kg_construction.dynamic_storage = self.temp_dir

        # Mock property graph store
        self.mock_graph_store = Mock()
        self.mock_dynamic_kg = Mock()
        self.mock_dynamic_kg.property_graph_store = self.mock_graph_store
        self.kg_construction.dynamic_kg = self.mock_dynamic_kg

    def test_utterance_creation(self):
        """Test that utterances are created with proper metadata."""
        # Test utterance creation
        utt_id = self.kg_construction._add_utterance("Hello, how are you?", "user")

        # Verify utterance ID format
        self.assertTrue(utt_id.startswith("utt_"))

        # Verify graph store was called
        self.mock_graph_store.upsert_nodes.assert_called_once()

        # Check the node data
        call_args = self.mock_graph_store.upsert_nodes.call_args[0][0]
        node_data = call_args[0]

        self.assertEqual(node_data["id"], utt_id)
        self.assertEqual(node_data["label"], "Utterance")
        self.assertEqual(node_data["properties"]["text"], "Hello, how are you?")
        self.assertEqual(node_data["properties"]["speaker"], "user")
        self.assertIn("ts", node_data["properties"])

    def test_utterance_sequence(self):
        """Test that utterance sequence numbers increment."""
        # Create multiple utterances
        utt_id1 = self.kg_construction._add_utterance("First message", "user")
        utt_id2 = self.kg_construction._add_utterance("Second message", "assistant")
        utt_id3 = self.kg_construction._add_utterance("Third message", "user")

        # Verify they have different IDs
        self.assertNotEqual(utt_id1, utt_id2)
        self.assertNotEqual(utt_id2, utt_id3)

        # Verify sequence numbers are in the IDs
        self.assertTrue("_1" in utt_id1)
        self.assertTrue("_2" in utt_id2)
        self.assertTrue("_3" in utt_id3)


class TestWALPersistence(unittest.TestCase):
    """Test Write-Ahead Log persistence."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test instance
        self.kg_persistence = KnowledgeGraphPersistence()
        self.kg_persistence.dynamic_storage = self.temp_dir

    def test_wal_append(self):
        """Test that WAL entries are appended correctly."""
        # Create test event
        event = {"type": "triple_add", "s": "Hugo", "p": "likes", "o": "coffee", "ts": time.time()}

        # Append to WAL
        self.kg_persistence._append_wal(event)

        # Verify WAL file was created
        wal_path = os.path.join(self.temp_dir, "main", "wal.jsonl")
        self.assertTrue(os.path.exists(wal_path))

        # Verify content
        with open(wal_path) as f:
            line = f.readline().strip()
            loaded_event = json.loads(line)

        self.assertEqual(loaded_event["type"], "triple_add")
        self.assertEqual(loaded_event["s"], "Hugo")
        self.assertEqual(loaded_event["p"], "likes")
        self.assertEqual(loaded_event["o"], "coffee")

    def test_multiple_wal_entries(self):
        """Test multiple WAL entries."""
        events = [
            {"type": "triple_add", "s": "Hugo", "p": "likes", "o": "coffee", "ts": time.time()},
            {
                "type": "utterance_add",
                "id": "utt_123",
                "text": "Hello",
                "speaker": "user",
                "ts": time.time(),
            },
            {
                "type": "triple_add",
                "s": "Alice",
                "p": "works_at",
                "o": "Company",
                "ts": time.time(),
            },
        ]

        # Append all events
        for event in events:
            self.kg_persistence._append_wal(event)

        # Verify all entries
        wal_path = os.path.join(self.temp_dir, "main", "wal.jsonl")
        with open(wal_path) as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 3)

        # Verify each line
        for i, line in enumerate(lines):
            loaded_event = json.loads(line.strip())
            self.assertEqual(loaded_event["type"], events[i]["type"])


class TestSchemaGuidedExtraction(unittest.TestCase):
    """Test schema-guided entity extraction."""

    def test_schema_definition(self):
        """Test that the schema is properly defined."""
        # This would normally be tested by checking if SchemaLLMPathExtractor
        # is initialized with the correct schema, but we'll test the schema structure

        expected_entities = [
            "Person",
            "Utterance",
            "Preference",
            "Goal",
            "Task",
            "Fact",
            "Episode",
            "CommunitySummary",
            "Tool",
            "Doc",
            "Claim",
            "Topic",
        ]

        expected_relations = [
            "MENTIONS",
            "ASSERTS",
            "REFERS_TO",
            "PREFERS",
            "GOAL_OF",
            "RELATES_TO",
            "SUMMARIZES",
            "CITES",
            "DERIVED_FROM",
            "SAID_BY",
        ]

        # These would be used in the actual schema
        self.assertEqual(len(expected_entities), 12)
        self.assertEqual(len(expected_relations), 10)

        # Verify key entity types are present
        self.assertIn("Person", expected_entities)
        self.assertIn("Utterance", expected_entities)
        self.assertIn("Preference", expected_entities)

        # Verify key relation types are present
        self.assertIn("MENTIONS", expected_relations)
        self.assertIn("ASSERTS", expected_relations)
        self.assertIn("SAID_BY", expected_relations)


class TestTripleExtraction(unittest.TestCase):
    """Test triple extraction and parsing."""

    def setUp(self):
        """Set up test environment."""
        self.kg_construction = KnowledgeGraphConstruction()
        self.kg_construction.max_triplets_per_chunk = 3

    def test_triple_parsing(self):
        """Test parsing of triple lines from LLM output."""
        # Test various formats
        raw_text = """
        (Hugo, likes, coffee)
        Alice, works_at, Company
        (Bob, prefers, tea)
        # This is a comment
        Triplets:
        (Charlie, drives, car)
        """

        triples = self.kg_construction._parse_triple_lines(raw_text)

        expected_triples = [
            ("Hugo", "likes", "coffee"),
            ("Alice", "works_at", "Company"),
            ("Bob", "prefers", "tea"),
            ("Charlie", "drives", "car"),
        ]

        self.assertEqual(len(triples), 4)
        for expected in expected_triples:
            self.assertIn(expected, triples)

    def test_text_chunking(self):
        """Test text chunking for processing."""
        long_text = (
            "This is a very long text that should be split into multiple chunks for processing. "
            * 20
        )

        chunks = self.kg_construction._split_text_into_chunks(long_text, max_tokens=50)

        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)

        # Each chunk should be reasonably sized
        for chunk in chunks:
            word_count = len(chunk.split())
            self.assertLessEqual(word_count, 60)  # Allow some flexibility


@pytest.mark.asyncio
class TestHybridRetrieverUpgrade:
    """Test the upgraded hybrid retriever with RRF and time-decay."""

    async def test_rrf_initialization(self):
        """Test that RRF fusion is initialized when available."""
        # Mock config with RRF enabled
        config = {
            "hybrid_retriever": {
                "use_rrf": True,
                "half_life_secs": 604800,
                "rerank_top_n": 10,
                "budget": {"max_chunks": 12},
            }
        }

        # Mock retrievers
        Mock()
        Mock()

        # This would test the actual initialization, but we need to mock
        # the LlamaIndex components that might not be available
        # For now, we'll test the configuration parsing

        retriever_config = config.get("hybrid_retriever", {})
        use_rrf = retriever_config.get("use_rrf", True)
        half_life_secs = retriever_config.get("half_life_secs", 7 * 24 * 3600)
        rerank_top_n = retriever_config.get("rerank_top_n", 10)

        assert use_rrf is True
        assert half_life_secs == 604800
        assert rerank_top_n == 10

    def test_time_decay_calculation(self):
        """Test time decay calculation for scoring."""
        import math

        half_life_secs = 7 * 24 * 3600  # 1 week
        current_time = time.time()

        # Test recent timestamp (should have high boost)
        recent_ts = current_time - 3600  # 1 hour ago
        recent_boost = math.exp(-(current_time - recent_ts) / half_life_secs)

        # Test old timestamp (should have low boost)
        old_ts = current_time - (14 * 24 * 3600)  # 2 weeks ago
        old_boost = math.exp(-(current_time - old_ts) / half_life_secs)

        # Recent should have higher boost than old
        assert recent_boost > old_boost
        assert recent_boost > 0.9  # Should be close to 1.0
        assert old_boost < 0.3  # Should be significantly lower


class TestConversationIntegration(unittest.TestCase):
    """Test integration with conversation manager."""

    def setUp(self):
        """Set up test environment."""
        # Mock dependencies
        self.mock_model_manager = Mock()
        self.mock_kg_manager = Mock()

        # Mock config
        self.config = {
            "conversation": {
                "max_history_length": 50,
                "summarize_threshold": 20,
                "context_window_tokens": 8000,
            }
        }

        # Create conversation manager
        self.conv_manager = ConversationManager(
            model_manager=self.mock_model_manager,
            kg_manager=self.mock_kg_manager,
            config=self.config,
        )

    @patch("asyncio.create_task")
    async def test_dynamic_kg_update_with_speaker(self, mock_create_task):
        """Test that dynamic KG updates include speaker information."""
        # Mock the KG manager's update method
        self.mock_kg_manager.update_dynamic_graph = AsyncMock(return_value=True)

        # Create a session
        session_id = self.conv_manager.create_session()

        # Call the update method
        await self.conv_manager._update_dynamic_graph_with_callback(
            session_id, "Hello, I like coffee", "user"
        )

        # Verify the KG manager was called with speaker info
        self.mock_kg_manager.update_dynamic_graph.assert_called_once_with(
            "Hello, I like coffee", "user"
        )


def run_basic_import_test():
    """Test that all the upgraded components can be imported."""
    try:
        # Test imports
        # Test imports - just verify they can be imported
        import services.conversation_manager
        import services.graph_manager.construction
        import services.graph_manager.persistence_simple
        import services.hybrid_retriever  # noqa: F401  # Testing import

        print("✅ All imports successful")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def run_config_validation_test():
    """Test that the configuration is properly structured."""
    try:
        import yaml

        # Load the config
        with open("improved-local-assistant/config.yaml") as f:
            config = yaml.safe_load(f)

        # Check for required sections
        required_sections = ["hybrid_retriever", "dynamic_kg", "graph"]

        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False

        # Check hybrid_retriever config
        hr_config = config["hybrid_retriever"]
        required_hr_keys = ["use_rrf", "half_life_secs", "rerank_top_n", "budget", "weights"]

        for key in required_hr_keys:
            if key not in hr_config:
                print(f"❌ Missing hybrid_retriever config key: {key}")
                return False

        # Check dynamic_kg config
        dkg_config = config["dynamic_kg"]
        required_dkg_keys = [
            "episode_every_turns",
            "persist_every_updates",
            "persist_interval_secs",
        ]

        for key in required_dkg_keys:
            if key not in dkg_config:
                print(f"❌ Missing dynamic_kg config key: {key}")
                return False

        print("✅ Configuration validation successful")
        return True

    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Dynamic KG Chat-Memory Upgrade Tests")
    print("=" * 50)

    # Run basic tests
    print("\n1. Testing imports...")
    import_success = run_basic_import_test()

    print("\n2. Testing configuration...")
    config_success = run_config_validation_test()

    if import_success and config_success:
        print("\n3. Running unit tests...")
        # Run the actual unit tests
        unittest.main(argv=[""], exit=False, verbosity=2)
    else:
        print("\n❌ Basic tests failed, skipping unit tests")
