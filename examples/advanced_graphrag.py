#!/usr/bin/env python3
"""Advanced GraphRAG usage examples.

Demonstrates sophisticated knowledge graph operations and retrieval patterns.
"""

import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.graph_manager.query import GraphQueryManager
from services.hybrid_retriever import HybridRetriever


class GraphRAGExample:
    """Example class demonstrating advanced GraphRAG operations."""

    def __init__(self) -> None:
        """Initialize the GraphRAG example."""
        self.query_manager = GraphQueryManager()
        self.retriever = HybridRetriever()

    def demonstrate_entity_extraction(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships from text.

        Args:
            text: Input text for entity extraction

        Returns:
            Dictionary containing extracted entities and relationships
        """
        # This would integrate with your actual extraction pipeline
        print(f"Extracting entities from: {text[:100]}...")

        # Simulated extraction result
        return {
            "entities": [
                {"name": "Machine Learning", "type": "concept"},
                {"name": "Neural Networks", "type": "technology"},
            ],
            "relationships": [
                {"source": "Machine Learning", "target": "Neural Networks", "type": "uses"}
            ],
        }

    def demonstrate_hybrid_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Demonstrate hybrid retrieval combining multiple search methods.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of retrieved documents with scores
        """
        print(f"Performing hybrid retrieval for: {query}")

        # This would use your actual hybrid retriever
        results = [
            {
                "content": "Machine learning is a subset of artificial intelligence...",
                "source": "AI Fundamentals",
                "score": 0.95,
                "method": "vector_similarity",
            },
            {
                "content": "Neural networks are computing systems inspired by biological neural networks...",
                "source": "Deep Learning Guide",
                "score": 0.87,
                "method": "graph_traversal",
            },
        ]

        return results[:top_k]

    def demonstrate_conversational_memory(self, session_id: str) -> Dict[str, Any]:
        """Show how conversational memory enhances responses.

        Args:
            session_id: Conversation session identifier

        Returns:
            Memory context for the session
        """
        print(f"Retrieving conversational memory for session: {session_id}")

        # Simulated memory context
        return {
            "recent_entities": ["Machine Learning", "Python", "Data Science"],
            "conversation_topics": ["AI fundamentals", "Programming"],
            "user_preferences": {"detail_level": "intermediate"},
            "session_length": 15,  # minutes
        }


def main() -> None:
    """Run the advanced GraphRAG examples."""
    example = GraphRAGExample()

    print("=== Advanced GraphRAG Examples ===\n")

    # Entity extraction example
    sample_text = """
    Machine learning is a powerful subset of artificial intelligence that enables 
    computers to learn and improve from experience without being explicitly programmed.
    Neural networks, inspired by the human brain, are a key technology in deep learning.
    """

    entities = example.demonstrate_entity_extraction(sample_text)
    print(f"Extracted entities: {entities}\n")

    # Hybrid retrieval example
    query = "How do neural networks work in machine learning?"
    results = example.demonstrate_hybrid_retrieval(query)

    print("Hybrid retrieval results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. [{result['method']}] {result['content'][:80]}...")
        print(f"     Score: {result['score']}, Source: {result['source']}\n")

    # Conversational memory example
    memory = example.demonstrate_conversational_memory("user-123")
    print(f"Conversational memory: {memory}")


if __name__ == "__main__":
    main()
