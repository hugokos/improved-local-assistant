#!/usr/bin/env python3
"""
Build graph registry with embeddings for fast semantic routing.

This script scans the prebuilt graphs directory and creates a registry
with graph metadata and embedding vectors for fast routing.
"""

import asyncio
import json
import logging
import os
import pathlib
import sys
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def build_registry():
    """Build the graph registry with embeddings."""
    
    print("🔧 Building Graph Registry")
    print("=" * 50)
    
    try:
        # Initialize embedder
        print("Initializing embedder...")
        embedder = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
            normalize=True,
            embed_batch_size=10
        )
        
        # Scan prebuilt graphs directory
        graphs_dir = pathlib.Path("./data/prebuilt_graphs")
        registry = {}
        
        if not graphs_dir.exists():
            print(f"❌ Graphs directory not found: {graphs_dir}")
            return False
        
        print(f"Scanning graphs in: {graphs_dir}")
        
        # Process each graph directory
        for graph_path in graphs_dir.iterdir():
            if not graph_path.is_dir():
                continue
            
            graph_name = graph_path.name
            print(f"Processing graph: {graph_name}")
            
            try:
                # Read metadata if available
                meta_file = graph_path / "meta.json"
                graph_meta_file = graph_path / "graph_meta.json"
                
                meta = {}
                graph_type = "simple"  # default
                
                if meta_file.exists():
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                
                if graph_meta_file.exists():
                    with open(graph_meta_file, 'r', encoding='utf-8') as f:
                        graph_meta = json.load(f)
                        graph_type = graph_meta.get("graph_type", "simple")
                        meta.update(graph_meta)
                
                # Read README if available for description
                readme_file = graph_path / "README.md"
                description = ""
                if readme_file.exists():
                    with open(readme_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract first paragraph as description
                        lines = content.split('\n')
                        for line in lines:
                            if line.strip() and not line.startswith('#'):
                                description = line.strip()
                                break
                
                # For property graphs, try to use node embeddings from kg.json
                kg_file = graph_path / "kg.json"
                node_embeddings = []
                
                if graph_type == "property" and kg_file.exists():
                    try:
                        with open(kg_file, 'r', encoding='utf-8') as f:
                            kg_data = json.load(f)
                        
                        # Extract entity names from nodes for embedding
                        if "nodes" in kg_data:
                            entity_names = []
                            for node_id, node in kg_data["nodes"].items():
                                if "properties" in node:
                                    name = node["properties"].get("original_name") or node["properties"].get("name", "")
                                    if name:
                                        entity_names.append(name)
                            
                            if entity_names:
                                # Use top entities for graph representation
                                embedding_text = f"{graph_name} contains: {', '.join(entity_names[:10])}"
                                print(f"  Using property graph entities for embedding: {len(entity_names)} entities")
                            else:
                                embedding_text = f"{graph_name} {description}".strip()
                        else:
                            embedding_text = f"{graph_name} {description}".strip()
                    except Exception as e:
                        print(f"  Warning: Could not read kg.json for {graph_name}: {e}")
                        embedding_text = f"{graph_name} {description}".strip()
                else:
                    # Create embedding text from graph name and description
                    embedding_text = f"{graph_name} {description}".strip()
                
                if not embedding_text:
                    embedding_text = graph_name
                
                print(f"  Generating embedding for: {embedding_text[:100]}...")
                
                # Generate embedding
                embedding = embedder.get_text_embedding(embedding_text)
                
                # Build registry entry
                registry_entry = {
                    "vector": embedding,
                    "description": description,
                    "embedding_text": embedding_text,
                    "path": str(graph_path),
                    "graph_type": graph_type,
                    "updated_at": datetime.now().isoformat(),
                    "meta": meta
                }
                
                # Add graph statistics if available
                if "nodes" in meta or "edges" in meta:
                    registry_entry["stats"] = {
                        "nodes": meta.get("nodes", 0),
                        "edges": meta.get("edges", 0)
                    }
                
                registry[graph_name] = registry_entry
                print(f"  ✅ Added {graph_name} to registry")
                
            except Exception as e:
                print(f"  ❌ Error processing {graph_name}: {e}")
                continue
        
        if not registry:
            print("❌ No graphs found to add to registry")
            return False
        
        # Save registry
        registry_path = "data/graph_registry.json"
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Registry saved to: {registry_path}")
        print(f"📊 Registry contains {len(registry)} graphs:")
        
        for name, entry in registry.items():
            desc = entry.get("description", "No description")[:50]
            print(f"  - {name}: {desc}...")
        
        # Validate registry
        print("\n🔍 Validating registry...")
        
        # Check vector dimensions
        vector_dims = set()
        for name, entry in registry.items():
            if "vector" in entry:
                vector_dims.add(len(entry["vector"]))
        
        if len(vector_dims) == 1:
            dim = list(vector_dims)[0]
            print(f"✅ All vectors have consistent dimension: {dim}")
        else:
            print(f"❌ Inconsistent vector dimensions: {vector_dims}")
            return False
        
        print("✅ Registry validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error building registry: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_registry():
    """Test the built registry."""
    
    print("\n🧪 Testing Registry")
    print("=" * 30)
    
    try:
        # Load registry
        registry_path = "data/graph_registry.json"
        if not os.path.exists(registry_path):
            print("❌ Registry file not found")
            return False
        
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        print(f"📊 Loaded registry with {len(registry)} graphs")
        
        # Test routing
        from services.graph_router import GraphRouter
        
        config = {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "registry_path": registry_path,
            "data_path": "data/prebuilt_graphs"
        }
        
        router = GraphRouter(config)
        
        if not await router.initialize():
            print("❌ Failed to initialize router")
            return False
        
        # Test queries
        test_queries = [
            "survival techniques",
            "finding water in wilderness",
            "edible plants",
            "building shelter"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: {query}")
            
            top_graphs, scores = router.route(query, k=3)
            
            if top_graphs:
                print("  Top matches:")
                for graph, score in zip(top_graphs, scores):
                    print(f"    {graph}: {score:.3f}")
            else:
                print("  No matches found")
        
        await router.close()
        
        print("\n✅ Registry test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error testing registry: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    
    # Build registry
    build_success = await build_registry()
    
    if build_success:
        # Test registry
        test_success = await test_registry()
        
        if test_success:
            print("\n🎉 Graph registry built and tested successfully!")
            print("The GraphRouter is ready for production use.")
        else:
            print("\n⚠️  Registry built but testing failed")
    else:
        print("\n❌ Failed to build registry")
    
    return build_success


if __name__ == "__main__":
    asyncio.run(main())