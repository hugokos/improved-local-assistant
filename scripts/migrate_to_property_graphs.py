#!/usr/bin/env python3
"""
Migration script to convert existing simple graphs to property graphs.

This script converts existing triples.json files to the new kg.json property graph format
and updates metadata to indicate the graph type.
"""

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_triples_to_property_graph(triples: list[tuple[str, str, str]]) -> dict[str, Any]:
    """
    Convert flat triples to property graph format.

    Args:
        triples: List of (subject, predicate, object) tuples

    Returns:
        Property graph data with nodes and edges
    """
    nodes = {}
    edges = []
    entity_to_id = {}

    # Create nodes for all unique entities
    entities = set()
    for subj, pred, obj in triples:
        entities.add(subj)
        entities.add(obj)

    for entity in entities:
        node_id = str(uuid.uuid4())
        entity_to_id[entity] = node_id

        nodes[node_id] = {
            "label": "ENTITY",
            "properties": {
                "original_name": entity,
                "canonical_name": entity.lower().strip(),
                "context_count": 0,  # Would need to be calculated from actual context
                "migrated_from_triples": True,
            },
        }

    # Create edges for all relationships
    for subj, pred, obj in triples:
        src_id = entity_to_id[subj]
        tgt_id = entity_to_id[obj]

        edges.append(
            {
                "source_id": src_id,
                "target_id": tgt_id,
                "label": pred,
                "properties": {
                    "src_display": subj,
                    "tgt_display": obj,
                    "chunk_id": 0,  # Default chunk_id for migrated data
                    "migrated_from_triples": True,
                },
            }
        )

    return {"nodes": nodes, "edges": edges}


def migrate_graph_directory(graph_path: Path) -> bool:
    """
    Migrate a single graph directory from simple to property format.

    Args:
        graph_path: Path to the graph directory

    Returns:
        True if migration was successful
    """
    logger.info(f"Migrating graph: {graph_path.name}")

    try:
        # Check if already migrated
        kg_json_path = graph_path / "kg.json"
        graph_meta_path = graph_path / "graph_meta.json"

        if kg_json_path.exists() and graph_meta_path.exists():
            with open(graph_meta_path, encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("graph_type") == "property":
                    logger.info("  ✅ Already migrated to property graph")
                    return True

        # Look for triples.json
        triples_path = graph_path / "triples.json"
        if not triples_path.exists():
            logger.warning("  ⚠️ No triples.json found, skipping")
            return False

        # Load triples
        with open(triples_path, encoding="utf-8") as f:
            triples_data = json.load(f)

        # Convert to tuples if needed
        if triples_data and isinstance(triples_data[0], list):
            triples = [(t[0], t[1], t[2]) for t in triples_data if len(t) >= 3]
        else:
            logger.warning(f"  ⚠️ Unexpected triples format in {triples_path}")
            return False

        logger.info(f"  Converting {len(triples)} triples to property graph...")

        # Convert to property graph format
        property_graph = convert_triples_to_property_graph(triples)

        # Write kg.json
        with open(kg_json_path, "w", encoding="utf-8") as f:
            json.dump(property_graph, f, ensure_ascii=False, indent=2)

        # Create/update graph_meta.json
        graph_meta = {
            "graph_type": "property",
            "migrated_from": "simple",
            "migration_date": "2025-01-03",
            "nodes_count": len(property_graph["nodes"]),
            "edges_count": len(property_graph["edges"]),
            "original_triples_count": len(triples),
        }

        with open(graph_meta_path, "w", encoding="utf-8") as f:
            json.dump(graph_meta, f, ensure_ascii=False, indent=2)

        # Keep original triples.json for compatibility
        logger.info("  ✅ Migrated successfully:")
        logger.info(f"     - Nodes: {len(property_graph['nodes'])}")
        logger.info(f"     - Edges: {len(property_graph['edges'])}")
        logger.info(f"     - Original triples: {len(triples)}")

        return True

    except Exception as e:
        logger.error(f"  ❌ Migration failed: {e}")
        return False


def migrate_all_graphs(graphs_dir: Path) -> bool:
    """
    Migrate all graphs in the prebuilt graphs directory.

    Args:
        graphs_dir: Path to the prebuilt graphs directory

    Returns:
        True if all migrations were successful
    """
    if not graphs_dir.exists():
        logger.error(f"Graphs directory not found: {graphs_dir}")
        return False

    logger.info(f"Scanning for graphs in: {graphs_dir}")

    success_count = 0
    total_count = 0

    for graph_path in graphs_dir.iterdir():
        if not graph_path.is_dir():
            continue

        total_count += 1
        if migrate_graph_directory(graph_path):
            success_count += 1

    logger.info(f"Migration complete: {success_count}/{total_count} graphs migrated")
    return success_count == total_count


def main():
    """Main migration function."""

    logger.info("🔄 Starting Property Graph Migration")
    logger.info("=" * 50)

    # Default paths
    graphs_dir = Path("./data/prebuilt_graphs")

    # Allow custom path via command line
    if len(sys.argv) > 1:
        graphs_dir = Path(sys.argv[1])

    logger.info(f"Migrating graphs in: {graphs_dir}")

    try:
        success = migrate_all_graphs(graphs_dir)

        if success:
            logger.info("🎉 Migration completed successfully!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Update config.yaml to set graph.type: property")
            logger.info("2. Run test_property_graph_conversion.py to verify")
            logger.info("3. Rebuild graph registry with build_graph_registry.py")
            return 0
        else:
            logger.error("❌ Migration completed with errors")
            return 1

    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
