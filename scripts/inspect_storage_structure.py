#!/usr/bin/env python3
"""
Simple script to inspect the current storage structure and data format.

This script shows:
1. How data is stored in the filesystem
2. The structure of JSON storage files
3. Current graph statistics
4. Storage file sizes and growth patterns
"""

import builtins
import contextlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def inspect_json_file(file_path: Path, max_preview_size: int = 1000):
    """Inspect a JSON file and show its structure."""
    try:
        size = file_path.stat().st_size
        modified = datetime.fromtimestamp(file_path.stat().st_mtime)

        print(f"    📄 {file_path.name}")
        print(f"       Size: {format_size(size)}")
        print(f"       Modified: {modified}")

        if size == 0:
            print("       Content: Empty file")
            return

        if size > 10 * 1024 * 1024:  # 10MB
            print("       Content: File too large to inspect")
            return

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            print("       Content: Empty JSON")
            return

        try:
            data = json.loads(content)

            if isinstance(data, dict):
                keys = list(data.keys())
                print(f"       Structure: Dict with {len(keys)} keys")
                if keys:
                    print(f"       Top keys: {keys[:5]}")

                # Special handling for known file types
                if file_path.name == "docstore.json":
                    docs = data.get("docstore/data", {})
                    print(f"       Documents: {len(docs)}")

                elif file_path.name == "graph_store.json":
                    graph_data = data.get("graph_store/data", {})
                    rel_map = graph_data.get("rel_map", {})
                    total_relations = sum(len(rels) for rels in rel_map.values())
                    print(f"       Entities: {len(rel_map)}")
                    print(f"       Relations: {total_relations}")

                elif file_path.name == "property_graph_store.json":
                    nodes = data.get("nodes", [])
                    edges = data.get("edges", [])
                    print(f"       Nodes: {len(nodes)}")
                    print(f"       Edges: {len(edges)}")

                    if nodes:
                        node_types = set()
                        for node in nodes[:10]:  # Sample first 10
                            if isinstance(node, dict) and "type" in node:
                                node_types.add(node["type"])
                        if node_types:
                            print(f"       Node types: {list(node_types)}")

                elif file_path.name == "index_store.json":
                    indices = data.get("index_store/data", {})
                    print(f"       Indices: {len(indices)}")
                    for idx_id, idx_data in list(indices.items())[:3]:
                        idx_type = idx_data.get("__type__", "unknown")
                        print(f"         - {idx_id}: {idx_type}")

                elif "vector_store" in file_path.name:
                    vector_data = data.get("vector_store/data", {})
                    embedding_dict = vector_data.get("embedding_dict", {})
                    print(f"       Vectors: {len(embedding_dict)}")

                    if embedding_dict:
                        # Sample a vector to get dimension
                        sample_vector = next(iter(embedding_dict.values()), [])
                        if sample_vector:
                            print(f"       Vector dimension: {len(sample_vector)}")

            elif isinstance(data, list):
                print(f"       Structure: List with {len(data)} items")
                if data and isinstance(data[0], dict):
                    sample_keys = list(data[0].keys())
                    print(f"       Item keys: {sample_keys[:5]}")

            else:
                print(f"       Structure: {type(data).__name__}")

        except json.JSONDecodeError as e:
            print(f"       Content: Invalid JSON - {e}")
            # Show first few characters
            preview = content[:max_preview_size]
            print(f"       Preview: {preview}...")

    except Exception as e:
        print(f"    ❌ {file_path.name}: Error - {e}")


def inspect_directory(directory: Path, title: str):
    """Inspect a directory and its contents."""
    print(f"\n{'='*60}")
    print(f"📁 {title}")
    print(f"   Path: {directory}")
    print(f"{'='*60}")

    if not directory.exists():
        print("   ❌ Directory does not exist")
        return

    # Get all files and subdirectories
    items = list(directory.iterdir())
    files = [item for item in items if item.is_file()]
    subdirs = [item for item in items if item.is_dir()]

    if not items:
        print("   📂 Empty directory")
        return

    # Show subdirectories
    if subdirs:
        print(f"\n   📂 Subdirectories ({len(subdirs)}):")
        for subdir in sorted(subdirs):
            item_count = len(list(subdir.iterdir())) if subdir.exists() else 0
            print(f"      📁 {subdir.name}/ ({item_count} items)")

    # Show files
    if files:
        print(f"\n   📄 Files ({len(files)}):")
        total_size = 0

        for file_path in sorted(files):
            if file_path.suffix == ".json":
                inspect_json_file(file_path)
            else:
                try:
                    size = file_path.stat().st_size
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    print(f"    📄 {file_path.name}")
                    print(f"       Size: {format_size(size)}")
                    print(f"       Modified: {modified}")
                except Exception as e:
                    print(f"    ❌ {file_path.name}: Error - {e}")

            with contextlib.suppress(builtins.BaseException):
                total_size += file_path.stat().st_size

        print(f"\n   📊 Total size: {format_size(total_size)}")


def show_storage_overview():
    """Show overview of all storage locations."""
    print("🔍 IMPROVED LOCAL ASSISTANT - STORAGE STRUCTURE INSPECTION")
    print("=" * 80)

    # Main data directory
    data_path = Path("data")
    if data_path.exists():
        print(f"\n📁 Main Data Directory: {data_path.absolute()}")

        # List all subdirectories
        subdirs = [item for item in data_path.iterdir() if item.is_dir()]
        files = [item for item in data_path.iterdir() if item.is_file()]

        if subdirs:
            print(f"   📂 Subdirectories: {[d.name for d in subdirs]}")
        if files:
            print(f"   📄 Files: {[f.name for f in files]}")

    # Inspect key directories
    directories_to_inspect = [
        (Path("data/dynamic_graph"), "Dynamic Graph Storage"),
        (Path("data/dynamic_graph/main"), "Dynamic Graph Main Index"),
        (Path("data/prebuilt_graphs"), "Prebuilt Graphs"),
        (Path("data/prebuilt_graphs/survivalist"), "Survivalist Graph"),
        (Path("data/sessions"), "Session Storage"),
    ]

    for directory, title in directories_to_inspect:
        inspect_directory(directory, title)

    # Show registry files
    registry_files = [
        Path("data/graph_registry.json"),
        Path("data/kg_cache.json"),
    ]

    print(f"\n{'='*60}")
    print("📋 Registry and Cache Files")
    print(f"{'='*60}")

    for file_path in registry_files:
        if file_path.exists():
            inspect_json_file(file_path)
        else:
            print(f"    ❌ {file_path.name}: Not found")


def show_storage_summary():
    """Show a summary of storage usage."""
    print(f"\n{'='*60}")
    print("📊 STORAGE SUMMARY")
    print(f"{'='*60}")

    # Calculate total storage usage
    data_path = Path("data")
    if not data_path.exists():
        print("❌ Data directory not found")
        return

    total_size = 0
    file_count = 0
    dir_count = 0

    def calculate_size(path: Path):
        nonlocal total_size, file_count, dir_count

        try:
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1
            elif path.is_dir():
                dir_count += 1
                for item in path.iterdir():
                    calculate_size(item)
        except Exception as e:
            logger.debug(f"Error calculating size for {path}: {e}")

    calculate_size(data_path)

    print(f"📁 Total directories: {dir_count}")
    print(f"📄 Total files: {file_count}")
    print(f"💾 Total storage used: {format_size(total_size)}")

    # Break down by major categories
    categories = {
        "Dynamic Graph": Path("data/dynamic_graph"),
        "Prebuilt Graphs": Path("data/prebuilt_graphs"),
        "Sessions": Path("data/sessions"),
        "Other": None,  # Will be calculated as remainder
    }

    category_sizes = {}
    other_size = total_size

    for category, path in categories.items():
        if path and path.exists():
            cat_size = 0
            cat_files = 0

            def calc_cat_size(p: Path):
                nonlocal cat_size, cat_files
                try:
                    if p.is_file():
                        cat_size += p.stat().st_size
                        cat_files += 1
                    elif p.is_dir():
                        for item in p.iterdir():
                            calc_cat_size(item)
                except Exception:
                    pass

            calc_cat_size(path)
            category_sizes[category] = (cat_size, cat_files)
            other_size -= cat_size

    category_sizes["Other"] = (other_size, 0)

    print("\n📊 Storage breakdown:")
    for category, (size, files) in category_sizes.items():
        if size > 0:
            percentage = (size / total_size * 100) if total_size > 0 else 0
            print(f"   {category}: {format_size(size)} ({percentage:.1f}%) - {files} files")


def main():
    """Main inspection function."""
    try:
        show_storage_overview()
        show_storage_summary()

        print(f"\n{'='*80}")
        print("✅ Storage inspection completed")
        print(f"{'='*80}")

    except Exception as e:
        logger.error(f"Error during inspection: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
