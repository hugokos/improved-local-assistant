#!/usr/bin/env python3
"""
Download prebuilt knowledge graphs for the Improved Local Assistant.

This script downloads compressed knowledge graph archives from GitHub releases
or other configured sources and extracts them to the appropriate directories.
"""

import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

# Configuration
GRAPHS_CONFIG = {
    "source": "github_releases",  # github_releases, s3, direct_url
    "repository": "hugokos/improved-local-assistant-graphs",
    "release_tag": "graphs-v1.0.0",
    "base_url": "https://github.com/hugokos/improved-local-assistant-graphs/releases/download",
    "graphs_dir": "./data/graphs",
    "available_graphs": [
        {
            "name": "survivalist",
            "filename": "survivalist-knowledge-v1.0.tar.gz",
            "description": "Survivalist and outdoor knowledge base",
            "size": "45MB",
            "entities": "2,847",
            "relationships": "8,234",
            "checksum": "sha256:abc123...",
        }
    ],
}


def print_banner():
    """Print download banner."""
    print("📦 Improved Local Assistant - Knowledge Graph Downloader")
    print("=" * 60)


def list_available_graphs():
    """List all available knowledge graphs."""
    print("\n🗂️  Available Knowledge Graphs:")
    print("-" * 40)

    for i, graph in enumerate(GRAPHS_CONFIG["available_graphs"], 1):
        print(f"{i}. {graph['name'].title()}")
        print(f"   Description: {graph['description']}")
        print(
            f"   Size: {graph['size']} | Entities: {graph['entities']} | Relationships: {graph['relationships']}"
        )
        print()


def download_file(url: str, filepath: Path, description: str = "") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"📥 Downloading {description}...")
        print(f"   URL: {url}")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(
                            f"\r   Progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)",
                            end="",
                        )

        print(f"\n✅ Downloaded: {filepath}")
        return True

    except Exception as e:
        print(f"\n❌ Error downloading {description}: {e}")
        return False


def verify_checksum(filepath: Path, expected_checksum: str) -> bool:
    """Verify file checksum."""
    if not expected_checksum or expected_checksum == "sha256:abc123...":
        print("   ⚠️  Checksum verification skipped (not configured)")
        return True

    try:
        hash_type, expected_hash = expected_checksum.split(":", 1)

        if hash_type == "sha256":
            hasher = hashlib.sha256()
        else:
            print(f"   ⚠️  Unsupported hash type: {hash_type}")
            return True

        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        actual_hash = hasher.hexdigest()

        if actual_hash == expected_hash:
            print("   ✅ Checksum verified")
            return True
        else:
            print("   ❌ Checksum mismatch!")
            print(f"      Expected: {expected_hash}")
            print(f"      Actual:   {actual_hash}")
            return False

    except Exception as e:
        print(f"   ⚠️  Checksum verification failed: {e}")
        return True  # Don't fail download for checksum issues


def extract_archive(filepath: Path, extract_dir: Path) -> bool:
    """Extract compressed archive."""
    try:
        print(f"📂 Extracting {filepath.name}...")

        if filepath.suffix == ".gz" and filepath.stem.endswith(".tar"):
            # .tar.gz file
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(extract_dir)
        elif filepath.suffix == ".zip":
            # .zip file
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        else:
            print(f"   ⚠️  Unsupported archive format: {filepath.suffix}")
            return False

        print(f"   ✅ Extracted to: {extract_dir}")

        # Remove archive file after successful extraction
        filepath.unlink()
        print(f"   🗑️  Removed archive: {filepath.name}")

        return True

    except Exception as e:
        print(f"   ❌ Error extracting {filepath.name}: {e}")
        return False


def build_download_url(graph: dict) -> str:
    """Build download URL based on configuration."""
    config = GRAPHS_CONFIG

    if config["source"] == "github_releases":
        return f"{config['base_url']}/{config['release_tag']}/{graph['filename']}"
    elif config["source"] == "s3":
        return f"{config['base_url']}/{graph['filename']}"
    else:
        return f"{config['base_url']}/{graph['filename']}"


def download_graph(graph: dict, graphs_dir: Path) -> bool:
    """Download and extract a single knowledge graph."""
    print(f"\n🔄 Processing: {graph['name'].title()}")
    print(f"   {graph['description']}")

    # Create graphs directory
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # Build download URL
    url = build_download_url(graph)
    filepath = graphs_dir / graph["filename"]

    # Check if graph already exists
    graph_dir = graphs_dir / graph["name"]
    if graph_dir.exists():
        print(f"   ⚠️  Graph already exists: {graph_dir}")
        response = input("   Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("   ⏭️  Skipped")
            return True

    # Download file
    if not download_file(url, filepath, graph["name"]):
        return False

    # Verify checksum
    if not verify_checksum(filepath, graph.get("checksum", "")):
        print("   ⚠️  Continuing despite checksum issues...")

    # Extract archive
    if not extract_archive(filepath, graphs_dir):
        return False

    print(f"   ✅ Successfully installed: {graph['name']}")
    return True


def download_selected_graphs(graph_names: list[str]):
    """Download selected knowledge graphs."""
    graphs_dir = Path(GRAPHS_CONFIG["graphs_dir"])
    available_graphs = {g["name"]: g for g in GRAPHS_CONFIG["available_graphs"]}

    success_count = 0
    total_count = len(graph_names)

    for graph_name in graph_names:
        if graph_name not in available_graphs:
            print(f"\n❌ Unknown graph: {graph_name}")
            print(f"   Available graphs: {', '.join(available_graphs.keys())}")
            continue

        graph = available_graphs[graph_name]
        if download_graph(graph, graphs_dir):
            success_count += 1

    # Summary
    print("\n📊 Download Summary:")
    print(f"   ✅ Successful: {success_count}/{total_count}")
    print(f"   📁 Graphs directory: {graphs_dir.absolute()}")

    if success_count > 0:
        print("\n🚀 Ready to use! Start the application with:")
        print("   python run_app.py")


def main():
    """Main entry point."""
    print_banner()

    # Parse command line arguments
    if len(sys.argv) == 1:
        # Interactive mode
        list_available_graphs()

        print("📝 Select graphs to download:")
        print("   • Enter graph names separated by spaces (e.g., 'survivalist medical')")
        print("   • Enter 'all' to download all graphs")
        print("   • Enter 'list' to see available graphs again")
        print("   • Press Enter to exit")

        while True:
            selection = input("\n> ").strip().lower()

            if not selection:
                print("👋 Goodbye!")
                return
            elif selection == "list":
                list_available_graphs()
                continue
            elif selection == "all":
                graph_names = [g["name"] for g in GRAPHS_CONFIG["available_graphs"]]
                break
            else:
                graph_names = selection.split()
                break

    elif sys.argv[1] in ["--help", "-h"]:
        print("\nUsage:")
        print("  python scripts/download_graphs.py [graph_names...]")
        print("  python scripts/download_graphs.py all")
        print("  python scripts/download_graphs.py --list")
        print("\nExamples:")
        print("  python scripts/download_graphs.py survivalist")
        print("  python scripts/download_graphs.py survivalist medical")
        print("  python scripts/download_graphs.py all")
        return

    elif sys.argv[1] == "--list":
        list_available_graphs()
        return

    elif sys.argv[1] == "all":
        graph_names = [g["name"] for g in GRAPHS_CONFIG["available_graphs"]]

    else:
        graph_names = sys.argv[1:]

    # Download selected graphs
    download_selected_graphs(graph_names)


if __name__ == "__main__":
    main()
